from __future__ import annotations

import ast
import json

from src.gemini_client import GeminiClient
from src.models import RankedArticle, StorySummary, UserProfile
from src.utils.article_fetcher import fetch_article_text
from src.utils.display_text import (
    coerce_dict,
    humanize_single_line,
    normalize_loose_json_string_field,
)
from src.utils.text_utils import fallback_key_points, is_likely_code_noise


class SummarizerAgent:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client
        self.last_errors: list[dict[str, str]] = []

    def run(self, articles: list[RankedArticle], profile: UserProfile) -> list[StorySummary]:
        self.last_errors = []
        if not self.gemini_client.available:
            # No deterministic fallback-summary mode: summaries require LLM generation.
            self.last_errors.append({"reason": "gemini_unavailable"})
            return []
        results: list[StorySummary] = []
        for article in articles:
            article_text = article.article_text or fetch_article_text(article.url)
            if article_text and is_likely_code_noise(article_text):
                article_text = None
            try:
                results.append(self._summarize_with_gemini(article, article_text, profile))
            except (RuntimeError, ValueError, TypeError) as exc:
                # If one article fails JSON generation/parsing, skip it rather than emitting
                # a deterministic fallback summary.
                self.last_errors.append(
                    {
                        "title": (article.title or "")[:140],
                        "url": article.url,
                        "reason": str(exc)[:220],
                    }
                )
                continue
        return results

    def _summarize_with_gemini(
        self, article: RankedArticle, article_text: str | None, profile: UserProfile
    ) -> StorySummary:
        lens = (profile.combined_interest_context or "").strip()
        lens_line = f"Editorial lens (interests read as one bundle): {lens}\n" if lens else ""
        prompt = (
            "You are a grounded news summarizer. "
            f"The user's interests are: {profile.interests}. "
            f"{lens_line}"
            "Use your judgment to highlight what matters for those interests, grounded only in the provided "
            "title, snippet, and article text. Do not invent facts. "
            "When article text is available, write clear, direct bullet points. "
            "If details are sparse, say so explicitly. "
            "Provide deeper synthesis from the full article text, not just snippet paraphrase. "
            "why_it_matters must be 2-4 plain sentences (prose), never a JSON object or Python dict. "
            "content_summary_points and interest_summary_points must be JSON arrays of short strings "
            "(each bullet one string), not objects keyed by interest name. "
            "The number of bullets should be dynamic based on article depth and available evidence. "
            "Generate content_summary_points directly from core article substance. "
            "Generate interest_summary_points independently from the same article, focused on user interests; "
            "some overlap with content_summary_points is acceptable when justified. "
            "Return strict JSON with keys: headline, one_sentence_summary, content_summary_points, "
            "interest_summary_points, "
            "why_it_matters, uncertainty_or_limitations.\n\n"
            f"Title: {article.title}\n"
            f"Source: {article.source}\n"
            f"URL: {article.url}\n"
            f"Snippet: {article.snippet}\n"
            f"Article text: {article_text or 'Unavailable'}\n"
        )
        raw = self.gemini_client.generate_json(prompt)
        data = _coerce_summary_payload(raw)
        content_pts = _normalize_points_from_unknown(data.get("content_summary_points"))
        interest_pts = _normalize_points_from_unknown(data.get("interest_summary_points"))
        fallback_body = article_text or article.snippet
        if not content_pts:
            content_pts = fallback_key_points(fallback_body, max_points=6)
        if not interest_pts:
            interest_pts = _fallback_interest_points(fallback_body, profile.interests)
        return StorySummary(
            headline=humanize_single_line(data.get("headline", article.title), max_len=240) or article.title,
            one_sentence_summary=humanize_single_line(
                data.get("one_sentence_summary", article.snippet),
                max_len=320,
            )
            or article.snippet,
            content_summary_points=content_pts,
            interest_summary_points=interest_pts,
            why_it_matters=normalize_loose_json_string_field(
                data.get("why_it_matters", _fallback_why_it_matters(fallback_body, profile.interests))
            )
            or _fallback_why_it_matters(fallback_body, profile.interests),
            published_date=humanize_single_line(article.published_date or "", max_len=80),
            source_url=article.url,
            source_name=article.source,
            uncertainty_or_limitations=normalize_loose_json_string_field(
                data.get(
                    "uncertainty_or_limitations",
                    "Generated from limited source snippet; verify with original article.",
                )
            )
            or "Generated from limited source snippet; verify with original article.",
        )


def _fallback_interest_points(text: str, interests: list[str]) -> list[str]:
    """Offline fallback: try to map key sentences to user interests first."""
    if not interests:
        return fallback_key_points(text, max_points=5)[:5]
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    points: list[str] = []
    lower_sentences = [s.lower() for s in sentences]
    for interest in interests:
        interest_l = interest.lower().strip()
        if not interest_l:
            continue
        chosen = ""
        for idx, s in enumerate(lower_sentences):
            if interest_l in s:
                chosen = sentences[idx]
                break
        if chosen:
            points.append(f"- {interest}: {chosen}")
    if not points:
        return fallback_key_points(text, max_points=5)[:5]
    return points[:5]


def _fallback_why_it_matters(text: str, interests: list[str]) -> str:
    key_pts = fallback_key_points(text, max_points=2)
    joined_interests = ", ".join(interests) if interests else "your priorities"
    lead = (
        "This matters because it carries concrete implications for "
        f"{joined_interests}, including near-term execution signals and market positioning."
    )
    supporting = " ".join(p.removeprefix("- ").strip() for p in key_pts if p.strip())
    return f"{lead} {supporting}".strip()


def _normalize_points_from_unknown(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        t = raw.strip()
        if t.startswith("[") or t.startswith("{"):
            try:
                loaded = json.loads(t)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                return _normalize_points_from_unknown(loaded)
            if isinstance(loaded, list):
                return _normalize_points(loaded)
    parsed_dict = coerce_dict(raw)
    if parsed_dict:
        pairs = [
            f"{k}: {normalize_loose_json_string_field(v)}"
            for k, v in parsed_dict.items()
            if normalize_loose_json_string_field(v)
        ]
        return _normalize_points(pairs)
    if isinstance(raw, list):
        return _normalize_points(raw)
    if isinstance(raw, str) and raw.strip():
        parsed_again = coerce_dict(raw)
        if parsed_again:
            return _normalize_points_from_unknown(parsed_again)
        return _normalize_points([raw])
    return []


def _coerce_summary_payload(raw: object) -> dict[str, object]:
    """
    Gemini may occasionally return a top-level list/string despite prompt instructions.
    Normalize to a dict so downstream `.get(...)` calls are safe.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        # Treat list as content points when model omitted object wrapper.
        pts = _normalize_points(raw)
        return {
            "headline": "",
            "one_sentence_summary": "",
            "content_summary_points": pts,
            "interest_summary_points": [],
            "why_it_matters": "",
            "uncertainty_or_limitations": "Model returned non-standard JSON payload.",
        }
    if isinstance(raw, str):
        parsed = coerce_dict(raw)
        if parsed:
            return parsed
        return {
            "headline": "",
            "one_sentence_summary": raw,
            "content_summary_points": [],
            "interest_summary_points": [],
            "why_it_matters": "",
            "uncertainty_or_limitations": "Model returned string payload instead of object.",
        }
    return {
        "headline": "",
        "one_sentence_summary": "",
        "content_summary_points": [],
        "interest_summary_points": [],
        "why_it_matters": "",
        "uncertainty_or_limitations": "Model returned unsupported JSON payload type.",
    }


def _strip_leading_bullet_markers(text: str) -> str:
    t = text.strip()
    while t:
        if t.startswith("- "):
            t = t[2:].strip()
            continue
        if t.startswith("--"):
            t = t[2:].strip()
            continue
        # Single leading dash as bullet marker, not a negative number (e.g. -660).
        if t.startswith("-") and len(t) > 1 and not t[1].isdigit():
            t = t[1:].strip()
            continue
        if t.startswith("•") or t.startswith("*"):
            t = t[1:].strip()
            continue
        break
    return t


def _normalize_points(raw_points: object) -> list[str]:
    if not isinstance(raw_points, list):
        return []
    normalized: list[str] = []
    for item in raw_points:
        text = _item_to_text(item)
        if text:
            body = _strip_leading_bullet_markers(text)
            if body:
                normalized.append(f"- {body}")
    return normalized


def _item_to_text(item: object) -> str:
    if isinstance(item, dict):
        interest = str(item.get("interest", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if interest and summary:
            return f"{interest}: {summary}"
        return summary or interest or ""
    if isinstance(item, str):
        candidate = item.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return _item_to_text(parsed)
            except (ValueError, SyntaxError):
                pass
        return candidate
    return str(item).strip()
