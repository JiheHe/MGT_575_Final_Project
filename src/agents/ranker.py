from __future__ import annotations

import re

from src.gemini_client import GeminiClient
from src.models import ArticleCandidate, RankedArticle, UserProfile


class RankingAgent:
    """Semantic article ranking via LLM on RSS metadata only (single pass)."""

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def run(
        self, articles: list[ArticleCandidate], profile: UserProfile, limit: int | None = None
    ) -> list[RankedArticle]:
        deduped = self._dedupe(articles)
        if self.gemini_client.available:
            try:
                ranked = self._rank_with_gemini(deduped, profile)
                return ranked[:limit] if limit else ranked
            except (RuntimeError, ValueError, TypeError):
                pass
        ranked = self._rank_with_boundary_fallback(deduped, profile)
        return ranked[:limit] if limit else ranked

    def _dedupe(self, articles: list[ArticleCandidate]) -> list[ArticleCandidate]:
        seen: set[str] = set()
        kept: list[ArticleCandidate] = []
        for article in articles:
            key = f"{article.url.strip().lower()}::{article.title.strip().lower()}"
            if key not in seen:
                seen.add(key)
                kept.append(article)
        return kept

    def _rank_with_boundary_fallback(
        self, articles: list[ArticleCandidate], profile: UserProfile
    ) -> list[RankedArticle]:
        """Only when Gemini is unavailable: word-boundary matches on interests (e.g. \\bai\\b not 'chai')."""
        patterns = [
            re.compile(rf"\b{re.escape(term.lower())}\b")
            for term in profile.interests
            if term.strip()
        ]
        ranked: list[RankedArticle] = []
        for article in articles:
            text = f"{article.title} {article.snippet} {article.topic}".lower()
            matches = sum(1 for p in patterns if p.search(text))
            score = float(matches * 20)
            ranked.append(
                RankedArticle(
                    **article.model_dump(),
                    relevance_score=score,
                    relevance_reason="Boundary-match fallback (LLM unavailable).",
                )
            )
        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked

    def _rank_with_gemini(
        self, articles: list[ArticleCandidate], profile: UserProfile
    ) -> list[RankedArticle]:
        """Uses ``GeminiClient.generate_json`` → same text model + ``GEMINI_TEXT_FALLBACK_MODELS`` as rest of app."""
        score_map: dict[str, tuple[float, str]] = {}
        for batch in _chunks(articles, 8):
            payload = [
                {
                    "url": a.url,
                    "title": a.title,
                    "source": a.source,
                    "snippet": a.snippet[:400],
                    "topic": a.topic,
                    "published_date": a.published_date,
                }
                for a in batch
            ]
            lens = (profile.combined_interest_context or "").strip()
            lens_block = f"How to read the interest bundle together: {lens}\n\n" if lens else ""
            prompt = (
                "You are a semantic ranking agent for a personalized news briefing. "
                "Score each article by relevance to the user's stated interests using your general knowledge "
                "of how topics relate (no synonym lists required). Prefer articles that genuinely connect to "
                "the user's interest bundle as a whole; penalize stories that are only tangentially related "
                "(e.g., a story whose topic merely shares a keyword with one interest but does not reinforce "
                "the rest of the bundle). Apply this judgment to whatever interests the user provides; do not "
                "assume any specific topic, industry, or vocabulary. "
                "Return strict JSON: {\"scored_articles\":[{"
                "\"url\",\"relevance_score_0_to_100\",\"matched_interests\",\"relevance_reason\"}]} "
                "relevance_score_0_to_100 is 0-100. matched_interests is an array of which user interests "
                "the article substantively supports.\n\n"
                f"{lens_block}"
                f"Interests: {profile.interests}\n"
                f"Articles: {payload}"
            )
            data = self.gemini_client.generate_json(prompt)
            for item in data.get("scored_articles", []):
                url = str(item.get("url", ""))
                relevance = float(item.get("relevance_score_0_to_100", 0.0))
                matched = item.get("matched_interests", [])
                matched_text = ", ".join(str(x) for x in matched) if isinstance(matched, list) else str(matched)
                reason = str(item.get("relevance_reason", "Gemini semantic ranking"))
                score_map[url] = (
                    relevance,
                    f"{reason} | matched_interests: {matched_text}",
                )
        ranked: list[RankedArticle] = []
        for article in articles:
            score, reason = score_map.get(article.url, (0.0, "Not scored by model"))
            ranked.append(
                RankedArticle(
                    **article.model_dump(),
                    relevance_score=score,
                    relevance_reason=reason,
                )
            )
        ranked.sort(key=lambda item: item.relevance_score, reverse=True)
        return ranked


def _chunks(items: list[ArticleCandidate], size: int) -> list[list[ArticleCandidate]]:
    return [items[i : i + size] for i in range(0, len(items), size)]
