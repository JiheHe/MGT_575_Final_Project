from __future__ import annotations

import html
import re


def normalize_interests(raw_interests: str) -> list[str]:
    parts = [item.strip().lower() for item in raw_interests.split(",")]
    cleaned = [re.sub(r"\s+", " ", item) for item in parts if item]
    # Preserve order, remove duplicates.
    deduped: list[str] = []
    seen: set[str] = set()
    for item in cleaned:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped


def strip_html(text: str) -> str:
    raw = html.unescape(text or "")
    # Remove tags and collapse whitespace.
    no_tags = re.sub(r"<[^>]+>", " ", raw)
    collapsed = re.sub(r"\s+", " ", no_tags).strip()
    return collapsed


def clean_news_text(text: str, max_len: int = 1200) -> str:
    cleaned = strip_html(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rsplit(" ", 1)[0]
    return cleaned


def is_likely_code_noise(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    noisy_markers = [
        "window.wiz_global_data",
        "subscribewithgoogle",
        "function(",
        "__next_data__",
        "webpack",
        "{\"error\":",
        "var ",
        "=>",
    ]
    if any(marker in t for marker in noisy_markers):
        return True
    symbol_count = len(re.findall(r"[\{\}\[\]<>_=;\"']", t))
    alpha_count = len(re.findall(r"[a-z]", t))
    if alpha_count == 0:
        return True
    return symbol_count > alpha_count * 0.35


def looks_like_breadcrumb_title(title: str) -> bool:
    t = (title or "").strip()
    return t.count(">") >= 2 or len(t) > 140


def fallback_key_points(snippet: str, *, max_points: int = 5) -> list[str]:
    text = clean_news_text(snippet).strip()
    if not text:
        return ["Source snippet is limited; details may be incomplete."]
    clauses = [seg.strip(" .") for seg in re.split(r"[.;!?]", text) if seg.strip()]
    cleaned_clauses: list[str] = []
    for clause in clauses:
        if len(clause) < 24:
            continue
        if re.fullmatch(r"[\d,\-\s]+", clause):
            continue
        cleaned_clauses.append(clause)
    points = cleaned_clauses[:max_points] if cleaned_clauses else [text]
    return [f"- {p}" if not p.startswith("-") else p for p in points]
