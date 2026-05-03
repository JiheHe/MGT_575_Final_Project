from __future__ import annotations

from datetime import datetime, timezone

import feedparser
import requests
from urllib.parse import parse_qs, unquote, urlparse

from src.config import AppConfig
from src.models import ArticleCandidate, UserProfile
from src.utils.text_utils import clean_news_text, is_likely_code_noise, strip_html


class NewsRetrieverAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, profile: UserProfile) -> tuple[list[ArticleCandidate], dict[str, object]]:
        # Generic discovery sources (interest-driven query), no fixed publisher whitelist.
        google, google_meta = self._google_news_rss_search(profile)
        bing, bing_meta = self._bing_news_rss_search(profile)
        merged = _merge_dedupe(google + bing, limit=profile.candidate_pool_size)
        dedup_dropped = len(google) + len(bing) - len(merged)
        if merged:
            return merged, {
                "mode": "live_generic_rss",
                "requested_pool": profile.candidate_pool_size,
                "count": len(merged),
                "dedup_dropped": dedup_dropped,
                "sources": {
                    "google_rss": google_meta,
                    "bing_rss": bing_meta,
                },
            }
        return [], {
            "mode": "live_search_failed",
            "requested_pool": profile.candidate_pool_size,
            "count": 0,
            "sources": {
                "google_rss": google_meta,
                "bing_rss": bing_meta,
            },
        }

    def _google_news_rss_search(self, profile: UserProfile) -> tuple[list[ArticleCandidate], dict[str, object]]:
        hint = (profile.search_rss_query_hint or "").strip()
        query = hint if hint else " OR ".join(profile.interests)
        rss_url = (
            "https://news.google.com/rss/search?"
            f"q={query.replace(' ', '+')}+when:7d&hl=en-{profile.region}&gl={profile.region}&ceid={profile.region}:en"
        )
        feed = feedparser.parse(rss_url)
        results: list[ArticleCandidate] = []
        resolved_redirect = 0
        unresolved_redirect = 0
        dropped_empty = 0
        max_scan = profile.candidate_pool_size * 5
        for entry in feed.entries[:max_scan]:
            title = clean_news_text(strip_html(getattr(entry, "title", "").strip()), max_len=220)
            raw_link = getattr(entry, "link", "").strip()
            link = raw_link
            publisher_url_resolved: bool | None = None
            if _is_google_redirect_url(raw_link):
                resolved = _resolve_google_redirect_url(raw_link)
                if resolved and not _is_google_redirect_url(resolved):
                    link = resolved
                    publisher_url_resolved = True
                    resolved_redirect += 1
                else:
                    unresolved_redirect += 1
                    continue
            source = "Google News RSS"
            if hasattr(entry, "source") and getattr(entry.source, "title", None):
                source = str(entry.source.title)
            if source.strip().lower() in {"google news rss", "google news"}:
                derived = _friendly_source_from_url(link)
                if derived:
                    source = derived
            snippet_raw = getattr(entry, "summary", "").strip()
            snippet = clean_news_text(strip_html(snippet_raw), max_len=600) or "Summary unavailable."
            if is_likely_code_noise(snippet):
                snippet = "Summary unavailable from feed; using headline context."
            published_date = (getattr(entry, "published", "") or "").strip()[:16] or datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M")
            topic = profile.interests[0] if profile.interests else "general"
            if title and link:
                results.append(
                    ArticleCandidate(
                        title=title,
                        source=source,
                        url=link,
                        raw_url=raw_link,
                        published_date=published_date,
                        snippet=snippet,
                        topic=topic,
                        rss_found=True,
                        source_feed="google_rss",
                        publisher_url_resolved=publisher_url_resolved,
                    )
                )
                if len(results) >= profile.candidate_pool_size:
                    break
            else:
                dropped_empty += 1
        return results, {
            "feed_entries": len(feed.entries),
            "scanned": min(len(feed.entries), max_scan),
            "kept": len(results),
            "resolved_google_redirect": resolved_redirect,
            "dropped_unresolved_google_redirect": unresolved_redirect,
            "dropped_empty": dropped_empty,
        }

    def _bing_news_rss_search(self, profile: UserProfile) -> tuple[list[ArticleCandidate], dict[str, object]]:
        hint = (profile.search_rss_query_hint or "").strip()
        if not hint:
            return [], {
                "feed_entries": 0,
                "scanned": 0,
                "kept": 0,
                "dropped_unresolved_apiclick": 0,
                "dropped_code_noise": 0,
                "dropped_empty": 0,
                "query_mode": "strict_hint_only_no_fallback",
                "skipped_reason": "missing_search_rss_query_hint",
            }
        query = hint
        rss_url = f"https://www.bing.com/news/search?q={query.replace(' ', '+')}&format=RSS"
        results: list[ArticleCandidate] = []
        feed = feedparser.parse(rss_url)
        dropped_apiclick = 0
        dropped_noise = 0
        dropped_empty = 0
        max_scan = profile.candidate_pool_size * 5
        for entry in feed.entries[:max_scan]:
            title = clean_news_text(strip_html(getattr(entry, "title", "").strip()), max_len=220)
            raw_link = getattr(entry, "link", "").strip()
            link = _decode_bing_apiclick_url(raw_link) or raw_link
            resolved = bool(raw_link and link and raw_link != link and "bing.com/news/apiclick" in raw_link.lower())
            if not link or "bing.com/news/apiclick" in link.lower():
                dropped_apiclick += 1
                continue
            snippet_raw = getattr(entry, "summary", "").strip() or getattr(entry, "description", "").strip()
            snippet = clean_news_text(strip_html(snippet_raw), max_len=800) or "Summary unavailable."
            if is_likely_code_noise(snippet):
                dropped_noise += 1
                continue
            published_date = (getattr(entry, "published", "") or "").strip()[:16] or datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%dT%H:%M")
            topic = profile.interests[0] if profile.interests else "general"
            if title and link:
                results.append(
                    ArticleCandidate(
                        title=title,
                        source=_friendly_source_from_url(link) or "Bing News RSS",
                        url=link,
                        raw_url=raw_link,
                        published_date=published_date,
                        snippet=snippet,
                        topic=topic,
                        rss_found=True,
                        source_feed="bing_rss",
                        publisher_url_resolved=resolved,
                    )
                )
                if len(results) >= profile.candidate_pool_size:
                    break
            else:
                dropped_empty += 1
        return results, {
            "feed_entries": len(feed.entries),
            "scanned": min(len(feed.entries), max_scan),
            "kept": len(results),
            "dropped_unresolved_apiclick": dropped_apiclick,
            "dropped_code_noise": dropped_noise,
            "dropped_empty": dropped_empty,
            "query_mode": "strict_hint_only_no_fallback",
        }


def _friendly_source_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    host = host[4:] if host.startswith("www.") else host
    if not host:
        return ""
    if host.endswith("cnbc.com"):
        return "CNBC"
    if host.endswith("reuters.com"):
        return "Reuters"
    if host.endswith("techcrunch.com"):
        return "TechCrunch"
    if host.endswith("theverge.com"):
        return "The Verge"
    parts = host.split(".")
    if len(parts) >= 2 and parts[-2]:
        return parts[-2].replace("-", " ").title()
    return host


def _is_google_redirect_url(url: str) -> bool:
    low = (url or "").lower()
    return "news.google.com/rss/articles/" in low or "news.google.com/articles/" in low


def _decode_bing_apiclick_url(url: str) -> str:
    if "bing.com/news/apiclick" not in (url or "").lower():
        return url
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    embedded = qs.get("url", [])
    if not embedded:
        return url
    return unquote(embedded[0]).strip()


def _resolve_google_redirect_url(url: str, timeout_sec: int = 6) -> str | None:
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    for key in ("url", "u", "q"):
        value = qs.get(key, [])
        if value and value[0]:
            candidate = unquote(value[0]).strip()
            if candidate.startswith("http://") or candidate.startswith("https://"):
                return candidate
    try:
        # Best-effort: follow redirects and keep non-Google final URL.
        response = requests.get(
            url,
            timeout=timeout_sec,
            allow_redirects=True,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                )
            },
        )
        final_url = str(getattr(response, "url", "") or "").strip()
        if final_url and not _is_google_redirect_url(final_url) and "news.google.com/" not in final_url.lower():
            return final_url
    except (requests.RequestException, ValueError):
        return None
    return None


def _merge_dedupe(items: list[ArticleCandidate], *, limit: int) -> list[ArticleCandidate]:
    out: list[ArticleCandidate] = []
    seen: set[str] = set()
    for item in items:
        key = f"{item.url.strip().lower()}::{item.title.strip().lower()}"
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out
