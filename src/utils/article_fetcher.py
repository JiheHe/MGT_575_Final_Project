from __future__ import annotations

from html.parser import HTMLParser
import requests

from src.utils.text_utils import clean_news_text, is_likely_code_noise

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover
    BeautifulSoup = None  # type: ignore
try:
    import trafilatura
except ImportError:  # pragma: no cover
    trafilatura = None  # type: ignore


def fetch_article_text(url: str, timeout_sec: int = 8) -> str | None:
    text, _ = fetch_article_text_with_reason(url, timeout_sec=timeout_sec)
    return text


def fetch_article_text_with_reason(url: str, timeout_sec: int = 8) -> tuple[str | None, str]:
    try:
        response = requests.get(
            url,
            timeout=timeout_sec,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36"
                )
            },
        )
        if response.status_code != 200:
            return None, f"http_status_{response.status_code}"
        trafilatura_text = _extract_with_trafilatura(response.text)
        if trafilatura_text:
            return trafilatura_text, "ok_trafilatura"
        if BeautifulSoup is None:
            fallback_text = _extract_with_htmlparser(response.text)
            if fallback_text:
                return fallback_text, "ok_htmlparser_fallback"
            return None, "bs4_missing"
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        containers = soup.find_all(["article", "main"])
        paragraphs: list[str] = []
        if containers:
            for container in containers:
                paragraphs.extend(
                    [clean_news_text(p.get_text(" ", strip=True), max_len=500) for p in container.find_all("p")]
                )
        else:
            paragraphs = [clean_news_text(p.get_text(" ", strip=True), max_len=500) for p in soup.find_all("p")]

        # Keep threshold moderate so valid short paragraphs are not discarded.
        useful = [p for p in paragraphs if len(p) > 35 and not is_likely_code_noise(p)]
        if not useful:
            total = len(paragraphs)
            too_short = sum(1 for p in paragraphs if len(p) <= 35)
            code_noise = sum(1 for p in paragraphs if p and is_likely_code_noise(p))
            empty = sum(1 for p in paragraphs if not p)
            return (
                None,
                (
                    "no_useful_paragraphs"
                    f"(total_p={total},too_short={too_short},code_noise={code_noise},empty={empty})"
                ),
            )
        # Keep substantially more content so downstream summaries are based on fuller context.
        joined = "\n".join(useful[:120])[:40000]
        if is_likely_code_noise(joined):
            return None, "code_noise_detected"
        return joined, "ok"
    except (requests.RequestException, ValueError):
        return None, "request_exception"


class _ParagraphHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_p = False
        self._curr: list[str] = []
        self.paragraphs: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        if tag.lower() == "p":
            self.in_p = True
            self._curr = []

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag.lower() == "p" and self.in_p:
            raw = "".join(self._curr).strip()
            cleaned = clean_news_text(raw, max_len=500)
            if len(cleaned) > 35 and not is_likely_code_noise(cleaned):
                self.paragraphs.append(cleaned)
            self.in_p = False
            self._curr = []

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self.in_p and data:
            self._curr.append(data)


def _extract_with_htmlparser(html: str) -> str | None:
    parser = _ParagraphHTMLParser()
    try:
        parser.feed(html)
    except (ValueError, RuntimeError):
        return None
    if not parser.paragraphs:
        return None
    joined = "\n".join(parser.paragraphs[:120])[:40000]
    if is_likely_code_noise(joined):
        return None
    return joined


def _extract_with_trafilatura(html: str) -> str | None:
    if trafilatura is None:
        return None
    try:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
    except (ValueError, RuntimeError, TypeError):
        return None
    text = clean_news_text((extracted or "").strip(), max_len=40000)
    if not text:
        return None
    # Keep large line budget; trafilatura can return dense single-line article bodies.
    paragraphs = [clean_news_text(p.strip(), max_len=8000) for p in text.splitlines() if p.strip()]
    useful = [p for p in paragraphs if len(p) > 35 and not is_likely_code_noise(p)]
    if not useful:
        return None
    joined = "\n".join(useful[:120])[:40000]
    if is_likely_code_noise(joined):
        return None
    return joined
