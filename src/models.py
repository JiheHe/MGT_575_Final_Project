from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    interests: list[str]
    region: str = "US"
    max_stories: int = 3
    candidate_pool_size: int = 20
    persona: str = "Lisa - professional anchor"
    tone: str = "professional"
    generate_images: bool = True
    generate_audio: bool = False
    generate_video: bool = False
    max_video_seconds: int = 60
    use_live_search: bool = False
    # Populated by InterestContextAgent: how to read the interest bundle together.
    interest_context_by_interest: dict[str, str] = Field(default_factory=dict)
    combined_interest_context: str = ""
    search_rss_query_hint: str = ""


class ArticleCandidate(BaseModel):
    title: str
    source: str
    url: str
    raw_url: str = ""
    published_date: str
    snippet: str
    topic: str = "general"
    article_text: str | None = None
    rss_found: bool = True
    source_feed: str = ""
    publisher_url_resolved: bool | None = None


class RankedArticle(ArticleCandidate):
    relevance_score: float = 0.0
    relevance_reason: str = "Keyword overlap fallback"


class StorySummary(BaseModel):
    headline: str
    one_sentence_summary: str
    content_summary_points: list[str] = Field(default_factory=list)
    interest_summary_points: list[str] = Field(default_factory=list)
    why_it_matters: str
    published_date: str = ""
    source_url: str
    source_name: str
    uncertainty_or_limitations: str


class BroadcastScript(BaseModel):
    opening: str
    story_segments: list[str] = Field(default_factory=list)
    closing: str
    full_script: str


class VisualPrompt(BaseModel):
    headline: str
    prompt: str
    style_notes: str


class BriefingResult(BaseModel):
    user_profile: UserProfile
    candidate_articles: list[ArticleCandidate] = Field(default_factory=list)
    ranked_candidates: list[RankedArticle] = Field(default_factory=list)
    selected_articles: list[RankedArticle] = Field(default_factory=list)
    summaries: list[StorySummary] = Field(default_factory=list)
    broadcast_script: BroadcastScript
    visual_prompts: list[VisualPrompt] = Field(default_factory=list)
    image_paths: list[str] = Field(default_factory=list)
    audio_path: str | None = None
    video_bytes: bytes | None = None
    agent_trace: list[dict[str, Any]] = Field(default_factory=list)
