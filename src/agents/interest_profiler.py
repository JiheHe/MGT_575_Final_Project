from __future__ import annotations

from src.models import UserProfile
from src.utils.text_utils import normalize_interests


class InterestProfilerAgent:
    def run(
        self,
        interests_raw: str,
        region: str,
        max_stories: int,
        candidate_pool_size: int,
        persona: str,
        tone: str,
        generate_images: bool,
        generate_audio: bool,
        generate_video: bool,
        max_video_seconds: int,
        use_live_search: bool,
    ) -> UserProfile:
        interests = normalize_interests(interests_raw)
        if not interests:
            interests = ["technology", "business"]
        return UserProfile(
            interests=interests,
            region=region.strip().upper() if region else "US",
            max_stories=max(1, min(int(max_stories), 8)),
            candidate_pool_size=max(5, min(int(candidate_pool_size), 50)),
            persona=persona,
            tone=tone,
            generate_images=generate_images,
            generate_audio=generate_audio,
            generate_video=generate_video,
            max_video_seconds=max(0, int(max_video_seconds)),
            use_live_search=use_live_search,
        )
