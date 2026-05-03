from __future__ import annotations

from collections import Counter
from collections.abc import Callable

from src.agents.image_generator import ImageGeneratorAgent
from src.agents.interest_context import InterestContextAgent
from src.agents.interest_profiler import InterestProfilerAgent
from src.agents.news_retriever import NewsRetrieverAgent
from src.agents.ranker import RankingAgent
from src.agents.script_writer import ScriptWriterAgent
from src.agents.summarizer import SummarizerAgent
from src.agents.video_generator import VideoGeneratorAgent
from src.agents.visual_prompt_writer import VisualPromptWriterAgent
from src.agents.voice_generator import VoiceGeneratorAgent
from src.config import AppConfig
from src.gemini_client import GeminiClient
from src.models import BriefingResult, BroadcastScript, StorySummary, UserProfile, VisualPrompt
from src.utils.article_fetcher import fetch_article_text_with_reason
from src.utils.media_workspace import clear_child_files, reset_briefing_session_media


class NewsBuddyOrchestrator:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.gemini_client = GeminiClient(config)
        self.interest_profiler = InterestProfilerAgent()
        self.interest_context = InterestContextAgent(self.gemini_client)
        self.retriever = NewsRetrieverAgent(config)
        self.ranker = RankingAgent(self.gemini_client)
        self.summarizer = SummarizerAgent(self.gemini_client)
        self.script_writer = ScriptWriterAgent(self.gemini_client)
        self.visual_prompt_writer = VisualPromptWriterAgent(self.gemini_client)
        self.image_generator = ImageGeneratorAgent(config, self.gemini_client)
        self.voice_generator = VoiceGeneratorAgent(config, self.gemini_client)
        self.video_generator = VideoGeneratorAgent(config)

    def gather_summaries_only(
        self,
        *,
        interests_raw: str,
        region: str,
        max_stories: int,
        candidate_pool_size: int,
        persona: str,
        tone: str,
        max_video_seconds: int = 0,
        use_live_search: bool,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[BriefingResult, list[dict[str, object]]]:
        """Run through dual summaries; no script, images, audio, or video."""
        trace: list[dict[str, object]] = []
        reset_briefing_session_media(self.config)
        self.gemini_client.reset_model_trace()
        _notify(progress_callback, "Interest profiling", "Normalizing user interests and briefing preferences.")
        profile = self.interest_profiler.run(
            interests_raw=interests_raw,
            region=region,
            max_stories=max_stories,
            candidate_pool_size=candidate_pool_size,
            persona=persona,
            tone=tone,
            generate_images=False,
            generate_audio=False,
            generate_video=False,
            max_video_seconds=max_video_seconds,
            use_live_search=use_live_search,
        )
        trace.append({"agent": "InterestProfilerAgent", "output": profile.model_dump_json(indent=2)})

        _notify(
            progress_callback,
            "Interest context",
            "Interpreting how your interests fit together for search and ranking.",
        )
        profile = self.interest_context.run(profile)
        trace.append(
            {
                "agent": "InterestContextAgent",
                "output": profile.model_dump_json(
                    include={
                        "interests",
                        "interest_context_by_interest",
                        "combined_interest_context",
                        "search_rss_query_hint",
                    },
                    indent=2,
                ),
            }
        )

        _notify(progress_callback, "News retrieval", "Collecting candidate stories from live RSS.")
        candidates, retrieval_meta = self.retriever.run(profile)
        trace.append({"agent": "NewsRetrieverAgent", "output": str(retrieval_meta)})

        _notify(progress_callback, "Story ranking", "Scoring relevance and sorting story candidates.")
        ranked_all = self.ranker.run(candidates, profile, limit=None)
        trace.append({"agent": "RankingAgent", "output": f"Ranked {len(ranked_all)} candidate stories."})

        _notify(
            progress_callback,
            "Content accessibility check",
            "Attempting article-text retrieval top-down; only stories with extracted full text are kept.",
        )
        selected, readability_stats, scanned_count, readability_decisions = self._select_readable_articles(
            ranked_all=ranked_all,
            max_stories=profile.max_stories,
            require_readable=True,
        )
        extracted_count = sum(1 for a in selected if getattr(a, "article_text", None))
        trace.append(
            {
                "agent": "ReadabilityFilter",
                "output": (
                    f"Selected {len(selected)} / requested {profile.max_stories} candidates. "
                    f"Candidates scanned: {scanned_count} / ranked {len(ranked_all)}. "
                    f"Full text extracted for: {extracted_count}. "
                    f"Failures by reason: {dict(readability_stats)}"
                ),
            }
        )
        trace.append({"agent": "ReadabilityDecisions", "output": str(readability_decisions)})

        _notify(progress_callback, "Content understanding", "Extracting article text and generating dual summaries.")
        summaries = self.summarizer.run(selected, profile)
        summarize_failures = len(getattr(self.summarizer, "last_errors", []))
        trace.append(
            {
                "agent": "SummarizerAgent",
                "output": (
                    f"Generated {len(summaries)} grounded summaries from {len(selected)} selected articles. "
                    f"Summarization failures: {summarize_failures}."
                ),
            }
        )
        if summarize_failures:
            trace.append(
                {
                    "agent": "SummarizerFailures",
                    "output": str(getattr(self.summarizer, "last_errors", [])),
                }
            )
        selected_metrics = []
        for article, summary in zip(selected, summaries):
            text = getattr(article, "article_text", "") or ""
            selected_metrics.append(
                {
                    "title": getattr(article, "title", "")[:140],
                    "source": getattr(article, "source", ""),
                    "url": getattr(article, "url", ""),
                    "relevance_score": round(float(getattr(article, "relevance_score", 0.0)), 2),
                    "extracted_chars": len(text),
                    "paragraphs_estimate": text.count("\n") + 1 if text else 0,
                    "summary_mode": "gemini",
                    "content_points": len(getattr(summary, "content_summary_points", []) or []),
                    "interest_points": len(getattr(summary, "interest_summary_points", []) or []),
                }
            )
        trace.append({"agent": "SelectedArticleMetrics", "output": str(selected_metrics)})
        trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})

        empty_script = BroadcastScript(opening="", story_segments=[], closing="", full_script="")
        result = BriefingResult(
            user_profile=profile,
            candidate_articles=candidates,
            ranked_candidates=ranked_all,
            selected_articles=selected,
            summaries=summaries,
            broadcast_script=empty_script,
            visual_prompts=[],
            image_paths=[],
            audio_path=None,
            video_bytes=None,
            agent_trace=trace,
        )
        return result, trace

    def generate_anchor_script(
        self,
        profile: UserProfile,
        summaries: list[StorySummary],
        *,
        generate_video: bool = False,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[BroadcastScript, list[dict[str, object]]]:
        trace: list[dict[str, object]] = []
        p = profile.model_copy(
            update={
                "generate_video": generate_video,
                "max_video_seconds": profile.max_video_seconds,
            }
        )
        _notify(progress_callback, "Script writing", "Building a personalized anchor script from story summaries.")
        script = self.script_writer.run(summaries, p)
        sw_failures = getattr(self.script_writer, "last_errors", [])
        if sw_failures:
            trace.append({"agent": "ScriptWriterFailures", "output": str(sw_failures)})
        trace.append(
            {
                "agent": "ScriptWriterAgent",
                "output": (
                    f"Broadcast script: {len(script.story_segments)} segments for {len(summaries)} stories; "
                    f"full_script_chars={len(script.full_script or '')}."
                ),
            }
        )
        trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})
        return script, trace

    def generate_story_images(
        self,
        profile: UserProfile,
        summaries: list[StorySummary],
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[list[VisualPrompt], list[str], list[dict[str, object]]]:
        trace: list[dict[str, object]] = []
        clear_child_files(self.config.generated_images_dir)
        _notify(progress_callback, "Visual planning", "Drafting visual prompts consistent with the selected stories.")
        visual_prompts = self.visual_prompt_writer.run(summaries, profile)
        trace.append({"agent": "VisualPromptWriterAgent", "output": f"Generated {len(visual_prompts)} visual prompts."})
        _notify(progress_callback, "Image generation", "Rendering story visuals with model fallback.")
        image_paths = self.image_generator.run(visual_prompts)
        trace.append({"agent": "ImageGeneratorAgent", "output": f"Saved {len(image_paths)} images."})
        trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})
        return visual_prompts, image_paths, trace

    def generate_voice_audio(
        self,
        profile: UserProfile,
        script: BroadcastScript,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[str | None, list[dict[str, object]]]:
        trace: list[dict[str, object]] = []
        clear_child_files(self.config.generated_audio_dir)
        _notify(progress_callback, "Voice generation", "Synthesizing audio briefing with voice model fallback.")
        audio_path = self.voice_generator.run(script, profile)
        trace.append(
            {
                "agent": "VoiceGeneratorAgent",
                "output": audio_path or "Voice generation unavailable in MVP fallback mode.",
            }
        )
        trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})
        return audio_path, trace

    def generate_storyboard_video(
        self,
        profile: UserProfile,
        script: BroadcastScript,
        summaries: list[StorySummary],
        image_paths: list[str],
        audio_path: str | None,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[bytes | None, list[dict[str, object]]]:
        trace: list[dict[str, object]] = []
        if not image_paths or not audio_path:
            trace.append(
                {
                    "agent": "VideoGeneratorAgent",
                    "output": "Skipped: video requires generated images and narration audio.",
                }
            )
            trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})
            return None, trace
        _notify(
            progress_callback,
            "Video generation",
            "Building a storyboard-style short video from images and narration.",
        )
        video_bytes = self.video_generator.run(
            image_paths=image_paths,
            audio_path=audio_path,
            script=script,
            summaries=summaries,
            max_seconds=profile.max_video_seconds,
        )
        trace.append(
            {
                "agent": "VideoGeneratorAgent",
                "output": (
                    f"Generated in-memory video bytes: {len(video_bytes)}"
                    if video_bytes
                    else getattr(self.video_generator, "last_error", "Video generation unavailable.")
                ),
            }
        )
        trace.append({"agent": "ModelRoutingReport", "output": self.gemini_client.get_model_trace()})
        return video_bytes, trace

    def generate_narration_and_storyboard_video(
        self,
        profile: UserProfile,
        script: BroadcastScript,
        summaries: list[StorySummary],
        image_paths: list[str],
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> tuple[str | None, bytes | None, list[dict[str, object]]]:
        """Fresh narration WAV, then storyboard MP4 bytes (same script + images + new audio)."""
        audio_path, t_audio = self.generate_voice_audio(profile, script, progress_callback=progress_callback)
        video_bytes, t_video = self.generate_storyboard_video(
            profile,
            script,
            summaries,
            image_paths,
            audio_path,
            progress_callback=progress_callback,
        )
        return audio_path, video_bytes, t_audio + t_video

    def run(
        self,
        *,
        interests_raw: str,
        region: str,
        max_stories: int,
        candidate_pool_size: int,
        persona: str,
        tone: str,
        generate_images: bool,
        generate_audio: bool,
        generate_video: bool,
        max_video_seconds: int = 0,
        use_live_search: bool,
        progress_callback: Callable[[str, str], None] | None = None,
    ) -> BriefingResult:
        """Full pipeline (same stages as stepwise UI, in production order)."""
        base, trace = self.gather_summaries_only(
            interests_raw=interests_raw,
            region=region,
            max_stories=max_stories,
            candidate_pool_size=candidate_pool_size,
            persona=persona,
            tone=tone,
            max_video_seconds=max_video_seconds,
            use_live_search=use_live_search,
            progress_callback=progress_callback,
        )
        profile = base.user_profile.model_copy(
            update={
                "generate_images": generate_images,
                "generate_audio": generate_audio,
                "generate_video": generate_video,
                "max_video_seconds": max_video_seconds,
            }
        )
        script, t_script = self.generate_anchor_script(
            profile,
            base.summaries,
            generate_video=generate_video,
            progress_callback=progress_callback,
        )
        trace.extend(t_script)
        visual_prompts: list[VisualPrompt] = []
        image_paths: list[str] = []
        if generate_images:
            visual_prompts, image_paths, t_img = self.generate_story_images(
                profile, base.summaries, progress_callback=progress_callback
            )
            trace.extend(t_img)
        audio_path: str | None = None
        if generate_audio:
            audio_path, t_audio = self.generate_voice_audio(profile, script, progress_callback=progress_callback)
            trace.extend(t_audio)
        video_bytes: bytes | None = None
        if generate_video and image_paths and audio_path:
            video_bytes, t_vid = self.generate_storyboard_video(
                profile,
                script,
                base.summaries,
                image_paths,
                audio_path,
                progress_callback=progress_callback,
            )
            trace.extend(t_vid)
        elif generate_video:
            trace.append(
                {
                    "agent": "VideoGeneratorAgent",
                    "output": "Skipped: video requires generated images and narration audio.",
                }
            )

        return BriefingResult(
            user_profile=profile,
            candidate_articles=base.candidate_articles,
            ranked_candidates=base.ranked_candidates,
            selected_articles=base.selected_articles,
            summaries=base.summaries,
            broadcast_script=script,
            visual_prompts=visual_prompts,
            image_paths=image_paths,
            audio_path=audio_path,
            video_bytes=video_bytes,
            agent_trace=trace,
        )

    def _select_readable_articles(
        self, ranked_all: list, max_stories: int, require_readable: bool
    ) -> tuple[list, Counter, int, list[dict[str, object]]]:
        selected = []
        failures: Counter = Counter()
        scanned = 0
        decisions: list[dict[str, object]] = []
        for article in ranked_all:
            scanned += 1
            article_text, reason = fetch_article_text_with_reason(article.url)
            if article_text:
                is_valid, judge_reason = self._is_thorough_article_text(
                    title=getattr(article, "title", ""),
                    url=getattr(article, "url", ""),
                    article_text=article_text,
                )
                if is_valid:
                    article.article_text = article_text
                    selected.append(article)
                    decisions.append(
                        {
                            "rank": scanned,
                            "selected": "yes",
                            "title": getattr(article, "title", "")[:140],
                            "source": getattr(article, "source", ""),
                            "url": getattr(article, "url", ""),
                            "relevance_score": round(float(getattr(article, "relevance_score", 0.0)), 2),
                            "rss_found": "yes" if getattr(article, "rss_found", False) else "no",
                            "publisher_url_resolved": _yes_no_unknown(
                                getattr(article, "publisher_url_resolved", None)
                            ),
                            "full_text_extracted": "yes",
                            "text_length_chars": len(article_text),
                            "quality_gate": "passed",
                            "quality_reason": "ok",
                            "summary_basis": "full article",
                            "fetch_reason": reason,
                        }
                    )
                else:
                    failures[judge_reason] += 1
                    decisions.append(
                        {
                            "rank": scanned,
                            "selected": "no",
                            "title": getattr(article, "title", "")[:140],
                            "source": getattr(article, "source", ""),
                            "url": getattr(article, "url", ""),
                            "relevance_score": round(float(getattr(article, "relevance_score", 0.0)), 2),
                            "rss_found": "yes" if getattr(article, "rss_found", False) else "no",
                            "publisher_url_resolved": _yes_no_unknown(
                                getattr(article, "publisher_url_resolved", None)
                            ),
                            "full_text_extracted": "yes",
                            "text_length_chars": len(article_text),
                            "quality_gate": "failed",
                            "quality_reason": judge_reason,
                            "summary_basis": "rss snippet",
                            "fetch_reason": reason,
                        }
                    )
            elif not require_readable:
                selected.append(article)
                decisions.append(
                    {
                        "rank": scanned,
                        "selected": "yes",
                        "title": getattr(article, "title", "")[:140],
                        "source": getattr(article, "source", ""),
                        "url": getattr(article, "url", ""),
                        "relevance_score": round(float(getattr(article, "relevance_score", 0.0)), 2),
                        "rss_found": "yes" if getattr(article, "rss_found", False) else "no",
                        "publisher_url_resolved": _yes_no_unknown(
                            getattr(article, "publisher_url_resolved", None)
                        ),
                        "full_text_extracted": "no",
                        "text_length_chars": 0,
                        "quality_gate": "skipped",
                        "quality_reason": "require_readable_disabled",
                        "summary_basis": "rss snippet",
                        "fetch_reason": reason,
                    }
                )
            else:
                failures[reason] += 1
                decisions.append(
                    {
                        "rank": scanned,
                        "selected": "no",
                        "title": getattr(article, "title", "")[:140],
                        "source": getattr(article, "source", ""),
                        "url": getattr(article, "url", ""),
                        "relevance_score": round(float(getattr(article, "relevance_score", 0.0)), 2),
                        "rss_found": "yes" if getattr(article, "rss_found", False) else "no",
                        "publisher_url_resolved": _yes_no_unknown(
                            getattr(article, "publisher_url_resolved", None)
                        ),
                        "full_text_extracted": "no",
                        "text_length_chars": 0,
                        "quality_gate": "failed",
                        "quality_reason": reason,
                        "summary_basis": "rss snippet",
                    }
                )
            if len(selected) >= max_stories:
                break
        return selected, failures, scanned, decisions

    def _is_thorough_article_text(self, *, title: str, url: str, article_text: str) -> tuple[bool, str]:
        # Fast heuristics first.
        txt = (article_text or "").strip()
        if len(txt) < 500:
            return False, "article_too_short"
        # Some extractors return dense single-block text; accept if length is substantial.
        if txt.count("\n") < 2 and len(txt) < 1000:
            return False, "article_not_structured"
        # Optional LLM judge for quality and validity.
        if self.gemini_client.available:
            try:
                prompt = (
                    "Judge whether extracted article text is a valid and sufficiently thorough news article body. "
                    "Reject nav boilerplate, promo pages, sparse blurbs, and malformed extraction. "
                    "Return strict JSON: {\"is_valid\": true|false, \"reason\": \"<short_snake_case_reason>\"}. "
                    "Keep reason short (2-8 words max).\n\n"
                    f"Title: {title}\nURL: {url}\n"
                    f"Extracted text sample (first 6000 chars): {txt[:6000]}"
                )
                data = self.gemini_client.generate_json(prompt)
                is_valid = bool(data.get("is_valid", False))
                reason = str(data.get("reason", "llm_judge_rejected")).strip().lower().replace(" ", "_")
                reason = "".join(ch for ch in reason if ch.isalnum() or ch == "_")[:80]
                return is_valid, reason if reason else "llm_judge_rejected"
            except (RuntimeError, ValueError, TypeError):
                pass
        return True, "ok"


def _notify(callback: Callable[[str, str], None] | None, stage: str, detail: str) -> None:
    if callback:
        callback(stage, detail)


def _yes_no_unknown(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"
