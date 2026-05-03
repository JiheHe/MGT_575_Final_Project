from __future__ import annotations

import ast
import json

from src.gemini_client import GeminiClient
from src.models import BroadcastScript, StorySummary, UserProfile
from src.utils.display_text import normalize_loose_json_string_field


def _word_count(text: str) -> int:
    return len(text.split())


def _estimate_spoken_seconds_from_words(words: int) -> int:
    # Conservative pacing for TTS + anchor-style narration.
    return max(1, int(round(words / 2.35)))


def _script_word_count(opening: str, segments: list[str], closing: str) -> int:
    return _word_count(" ".join([opening, *segments, closing]).strip())


def _compact_story(item: StorySummary) -> dict[str, object]:
    return {
        "headline": item.headline,
        "source_name": item.source_name,
        "one_sentence_summary": item.one_sentence_summary,
        "why_it_matters": item.why_it_matters,
        "content_summary_points": item.content_summary_points[:5],
        "interest_summary_points": item.interest_summary_points[:5],
    }


def _normalize_story_segments(raw_segments: object) -> list[str]:
    if not isinstance(raw_segments, list):
        return []
    output: list[str] = []
    for item in raw_segments:
        text = _segment_to_text(item)
        if text:
            output.append(text)
    return output


def _segment_to_text(item: object) -> str:
    if isinstance(item, dict):
        title = str(item.get("title", "")).strip()
        script = str(item.get("script", "")).strip()
        if title and script:
            return f"{title}: {script}"
        return script or title
    if isinstance(item, str):
        candidate = item.strip()
        if candidate.startswith("{") and candidate.endswith("}"):
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return _segment_to_text(parsed)
            except (ValueError, SyntaxError):
                pass
        return candidate
    return str(item).strip()


def _coerce_opening_closing(raw: object) -> str:
    s = normalize_loose_json_string_field(raw)
    return s.strip()


def _validate_script_parts(
    opening: str,
    segments: list[str],
    closing: str,
    summaries: list[StorySummary],
    *,
    min_opening_words: int,
    min_closing_words: int,
    min_segment_words: int,
) -> list[str]:
    reasons: list[str] = []
    if not opening:
        reasons.append("opening is empty")
    elif _word_count(opening) < min_opening_words:
        reasons.append(
            f"opening too brief ({_word_count(opening)} words; need at least {min_opening_words})"
        )
    if not closing:
        reasons.append("closing is empty")
    elif _word_count(closing) < min_closing_words:
        reasons.append(
            f"closing too brief ({_word_count(closing)} words; need at least {min_closing_words})"
        )
    if len(segments) != len(summaries):
        reasons.append(
            f"story_segments length {len(segments)} does not match story count {len(summaries)}"
        )
    for i, seg in enumerate(segments):
        if _word_count(seg) < min_segment_words:
            reasons.append(
                f"story segment {i + 1} too brief ({_word_count(seg)} words; need at least {min_segment_words})"
            )
    return reasons


class ScriptWriterAgent:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client
        self.last_errors: list[dict[str, str]] = []

    def run(self, summaries: list[StorySummary], profile: UserProfile) -> BroadcastScript:
        self.last_errors = []
        if not summaries:
            return BroadcastScript(
                opening="No readable stories were available for this run.",
                story_segments=[],
                closing="Try adjusting interests, region, or disabling live search.",
                full_script=(
                    "No readable stories were available for this run.\n\n"
                    "Try adjusting interests, region, or disabling live search."
                ),
            )
        if not self.gemini_client.available:
            self.last_errors.append({"reason": "Gemini client not configured (missing API key)."})
            return self._error_script(
                "Broadcast script was not generated: Gemini is not configured. "
                "See Agent trace → ScriptWriterFailures."
            )
        return self._script_with_gemini(summaries, profile)

    def _error_script(self, message: str) -> BroadcastScript:
        return BroadcastScript(
            opening="",
            story_segments=[],
            closing="",
            full_script=message,
        )

    def _script_with_gemini(
        self, summaries: list[StorySummary], profile: UserProfile
    ) -> BroadcastScript:
        n = len(summaries)
        is_video_mode = bool(getattr(profile, "generate_video", False))
        min_opening_words = 12 if is_video_mode else 14
        min_closing_words = 10 if is_video_mode else 14
        min_segment_words = 10 if is_video_mode else 42
        raw_target_seconds = int(getattr(profile, "max_video_seconds", 0) or 0)
        enforce_duration = bool(is_video_mode and raw_target_seconds > 0)
        target_seconds = raw_target_seconds if enforce_duration else 0
        target_words = int(target_seconds * 2.35) if enforce_duration else 0
        hard_word_limit = int(target_words * 1.05) if enforce_duration else 0
        payload = [_compact_story(s) for s in summaries]
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        segment_rule = (
            "Focus each segment on the most decision-relevant interest_summary_points first, then why_it_matters. "
            "Segment lengths can vary by story importance. "
            if is_video_mode
            else "Each segment must be 90–200 spoken words: "
        )
        opening_rule = "opening: 18–35 words." if is_video_mode else "opening: 35–70 words;"
        closing_rule = "closing: 12–28 words." if is_video_mode else "closing: 35–70 words;"
        video_duration_rule = (
            f"Total full script should fit under {target_seconds} seconds spoken "
            f"(target <= {target_words} words, hard cap {hard_word_limit} words). "
            if enforce_duration
            else ""
        )
        base_rules = (
            f"You write a spoken news briefing. There are exactly {n} stories (fixed order). "
            f"Anchor persona: {profile.persona}. Tone: {profile.tone}. "
            f"Audience interests: {profile.interests}. "
            "Use only facts that appear in each story's JSON fields; do not invent statistics or URLs. "
            "Avoid robotic meta-lines like 'I am your AI News Buddy'. "
            "Return STRICT JSON with keys only: opening (string), story_segments (array of strings), "
            f"closing (string). story_segments MUST contain EXACTLY {n} elements. "
            f"Element i must cover only story i (1-based: story {n} is last), preserving order. "
            f"{segment_rule}name the main entity from the headline, "
            "give at least two concrete facts drawn from the summaries (names, numbers, regions), "
            "tie briefly to why_it_matters, and mention the source_name once in plain words. "
            f"{opening_rule} frames the briefing without reading every headline. "
            f"{closing_rule} thanks the listener and points them to linked sources for detail. "
            f"{video_duration_rule}"
            "Do not leave story_segments empty or shorter than the story count. "
            "Stories JSON:\n"
        )
        prompt = base_rules + payload_json

        try:
            data = self.gemini_client.generate_json(prompt)
        except (RuntimeError, ValueError, TypeError) as exc:
            self.last_errors.append({"stage": "primary_json", "reason": repr(exc)})
            return self._error_script(
                "Broadcast script was not generated: Gemini JSON request failed. "
                "See Agent trace → ScriptWriterFailures."
            )
        if not isinstance(data, dict):
            self.last_errors.append(
                {"stage": "primary_json", "reason": f"expected JSON object, got {type(data).__name__}"}
            )
            return self._error_script(
                "Broadcast script was not generated: model returned non-object JSON. "
                "See Agent trace → ScriptWriterFailures."
            )

        opening, segments, closing = self._parse_script_payload(data)
        reasons = _validate_script_parts(
            opening,
            segments,
            closing,
            summaries,
            min_opening_words=min_opening_words,
            min_closing_words=min_closing_words,
            min_segment_words=min_segment_words,
        )
        if not reasons:
            if enforce_duration:
                opening, segments, closing = self._maybe_refine_for_duration(
                    opening=opening,
                    segments=segments,
                    closing=closing,
                    profile=profile,
                    payload_json=payload_json,
                    target_seconds=target_seconds,
                    hard_word_limit=hard_word_limit,
                    story_count=n,
                )
            full_script = "\n\n".join([opening, *segments, closing])
            return BroadcastScript(
                opening=opening,
                story_segments=segments,
                closing=closing,
                full_script=full_script,
            )

        self.last_errors.append({"stage": "primary_validation", "reasons": "; ".join(reasons)})
        repair = (
            "Your previous JSON failed validation:\n"
            + "\n".join(f"- {r}" for r in reasons)
            + f"\n\nRegenerate the ENTIRE JSON object with keys opening, story_segments, closing. "
            f"story_segments MUST have length {n}. "
            f"Each segment must be >= {min_segment_words} words using ONLY the facts below. "
            "Stories JSON:\n"
            + payload_json
        )
        try:
            data2 = self.gemini_client.generate_json(repair)
        except (RuntimeError, ValueError, TypeError) as exc:
            self.last_errors.append({"stage": "repair_json", "reason": repr(exc)})
            return self._error_script(
                "Broadcast script was not generated: repair JSON request failed. "
                "See Agent trace → ScriptWriterFailures."
            )
        if not isinstance(data2, dict):
            self.last_errors.append(
                {"stage": "repair_json", "reason": f"expected JSON object, got {type(data2).__name__}"}
            )
            return self._error_script(
                "Broadcast script was not generated: repair pass returned non-object JSON. "
                "See Agent trace → ScriptWriterFailures."
            )

        opening2, segments2, closing2 = self._parse_script_payload(data2)
        reasons2 = _validate_script_parts(
            opening2,
            segments2,
            closing2,
            summaries,
            min_opening_words=min_opening_words,
            min_closing_words=min_closing_words,
            min_segment_words=min_segment_words,
        )
        if reasons2:
            self.last_errors.append({"stage": "repair_validation", "reasons": "; ".join(reasons2)})
            return self._error_script(
                "Broadcast script was not generated: model output did not include one full segment "
                f"per story after a repair attempt ({n} stories expected). "
                "See Agent trace → ScriptWriterFailures."
            )

        if enforce_duration:
            opening2, segments2, closing2 = self._maybe_refine_for_duration(
                opening=opening2,
                segments=segments2,
                closing=closing2,
                profile=profile,
                payload_json=payload_json,
                target_seconds=target_seconds,
                hard_word_limit=hard_word_limit,
                story_count=n,
            )
        full_script = "\n\n".join([opening2, *segments2, closing2])
        return BroadcastScript(
            opening=opening2,
            story_segments=segments2,
            closing=closing2,
            full_script=full_script,
        )

    def _parse_script_payload(self, data: dict[str, object]) -> tuple[str, list[str], str]:
        opening = _coerce_opening_closing(data.get("opening", ""))
        closing = _coerce_opening_closing(data.get("closing", ""))
        story_segments = _normalize_story_segments(data.get("story_segments", []))
        story_segments = [
            (normalize_loose_json_string_field(s) or str(s)).strip()
            for s in story_segments
            if str(s).strip()
        ]
        return opening, story_segments, closing

    def _maybe_refine_for_duration(
        self,
        *,
        opening: str,
        segments: list[str],
        closing: str,
        profile: UserProfile,
        payload_json: str,
        target_seconds: int,
        hard_word_limit: int,
        story_count: int,
    ) -> tuple[str, list[str], str]:
        if not bool(getattr(profile, "generate_video", False)):
            return opening, segments, closing
        words = _script_word_count(opening, segments, closing)
        if words <= hard_word_limit:
            return opening, segments, closing

        current = {
            "opening": opening,
            "story_segments": segments,
            "closing": closing,
        }
        for attempt in range(1, 4):
            over_by = max(0, words - hard_word_limit)
            prompt = (
                "You are revising a news briefing script to fit a hard runtime cap while preserving core meaning. "
                "Return STRICT JSON with keys only: opening, story_segments, closing. "
                f"story_segments length MUST stay exactly {story_count}. "
                "Do not invent facts. Keep ordering of stories unchanged. "
                "You may shorten less important stories more aggressively and keep more detail for high-impact stories. "
                f"Target duration <= {target_seconds} seconds, hard cap {hard_word_limit} words. "
                f"Current draft is {words} words (over by {over_by}). "
                "Current draft JSON:\n"
                f"{json.dumps(current, ensure_ascii=False)}\n\n"
                "Story fact constraints JSON:\n"
                f"{payload_json}"
            )
            try:
                revised = self.gemini_client.generate_json(prompt)
            except (RuntimeError, ValueError, TypeError) as exc:
                self.last_errors.append({"stage": f"duration_refine_{attempt}", "reason": repr(exc)})
                break
            if not isinstance(revised, dict):
                self.last_errors.append(
                    {
                        "stage": f"duration_refine_{attempt}",
                        "reason": f"expected JSON object, got {type(revised).__name__}",
                    }
                )
                break
            r_opening, r_segments, r_closing = self._parse_script_payload(revised)
            if len(r_segments) != story_count or not r_opening or not r_closing:
                self.last_errors.append(
                    {
                        "stage": f"duration_refine_{attempt}",
                        "reason": "invalid refined structure",
                    }
                )
                continue
            words = _script_word_count(r_opening, r_segments, r_closing)
            opening, segments, closing = r_opening, r_segments, r_closing
            current = {
                "opening": opening,
                "story_segments": segments,
                "closing": closing,
            }
            if words <= hard_word_limit:
                break
        if words > hard_word_limit:
            self.last_errors.append(
                {
                    "stage": "duration_refine_limit_not_met",
                    "reason": (
                        f"final_words={words}, est_seconds={_estimate_spoken_seconds_from_words(words)}, "
                        f"target_seconds={target_seconds}"
                    ),
                }
            )
        return opening, segments, closing
