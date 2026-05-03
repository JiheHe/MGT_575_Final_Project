from __future__ import annotations

import json

from src.gemini_client import GeminiClient
from src.models import UserProfile


class InterestContextAgent:
    """
    Interprets the user's interest bundle as one unified lens so each term is read in
    the context of the others, and produces hints for retrieval, ranking, and summarization.
    """

    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def run(self, profile: UserProfile) -> UserProfile:
        interests = profile.interests
        if not interests:
            return profile

        if self.gemini_client.available:
            try:
                return self._enrich_with_gemini(profile)
            except (RuntimeError, ValueError, TypeError):
                pass

        # Offline fallback (only used when Gemini is unavailable).
        # Generic per-interest treatment: every term is interpreted as part of the
        # user's bundle, with no hand-tuned rules for any specific topic.
        joined = ", ".join(interests)
        per: dict[str, str] = {}
        for item in interests:
            label = item.strip()
            if not label:
                continue
            per[label] = (
                f"{label} is interpreted as part of your unified bundle ({joined}); "
                "the app prioritizes stories that connect this term to the other items in your bundle "
                "rather than coverage where it appears in isolation."
            )
        combined = (
            "Your interests are treated as one unified briefing lens. "
            "Ambiguous terms are interpreted using the full bundle so retrieval and ranking focus on "
            "stories where the topics reinforce each other rather than where any single term appears alone."
        )
        return profile.model_copy(
            update={
                "interest_context_by_interest": per,
                "combined_interest_context": combined,
                "search_rss_query_hint": " ".join(i.strip() for i in interests if i.strip()),
            }
        )

    def _enrich_with_gemini(self, profile: UserProfile) -> UserProfile:
        interests = profile.interests
        prompt = (
            "You help a news app interpret a user's comma-separated interests as one coherent bundle. "
            "Disambiguate ambiguous words using the full list: an interest's meaning should be inferred "
            "from how it relates to the user's other interests in this bundle, not from its most common "
            "stand-alone meaning. Apply this principle to whatever interests the user provides; do not "
            "assume any specific topic, industry, or vocabulary. "
            "Stay concise. Return strict JSON with keys:\n"
            "- per_interest: object mapping each interest string (exact keys below) to an object with:\n"
            "  - definition: a 1-2 sentence contextual definition of how to read that term inside the bundle\n"
            "  - search_terms: an array of 4-9 short terms/phrases that should appear in relevant RSS items\n"
            "- combined_paragraph: 2-4 sentences describing the unified editorial lens (include the above definitions)\n"
            "- rss_search_query_hint: a single short RSS query string that uses the most important concepts from search_terms "
            "(no boolean operators required; just a natural keyword query)\n\n"
            f"Interests (use these exact keys in per_interest): {json.dumps(interests)}\n"
            f"Region hint: {profile.region}\n"
        )
        data = self.gemini_client.generate_json(prompt)

        per_raw = data.get("per_interest", {})
        per: dict[str, str] = {}
        all_terms: list[str] = []
        if isinstance(per_raw, dict):
            for key in interests:
                # Case-insensitive key matching for robustness.
                v = per_raw.get(key)
                if v is None:
                    for rk, rv in per_raw.items():
                        if str(rk).strip().lower() == key.lower():
                            v = rv
                            break
                if isinstance(v, dict):
                    definition = str(v.get("definition", "")).strip()
                    if definition:
                        per[key] = definition
                    st = v.get("search_terms", [])
                    if isinstance(st, list):
                        for t in st:
                            if t and isinstance(t, str) and t.strip():
                                all_terms.append(t.strip())

        for key in interests:
            per.setdefault(
                key,
                f"{key} is interpreted in context of the full bundle so retrieval and ranking favor mutually reinforcing stories.",
            )

        combined = str(data.get("combined_paragraph", "")).strip() or (
            f"Unified editorial lens: interpret {', '.join(interests)} together so ambiguous terms resolve using the full list."
        )
        hint = str(data.get("rss_search_query_hint", "")).strip()
        if not hint:
            # If model doesn't return a query hint, derive one from search terms.
            hint = " ".join(all_terms[:10]) if all_terms else " ".join(interests)
        return profile.model_copy(
            update={
                "interest_context_by_interest": per,
                "combined_interest_context": combined,
                "search_rss_query_hint": hint,
            }
        )
