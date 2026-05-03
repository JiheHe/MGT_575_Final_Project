from __future__ import annotations

import json

from src.gemini_client import GeminiClient
from src.models import UserProfile


class InterestContextAgent:
    """
    Interprets the user's interest bundle (e.g. how 'markets' should be read alongside 'AI' and 'robotics')
    and produces hints for retrieval, ranking, and summarization.
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

        # Offline fallback: brief, human-friendly contextual definitions.
        # This is only used when Gemini is unavailable.
        joined = ", ".join(interests)
        has_ai = any(i.lower() == "ai" for i in interests)
        has_robotics = any(i.lower() == "robotics" for i in interests)
        has_markets = any(i.lower() == "markets" for i in interests)
        ai_robotics_lens = has_ai and has_robotics

        def _mk(defn: str) -> str:
            return f"{defn} (in your bundle: {joined})"

        per: dict[str, str] = {}
        for item in interests:
            low = item.lower().strip()
            if low == "ai":
                per[item] = _mk(
                    "AI refers to machine learning, automation, and deployed intelligence (including AI software and AI-enabling chips) that can affect products and operations."
                )
            elif low == "robotics":
                per[item] = _mk(
                    "Robotics refers to robots, automation systems, and industrial/consumer robotics deployments that create operational change and measurable adoption signals."
                )
            elif low == "markets":
                per[item] = _mk(
                    "Markets refers to financial and sector-moving developments tied to AI/robotics—e.g., investor sentiment, company performance, supply-chain implications, and capital allocation—rather than unrelated commodity price moves."
                )
            else:
                per[item] = _mk(
                    f"{item} is treated as a briefing lens; the app will prioritize stories that connect it to the other items in your bundle."
                )

        combined = (
            "Your interests are treated as one unified briefing lens. "
            "Ambiguous terms are interpreted using the full bundle so the search and ranking focus on stories where the topics reinforce each other. "
        )
        if ai_robotics_lens and has_markets:
            combined += "For example, 'markets' is read as AI/robotics-linked market dynamics (investors, sector results, and supply-chain signals)."
        return profile.model_copy(
            update={
                "interest_context_by_interest": per,
                "combined_interest_context": combined,
                "search_rss_query_hint": " ".join(
                    [
                        "AI",
                        "robotics" if has_robotics else "",
                        "markets" if has_markets else "",
                        "investors",
                        "sector",
                        "adoption",
                    ]
                ).strip(),
            }
        )

    def _enrich_with_gemini(self, profile: UserProfile) -> UserProfile:
        interests = profile.interests
        prompt = (
            "You help a news app interpret a user's comma-separated interests as one coherent bundle. "
            "Disambiguate ambiguous words using the full list (e.g. 'markets' with 'AI' and 'robotics' "
            "should be about AI/robotics-linked market dynamics and investor/sector signals, not generic oil/commodity price stories). "
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
