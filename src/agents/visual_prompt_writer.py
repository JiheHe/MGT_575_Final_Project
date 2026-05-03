from __future__ import annotations

from src.gemini_client import GeminiClient
from src.models import StorySummary, UserProfile, VisualPrompt


class VisualPromptWriterAgent:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini_client = gemini_client

    def run(self, summaries: list[StorySummary], profile: UserProfile) -> list[VisualPrompt]:
        prompts: list[VisualPrompt] = []
        for summary in summaries:
            if self.gemini_client.available:
                try:
                    prompts.append(self._with_gemini(summary, profile))
                    continue
                except Exception:
                    pass
            prompts.append(self._fallback(summary))
        return prompts

    def _with_gemini(self, summary: StorySummary, profile: UserProfile) -> VisualPrompt:
        prompt = (
            "Create one safe editorial illustration prompt for a news dashboard. "
            "No real person likeness, no logos, no fake evidence style photography. "
            "Prefer abstract, symbolic, infographic-like scene. "
            "Return strict JSON keys: prompt, style_notes.\n\n"
            f"Persona: {profile.persona}\nTone: {profile.tone}\nStory: {summary.model_dump()}"
        )
        data = self.gemini_client.generate_json(prompt)
        return VisualPrompt(
            headline=summary.headline,
            prompt=str(data.get("prompt", self._fallback(summary).prompt)),
            style_notes=str(
                data.get(
                    "style_notes",
                    "Modern news graphic, deep navy background, cyan accents, crisp white text.",
                )
            ),
        )

    def _fallback(self, summary: StorySummary) -> VisualPrompt:
        return VisualPrompt(
            headline=summary.headline,
            prompt=(
                "Editorial illustration for a news segment. Abstract symbols representing: "
                f"{summary.headline}. Modern news graphic design, deep navy backdrop, cyan electric highlights, "
                "clean geometric overlays, no faces, no logos, no photorealism."
            ),
            style_notes="Branded AI News Buddy style with navy/cyan palette.",
        )
