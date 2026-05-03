from __future__ import annotations

from datetime import datetime
from pathlib import Path

from src.config import AppConfig
from src.gemini_client import GeminiClient
from src.models import VisualPrompt
from src.utils.placeholder_image import create_placeholder_news_image


class ImageGeneratorAgent:
    def __init__(self, config: AppConfig, gemini_client: GeminiClient) -> None:
        self.config = config
        self.gemini_client = gemini_client

    def run(self, prompts: list[VisualPrompt]) -> list[str]:
        image_paths: list[str] = []
        for idx, prompt in enumerate(prompts, start=1):
            file_path = self.config.generated_images_dir / f"story_{idx}_{_ts()}.png"
            generated = self._generate_or_fallback(prompt, file_path)
            image_paths.append(generated)
        return image_paths

    def _generate_or_fallback(self, prompt: VisualPrompt, file_path: Path) -> str:
        if self.gemini_client.available:
            try:
                img_bytes = self.gemini_client.generate_image_png_bytes(
                    _build_photo_prompt(prompt)
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_bytes(img_bytes)
                return str(file_path)
            except (RuntimeError, ValueError, OSError):
                return self._fallback_image(prompt, file_path)
        return self._fallback_image(prompt, file_path)

    def _fallback_image(self, prompt: VisualPrompt, file_path: Path) -> str:
        return create_placeholder_news_image(prompt.headline, file_path)


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _build_photo_prompt(vp: VisualPrompt) -> str:
    # User requested “real photo”; keep it photorealistic but non-deceptive: no real people, no logos.
    return (
        "Generate a photorealistic, original news-style image for a briefing card. "
        "Do NOT include any real person likeness, no recognizable public figures, no logos, no brand names, "
        "no text overlays, and no misleading evidence-like scenes. "
        "Prefer symbolic objects, environments, and abstract-but-photoreal composition.\n\n"
        f"Headline: {vp.headline}\n"
        f"Creative direction: {vp.prompt}\n"
        f"Style notes: {vp.style_notes}\n"
        "Aspect ratio 16:9."
    )
