from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def create_placeholder_news_image(
    headline: str,
    output_path: Path,
    label: str = "AI News Buddy",
) -> str:
    width, height = 1024, 576
    background = (8, 22, 48)
    accent = (0, 224, 255)
    white = (245, 248, 255)

    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)

    draw.rectangle((14, 14, width - 14, height - 14), outline=accent, width=4)
    draw.line((14, 88, width - 14, 88), fill=accent, width=3)

    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        body_font = ImageFont.truetype("arial.ttf", 28)
    except OSError:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    draw.text((34, 30), label, font=title_font, fill=accent)
    draw.text((34, 116), "Editorial Illustration", font=body_font, fill=white)

    safe_headline = headline if headline else "News headline unavailable"
    wrapped = _wrap_text(safe_headline, max_chars=50)
    draw.multiline_text((34, 180), wrapped, font=body_font, fill=white, spacing=12)
    draw.text((34, 510), "Generated fallback visual (not a real photo)", font=body_font, fill=accent)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path)


def _wrap_text(text: str, max_chars: int = 50) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join(current + [word]).strip()
        if len(candidate) <= max_chars:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:6])
