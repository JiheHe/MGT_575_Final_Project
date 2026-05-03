from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    cleaned = value.strip()
    return cleaned if cleaned else default


@dataclass(frozen=True)
class AppConfig:
    gemini_api_key: str | None
    gemini_text_models: list[str]
    gemini_image_models: list[str]
    gemini_voice_models: list[str]
    search_api_key: str | None
    search_provider: str
    data_dir: Path
    generated_images_dir: Path
    generated_audio_dir: Path
    generated_video_dir: Path


def _env_csv(name: str, default: str) -> list[str]:
    raw = _env(name, default) or default
    return [item.strip() for item in raw.split(",") if item.strip()]


def _sanitize_text_model_chain(models: list[str]) -> list[str]:
    """
    Keep modern text models first and drop deprecated 2.0 fallbacks that can
    produce 404 for some accounts.
    """
    deprecated = {
        "gemini-2.0-flash",
        "gemini-2.0-flash-001",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-lite-001",
    }
    preferred_order = [
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-2.5-pro",
        "gemini-2.5-flash-lite",
    ]
    cleaned: list[str] = []
    seen: set[str] = set()
    for model in models:
        m = model.strip()
        if not m or m in seen or m in deprecated:
            continue
        seen.add(m)
        cleaned.append(m)
    rank: dict[str, int] = {}
    for idx, name in enumerate(preferred_order):
        if name not in rank:
            rank[name] = idx
    cleaned.sort(key=lambda m: rank.get(m, 10_000))
    return cleaned


def _env_chain(primary_name: str, fallback_name: str, chain_name: str, default_chain: str) -> list[str]:
    """
    Resolve model chain from new *_MODELS var first.
    Backward-compatible with legacy PRIMARY + FALLBACK vars.
    """
    chain_raw = _env(chain_name)
    if chain_raw:
        return _env_csv(chain_name, default_chain)
    primary = _env(primary_name)
    fallback = _env_csv(fallback_name, default_chain)
    if primary:
        return [primary, *fallback]
    return fallback


def load_config() -> AppConfig:
    text_chain_raw = _env_chain(
        "GEMINI_TEXT_MODEL",
        "GEMINI_TEXT_FALLBACK_MODELS",
        "GEMINI_TEXT_MODELS",
        "gemini-flash-latest,gemini-2.5-pro,gemini-2.5-flash-lite",
    )
    text_chain = _sanitize_text_model_chain(text_chain_raw)
    image_chain = _env_chain(
        "GEMINI_IMAGE_MODEL",
        "GEMINI_IMAGE_FALLBACK_MODELS",
        "GEMINI_IMAGE_MODELS",
        "imagen-4.0-fast-generate-001,gemini-2.5-flash-image",
    )
    voice_chain = _env_chain(
        "GEMINI_VOICE_MODEL",
        "GEMINI_VOICE_FALLBACK_MODELS",
        "GEMINI_VOICE_MODELS",
        "gemini-2.5-flash-preview-tts,gemini-2.5-pro-preview-tts",
    )
    cfg = AppConfig(
        gemini_api_key=_env("GEMINI_API_KEY"),
        gemini_text_models=text_chain or ["gemini-2.5-flash"],
        gemini_image_models=image_chain or ["imagen-4.0-fast-generate-001"],
        gemini_voice_models=voice_chain or ["gemini-2.5-flash-preview-tts"],
        search_api_key=_env("SEARCH_API_KEY"),
        search_provider=(_env("SEARCH_PROVIDER", "rss") or "rss").lower(),
        data_dir=ROOT_DIR / "data",
        generated_images_dir=ROOT_DIR / "generated" / "images",
        generated_audio_dir=ROOT_DIR / "generated" / "audio",
        generated_video_dir=ROOT_DIR / "generated" / "video",
    )
    cfg.generated_images_dir.mkdir(parents=True, exist_ok=True)
    cfg.generated_audio_dir.mkdir(parents=True, exist_ok=True)
    cfg.generated_video_dir.mkdir(parents=True, exist_ok=True)
    return cfg
