from __future__ import annotations

from datetime import datetime

from src.config import AppConfig
from src.gemini_client import GeminiClient
from src.models import BroadcastScript, UserProfile


class VoiceGeneratorAgent:
    def __init__(self, config: AppConfig, gemini_client: GeminiClient) -> None:
        self.config = config
        self.gemini_client = gemini_client

    def run(self, script: BroadcastScript, profile: UserProfile) -> str | None:
        if not self.gemini_client.available:
            return None
        try:
            voice = _voice_for_persona(profile.persona)
            wav_bytes = self.gemini_client.generate_tts_wav_bytes(
                _tts_prompt(script.full_script, profile),
                voice_name=voice,
            )
            out_path = self.config.generated_audio_dir / f"briefing_{_ts()}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(wav_bytes)
            return str(out_path)
        except (RuntimeError, ValueError, OSError):
            return None


def _tts_prompt(script_text: str, profile: UserProfile) -> str:
    return (
        f"Read this as a {profile.tone} {profile.persona} news briefing. "
        "Natural pacing, clear enunciation.\n\n"
        f"{script_text}"
    )


def _voice_for_persona(persona: str) -> str:
    p = (persona or "").lower()
    if "friendly" in p:
        return "Puck"
    return "Kore"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
