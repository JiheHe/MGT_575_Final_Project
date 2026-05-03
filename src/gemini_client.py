from __future__ import annotations

import json
from io import BytesIO
from typing import Any, TYPE_CHECKING, cast

from PIL import Image

from src.config import AppConfig

try:
    from google import genai  # type: ignore  # pylint: disable=no-name-in-module
    from google.genai import types  # type: ignore  # pylint: disable=import-error,no-name-in-module
except Exception:  # pragma: no cover  # pylint: disable=broad-exception-caught
    genai = None  # type: ignore
    types = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover
    from google.genai import types as _types  # noqa: PLC0415


class GeminiClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.available = bool(config.gemini_api_key)
        self._client = (
            genai.Client(api_key=config.gemini_api_key)  # type: ignore[attr-defined]
            if (config.gemini_api_key and genai is not None)
            else None
        )
        self._model_usage: dict[str, str | None] = {"text": None, "image": None, "voice": None}
        self._model_failures: dict[str, list[dict[str, str]]] = {
            "text": [],
            "image": [],
            "voice": [],
        }

    def generate_text(
        self,
        prompt: str,
        *,
        model: str | None = None,
        response_mime_type: str | None = None,
    ) -> str:
        if not self.available or not self._client or genai is None:
            raise RuntimeError("Gemini API key is not configured.")
        candidates = _dedupe_models(([model] if model else []) + self.config.gemini_text_models)
        kwargs: dict[str, Any] = {}
        if response_mime_type:
            kwargs["response_mime_type"] = response_mime_type
        last_error: Exception | None = None
        for chosen_model in candidates:
            try:
                response = self._client.models.generate_content(
                    model=chosen_model,
                    contents=prompt,
                    config=kwargs or None,
                )
                self._record_success("text", chosen_model)
                return (response.text or "").strip()
            except Exception as exc:  # pylint: disable=broad-exception-caught
                last_error = exc
                self._record_failure("text", chosen_model, exc)
                continue
        raise RuntimeError(f"Text generation failed across model fallbacks: {last_error}")

    def generate_json(self, prompt: str, *, model: str | None = None) -> dict[str, Any]:
        raw = self.generate_text(prompt, model=model, response_mime_type="application/json")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Gemini JSON parse failed: {exc}") from exc

    def generate_image_png_bytes(self, prompt: str, *, model: str | None = None) -> bytes:
        if not self.available or not self._client or types is None:
            raise RuntimeError("Gemini API key is not configured.")
        candidates = _dedupe_models(([model] if model else []) + self.config.gemini_image_models)
        last_error: Exception | None = None
        for m in candidates:
            chosen_model = m.replace("models/", "")
            try:
                if "imagen" in chosen_model.lower():
                    output = self._generate_with_imagen(prompt, chosen_model)
                    self._record_success("image", chosen_model)
                    return output
                output = self._generate_with_gemini_image(prompt, chosen_model)
                self._record_success("image", chosen_model)
                return output
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                last_error = exc
                self._record_failure("image", chosen_model, exc)
                # Keep trying alternative models on not-found/transient overload.
                if _is_retryable_model_error(exc):
                    continue
                continue
        raise RuntimeError(f"Image generation failed across model fallbacks: {last_error}")

    def _generate_with_gemini_image(self, prompt: str, model: str) -> bytes:
        assert types is not None
        response = self._client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["Text", "Image"]),  # type: ignore[attr-defined]
        )
        for part in (response.candidates[0].content.parts if response.candidates else []):
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                data = part.inline_data.data
                image = Image.open(BytesIO(data))
                out = BytesIO()
                image.save(out, format="PNG")
                return out.getvalue()
        raise RuntimeError("No image bytes returned by Gemini image model.")

    def _generate_with_imagen(self, prompt: str, model: str) -> bytes:
        assert types is not None
        response = self._client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(number_of_images=1),  # type: ignore[attr-defined]
        )
        if not getattr(response, "generated_images", None):
            raise RuntimeError("No images returned by Imagen model.")
        img0 = response.generated_images[0]
        image_bytes = img0.image.image_bytes
        image = Image.open(BytesIO(image_bytes))
        out = BytesIO()
        image.save(out, format="PNG")
        return out.getvalue()

    def generate_tts_wav_bytes(
        self,
        text: str,
        *,
        model: str | None = None,
        voice_name: str = "Kore",
        sample_rate_hz: int = 24000,
    ) -> bytes:
        if not self.available or not self._client or types is None:
            raise RuntimeError("Gemini API key is not configured.")
        candidates = _dedupe_models(([model] if model else []) + self.config.gemini_voice_models)
        last_error: Exception | None = None
        for chosen_model in candidates:
            try:
                response = self._client.models.generate_content(
                    model=chosen_model,
                    contents=text,
                    config=types.GenerateContentConfig(  # type: ignore[attr-defined]
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(  # type: ignore[attr-defined]
                            voice_config=types.VoiceConfig(  # type: ignore[attr-defined]
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(  # type: ignore[attr-defined]
                                    voice_name=voice_name
                                )
                            )
                        )
                    ),
                )
                if not response.candidates:
                    raise RuntimeError("No TTS candidates returned.")
                parts = response.candidates[0].content.parts
                if (
                    not parts
                    or not getattr(parts[0], "inline_data", None)
                    or not parts[0].inline_data.data
                ):
                    raise RuntimeError("No audio bytes returned from TTS model.")
                pcm = parts[0].inline_data.data
                self._record_success("voice", chosen_model)
                return _pcm_to_wav_bytes(pcm, sample_rate_hz=sample_rate_hz)
            except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
                last_error = exc
                self._record_failure("voice", chosen_model, exc)
                if _is_retryable_model_error(exc):
                    continue
                continue
        raise RuntimeError(f"TTS generation failed across model fallbacks: {last_error}")

    def get_model_trace(self) -> dict[str, Any]:
        return {
            "used_models": dict(self._model_usage),
            "failures": {k: list(v) for k, v in self._model_failures.items()},
            "high_demand_hits": {
                k: [f for f in v if "high demand" in f.get("error", "").lower() or "503" in f.get("error", "")]
                for k, v in self._model_failures.items()
            },
        }

    def reset_model_trace(self) -> None:
        self._model_usage = {"text": None, "image": None, "voice": None}
        self._model_failures = {"text": [], "image": [], "voice": []}

    def _record_success(self, modality: str, model: str) -> None:
        self._model_usage[modality] = model

    def _record_failure(self, modality: str, model: str, exc: Exception) -> None:
        self._model_failures[modality].append(
            {
                "model": model,
                "error": str(exc)[:240],
            }
        )


def _pcm_to_wav_bytes(pcm: bytes, *, sample_rate_hz: int = 24000) -> bytes:
    import wave

    out = BytesIO()
    wf_any = cast(Any, wave.open(out, "wb"))
    with wf_any as wf:
        wf.setnchannels(1)  # pylint: disable=no-member
        wf.setsampwidth(2)  # 16-bit PCM  # pylint: disable=no-member
        wf.setframerate(sample_rate_hz)  # pylint: disable=no-member
        wf.writeframes(pcm)  # pylint: disable=no-member
    return out.getvalue()


def _is_retryable_model_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "503" in msg
        or "unavailable" in msg
        or "high demand" in msg
        or "resource exhausted" in msg
        or "429" in msg
        or "404" in msg
        or "not found" in msg
    )


def _dedupe_models(models: list[str | None]) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for m in models:
        if not m:
            continue
        key = m.strip()
        if key and key not in seen:
            seen.add(key)
            cleaned.append(key)
    return cleaned
