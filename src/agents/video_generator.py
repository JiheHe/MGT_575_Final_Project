from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import tempfile

from PIL import Image, ImageDraw, ImageFont

from src.config import AppConfig
from src.models import BroadcastScript, StorySummary

try:
    from moviepy import AudioFileClip  # type: ignore
except ImportError:  # pragma: no cover
    try:
        from moviepy.editor import AudioFileClip  # type: ignore
    except ImportError:
        AudioFileClip = None  # type: ignore

try:
    import imageio_ffmpeg
except ImportError:  # pragma: no cover
    imageio_ffmpeg = None  # type: ignore


class VideoGeneratorAgent:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.last_error: str = ""

    def run(
        self,
        *,
        image_paths: list[str],
        audio_path: str | None,
        script: BroadcastScript,
        summaries: list[StorySummary],
        max_seconds: int,
    ) -> bytes | None:
        self.last_error = ""
        if not image_paths:
            self.last_error = "No generated images available for video storyboard."
            return None
        if AudioFileClip is None or imageio_ffmpeg is None:
            self.last_error = "moviepy/imageio-ffmpeg is not installed."
            return None
        if not audio_path:
            self.last_error = "No narration audio available for video storyboard."
            return None

        max_seconds = int(max_seconds)
        temp_paths: list[Path] = []
        out_path = Path(tempfile.gettempdir()) / f"briefing_{_ts()}.mp4"
        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())  # type: ignore[union-attr]
        try:
            audio_clip = AudioFileClip(audio_path)
            audio_duration = float(audio_clip.duration)
            target_duration = min(audio_duration, float(max_seconds)) if max_seconds > 0 else audio_duration
            if target_duration <= 0:
                self.last_error = "Narration audio duration is invalid."
                return None
            segment_durations = _compute_segment_durations(
                target_duration=target_duration,
                image_count=len(image_paths),
                script=script,
            )
            for idx, img_path in enumerate(image_paths):
                title = summaries[idx].headline if idx < len(summaries) else f"Story {idx + 1}"
                framed = self._render_storyboard_frame(
                    image_path=Path(img_path),
                    title=title,
                    idx=idx + 1,
                    total=max(1, len(image_paths)),
                )
                temp_paths.append(framed)
            if not temp_paths:
                self.last_error = "Failed to build storyboard frames."
                return None
            concat_file = Path(tempfile.gettempdir()) / f"concat_{_ts()}.txt"
            lines: list[str] = ["ffconcat version 1.0"]
            for i, frame in enumerate(temp_paths):
                lines.append(f"file '{frame.as_posix()}'")
                lines.append(f"duration {segment_durations[i]:.3f}")
            # Repeat last frame once so ffmpeg keeps final duration entry.
            lines.append(f"file '{temp_paths[-1].as_posix()}'")
            concat_file.write_text("\n".join(lines), encoding="utf-8")
            cmd = [
                str(ffmpeg_path),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_file),
                "-i",
                str(audio_path),
                "-t",
                f"{target_duration:.3f}",
                "-vf",
                "scale=1280:720,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-tune",
                "stillimage",
                "-c:a",
                "aac",
                str(out_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            concat_file.unlink(missing_ok=True)
            if proc.returncode != 0 or not out_path.exists():
                self.last_error = f"ffmpeg video build failed: {proc.stderr[-500:]}"
                return None
            data = out_path.read_bytes()
            out_path.unlink(missing_ok=True)
            return data
        except (OSError, ValueError, RuntimeError) as exc:
            self.last_error = f"Video generation error: {exc}"
            return None
        finally:
            try:
                audio_clip.close()  # type: ignore[name-defined]
            except (AttributeError, OSError, RuntimeError, ValueError):
                pass
            for tmp in temp_paths:
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def _render_storyboard_frame(self, *, image_path: Path, title: str, idx: int, total: int) -> Path:
        base = Image.open(image_path).convert("RGB")
        canvas = base.resize((1280, 720))
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([(0, 560), (1280, 720)], fill=(0, 0, 0, 170))
        font = ImageFont.load_default()
        title_text = _truncate(title, 120)
        draw.text((32, 585), f"Story {idx}/{total}", fill=(235, 235, 235, 255), font=font)
        draw.text((32, 620), title_text, fill=(255, 255, 255, 255), font=font)
        framed = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        out = Path(tempfile.gettempdir()) / f"frame_{_ts()}_{idx}.jpg"
        framed.save(out, format="JPEG", quality=90)
        return out


def _truncate(text: str, max_len: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rsplit(" ", 1)[0] + "…"


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _compute_segment_durations(*, target_duration: float, image_count: int, script: BroadcastScript) -> list[float]:
    if image_count <= 0:
        return []
    if image_count == 1:
        return [target_duration]
    segs = list(script.story_segments or [])
    if not segs:
        even = target_duration / image_count
        return [even] * image_count
    weights: list[float] = []
    opening_words = _word_count(script.opening)
    closing_words = _word_count(script.closing)
    for i in range(image_count):
        seg_words = _word_count(segs[i]) if i < len(segs) else 16
        if i == 0:
            seg_words += opening_words
        if i == image_count - 1:
            seg_words += closing_words
        weights.append(max(8.0, float(seg_words)))
    w_sum = sum(weights) or float(image_count)
    durations = [target_duration * (w / w_sum) for w in weights]
    min_seg = min(2.5, target_duration / max(1, image_count))
    durations = [max(min_seg, d) for d in durations]
    scale = target_duration / sum(durations)
    return [d * scale for d in durations]


def _word_count(text: str) -> int:
    return len((text or "").split())
