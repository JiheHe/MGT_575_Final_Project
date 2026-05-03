"""Clear generated briefing media so each run stays fast and disk use stays bounded."""

from __future__ import annotations

from pathlib import Path

from src.config import AppConfig, ROOT_DIR


def clear_child_files(directory: Path) -> None:
    """Remove regular files directly under directory (flat generated folders)."""
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file():
            try:
                path.unlink()
            except OSError:
                pass


def sweep_stale_root_media_artifacts(project_root: Path | None = None) -> None:
    """
    Remove editor / moviepy temp exports that landed in the project root by mistake.
    Keeps normal source files untouched (only obvious temp/briefing junk).
    """
    root = project_root or ROOT_DIR
    if not root.is_dir():
        return
    for path in root.iterdir():
        if not path.is_file():
            continue
        name = path.name
        lower = name.lower()
        if "temp_mpy" in lower or "temp_wvf" in lower:
            try:
                path.unlink()
            except OSError:
                pass
            continue
        if name.startswith("briefing_") and "TEMP" in name.upper():
            try:
                path.unlink()
            except OSError:
                pass


def reset_briefing_session_media(config: AppConfig) -> None:
    """Wipe prior images/audio/video outputs and stray root temps (new 'Find News' session)."""
    clear_child_files(config.generated_images_dir)
    clear_child_files(config.generated_audio_dir)
    clear_child_files(config.generated_video_dir)
    sweep_stale_root_media_artifacts(ROOT_DIR)
