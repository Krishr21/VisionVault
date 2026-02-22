from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Tuple


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_audio(video_path: Path, output_audio_path: Path) -> Path:
    ensure_dir(output_audio_path.parent)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-q:a",
        "0",
        "-map",
        "a",
        str(output_audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg audio extraction failed: {e.stderr}") from e
    return output_audio_path


def extract_frames(video_path: Path, frames_dir: Path, fps: float = 1.0) -> Path:
    ensure_dir(frames_dir)
    frame_pattern = frames_dir / "frame_%06d.jpg"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        str(frame_pattern),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg frame extraction failed: {e.stderr}") from e
    return frames_dir


def frame_number_to_timestamp_seconds(frame_number: int, fps: float) -> float:
    # ffmpeg numbering starts from 1 in our template
    return max(0.0, (frame_number - 1) / fps)


def parse_frame_number(frame_filename: str) -> int:
    stem = Path(frame_filename).stem  # frame_000123
    return int(stem.split("_")[-1])
