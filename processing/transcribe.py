from __future__ import annotations

from pathlib import Path
from typing import List, Dict


def transcribe_audio(audio_path: Path, model_size: str = "base") -> List[Dict]:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size)
    segments, _ = model.transcribe(str(audio_path))

    output = []
    for seg in segments:
        output.append(
            {
                "start": float(seg.start),
                "end": float(seg.end),
                "text": (seg.text or "").strip(),
            }
        )
    return output
