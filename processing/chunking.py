from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Chunk:
    video_id: str
    start: float
    end: float
    transcript: str
    caption: str
    frame_file: str

    @property
    def text(self) -> str:
        return f"{self.transcript} {self.caption}".strip()


def align_transcript_and_captions(
    video_id: str,
    transcript_segments: List[Dict],
    frame_descriptions: List[Dict],
) -> List[Chunk]:
    chunks: List[Chunk] = []

    for seg in transcript_segments:
        s, e = float(seg["start"]), float(seg["end"])
        overlaps = [
            f for f in frame_descriptions if (f["timestamp"] >= s and f["timestamp"] <= e)
        ]
        if not overlaps:
            overlaps = sorted(
                frame_descriptions,
                key=lambda x: abs(x["timestamp"] - ((s + e) / 2.0)),
            )[:1]

        combined_caption = " | ".join(o["caption"] for o in overlaps)
        thumb = overlaps[0]["frame_file"] if overlaps else ""

        chunks.append(
            Chunk(
                video_id=video_id,
                start=s,
                end=e,
                transcript=(seg.get("text") or "").strip(),
                caption=combined_caption,
                frame_file=thumb,
            )
        )

    return chunks
