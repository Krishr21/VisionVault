from __future__ import annotations

import os
from pathlib import Path

from processing.pipeline import ingest_video, search


def main():
    sample = os.environ.get("VISIONVAULT_SAMPLE_VIDEO")
    if not sample:
        print("Set VISIONVAULT_SAMPLE_VIDEO to an absolute path of an mp4 file to run smoke test.")
        return

    result = ingest_video("local", sample, fps=0.5, max_frames=5)
    video_id = result["video_id"]
    hits = search(video_id, "what is happening", top_k=3)
    print("video_id:", video_id)
    print("hits:", len(hits))
    for h in hits:
        print(h["start"], h["end"], h.get("caption"))


if __name__ == "__main__":
    main()
