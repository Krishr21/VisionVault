from __future__ import annotations

import subprocess
from pathlib import Path


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def download_youtube(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Prefer a broadly-available format selection and allow yt-dlp to merge.
    # This avoids common 'Requested format is not available' issues.
    # Use mp4 when possible, otherwise fall back to best.
    fmt = "bv*+ba/best"
    base_cmd = [
        "yt-dlp",
        "-f",
        fmt,
        "--merge-output-format",
        "mp4",
        "--retries",
        "3",
        "--fragment-retries",
        "3",
        "--concurrent-fragments",
        "4",
        "-o",
        str(output_path),
        url,
    ]
    try:
        _run(base_cmd)
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or e.stdout or "").strip()

        # Common YouTube breakage: nsig extraction fails because yt-dlp is outdated or blocked.
        # Try a conservative fallback that avoids some signature processing for certain videos.
        if "nsig extraction failed" in msg or "downloaded file is empty" in msg or "Signature extraction" in msg:
            fallback_cmd = base_cmd[:]
            # This can help in some cases where signature decryption breaks.
            fallback_cmd.insert(1, "--extractor-args")
            fallback_cmd.insert(2, "youtube:player_client=android")
            try:
                _run(fallback_cmd)
            except subprocess.CalledProcessError as e2:
                msg2 = (e2.stderr or e2.stdout or "").strip()
                raise RuntimeError(
                    "yt-dlp failed (YouTube extraction). This usually means yt-dlp needs an update or the video is restricted. "
                    f"Details: {msg2}"
                ) from e2
        else:
            raise RuntimeError(f"yt-dlp failed: {msg}") from e

    # Validate output not empty
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(
            "yt-dlp reported success but the output file is empty. Try updating yt-dlp or use a different video."
        )
    return output_path
