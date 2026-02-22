from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
import json
import os

from app.schemas import IngestRequest, IngestResponse, SearchRequest, SearchResponse, SearchHit
from processing.pipeline import ingest_video, search, request_cancel_ingest

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    try:
        result = ingest_video(
            req.source_type,
            req.source,
            fps=req.fps,
            max_frames=req.max_frames,
            enable_captions=req.enable_captions,
        )
        return IngestResponse(video_id=result["video_id"], status="ok", chunks_indexed=result["chunks_indexed"])
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/cancel")
def cancel_ingest():
    """Best-effort cancellation for the currently running ingest in this process."""
    request_cancel_ingest()
    return {"status": "cancelling"}


@router.post("/search", response_model=SearchResponse)
def do_search(req: SearchRequest):
    if not req.video_id:
        raise HTTPException(status_code=400, detail="video_id is required for MVP")
    if not (req.query or "").strip():
        raise HTTPException(status_code=400, detail="query must not be empty")
    hits = search(req.video_id, req.query, top_k=req.top_k)
    return SearchResponse(
        query=req.query,
        hits=[
            SearchHit(
                video_id=h["video_id"],
                start=float(h["start"]),
                end=float(h["end"]),
                score=float(h["score"]),
                transcript=h.get("transcript", ""),
                caption=h.get("caption", ""),
                thumbnail_path=h.get("thumbnail_path"),
            )
            for h in hits
        ],
    )


@router.get("/videos/{video_id}/meta")
def get_video_meta(video_id: str):
    meta_path = Path(__file__).resolve().parents[1] / "data" / "meta" / video_id / "meta.json"
    if not meta_path.exists():
        raise HTTPException(status_code=404, detail=f"No metadata found for video_id={video_id}")
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read metadata: {e}")


@router.get("/videos/{video_id}/frames/{frame_file}")
def get_video_frame(video_id: str, frame_file: str):
    """Serve extracted frame thumbnails.

    We intentionally do not accept arbitrary filesystem paths.
    Allowed: filenames like 'frame_000010.jpg' within data/frames/<video_id>/.
    """

    if not frame_file.startswith("frame_") or ("/" in frame_file) or ("\\" in frame_file):
        raise HTTPException(status_code=400, detail="Invalid frame filename")
    if not (frame_file.endswith(".jpg") or frame_file.endswith(".jpeg")):
        raise HTTPException(status_code=400, detail="Invalid frame file extension")

    frames_dir = Path(__file__).resolve().parents[1] / "data" / "frames" / video_id
    frame_path = (frames_dir / frame_file).resolve()

    # Prevent path traversal: resolved path must remain inside frames_dir
    try:
        frame_path.relative_to(frames_dir.resolve())
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid frame path")

    if not frame_path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")

    return FileResponse(str(frame_path), media_type="image/jpeg")


@router.get("/health/vectorstore")
def health_vectorstore():
    """Reports which vector backend is active.

    - If QDRANT_URL is set, we use Qdrant and attempt a lightweight connectivity check.
    - Otherwise, we use local FAISS indexes.
    """
    qdrant_url = os.getenv("QDRANT_URL")
    if not qdrant_url:
        return {"backend": "faiss"}

    collection = os.getenv("QDRANT_COLLECTION", "visionvault_chunks")
    try:
        # Mirror QdrantStore naming convention for clarity.
        # NOTE: we can't know actual dim without loading the embedder; report base name here.
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, api_key=os.getenv("QDRANT_API_KEY"))
        # A fast call that validates auth + reachability.
        _ = client.get_collections()
        return {"backend": "qdrant", "url": qdrant_url, "collection_base": collection, "ok": True}
    except Exception as e:
        return {"backend": "qdrant", "url": qdrant_url, "collection_base": collection, "ok": False, "error": str(e)}
