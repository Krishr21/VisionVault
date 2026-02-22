from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional, Dict, List

from processing.download_video import download_youtube
from processing.extract_frames import extract_audio, extract_frames, parse_frame_number, frame_number_to_timestamp_seconds
from processing.transcribe import transcribe_audio
from processing.caption_frames import caption_frames
from processing.chunking import align_transcript_and_captions
from embeddings.embed import Embedder
from embeddings.faiss_index import FaissStore
from rag.llamaindex_rag import build_index_from_chunks
import os


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Simple in-process cancel flag for MVP. This cancels the currently running ingestion
# in this single-server process. (For multi-worker, we'd use Redis/DB.)
_CANCEL_INGEST = False


def request_cancel_ingest() -> None:
    global _CANCEL_INGEST
    _CANCEL_INGEST = True


def _check_cancel() -> None:
    if _CANCEL_INGEST:
        raise RuntimeError("Ingest cancelled")


def _paths(video_id: str) -> Dict[str, Path]:
    return {
        "videos": DATA_DIR / "videos" / video_id / "video.mp4",
        "audio": DATA_DIR / "audio" / video_id / "audio.mp3",
        "frames": DATA_DIR / "frames" / video_id,
        "transcript": DATA_DIR / "transcripts" / video_id / "transcript.json",
        "chunks": DATA_DIR / "chunks" / video_id / "chunks.json",
        "index": DATA_DIR / "index" / video_id,
        "meta": DATA_DIR / "meta" / video_id / "meta.json",
    }


def ingest_video(
    source_type: str,
    source: str,
    fps: float = 1.0,
    max_frames: Optional[int] = None,
    enable_captions: bool = True,
) -> Dict:
    global _CANCEL_INGEST
    _CANCEL_INGEST = False
    video_id = uuid.uuid4().hex[:10]
    p = _paths(video_id)
    for d in [
        DATA_DIR,
        p["videos"].parent,
        p["audio"].parent,
        p["frames"].parent,
        p["transcript"].parent,
        p["chunks"].parent,
        p["index"],
        p["meta"].parent,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    # Persist lightweight metadata so the UI can link out (e.g., open YouTube at timestamp).
    p["meta"].write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_type": source_type,
                "source": source,
                "fps": fps,
                "max_frames": max_frames,
                "enable_captions": enable_captions,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    video_path = p["videos"]
    if source_type == "youtube":
        _check_cancel()
        download_youtube(source, video_path)
    elif source_type == "local":
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"Local video not found: {src}")
        video_path.write_bytes(src.read_bytes())
    else:
        raise ValueError("source_type must be youtube or local")

    _check_cancel()
    audio_path = extract_audio(video_path, p["audio"])

    frame_files: List[Path] = []
    # Allow turning off frame extraction entirely for faster ingest.
    if fps > 0 and (max_frames is None or max_frames > 0):
        _check_cancel()
        frames_dir = extract_frames(video_path, p["frames"], fps=fps)
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        if max_frames is not None:
            frame_files = frame_files[: max_frames]

    _check_cancel()
    transcript = transcribe_audio(audio_path)
    p["transcript"].write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    _check_cancel()
    cap_map = {} if not enable_captions else caption_frames(frame_files)
    frame_desc: List[Dict] = []
    for fp in frame_files:
        n = parse_frame_number(fp.name)
        ts = frame_number_to_timestamp_seconds(n, fps)
        frame_desc.append({"frame_file": fp.name, "timestamp": ts, "caption": cap_map.get(fp.name, "")})

    chunks = align_transcript_and_captions(video_id, transcript, frame_desc)
    chunk_dicts = [
        {
            "video_id": c.video_id,
            "start": c.start,
            "end": c.end,
            "transcript": c.transcript,
            "caption": c.caption,
            "frame_file": c.frame_file,
            "text": c.text,
        }
        for c in chunks
    ]
    p["chunks"].write_text(json.dumps(chunk_dicts, ensure_ascii=False, indent=2), encoding="utf-8")

    embedder = Embedder()
    texts = [c["text"] for c in chunk_dicts]
    _check_cancel()
    embs = embedder.encode(texts)

    # Vector storage: FAISS (default) or Qdrant (if configured).
    # Qdrant stores one global collection and filters by video_id.
    if os.getenv("QDRANT_URL"):
        from vectorstore.qdrant_store import QdrantStore

        qstore = QdrantStore(dim=embs.shape[1])
        _check_cancel()
        qstore.upsert(embs, payloads=chunk_dicts)
    else:
        store = FaissStore(p["index"], dim=embs.shape[1])
        store.add(embs, metas=chunk_dicts)

    # Also persist a LlamaIndex-compatible index for RAG answer synthesis.
    # This writes into the same directory, adding its own artifacts.
    try:
        _check_cancel()
        build_index_from_chunks(chunk_dicts, p["index"])
    except Exception:
        # Don't fail ingestion if RAG index build fails; base search still works.
        pass

    return {"video_id": video_id, "chunks_indexed": len(chunk_dicts)}


def search(video_id: str, query: str, top_k: int = 5) -> List[Dict]:
    p = _paths(video_id)
    embedder = Embedder()
    q = embedder.encode([query])

    # Retrieve more candidates, then optionally rerank down to top_k.
    retrieve_k = int(os.getenv("RETRIEVE_K", str(max(top_k, 50))))

    # Filtering so we don't always return exactly top_k when results are weak.
    #
    # Controls (env):
    #   MIN_HIT_SCORE: absolute cutoff (default: 0)
    #   RELATIVE_MIN: keep hits with score >= best_score * RELATIVE_MIN (default: 0.90)
    #   DROPOFF_GAP: stop when score drops by >= this much vs previous hit (default: 0.08)
    #   MIN_RETURN_HITS: minimum hits to return even if thresholds filter too hard (default: 1)
    #
    # Note: score meaning differs by backend/model.
    min_score = float(os.getenv("MIN_HIT_SCORE", "0"))
    relative_min = float(os.getenv("RELATIVE_MIN", "0.90"))
    dropoff_gap = float(os.getenv("DROPOFF_GAP", "0.08"))
    min_return_hits = int(os.getenv("MIN_RETURN_HITS", "1"))

    def _effective_score(h: Dict) -> float:
        # Prefer reranker score if present.
        try:
            return float(h.get("rerank_score", h.get("score", 0.0)) or 0.0)
        except Exception:
            return 0.0

    def _filter_hits(hits_in: List[Dict], k: int) -> List[Dict]:
        if not hits_in:
            return []

        # Compute best score and apply relative+absolute thresholds.
        scores = [_effective_score(h) for h in hits_in]
        best = scores[0]
        if best <= 0:
            return hits_in[: min(k, max(min_return_hits, 0))]

        out: List[Dict] = []
        prev = best
        for h, s in zip(hits_in, scores):
            if s < min_score:
                continue
            if s < best * relative_min:
                break
            # Stop if there's a sharp cliff in relevance.
            if (prev - s) >= dropoff_gap:
                break
            out.append(h)
            prev = s
            if len(out) >= k:
                break

        if len(out) < min_return_hits:
            return hits_in[: min(k, max(min_return_hits, 0))]
        return out

    def _thumb_url(m: Dict) -> Optional[str]:
        ff = m.get("frame_file")
        if not ff:
            return None
        # Served by FastAPI route: /videos/{video_id}/frames/{frame_file}
        return f"/videos/{video_id}/frames/{ff}"

    # If Qdrant is configured, search there first.
    if os.getenv("QDRANT_URL"):
        from vectorstore.qdrant_store import QdrantStore

        qstore = QdrantStore(dim=q.shape[1])
        results = qstore.search(q, top_k=retrieve_k, video_id=video_id)
        hits: List[Dict] = []
        for score, payload in results:
            m = dict(payload)
            m["score"] = float(score)
            m["thumbnail_path"] = _thumb_url(m)
            hits.append(m)
        # Optional rerank
        try:
            from retrieval.rerank import Reranker

            reranker = Reranker()
            if reranker.cfg.enabled:
                hits = reranker.rerank(query, hits, top_k=top_k, text_key="text")
        except Exception:
            pass

        return _filter_hits(hits, top_k)

    # Fallback: local FAISS index
    store = FaissStore(p["index"], dim=q.shape[1])
    scores, idxs, metas = store.search(q, top_k=retrieve_k)

    hits: List[Dict] = []
    for score, idx in zip(scores.tolist(), idxs.tolist()):
        if idx < 0 or idx >= len(metas):
            continue
        m = metas[idx]
        m = dict(m)
        m["score"] = float(score)
        m["thumbnail_path"] = _thumb_url(m)
        hits.append(m)

    # Optional rerank
    try:
        from retrieval.rerank import Reranker

        reranker = Reranker()
        if reranker.cfg.enabled:
            hits = reranker.rerank(query, hits, top_k=top_k, text_key="text")
    except Exception:
        pass

    return _filter_hits(hits, top_k)
