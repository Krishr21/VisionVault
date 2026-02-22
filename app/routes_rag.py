from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.schemas_rag import SearchRagRequest, SearchRagResponse, RagSource
from rag.synthesize import answer_question


router_rag = APIRouter(prefix="/rag")


def _index_dir(video_id: str) -> Path:
    # Reuse existing data/index/<video_id> location
    return Path(__file__).resolve().parents[1] / "data" / "index" / video_id


@router_rag.post("/search", response_model=SearchRagResponse)
def rag_search(req: SearchRagRequest):
    idx = _index_dir(req.video_id)
    if not idx.exists():
        raise HTTPException(status_code=404, detail=f"Index not found for video_id={req.video_id}")

    try:
        answer, sources = answer_question(
            idx, req.query, top_k=req.top_k, llm_model=req.llm_model
        )
        return SearchRagResponse(
            query=req.query,
            answer=answer,
            sources=[RagSource(**s) for s in sources],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
