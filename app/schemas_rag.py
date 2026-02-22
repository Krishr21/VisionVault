from pydantic import BaseModel
from typing import List, Optional


class SearchRagRequest(BaseModel):
    query: str
    top_k: int = 5
    video_id: str
    llm_model: str = "llama3.1:8b"


class RagSource(BaseModel):
    start: float
    end: float
    transcript: str
    caption: str
    score: Optional[float] = None


class SearchRagResponse(BaseModel):
    query: str
    answer: str
    sources: List[RagSource]
