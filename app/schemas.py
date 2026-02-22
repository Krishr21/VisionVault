from pydantic import BaseModel, Field
from typing import Optional, List


class IngestRequest(BaseModel):
    source_type: str = Field(description="youtube|local")
    source: str = Field(description="YouTube URL or local absolute path")
    fps: float = 1.0
    max_frames: Optional[int] = None
    enable_captions: bool = Field(default=True, description="If false, skip frame captioning for faster ingest")


class IngestResponse(BaseModel):
    video_id: str
    status: str
    chunks_indexed: int


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    video_id: Optional[str] = None


class SearchHit(BaseModel):
    video_id: str
    start: float
    end: float
    score: float
    transcript: str
    caption: str
    thumbnail_path: Optional[str] = None


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]
