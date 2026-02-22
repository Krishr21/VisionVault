from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "qdrant-client is required for Qdrant support. Install it and set QDRANT_URL/QDRANT_API_KEY."
    ) from e


DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "visionvault_chunks")


@dataclass
class QdrantConfig:
    url: str
    api_key: Optional[str] = None
    collection: str = DEFAULT_COLLECTION


def _env_config() -> Optional[QdrantConfig]:
    url = os.getenv("QDRANT_URL")
    if not url:
        return None
    return QdrantConfig(url=url, api_key=os.getenv("QDRANT_API_KEY"), collection=DEFAULT_COLLECTION)


class QdrantStore:
    """Lightweight Qdrant-backed vector store.

    - One global collection
    - Filter by video_id
    - Stores full chunk text payload for MVP
    """

    def __init__(self, dim: int, cfg: Optional[QdrantConfig] = None):
        self.dim = dim
        self.cfg = cfg or _env_config()
        if not self.cfg:
            raise RuntimeError("QDRANT_URL env var not set; cannot use QdrantStore")
        self.client = QdrantClient(url=self.cfg.url, api_key=self.cfg.api_key)
        # If you switch embedding models, vector dimension can change.
        # Qdrant collections have a fixed dim, so we namespace collection by dim
        # to prevent hard failures (384 vs 768, etc.).
        base = self.cfg.collection
        self.collection = f"{base}_d{self.dim}"
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            existing = self.client.get_collections().collections
            if any(c.name == self.collection for c in existing):
                return
        except Exception:
            # If list fails, still try create (idempotent-ish).
            pass

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
            # Payload indexing improves filter performance.
            optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=20000),
        )
        # Create payload index for video_id for faster filtering.
        try:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name="video_id",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

    def upsert(self, vectors: np.ndarray, payloads: List[Dict[str, Any]]):
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        if vectors.ndim != 2:
            raise ValueError("vectors must be 2D array")
        if len(payloads) != vectors.shape[0]:
            raise ValueError("payload length must match vectors")

        def _as_qdrant_point_id(raw: Any, fallback_key: str) -> str:
            """Return a valid Qdrant point ID (UUID or unsigned int).

            Qdrant rejects arbitrary strings. We accept:
            - UUID strings
            - ints / numeric strings
            Otherwise we generate a deterministic UUID5 from fallback_key.
            """

            if raw is None:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, fallback_key))

            # ints are accepted
            if isinstance(raw, int):
                return raw  # type: ignore[return-value]

            s = str(raw).strip()
            if not s:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, fallback_key))

            # numeric strings are accepted
            if s.isdigit():
                return int(s)  # type: ignore[return-value]

            # UUID strings are accepted
            try:
                return str(uuid.UUID(s))
            except Exception:
                return str(uuid.uuid5(uuid.NAMESPACE_URL, fallback_key))

        points: List[qm.PointStruct] = []
        for i in range(vectors.shape[0]):
            # Prefer a stable id if provided, but ensure it's valid for Qdrant.
            raw_pid = payloads[i].get("point_id") or payloads[i].get("chunk_id")
            vid = payloads[i].get("video_id", "vid")
            start = payloads[i].get("start", "")
            end = payloads[i].get("end", "")
            fallback_key = f"{vid}:{start}:{end}:{i}"
            pid = _as_qdrant_point_id(raw_pid, fallback_key)
            points.append(qm.PointStruct(id=pid, vector=vectors[i].tolist(), payload=payloads[i]))

        self.client.upsert(collection_name=self.collection, points=points)

    def search(self, query_vec: np.ndarray, top_k: int = 5, video_id: Optional[str] = None) -> List[Tuple[float, Dict[str, Any]]]:
        if query_vec.ndim == 2:
            query_vec = query_vec[0]
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)

        flt = None
        if video_id:
            flt = qm.Filter(must=[qm.FieldCondition(key="video_id", match=qm.MatchValue(value=video_id))])

        res = self.client.search(
            collection_name=self.collection,
            query_vector=query_vec.tolist(),
            limit=top_k,
            with_payload=True,
            query_filter=flt,
        )

        out: List[Tuple[float, Dict[str, Any]]] = []
        for r in res:
            payload = dict(r.payload or {})
            out.append((float(r.score), payload))
        return out
