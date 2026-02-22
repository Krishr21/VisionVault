from __future__ import annotations

from typing import List
import os


class Embedder:
    def __init__(self, model_name: str | None = None):
        from sentence_transformers import SentenceTransformer

        # More accurate default than bge-small while still reasonably fast.
        # Can be overridden via env EMBED_MODEL or by passing model_name.
        resolved = model_name or os.getenv("EMBED_MODEL") or "BAAI/bge-base-en-v1.5"
        self.model_name = resolved
        self.model = SentenceTransformer(resolved)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, normalize_embeddings=True)
