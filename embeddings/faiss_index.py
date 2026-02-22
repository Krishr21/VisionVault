from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import faiss


class FaissStore:
    def __init__(self, index_dir: Path, dim: int):
        self.index_dir = index_dir
        self.dim = dim
        self.index_path = index_dir / "index.faiss"
        self.meta_path = index_dir / "meta.jsonl"
        self._index = None

    def _ensure_dir(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def _load(self):
        self._ensure_dir()
        if self._index is None:
            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
            else:
                self._index = faiss.IndexFlatIP(self.dim)

    def add(self, embeddings: np.ndarray, metas: List[Dict]):
        self._load()
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        self._index.add(embeddings)
        with self.meta_path.open("a", encoding="utf-8") as f:
            for m in metas:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")
        faiss.write_index(self._index, str(self.index_path))

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        self._load()
        if query_emb.ndim == 1:
            query_emb = query_emb[None, :]
        if query_emb.dtype != np.float32:
            query_emb = query_emb.astype(np.float32)
        scores, idxs = self._index.search(query_emb, top_k)

        metas = []
        if self.meta_path.exists():
            with self.meta_path.open("r", encoding="utf-8") as f:
                metas = [json.loads(line) for line in f if line.strip()]

        return scores[0], idxs[0], metas
