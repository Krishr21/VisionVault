from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class RerankConfig:
    enabled: bool = os.getenv("RERANK_ENABLE", "1").lower() not in {"0", "false", "no", "off"}
    model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    top_k: int = int(os.getenv("RERANK_TOP_K", "5"))
    candidate_k: int = int(os.getenv("RETRIEVE_K", "50"))


class Reranker:
    """Cross-encoder reranker for improving precision.

    Usage:
      reranker.rerank(query, candidates=[{"text": ..., **payload}], top_k=5)

    Returns candidates sorted by rerank_score (desc). If disabled/unavailable, falls back.
    """

    def __init__(self, cfg: Optional[RerankConfig] = None):
        self.cfg = cfg or RerankConfig()
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        if not self.cfg.enabled:
            self._model = False
            return
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.cfg.model)
        except Exception:
            # If model can't load (missing deps/CPU constraints), disable gracefully.
            self._model = False

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        self._load()
        k = top_k if top_k is not None else self.cfg.top_k

        if not candidates:
            return []

        # Fallback
        if self._model is False:
            return candidates[:k]

        model = self._model
        assert model is not None

        pairs: List[Tuple[str, str]] = []
        for c in candidates:
            t = (c.get(text_key) or "").strip()
            pairs.append((query, t))

        scores = model.predict(pairs)
        reranked = []
        for c, s in zip(candidates, scores):
            cc = dict(c)
            cc["rerank_score"] = float(s)
            reranked.append(cc)

        reranked.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return reranked[:k]
