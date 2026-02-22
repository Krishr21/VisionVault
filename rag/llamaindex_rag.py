from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

import faiss


def build_index_from_chunks(
    chunks: List[Dict],
    index_dir: Path,
    embed_model_name: str = "BAAI/bge-small-en",
) -> VectorStoreIndex:
    """Create and persist a LlamaIndex FAISS-backed index from our chunk dicts."""

    index_dir.mkdir(parents=True, exist_ok=True)

    docs: List[Document] = []
    for c in chunks:
        docs.append(
            Document(
                text=c.get("text") or "",
                metadata={
                    "video_id": c.get("video_id"),
                    "start": float(c.get("start", 0.0)),
                    "end": float(c.get("end", 0.0)),
                    "frame_file": c.get("frame_file"),
                    "transcript": c.get("transcript", ""),
                    "caption": c.get("caption", ""),
                },
            )
        )

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # Create a fresh FAISS index. We'll use inner-product over normalized embeddings.
    dim = embed_model.get_text_embedding("dimension_probe").__len__()
    faiss_index = faiss.IndexFlatIP(dim)

    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
    index.storage_context.persist(persist_dir=str(index_dir))
    return index


def load_index(
    index_dir: Path, embed_model_name: str = "BAAI/bge-small-en"
) -> VectorStoreIndex:
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    vector_store = FaissVectorStore.from_persist_dir(str(index_dir))
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=str(index_dir)
    )

    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, storage_context=storage_context, embed_model=embed_model
    )
