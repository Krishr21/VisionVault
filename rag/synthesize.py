from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from llama_index.core import Settings
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.llms.ollama import Ollama

from rag.llamaindex_rag import load_index


SYSTEM_PROMPT = """You are VisionVault, a video-context search assistant.
You answer using only the retrieved evidence from transcript and visual captions.
Always cite timestamps like [12.3s–18.4s]. If evidence is weak, say so."""


def answer_question(
    index_dir: Path,
    query: str,
    top_k: int = 5,
    llm_model: str = "llama3.1:8b",
) -> Tuple[str, List[Dict]]:
    index = load_index(index_dir)

    # LLM for synthesis (local via Ollama)
    Settings.llm = Ollama(model=llm_model, request_timeout=180.0)

    retriever = index.as_retriever(similarity_top_k=top_k)
    synth = get_response_synthesizer(
        response_mode="compact",
        text_qa_template=None,
    )

    engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=synth,
    )

    # Inject system instructions by prefixing query (simple + robust)
    response = engine.query(f"{SYSTEM_PROMPT}\n\nUser question: {query}")

    sources: List[Dict] = []
    for sn in getattr(response, "source_nodes", []) or []:
        md = sn.node.metadata or {}
        sources.append(
            {
                "start": float(md.get("start", 0.0)),
                "end": float(md.get("end", 0.0)),
                "transcript": md.get("transcript", ""),
                "caption": md.get("caption", ""),
                "score": getattr(sn, "score", None),
            }
        )

    return str(response), sources
