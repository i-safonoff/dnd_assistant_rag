import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # CPU, not GPU: vLLM's KV cache leaves ~750MB VRAM free at our gpu-memory-utilization
        # setting, not enough for bge-m3 to coexist without OOM. Embedding isn't latency-critical
        # enough (one-off ingestion, single-query retrieval) to fight vLLM for GPU memory.
        _model = SentenceTransformer(settings.embedding_model_name, device="cpu")
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)


def embed_query(text: str) -> np.ndarray:
    return embed_texts([text])[0]
