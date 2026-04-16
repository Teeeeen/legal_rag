"""Embedding 初始化 — 单例缓存"""

from langchain_ollama import OllamaEmbeddings
from app.config import settings

_embed_cache: dict[str, OllamaEmbeddings] = {}


def get_embeddings(model: str | None = None) -> OllamaEmbeddings:
    _model = model or settings.EMBEDDING_MODEL
    if _model not in _embed_cache:
        _embed_cache[_model] = OllamaEmbeddings(
            model=_model,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _embed_cache[_model]
