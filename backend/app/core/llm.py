"""LLM 初始化 — 实例缓存，避免在单 GPU 环境下重复创建连接"""

from langchain_ollama import ChatOllama
from app.config import settings

# 按 (model, temperature) 缓存 LLM 实例，本地部署场景下
# 同一参数组合复用同一连接，减少 Ollama 端的会话开销
_llm_cache: dict[tuple[str, float], ChatOllama] = {}


def get_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOllama:
    _model = model or settings.LLM_MODEL
    _temp = temperature if temperature is not None else settings.LLM_TEMPERATURE
    cache_key = (_model, _temp)

    if cache_key not in _llm_cache:
        _llm_cache[cache_key] = ChatOllama(
            model=_model,
            temperature=_temp,
            num_ctx=settings.LLM_NUM_CTX,
            base_url=settings.OLLAMA_BASE_URL,
        )
    return _llm_cache[cache_key]


def clear_llm_cache():
    """清空缓存（模型切换或测试时使用）"""
    _llm_cache.clear()
