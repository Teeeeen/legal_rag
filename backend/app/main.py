"""FastAPI 应用入口"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.api import chat, knowledge, performance

app = FastAPI(
    title="法律 RAG 系统",
    description="基于 LangChain + Qwen3:8B + BGE-M3 的法律智能问答系统",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(knowledge.router, prefix=settings.API_PREFIX)
app.include_router(performance.router, prefix=settings.API_PREFIX)


@app.on_event("startup")
async def warmup_local_models():
    """预热本地模型：Ollama 首次加载模型到 GPU 需要 10-30s（冷启动），
    在应用启动时主动触发加载，避免用户首次查询遭遇长等待。
    这是本地部署特有的优化——云端 API 不存在冷启动问题。"""
    import logging
    logger = logging.getLogger("uvicorn")
    try:
        logger.info("正在预热本地 LLM（加载模型到 GPU 显存）...")
        from app.core.llm import get_llm
        llm = get_llm()
        await llm.ainvoke("你好")
        logger.info(f"LLM 预热完成: {settings.LLM_MODEL}")

        logger.info("正在预热 Embedding 模型...")
        from app.core.embeddings import get_embeddings
        get_embeddings().embed_query("测试")
        logger.info(f"Embedding 预热完成: {settings.EMBEDDING_MODEL}")
    except Exception as e:
        logger.warning(f"模型预热失败（Ollama 可能未启动）: {e}")


@app.get("/")
async def root():
    return {
        "name": "法律 RAG 系统",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """全局健康检查 — 包含本地模型加载状态"""
    from app.services.kb_service import get_kb_stats
    from app.services.perf_service import get_ollama_status
    stats = get_kb_stats()
    ollama = await get_ollama_status()
    return {
        "status": "healthy",
        "models": {
            "llm": settings.LLM_MODEL,
            "embedding": settings.EMBEDDING_MODEL,
        },
        "ollama": ollama,
        "knowledge_base": stats,
    }
