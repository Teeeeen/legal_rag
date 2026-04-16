"""RAG 主服务 — 向后兼容的入口，委托给 RAGPipeline 执行"""

from app.models.schemas import ChatResponse
from app.services.pipeline import (
    RAGPipeline,
    PipelineConfig,
    QueryTransformStrategy,
    RerankStrategy,
    GenerationStrategy,
    resolve_collections,
    build_context,
)


async def rag_query(
    question: str,
    use_rerank: bool = True,
    use_query_rewrite: bool = False,
    top_k: int = 5,
    collection: str = "all",
    # --- 高级选项 ---
    query_transform: str = "none",
    rerank_strategy: str = "simple",
    generation_strategy: str = "standard",
    use_kg: bool = False,
) -> ChatResponse:
    """
    执行完整的 RAG 查询流程（向后兼容入口）

    遗留参数映射：
    - use_rerank=False  → rerank_strategy="none"
    - use_query_rewrite=True → query_transform="multi_query"（如果 query_transform 仍为 "none"）
    """
    # 遗留参数向新策略的映射
    effective_rerank = rerank_strategy
    if not use_rerank:
        effective_rerank = "none"

    effective_query_transform = query_transform
    if use_query_rewrite and query_transform == "none":
        effective_query_transform = "multi_query"

    config = PipelineConfig(
        query_transform=QueryTransformStrategy(effective_query_transform),
        rerank_strategy=RerankStrategy(effective_rerank),
        generation_strategy=GenerationStrategy(generation_strategy),
        use_kg=use_kg,
        top_k=top_k,
        collection_names=resolve_collections(collection),
    )

    pipeline = RAGPipeline(config)
    return await pipeline.execute(question)
