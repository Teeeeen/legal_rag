"""管线配置与策略枚举测试"""

import pytest


def test_pipeline_config_defaults():
    """测试 PipelineConfig 默认值与旧行为一致"""
    from app.services.pipeline import PipelineConfig, QueryTransformStrategy, RerankStrategy, GenerationStrategy

    cfg = PipelineConfig()
    assert cfg.query_transform == QueryTransformStrategy.NONE
    assert cfg.rerank_strategy == RerankStrategy.SIMPLE
    assert cfg.generation_strategy == GenerationStrategy.STANDARD
    assert cfg.use_kg is False
    assert cfg.top_k == 5
    assert cfg.collection_names == ["laws", "cases"]


def test_pipeline_config_to_dict():
    """测试 PipelineConfig.to_dict()"""
    from app.services.pipeline import PipelineConfig

    cfg = PipelineConfig()
    d = cfg.to_dict()
    assert d["query_transform"] == "none"
    assert d["rerank_strategy"] == "simple"
    assert d["generation_strategy"] == "standard"
    assert d["use_kg"] is False


def test_strategy_enum_values():
    """测试策略枚举值覆盖所有 API 参数"""
    from app.services.pipeline import QueryTransformStrategy, RerankStrategy, GenerationStrategy

    # QueryTransform
    assert QueryTransformStrategy("none") == QueryTransformStrategy.NONE
    assert QueryTransformStrategy("multi_query") == QueryTransformStrategy.MULTI_QUERY
    assert QueryTransformStrategy("hyde") == QueryTransformStrategy.HYDE
    assert QueryTransformStrategy("decompose") == QueryTransformStrategy.DECOMPOSE
    assert QueryTransformStrategy("multi_query_hyde") == QueryTransformStrategy.MULTI_QUERY_HYDE

    # Rerank
    assert RerankStrategy("none") == RerankStrategy.NONE
    assert RerankStrategy("simple") == RerankStrategy.SIMPLE
    assert RerankStrategy("llm") == RerankStrategy.LLM

    # Generation
    assert GenerationStrategy("standard") == GenerationStrategy.STANDARD
    assert GenerationStrategy("chain_of_thought") == GenerationStrategy.COT
    assert GenerationStrategy("self_reflect") == GenerationStrategy.SELF_REFLECT
    assert GenerationStrategy("structured_legal") == GenerationStrategy.STRUCTURED


def test_backward_compat_use_rerank_false():
    """测试遗留参数 use_rerank=False 映射到 rerank_strategy='none'"""
    from app.services.pipeline import RerankStrategy

    # 模拟 rag_service.py 中的映射逻辑
    use_rerank = False
    rerank_strategy = "simple"

    effective = rerank_strategy
    if not use_rerank:
        effective = "none"

    assert RerankStrategy(effective) == RerankStrategy.NONE


def test_backward_compat_use_query_rewrite_true():
    """测试遗留参数 use_query_rewrite=True 映射到 query_transform='multi_query'"""
    from app.services.pipeline import QueryTransformStrategy

    use_query_rewrite = True
    query_transform = "none"

    effective = query_transform
    if use_query_rewrite and query_transform == "none":
        effective = "multi_query"

    assert QueryTransformStrategy(effective) == QueryTransformStrategy.MULTI_QUERY


def test_resolve_collections():
    """测试 collection 名称解析"""
    from app.services.pipeline import resolve_collections

    assert resolve_collections("laws") == ["laws"]
    assert resolve_collections("cases") == ["cases"]
    assert resolve_collections("all") == ["laws", "cases"]


def test_build_context_empty():
    """测试空文档列表的上下文构建"""
    from app.services.pipeline import build_context

    assert build_context([]) == "未找到相关参考资料。"


def test_build_context_with_docs():
    """测试有文档时的上下文构建"""
    from langchain_core.documents import Document
    from app.services.pipeline import build_context

    docs = [
        Document(page_content="测试法律条文内容", metadata={"doc_type": "law", "law_name": "测试法"}),
    ]
    ctx = build_context(docs)
    assert "来源1" in ctx
    assert "测试法律条文内容" in ctx
