"""RAG 可插拔管线 — 编排查询变换、检索、重排序、生成等阶段"""

import time
from dataclasses import dataclass, field, asdict
from enum import Enum

from langchain_core.documents import Document

from app.core.llm import get_llm
from app.core.retriever import get_hybrid_retriever
from app.services.reranker import simple_rerank, llm_rerank
from app.services.query_rewriter import multi_query_rewrite
from app.services.prompts import (
    LEGAL_QA_PROMPT,
    LEGAL_COT_PROMPT,
    LEGAL_STRUCTURED_PROMPT,
)
from app.utils.metadata import format_source_display
from app.models.schemas import ChatResponse, SourceDocument, StageMetrics
from app.config import settings


# ===================== 策略枚举 =====================

class QueryTransformStrategy(str, Enum):
    NONE = "none"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    DECOMPOSE = "decompose"
    MULTI_QUERY_HYDE = "multi_query_hyde"


class RerankStrategy(str, Enum):
    NONE = "none"
    SIMPLE = "simple"
    LLM = "llm"


class GenerationStrategy(str, Enum):
    STANDARD = "standard"
    COT = "chain_of_thought"
    SELF_REFLECT = "self_reflect"
    STRUCTURED = "structured_legal"


# ===================== 管线配置 =====================

@dataclass
class PipelineConfig:
    query_transform: QueryTransformStrategy = QueryTransformStrategy.NONE
    rerank_strategy: RerankStrategy = RerankStrategy.SIMPLE
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD
    use_kg: bool = False
    top_k: int = 5
    collection_names: list[str] = field(default_factory=lambda: ["laws", "cases"])

    def to_dict(self) -> dict:
        return {
            "query_transform": self.query_transform.value,
            "rerank_strategy": self.rerank_strategy.value,
            "generation_strategy": self.generation_strategy.value,
            "use_kg": self.use_kg,
            "top_k": self.top_k,
            "collection_names": self.collection_names,
        }


# ===================== RAG 管线 =====================

class RAGPipeline:
    """可插拔 RAG 管线，按配置调度各阶段策略"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics: dict = {
            "query_rewrite_ms": None,
            "retrieval_ms": 0,
            "rerank_ms": None,
            "generation_ms": 0,
            "total_ms": 0,
            "kg_lookup_ms": None,
            "self_reflect_ms": None,
            "was_corrected": False,
            "llm_calls": 0,
            "llm_calls_saved": 0,
        }

    async def execute(self, question: str) -> ChatResponse:
        total_start = time.time()

        # Stage 1: 查询变换
        search_queries, hyde_doc, rewritten_queries = await self._query_transform(question)
        self.metrics["llm_calls"] += self._count_transform_calls()

        # Stage 2: 检索（纯检索，零 LLM 调用）
        all_docs = await self._retrieve(search_queries, hyde_doc)

        # Stage 2.5: KG 查找（并入检索结果）
        kg_entities: list[str] = []
        if self.config.use_kg:
            kg_entities, kg_docs = await self._kg_lookup(question)
            # KG 精确匹配命中时节省 1 次 LLM 调用
            if kg_docs:
                self.metrics["llm_calls_saved"] += 1
                existing_keys = {hash(d.page_content[:200]) for d in all_docs}
                for kd in kg_docs:
                    key = hash(kd.page_content[:200])
                    if key not in existing_keys:
                        all_docs.insert(0, kd)
                        existing_keys.add(key)

        # Stage 3: 重排序
        reranked_docs = await self._rerank(question, all_docs)
        if self.config.rerank_strategy == RerankStrategy.LLM:
            self.metrics["llm_calls"] += 1
            # 批量重排序节省 N-1 次调用（相比逐文档评分）
            self.metrics["llm_calls_saved"] += max(0, len(all_docs) - 1)

        # Stage 4: 生成
        answer, was_corrected = await self._generate(question, reranked_docs)
        self.metrics["was_corrected"] = was_corrected
        self.metrics["llm_calls"] += 1  # 生成至少 1 次
        if was_corrected:
            self.metrics["llm_calls"] += 2  # 反思验证 + 修正

        # 总耗时
        self.metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)

        # 构建来源
        sources = []
        for doc in reranked_docs:
            sources.append(SourceDocument(
                content=doc.page_content[:500],
                metadata=doc.metadata,
                score=None,
            ))

        return ChatResponse(
            answer=answer,
            sources=sources,
            metrics=StageMetrics(**self.metrics),
            rewritten_queries=rewritten_queries,
            kg_entities=kg_entities or None,
            generation_strategy=self.config.generation_strategy.value,
            pipeline_config=self.config.to_dict(),
        )

    # ---- Stage 1: 查询变换 ----

    def _count_transform_calls(self) -> int:
        """统计查询变换阶段的 LLM 调用次数"""
        s = self.config.query_transform
        if s == QueryTransformStrategy.NONE:
            return 0
        if s == QueryTransformStrategy.MULTI_QUERY_HYDE:
            return 2  # 并行但各调一次
        return 1  # MULTI_QUERY / HYDE / DECOMPOSE 各一次

    async def _query_transform(self, question: str) -> tuple[list[str], str | None, list[str] | None]:
        """返回 (search_queries, hyde_doc_or_none, rewritten_queries_or_none)"""
        strategy = self.config.query_transform
        if strategy == QueryTransformStrategy.NONE:
            return [question], None, None

        t0 = time.time()
        search_queries = [question]
        hyde_doc = None
        rewritten_queries = None

        if strategy == QueryTransformStrategy.MULTI_QUERY:
            search_queries = await multi_query_rewrite(question)
            rewritten_queries = search_queries

        elif strategy == QueryTransformStrategy.HYDE:
            from app.services.hyde import hyde_transform
            original, hypo = await hyde_transform(question)
            search_queries = [original]
            hyde_doc = hypo

        elif strategy == QueryTransformStrategy.DECOMPOSE:
            from app.services.query_rewriter import decompose_query
            sub_qs = await decompose_query(question)
            search_queries = sub_qs
            rewritten_queries = sub_qs

        elif strategy == QueryTransformStrategy.MULTI_QUERY_HYDE:
            import asyncio
            from app.services.hyde import hyde_transform
            # 并行执行多查询重写和 HyDE 生成
            mq_task = asyncio.create_task(multi_query_rewrite(question))
            hyde_task = asyncio.create_task(hyde_transform(question))
            mq, (_, hypo) = await asyncio.gather(mq_task, hyde_task)
            search_queries = mq
            hyde_doc = hypo
            rewritten_queries = mq

        self.metrics["query_rewrite_ms"] = round((time.time() - t0) * 1000, 1)
        return search_queries, hyde_doc, rewritten_queries

    # ---- Stage 2: 检索 ----

    async def _retrieve(self, search_queries: list[str], hyde_doc: str | None) -> list[Document]:
        t0 = time.time()
        retriever = get_hybrid_retriever(self.config.collection_names)

        all_docs: list[Document] = []
        seen_contents: set[int] = set()

        if hyde_doc:
            # 使用 split query: BM25 用原始查询, 向量用假设文档
            docs = retriever.search_with_split_queries(
                bm25_query=search_queries[0],
                vector_query=hyde_doc,
            )
            for doc in docs:
                key = hash(doc.page_content[:200])
                if key not in seen_contents:
                    seen_contents.add(key)
                    all_docs.append(doc)

        # 常规多查询检索
        for q in search_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                key = hash(doc.page_content[:200])
                if key not in seen_contents:
                    seen_contents.add(key)
                    all_docs.append(doc)

        self.metrics["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        return all_docs

    # ---- Stage 2.5: KG 查找 ----

    async def _kg_lookup(self, question: str) -> tuple[list[str], list[Document]]:
        t0 = time.time()
        try:
            from app.services.kg_service import extract_crime_entities, kg_lookup
            entities = await extract_crime_entities(question)
            docs = kg_lookup(entities) if entities else []
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            return entities, docs
        except Exception:
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            return [], []

    # ---- Stage 3: 重排序 ----

    async def _rerank(self, question: str, all_docs: list[Document]) -> list[Document]:
        strategy = self.config.rerank_strategy
        top_k = self.config.top_k

        if strategy == RerankStrategy.NONE or not all_docs:
            return all_docs[:top_k]

        t0 = time.time()
        try:
            if strategy == RerankStrategy.SIMPLE:
                scored = simple_rerank(question, all_docs, top_k=top_k)
                reranked = [doc for doc, _ in scored]
            elif strategy == RerankStrategy.LLM:
                scored = await llm_rerank(question, all_docs, top_k=top_k)
                reranked = [doc for doc, _ in scored]
            else:
                reranked = all_docs[:top_k]
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return reranked
        except Exception:
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return all_docs[:top_k]

    # ---- Stage 4: 生成 ----

    async def _generate(self, question: str, docs: list[Document]) -> tuple[str, bool]:
        t0 = time.time()
        context = build_context(docs)
        strategy = self.config.generation_strategy

        # 选择 prompt
        if strategy == GenerationStrategy.COT:
            prompt = LEGAL_COT_PROMPT
        elif strategy == GenerationStrategy.STRUCTURED:
            prompt = LEGAL_STRUCTURED_PROMPT
        else:
            prompt = LEGAL_QA_PROMPT

        llm = get_llm()
        chain = prompt | llm
        response = await chain.ainvoke({"context": context, "question": question})
        answer = response.content
        was_corrected = False

        # 自我反思
        if strategy == GenerationStrategy.SELF_REFLECT:
            gen_ms = round((time.time() - t0) * 1000, 1)
            self.metrics["generation_ms"] = gen_ms

            t_reflect = time.time()
            from app.services.self_reflect import self_reflect_and_correct
            answer, was_corrected = await self_reflect_and_correct(
                question=question,
                initial_answer=answer,
                context=context,
                max_iterations=settings.SELF_REFLECT_MAX_ITER,
            )
            self.metrics["self_reflect_ms"] = round((time.time() - t_reflect) * 1000, 1)
            return answer, was_corrected

        self.metrics["generation_ms"] = round((time.time() - t0) * 1000, 1)
        return answer, was_corrected


# ===================== 公共辅助函数 =====================

def resolve_collections(collection: str) -> list[str]:
    """解析要检索的 collection"""
    if collection == "laws":
        return [settings.LAWS_COLLECTION]
    elif collection == "cases":
        return [settings.CASES_COLLECTION]
    else:
        return [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]


def build_context(docs: list[Document], max_length: int = 4000) -> str:
    """构建上下文文本 — 在本地 8B 模型有限的上下文窗口(8192 tokens)下，
    显式按文档优先级分配预算：KG 结构化知识 > 法律条文 > 案例。"""
    if not docs:
        return "未找到相关参考资料。"

    # 按 doc_type 优先级排序：kg > law/statute > case > other
    PRIORITY = {"kg": 0, "law": 1, "statute": 1, "case": 2}
    sorted_docs = sorted(docs, key=lambda d: PRIORITY.get(d.metadata.get("doc_type", ""), 3))

    parts = []
    total_len = 0
    for i, doc in enumerate(sorted_docs, 1):
        source_label = format_source_display(doc.metadata)
        segment = f"[来源{i}] {source_label}\n{doc.page_content}\n"
        if total_len + len(segment) > max_length:
            break
        parts.append(segment)
        total_len += len(segment)

    return "\n".join(parts)
