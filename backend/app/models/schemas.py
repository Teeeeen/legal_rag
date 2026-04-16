"""Pydantic 数据模型"""

from datetime import datetime
from pydantic import BaseModel, Field


# ===================== 通用响应包装 =====================

class APIResponse(BaseModel):
    code: int = 200
    data: object = None
    message: str = "success"


# ===================== 性能 =====================

class SystemInfo(BaseModel):
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float


# ===================== 质量评估 =====================

class RougeScores(BaseModel):
    rouge_1: float = 0.0
    rouge_2: float = 0.0
    rouge_l: float = 0.0


class RetrievalRelevanceScore(BaseModel):
    avg_relevance: float = 0.0
    relevant_doc_count: int = 0
    relevance_ratio: float = 0.0


class FaithfulnessScore(BaseModel):
    score: float = 0.0
    explanation: str = ""


class QualityMetrics(BaseModel):
    """单条查询的质量评估结果"""
    query: str
    rouge: RougeScores | None = None
    retrieval_relevance: RetrievalRelevanceScore | None = None
    faithfulness: FaithfulnessScore | None = None


class QualityAggregated(BaseModel):
    """汇总质量均值"""
    avg_rouge_1: float = 0.0
    avg_rouge_2: float = 0.0
    avg_rouge_l: float = 0.0
    avg_retrieval_relevance: float = 0.0
    avg_faithfulness: float = 0.0
    evaluated_count: int = 0


# ===================== 问答 =====================

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    use_rerank: bool = True
    use_query_rewrite: bool = False
    top_k: int = Field(5, ge=1, le=20)
    collection: str = Field("all", pattern=r"^(all|laws|cases)$")
    evaluate_quality: bool = False
    monitor_system: bool = False
    # --- Advanced pipeline options ---
    query_transform: str = Field("none", pattern=r"^(none|multi_query|hyde|decompose|multi_query_hyde)$")
    rerank_strategy: str = Field("simple", pattern=r"^(none|simple|llm)$")
    generation_strategy: str = Field("standard", pattern=r"^(standard|chain_of_thought|self_reflect|structured_legal)$")
    use_kg: bool = False


class SourceDocument(BaseModel):
    content: str
    metadata: dict = {}
    score: float | None = None


class StageMetrics(BaseModel):
    query_rewrite_ms: float | None = None
    retrieval_ms: float = 0
    rerank_ms: float | None = None
    generation_ms: float = 0
    total_ms: float = 0
    kg_lookup_ms: float | None = None
    self_reflect_ms: float | None = None
    was_corrected: bool = False
    # 本地模型调用效率指标
    llm_calls: int = 0
    llm_calls_saved: int = 0


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDocument] = []
    metrics: StageMetrics = StageMetrics()
    rewritten_queries: list[str] | None = None
    system_before: SystemInfo | None = None
    system_after: SystemInfo | None = None
    quality: QualityMetrics | None = None
    # --- Advanced pipeline fields ---
    kg_entities: list[str] | None = None
    generation_strategy: str | None = None
    pipeline_config: dict | None = None


# ===================== 知识库 =====================

class KnowledgeFileInfo(BaseModel):
    filename: str
    doc_type: str
    size_bytes: int
    chunk_count: int = 0


class KnowledgeStats(BaseModel):
    total_files: int = 0
    total_chunks: int = 0
    laws_chunks: int = 0
    cases_chunks: int = 0
    collections: list[str] = []


# ===================== 基准测试 =====================

class BenchmarkResult(BaseModel):
    system_info: SystemInfo
    total_queries: int
    avg_latency_ms: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    queries_per_second: float
    details: list[dict] = []


class BenchmarkResultV2(BaseModel):
    """包含质量评估的基准测试结果"""
    system_info: SystemInfo
    total_queries: int
    avg_latency_ms: float
    avg_retrieval_ms: float
    avg_generation_ms: float
    queries_per_second: float
    details: list[dict] = []
    quality: QualityAggregated | None = None
    quality_details: list[QualityMetrics] = []


# ===================== 报告 =====================

class ReportMeta(BaseModel):
    """报告列表项"""
    report_id: str
    created_at: str
    total_queries: int = 0
    avg_latency_ms: float = 0.0
    queries_per_second: float = 0.0
    has_quality: bool = False


class ReportFull(BaseModel):
    """完整性能报告"""
    report_id: str
    created_at: str
    system_snapshot: SystemInfo
    benchmark: dict
    knowledge_base: dict = {}
    lightweight_highlights: dict = {}


# ===================== 问答记录 =====================

class ChatRecordMeta(BaseModel):
    """问答性能记录列表项"""
    record_id: str
    created_at: str
    question: str
    total_ms: float = 0.0
    has_quality: bool = False
