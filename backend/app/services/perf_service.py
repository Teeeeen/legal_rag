"""性能测试服务"""

import time
import psutil
import httpx
from app.core.llm import get_llm
from app.core.embeddings import get_embeddings
from app.services.rag_service import rag_query
from app.config import settings
from app.models.schemas import (
    SystemInfo,
    BenchmarkResult,
    BenchmarkResultV2,
    SourceDocument,
)


LEGAL_TEST_QUERIES = [
    "民法典中关于合同成立的规定是什么？",
    "故意杀人罪的量刑标准是什么？",
    "劳动合同法中关于经济补偿金的规定",
    "交通事故损害赔偿的法律依据",
    "知识产权侵权的认定标准",
    "公司法中股东代表诉讼的条件",
    "行政诉讼的受案范围",
    "最高法关于民间借贷利率的指导案例",
]


def get_system_info() -> SystemInfo:
    """获取当前系统资源状态"""
    mem = psutil.virtual_memory()
    return SystemInfo(
        cpu_percent=psutil.cpu_percent(interval=0.5),
        memory_percent=mem.percent,
        memory_used_gb=round(mem.used / (1024**3), 2),
        memory_total_gb=round(mem.total / (1024**3), 2),
    )


async def get_ollama_status() -> dict:
    """获取 Ollama 本地模型加载状态和显存占用 —— 本地部署特有的资源监控"""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.OLLAMA_BASE_URL}/api/ps")
            resp.raise_for_status()
            data = resp.json()
        models = data.get("models", [])
        return {
            "status": "running",
            "loaded_models": [
                {
                    "name": m.get("name", ""),
                    "size_mb": round(m.get("size", 0) / 1024 / 1024, 1),
                    "vram_mb": round(m.get("size_vram", 0) / 1024 / 1024, 1),
                }
                for m in models
            ],
            "total_vram_mb": round(
                sum(m.get("size_vram", 0) for m in models) / 1024 / 1024, 1
            ),
            "model_count": len(models),
        }
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


async def run_benchmark(
    queries: list[str] | None = None,
    use_rerank: bool = True,
    evaluate_quality: bool = False,
) -> BenchmarkResultV2:
    """运行基准性能测试，可选质量评估"""
    test_queries = queries or LEGAL_TEST_QUERIES[:4]
    sys_info = get_system_info()

    details = []
    total_retrieval = 0.0
    total_generation = 0.0
    total_latency = 0.0

    # 质量评估相关
    quality_details = []

    for q in test_queries:
        t0 = time.time()
        try:
            result = await rag_query(question=q, use_rerank=use_rerank, use_query_rewrite=False)
            latency = (time.time() - t0) * 1000
            details.append({
                "query": q,
                "latency_ms": round(latency, 1),
                "retrieval_ms": result.metrics.retrieval_ms,
                "generation_ms": result.metrics.generation_ms,
                "sources_count": len(result.sources),
                "answer_length": len(result.answer),
            })
            total_retrieval += result.metrics.retrieval_ms
            total_generation += result.metrics.generation_ms
            total_latency += latency

            # 质量评估
            if evaluate_quality:
                from app.services.quality_service import evaluate_single_query
                qm = await evaluate_single_query(
                    query=q,
                    answer=result.answer,
                    sources=result.sources,
                )
                quality_details.append(qm)

        except Exception as e:
            details.append({
                "query": q,
                "error": str(e),
                "latency_ms": round((time.time() - t0) * 1000, 1),
            })
            total_latency += (time.time() - t0) * 1000

    n = len(test_queries)

    # 汇总质量指标
    quality_aggregated = None
    if evaluate_quality and quality_details:
        from app.services.quality_service import aggregate_quality
        quality_aggregated = aggregate_quality(quality_details)

    return BenchmarkResultV2(
        system_info=sys_info,
        total_queries=n,
        avg_latency_ms=round(total_latency / n, 1) if n else 0,
        avg_retrieval_ms=round(total_retrieval / n, 1) if n else 0,
        avg_generation_ms=round(total_generation / n, 1) if n else 0,
        queries_per_second=round(n / (total_latency / 1000), 2) if total_latency else 0,
        details=details,
        quality=quality_aggregated,
        quality_details=quality_details,
    )
