"""质量评估：采集 ROUGE-L / 检索相关性 / 忠实度 三维数据"""
import asyncio, json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pipeline import RAGPipeline, PipelineConfig, QueryTransformStrategy, RerankStrategy, GenerationStrategy
from app.services.quality_service import evaluate_single_query, aggregate_quality
from app.models.schemas import SourceDocument

# 8个有参考答案的问题，默认配置（标准模式）
EVAL_QUERIES = [
    "民法典中关于合同成立的规定是什么？",
    "故意杀人罪的量刑标准是什么？",
    "劳动合同法中关于经济补偿金的规定",
    "交通事故损害赔偿的法律依据",
    "知识产权侵权的认定标准",
    "公司法中股东代表诉讼的条件",
    "行政诉讼的受案范围",
    "最高法关于民间借贷利率的指导案例",
]

async def main():
    print("=== Warming up ===")
    from app.core.llm import get_llm
    from app.core.embeddings import get_embeddings
    await get_llm().ainvoke("你好")
    get_embeddings().embed_query("测试")
    print("=== Warmup done ===\n")

    all_quality = []
    for i, q in enumerate(EVAL_QUERIES):
        print(f"[{i+1}/{len(EVAL_QUERIES)}] {q[:30]}...", end=" ", flush=True)
        t0 = time.time()
        cfg = PipelineConfig(
            query_transform=QueryTransformStrategy.NONE,
            rerank_strategy=RerankStrategy.SIMPLE,
            generation_strategy=GenerationStrategy.STANDARD,
            use_kg=False, top_k=5,
            collection_names=["laws", "cases"],
        )
        pipe = RAGPipeline(cfg)
        resp = await pipe.execute(q)

        sources_for_eval = [
            SourceDocument(content=s.content, metadata=s.metadata, score=s.score)
            for s in resp.sources
        ]
        qm = await evaluate_single_query(q, resp.answer, sources_for_eval)
        all_quality.append(qm)
        elapsed = time.time() - t0

        rouge_str = f"R1={qm.rouge.rouge_1:.3f} R2={qm.rouge.rouge_2:.3f} RL={qm.rouge.rouge_l:.3f}" if qm.rouge else "no-ref"
        rel_str = f"rel={qm.retrieval_relevance.avg_relevance:.1f}" if qm.retrieval_relevance else "N/A"
        faith_str = f"faith={qm.faithfulness.score:.1f}" if qm.faithfulness else "N/A"
        print(f"{elapsed:.0f}s | {rouge_str} | {rel_str} | {faith_str}")

    agg = aggregate_quality(all_quality)
    print("\n" + "="*50)
    print(f"ROUGE-1:      {agg.avg_rouge_1:.4f}")
    print(f"ROUGE-2:      {agg.avg_rouge_2:.4f}")
    print(f"ROUGE-L:      {agg.avg_rouge_l:.4f}")
    print(f"检索相关性:   {agg.avg_retrieval_relevance:.2f} / 10")
    print(f"忠实度:       {agg.avg_faithfulness:.2f} / 10")
    print(f"评估条数:     {agg.evaluated_count}")

    details = []
    for qm in all_quality:
        details.append({
            "query": qm.query,
            "rouge": {"r1": qm.rouge.rouge_1, "r2": qm.rouge.rouge_2, "rl": qm.rouge.rouge_l} if qm.rouge else None,
            "retrieval_relevance": qm.retrieval_relevance.avg_relevance if qm.retrieval_relevance else None,
            "relevance_ratio": qm.retrieval_relevance.relevance_ratio if qm.retrieval_relevance else None,
            "faithfulness": qm.faithfulness.score if qm.faithfulness else None,
            "faithfulness_expl": qm.faithfulness.explanation if qm.faithfulness else None,
        })
    with open("/tmp/quality_eval_results.json", "w") as f:
        json.dump({"aggregated": {"rouge_1": agg.avg_rouge_1, "rouge_2": agg.avg_rouge_2, "rouge_l": agg.avg_rouge_l,
                                    "retrieval_relevance": agg.avg_retrieval_relevance, "faithfulness": agg.avg_faithfulness},
                    "details": details}, f, ensure_ascii=False, indent=2)
    print("\nSaved to /tmp/quality_eval_results.json")

asyncio.run(main())
