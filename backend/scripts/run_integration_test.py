"""集成测试：15个法律问题 × 不同策略组合，采集真实数据"""
import asyncio, time, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pipeline import (
    RAGPipeline, PipelineConfig,
    QueryTransformStrategy, RerankStrategy, GenerationStrategy,
)

# Strategy shorthand mapping
QT = {
    "none": QueryTransformStrategy.NONE,
    "multi_query": QueryTransformStrategy.MULTI_QUERY,
    "hyde": QueryTransformStrategy.HYDE,
    "decompose": QueryTransformStrategy.DECOMPOSE,
    "multi_query_hyde": QueryTransformStrategy.MULTI_QUERY_HYDE,
}
RR = {
    "none": RerankStrategy.NONE,
    "simple": RerankStrategy.SIMPLE,
    "llm": RerankStrategy.LLM,
}
GS = {
    "standard": GenerationStrategy.STANDARD,
    "chain_of_thought": GenerationStrategy.COT,
    "self_reflect": GenerationStrategy.SELF_REFLECT,
    "structured_legal": GenerationStrategy.STRUCTURED,
}

TEST_CASES = [
    # T01-T03: 基础问答 (无+简单+标准)
    {"id": "T01", "q": "民法典中关于合同成立的规定是什么？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T02", "q": "劳动合同法中关于经济补偿金的规定有哪些？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T03", "q": "行政诉讼的受案范围包括哪些？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    # T04-T06: 刑法+KG (KG+简单+CoT)
    {"id": "T04", "q": "故意杀人罪的量刑标准是什么？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "chain_of_thought", "use_kg": True}},
    {"id": "T05", "q": "盗窃罪的构成要件有哪些？入室盗窃如何加重处罚？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "chain_of_thought", "use_kg": True}},
    {"id": "T06", "q": "交通肇事罪和危险驾驶罪有什么区别？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": True}},
    # T07-T09: 口语化 (HyDE+简单+标准)
    {"id": "T07", "q": "打人了会怎么判？",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T08", "q": "借钱不还怎么起诉？",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T09", "q": "酒驾被抓了要坐牢吗？",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    # T10-T11: 复杂问题 (分解+简单+结构化)
    {"id": "T10", "q": "公司拖欠工资三个月，员工可以通过哪些法律途径维权？需要准备什么证据？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T11", "q": "网上购物买到假货，消费者可以要求几倍赔偿？相关法律依据是什么？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    # T12-T13: 高级组合 (多查询+HyDE+KG+自我修正)
    {"id": "T12", "q": "故意伤害致人重伤的，法律是怎么规定的？如果是正当防卫呢？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "simple", "generation_strategy": "self_reflect", "use_kg": True}},
    {"id": "T13", "q": "诈骗罪的立案标准是多少金额？电信诈骗有特殊规定吗？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "simple", "generation_strategy": "self_reflect", "use_kg": True}},
    # T14: 多查询+HyDE+LLM重排+标准
    {"id": "T14", "q": "知识产权侵权的认定标准是什么？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "llm", "generation_strategy": "standard", "use_kg": False}},
    # T15: 全策略
    {"id": "T15", "q": "民间借贷利率的法律上限是多少？超过部分是否受法律保护？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "llm", "generation_strategy": "self_reflect", "use_kg": True}},
]


async def run_one(tc):
    cfg = PipelineConfig(
        query_transform=QT[tc["cfg"]["query_transform"]],
        rerank_strategy=RR[tc["cfg"]["rerank_strategy"]],
        generation_strategy=GS[tc["cfg"]["generation_strategy"]],
        use_kg=tc["cfg"]["use_kg"],
        top_k=5,
        collection_names=["laws", "cases"],
    )
    pipe = RAGPipeline(cfg)
    t0 = time.time()
    try:
        resp = await pipe.execute(tc["q"])
        elapsed = time.time() - t0
        return {
            "id": tc["id"],
            "question": tc["q"],
            "strategy": f"{tc['cfg']['query_transform']}+{tc['cfg']['rerank_strategy']}+{tc['cfg']['generation_strategy']}" + ("+KG" if tc["cfg"]["use_kg"] else ""),
            "pass": True,
            "elapsed_s": round(elapsed, 1),
            "sources": len(resp.sources),
            "kg_entities": resp.kg_entities or [],
            "was_corrected": resp.metrics.was_corrected,
            "llm_calls": resp.metrics.llm_calls,
            "llm_calls_saved": resp.metrics.llm_calls_saved,
            "answer_snippet": resp.answer[:120].replace("\n", " "),
        }
    except Exception as e:
        elapsed = time.time() - t0
        return {
            "id": tc["id"],
            "question": tc["q"],
            "pass": False,
            "elapsed_s": round(elapsed, 1),
            "error": str(e)[:200],
        }


async def main():
    # Warmup: trigger model load
    print("=== Warming up models ===")
    from app.core.llm import get_llm
    from app.core.embeddings import get_embeddings
    llm = get_llm()
    await llm.ainvoke("你好")
    get_embeddings().embed_query("测试")
    print("=== Warmup done ===\n")

    results = []
    for i, tc in enumerate(TEST_CASES):
        print(f"[{i+1}/15] {tc['id']}: {tc['q'][:30]}... ", end="", flush=True)
        r = await run_one(tc)
        print(f"{'PASS' if r['pass'] else 'FAIL'} {r['elapsed_s']}s")
        results.append(r)

    print("\n=== RESULTS ===")
    for r in results:
        if r["pass"]:
            print(f"{r['id']} | {r['elapsed_s']:>5.1f}s | src={r['sources']} | llm={r['llm_calls']}/{r['llm_calls_saved']}saved | kg={r['kg_entities']} | corrected={r['was_corrected']} | {r['answer_snippet'][:60]}")
        else:
            print(f"{r['id']} | FAIL | {r.get('error','')[:80]}")

    with open("/tmp/integration_test_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nFull results saved to /tmp/integration_test_results.json")

asyncio.run(main())
