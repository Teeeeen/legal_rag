"""集成测试：40个法律问题 × 不同策略组合，采集真实数据"""
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
    # T16-T40: 扩展覆盖集（真实咨询语料 + 长尾领域）
    {"id": "T16", "q": "你好，请问交通事故发生了伤着住院治疗需要赔付是怎么样流程？",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T17", "q": "因再次酒驾而吊销驾驶证，有没有办法尽快拿到驾驶证。",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T18", "q": "借条一定要有保证人才算有法律效应么",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T19", "q": "信用卡还不起了，银行都有哪些处罚，可以协商么？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T20", "q": "注册资金200万是什么意思，是指实际资产吗",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T21", "q": "有限责任公司股东变更都需要什么资料，我自己去工商局能办理吗？需要花钱吗大概流程是什么样的？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T22", "q": "注册商标后有人用同样名称从事文化宣传和商业经营活动也不向注册人告知是否属于侵权属于侵权？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "llm", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T23", "q": "分享画家作品算侵权吗？我在堆糖网找的图片发表到XX讯这个软件上上可以吗",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "llm", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T24", "q": "申请复议是什么意思，申请复议了之后该怎么办？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T25", "q": "对消防处罚有异议，行政处罚说我们违反了第58条，其中的人员密集场所，未经消防安全检查擅自投入使用",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T26", "q": "个人购买社保10，单位买5年后满50岁的女性可以办理退休吗",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T27", "q": "私人老板请员工上班27个月后，员工自已辞职，9个月后又来上班，以前的工龄可以跟现在的一起算吗",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T28", "q": "在成都XX公司工作，自动离职他不发工资怎么办？",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T29", "q": "父母无遗嘱遗产房姐要继承析产弟",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T30", "q": "我父亲去世之前上午叫我找两个朋友做视频遗嘱，遗嘱说他把所有的遗产都留给他的儿子我没有说详细的遗产请问这个遗嘱有效吗？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T31", "q": "我想把我的房子过户给儿子，但儿子现在在国外留学，儿子不在可以过户吗？过户费咋算",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T32", "q": "在嘉兴秀洲区盛世豪庭135.08平方的房产，4个人共同所有，每人25%的份额法院已经认定这样的房产能强制执行拍卖吗？",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T33", "q": "民事诉讼的时效期是三年吗，过了三年还可以提起诉讼吗",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T34", "q": "起诉别人打官司首先要做什么，然后做什么，我想知道有哪几步骤",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T35", "q": "取保候审可以出国吗？就是在网上赌博做推广人员，然后被抓，去没有一个月现在想去国外旅游可以出去吗？",
     "cfg": {"query_transform": "multi_query_hyde", "rerank_strategy": "simple", "generation_strategy": "self_reflect", "use_kg": True}},
    {"id": "T36", "q": "因盗窃罪被拘留十天，后来派出所说取保候审，当时交了5000元取保金，受害人钱已退还是不是还要被判刑？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "chain_of_thought", "use_kg": True}},
    {"id": "T37", "q": "玩黑彩输了150万都是骗来的，范了什么法",
     "cfg": {"query_transform": "hyde", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T38", "q": "网上平台贷款受法律保护吗？？？",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
    {"id": "T39", "q": "母公司吸收合并子公司，原企业法律承续或分割情况的说明材料应如何填写",
     "cfg": {"query_transform": "decompose", "rerank_strategy": "simple", "generation_strategy": "structured_legal", "use_kg": False}},
    {"id": "T40", "q": "欠别人一万多别人按天算百分之0.35是不是放高利贷",
     "cfg": {"query_transform": "none", "rerank_strategy": "simple", "generation_strategy": "standard", "use_kg": False}},
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
        print(f"[{i+1}/{len(TEST_CASES)}] {tc['id']}: {tc['q'][:30]}... ", end="", flush=True)
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
