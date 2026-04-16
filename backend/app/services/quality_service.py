"""生成质量评估服务

三个评估维度:
1. ROUGE — rouge-chinese 库计算 ROUGE-1/2/L F1（有参考答案时）
2. 检索相关性 — LLM-as-judge 对每个检索文档评分
3. 忠实度 — LLM-as-judge 判断回答是否基于检索来源
"""

import json
from pathlib import Path

import jieba
from rouge_chinese import Rouge

from app.core.llm import get_llm
from app.models.schemas import (
    QualityMetrics,
    QualityAggregated,
    RougeScores,
    RetrievalRelevanceScore,
    FaithfulnessScore,
    SourceDocument,
)
from app.config import settings

# 懒加载参考答案
_reference_answers: dict[str, str] | None = None


def _load_reference_answers() -> dict[str, str]:
    """加载参考答案文件"""
    global _reference_answers
    if _reference_answers is not None:
        return _reference_answers

    ref_path = Path(settings.DATA_DIR) / "reference_answers.json"
    if ref_path.exists():
        with open(ref_path, "r", encoding="utf-8") as f:
            _reference_answers = json.load(f)
    else:
        _reference_answers = {}
    return _reference_answers


# ===================== ROUGE 评估 =====================

def compute_rouge(answer: str, reference: str) -> RougeScores:
    """使用 rouge-chinese 计算 ROUGE-1/2/L F1 分数"""
    # jieba 分词后用空格连接（rouge-chinese 要求）
    answer_seg = " ".join(jieba.cut(answer))
    reference_seg = " ".join(jieba.cut(reference))

    if not answer_seg.strip() or not reference_seg.strip():
        return RougeScores()

    rouge = Rouge()
    try:
        scores = rouge.get_scores(answer_seg, reference_seg)[0]
        return RougeScores(
            rouge_1=round(scores["rouge-1"]["f"], 4),
            rouge_2=round(scores["rouge-2"]["f"], 4),
            rouge_l=round(scores["rouge-l"]["f"], 4),
        )
    except Exception:
        return RougeScores()


# ===================== 检索相关性评估 =====================

_RELEVANCE_PROMPT = """请判断以下法律文本与用户问题的相关性。
用户问题：{query}

法律文本：
{text}

请只返回 0 到 10 之间的整数评分，不要返回其他内容。
评分标准：
- 0: 完全无关
- 5: 部分相关
- 10: 高度相关且直接回答了问题"""


async def evaluate_retrieval_relevance(
    query: str,
    sources: list[SourceDocument],
) -> RetrievalRelevanceScore:
    """使用 LLM 评估检索文档的相关性"""
    if not sources:
        return RetrievalRelevanceScore()

    llm = get_llm(temperature=0)
    scores: list[float] = []

    for src in sources:
        prompt = _RELEVANCE_PROMPT.format(
            query=query,
            text=src.content[:800],
        )
        try:
            resp = await llm.ainvoke(prompt)
            score_text = resp.content.strip()
            score = float("".join(c for c in score_text if c.isdigit() or c == ".") or "0")
            score = min(max(score, 0), 10)
        except Exception:
            score = 5.0
        scores.append(score)

    avg = sum(scores) / len(scores) if scores else 0.0
    relevant_count = sum(1 for s in scores if s >= 6)
    return RetrievalRelevanceScore(
        avg_relevance=round(avg, 2),
        relevant_doc_count=relevant_count,
        relevance_ratio=round(relevant_count / len(scores), 2) if scores else 0.0,
    )


# ===================== 忠实度评估 =====================

_FAITHFULNESS_PROMPT = """请判断以下回答是否忠实于提供的参考来源。

用户问题：{query}

参考来源：
{sources}

生成的回答：
{answer}

请从以下两个方面评估：
1. 回答中的信息是否都能在参考来源中找到依据
2. 回答是否存在编造或添加参考来源中没有的法律条文

请按以下格式返回（只返回两行，不要其他内容）：
评分：[0-10的整数]
说明：[一句话解释]"""


async def evaluate_faithfulness(
    query: str,
    answer: str,
    sources: list[SourceDocument],
) -> FaithfulnessScore:
    """使用 LLM 评估回答的忠实度"""
    if not sources or not answer:
        return FaithfulnessScore()

    llm = get_llm(temperature=0)
    sources_text = "\n".join(
        f"[来源{i+1}] {src.content[:400]}" for i, src in enumerate(sources[:5])
    )

    prompt = _FAITHFULNESS_PROMPT.format(
        query=query,
        sources=sources_text,
        answer=answer[:1000],
    )

    try:
        resp = await llm.ainvoke(prompt)
        text = resp.content.strip()
        lines = text.split("\n")
        score = 5.0
        explanation = ""
        for line in lines:
            if "评分" in line:
                digits = "".join(c for c in line if c.isdigit() or c == ".")
                if digits:
                    score = min(max(float(digits), 0), 10)
            if "说明" in line:
                explanation = line.split("：", 1)[-1].strip() if "：" in line else line.split(":", 1)[-1].strip()
        return FaithfulnessScore(score=round(score, 1), explanation=explanation)
    except Exception:
        return FaithfulnessScore(score=5.0, explanation="评估失败")


# ===================== 综合评估 =====================

async def evaluate_single_query(
    query: str,
    answer: str,
    sources: list[SourceDocument],
) -> QualityMetrics:
    """对单条查询执行完整的质量评估"""
    ref_answers = _load_reference_answers()

    # ROUGE（仅在有参考答案时计算）
    rouge = None
    if query in ref_answers:
        rouge = compute_rouge(answer, ref_answers[query])

    # 检索相关性
    retrieval_relevance = await evaluate_retrieval_relevance(query, sources)

    # 忠实度
    faithfulness = await evaluate_faithfulness(query, answer, sources)

    return QualityMetrics(
        query=query,
        rouge=rouge,
        retrieval_relevance=retrieval_relevance,
        faithfulness=faithfulness,
    )


def aggregate_quality(quality_list: list[QualityMetrics]) -> QualityAggregated:
    """汇总多条查询的质量评估结果"""
    if not quality_list:
        return QualityAggregated()

    rouge_1_vals = []
    rouge_2_vals = []
    rouge_l_vals = []
    relevance_vals = []
    faithfulness_vals = []

    for qm in quality_list:
        if qm.rouge:
            rouge_1_vals.append(qm.rouge.rouge_1)
            rouge_2_vals.append(qm.rouge.rouge_2)
            rouge_l_vals.append(qm.rouge.rouge_l)
        if qm.retrieval_relevance:
            relevance_vals.append(qm.retrieval_relevance.avg_relevance)
        if qm.faithfulness:
            faithfulness_vals.append(qm.faithfulness.score)

    def _avg(vals: list[float]) -> float:
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return QualityAggregated(
        avg_rouge_1=_avg(rouge_1_vals),
        avg_rouge_2=_avg(rouge_2_vals),
        avg_rouge_l=_avg(rouge_l_vals),
        avg_retrieval_relevance=_avg(relevance_vals),
        avg_faithfulness=_avg(faithfulness_vals),
        evaluated_count=len(quality_list),
    )
