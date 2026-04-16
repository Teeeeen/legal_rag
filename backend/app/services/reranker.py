"""重排序服务

提供两种重排序策略：
1. simple_rerank — 基于 jieba 分词的 Jaccard 相似度 + 元数据加分（零额外资源）
2. llm_rerank   — 使用 Qwen3:8B 批量评分（单次 LLM 调用，替代逐文档调用）
"""

import re
import jieba
from langchain_core.documents import Document
from app.core.llm import get_llm


# ===================== 简单重排序 =====================

def simple_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """基于关键词重叠 + 元数据加分的轻量重排序"""
    query_tokens = set(jieba.cut(query))

    scored: list[tuple[Document, float]] = []
    for doc in documents:
        doc_tokens = set(jieba.cut(doc.page_content))

        # Jaccard 相似度
        intersection = query_tokens & doc_tokens
        union = query_tokens | doc_tokens
        jaccard = len(intersection) / len(union) if union else 0.0

        # 元数据加分
        meta = doc.metadata
        bonus = 0.0
        law_name = meta.get("law_name", "")
        if law_name and law_name in query:
            bonus += 0.2
        article_num = meta.get("article_number", "")
        if article_num and article_num in query:
            bonus += 0.3
        guiding = meta.get("guiding_number", "")
        if guiding and guiding in query:
            bonus += 0.2

        scored.append((doc, jaccard + bonus))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ===================== LLM 重排序（批量单次调用） =====================

_BATCH_RERANK_PROMPT = """请判断以下多段法律文本与用户问题的相关性，为每段文本打分。

用户问题：{query}

{doc_list}

请严格按以下格式输出每段的评分（0-10整数），每行一个，不要输出其他内容：
1: 评分
2: 评分
...

评分标准：0=完全无关，5=部分相关，10=高度相关且直接回答了问题"""


async def llm_rerank(
    query: str,
    documents: list[Document],
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """使用 LLM 批量评分重排序（单次调用替代逐文档调用）

    优化策略：
    1. 先用 simple_rerank 预筛选到 top_k*2 篇，减少送入 LLM 的文档数
    2. 将所有候选文档拼入一个 prompt，单次 LLM 调用完成评分
    """
    # 预筛选：用 simple_rerank 取 top_k*2 篇候选
    pre_k = min(len(documents), top_k * 2, 8)
    pre_scored = simple_rerank(query, documents, top_k=pre_k)
    candidates = [doc for doc, _ in pre_scored]

    if not candidates:
        return []

    # 构建批量 prompt
    doc_parts = []
    for i, doc in enumerate(candidates, 1):
        text = doc.page_content[:300]
        meta = doc.metadata
        label = meta.get("law_name", "") or meta.get("guiding_number", "") or meta.get("source_file", "")
        doc_parts.append(f"[文本{i}] {label}\n{text}")

    doc_list = "\n\n".join(doc_parts)
    prompt = _BATCH_RERANK_PROMPT.format(query=query, doc_list=doc_list)

    llm = get_llm(temperature=0)
    try:
        resp = await llm.ainvoke(prompt)
        scores = _parse_batch_scores(resp.content, len(candidates))
    except Exception:
        # 回退到预筛选排序
        return pre_scored[:top_k]

    scored = list(zip(candidates, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def _parse_batch_scores(text: str, expected_count: int) -> list[float]:
    """解析批量评分输出，如 '1: 8\n2: 5\n3: 2'"""
    scores = [5.0] * expected_count  # 默认值

    for line in text.strip().split("\n"):
        match = re.match(r"(\d+)\s*[:：]\s*(\d+)", line.strip())
        if match:
            idx = int(match.group(1)) - 1
            score = float(match.group(2))
            if 0 <= idx < expected_count:
                scores[idx] = min(max(score, 0), 10)

    return scores
