"""查询重写服务

将用户的口语化查询扩展为多个专业法律检索查询。
"""

from app.core.llm import get_llm
from app.services.prompts import DECOMPOSE_PROMPT

_REWRITE_PROMPT = """你是一个法律检索专家。请将用户的问题改写为 3 个不同角度的法律检索查询，以便更全面地检索法律条文和指导案例。

要求：
1. 第一个查询侧重法律条文检索，使用标准法律术语
2. 第二个查询侧重司法解释或指导案例
3. 第三个查询使用不同的术语或角度表述

用户问题：{question}

请直接返回 3 个查询，每行一个，不要编号或其他格式："""


# ===================== 法律术语规范化映射 =====================

LEGAL_TERM_MAP: dict[str, str] = {
    "打人": "故意伤害",
    "杀人": "故意杀人",
    "偷东西": "盗窃",
    "抢东西": "抢劫",
    "骗钱": "诈骗",
    "借钱不还": "民间借贷纠纷",
    "欠钱不还": "民间借贷纠纷",
    "离婚分财产": "离婚财产分割",
    "拖欠工资": "劳动报酬争议",
    "不发工资": "劳动报酬争议",
    "开除": "解除劳动合同",
    "辞退": "解除劳动合同",
    "碰瓷": "敲诈勒索",
    "闯红灯": "违反交通信号灯",
    "酒驾": "危险驾驶",
    "醉驾": "危险驾驶",
    "撞人": "交通肇事",
    "卖假货": "销售伪劣产品",
    "老赖": "失信被执行人",
    "网贷": "网络借贷",
    "高利贷": "民间借贷利率",
}


def normalize_legal_terms(query: str) -> str:
    """将口语化表述替换为标准法律术语

    按 key 长度降序匹配，避免短 key 误替换长词的子串。
    跳过已被替换过的位置（目标术语已存在于 query 中时不替换）。
    """
    result = query
    # 按 key 长度降序排列，优先匹配更长的口语词
    sorted_items = sorted(LEGAL_TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for colloquial, formal in sorted_items:
        if colloquial in result and formal not in result:
            result = result.replace(colloquial, formal)
    return result


async def multi_query_rewrite(question: str) -> list[str]:
    """将用户问题改写为多个检索查询"""
    llm = get_llm(temperature=0.5)
    resp = await llm.ainvoke(_REWRITE_PROMPT.format(question=question))

    queries = [line.strip() for line in resp.content.strip().split("\n") if line.strip()]
    # 过滤掉空行和过短的行
    queries = [q for q in queries if len(q) > 4]

    if not queries:
        return [question]

    # 始终包含原始问题
    if question not in queries:
        queries.insert(0, question)

    return queries[:4]


async def decompose_query(question: str) -> list[str]:
    """将复杂法律问题分解为多个子问题"""
    llm = get_llm(temperature=0.3)
    chain = DECOMPOSE_PROMPT | llm
    resp = await chain.ainvoke({"question": question})

    sub_questions = [line.strip() for line in resp.content.strip().split("\n") if line.strip()]
    sub_questions = [q for q in sub_questions if len(q) > 4]

    if not sub_questions:
        return [question]

    # 始终包含原始问题
    if question not in sub_questions:
        sub_questions.insert(0, question)

    return sub_questions[:5]
