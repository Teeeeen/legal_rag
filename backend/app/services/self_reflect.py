"""自我反思与修正服务

让 LLM 验证初始回答的准确性，必要时进行修正。
"""

from app.core.llm import get_llm
from app.services.prompts import SELF_REFLECT_PROMPT, SELF_REFLECT_CORRECT_PROMPT


async def self_reflect_and_correct(
    question: str,
    initial_answer: str,
    context: str,
    max_iterations: int = 1,
) -> tuple[str, bool]:
    """
    验证并可能修正法律回答

    Args:
        question: 用户原始问题
        initial_answer: 初始生成的回答
        context: 参考资料上下文
        max_iterations: 最大修正轮次（默认 1 轮，控制延迟）

    Returns:
        (final_answer, was_corrected)
    """
    llm = get_llm(temperature=0)
    current_answer = initial_answer

    for _ in range(max_iterations):
        # 反思：检查回答准确性
        reflect_chain = SELF_REFLECT_PROMPT | llm
        reflect_resp = await reflect_chain.ainvoke({
            "context": context,
            "answer": current_answer,
        })
        reflection = reflect_resp.content.strip()

        # 判断是否需要修正
        if "准确" in reflection and "存在问题" not in reflection:
            return current_answer, current_answer != initial_answer

        # 修正：基于反思结果重新生成
        correct_chain = SELF_REFLECT_CORRECT_PROMPT | llm
        correct_resp = await correct_chain.ainvoke({
            "correction_guidance": reflection,
            "context": context,
            "question": question,
        })
        current_answer = correct_resp.content.strip()

    return current_answer, current_answer != initial_answer
