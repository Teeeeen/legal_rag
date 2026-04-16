"""HyDE (Hypothetical Document Embeddings) 查询变换

生成假设性法律文档用于向量检索，原始查询用于 BM25 检索。
"""

from app.core.llm import get_llm
from app.services.prompts import HYDE_PROMPT
from app.config import settings


async def hyde_transform(question: str) -> tuple[str, str | None]:
    """
    生成假设性法律文档

    Returns:
        (original_question, hypothetical_document)
    """
    try:
        llm = get_llm(temperature=settings.HYDE_TEMPERATURE)
        chain = HYDE_PROMPT | llm
        response = await chain.ainvoke({"question": question})
        hypothetical_doc = response.content.strip()
        if len(hypothetical_doc) < 10:
            return question, None
        return question, hypothetical_doc
    except Exception:
        return question, None
