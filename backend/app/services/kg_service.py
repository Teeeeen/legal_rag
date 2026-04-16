"""犯罪知识图谱服务

加载犯罪知识图谱数据，提供实体提取和知识查找功能。
"""

import re
from functools import lru_cache

from langchain_core.documents import Document

from app.core.llm import get_llm
from app.services.prompts import KG_ENTITY_EXTRACT_PROMPT
from app.config import settings


# ===================== KG 数据加载 =====================

@lru_cache(maxsize=1)
def _load_crime_kg() -> dict[str, dict]:
    """加载犯罪知识图谱文件，解析为 {罪名: {概念, 构成, 量刑, 法条}} 字典

    使用 lru_cache 全局缓存，只在首次调用时加载。
    """
    kg_path = settings.KG_DATA_PATH
    crime_map: dict[str, dict] = {}

    try:
        with open(kg_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return crime_map

    # 按 "【罪名】" 分割条目（首条可能有 === 前缀）
    entries = re.split(r"(?:={3,})?\n*【", content)
    for entry in entries:
        if not entry.strip():
            continue

        # 提取罪名（entry 此时以 "罪名】..." 开头）
        name_match = re.match(r"([^】]+)】", entry)
        if not name_match:
            continue
        crime_name = name_match.group(1).strip()

        # 提取各节内容
        sections: dict[str, str] = {}

        concept_match = re.search(r"概念与定义：\n(.*?)(?=\n犯罪构成特征：|\n认定与区分：|\n量刑处罚：|\n相关法条：|\Z)", entry, re.DOTALL)
        if concept_match:
            sections["概念与定义"] = concept_match.group(1).strip()

        elements_match = re.search(r"犯罪构成特征：\n(.*?)(?=\n认定与区分：|\n量刑处罚：|\n相关法条：|\Z)", entry, re.DOTALL)
        if elements_match:
            sections["犯罪构成"] = elements_match.group(1).strip()

        sentencing_match = re.search(r"量刑处罚：\n(.*?)(?=\n相关法条：|\Z)", entry, re.DOTALL)
        if sentencing_match:
            sections["量刑处罚"] = sentencing_match.group(1).strip()

        articles_match = re.search(r"相关法条：\n(.*?)(?=\Z)", entry, re.DOTALL)
        if articles_match:
            sections["相关法条"] = articles_match.group(1).strip()

        # 提取括号中的类别
        category_match = re.search(r"（([^）]+)）", entry[:100])
        if category_match:
            sections["类别"] = category_match.group(1).strip()

        crime_map[crime_name] = sections

    return crime_map


# ===================== 实体提取 =====================

async def extract_crime_entities(question: str) -> list[str]:
    """从用户问题中提取涉及的罪名，模糊匹配 KG 中的标准罪名"""
    kg = _load_crime_kg()
    if not kg:
        return []

    # 先尝试精确匹配（无需 LLM）
    matched = []
    for crime_name in kg:
        if crime_name in question:
            matched.append(crime_name)
    if matched:
        return matched[:3]

    # LLM 提取
    try:
        llm = get_llm(temperature=0)
        chain = KG_ENTITY_EXTRACT_PROMPT | llm
        resp = await chain.ainvoke({"question": question})
        raw_names = [line.strip() for line in resp.content.strip().split("\n") if line.strip()]

        if not raw_names or raw_names == ["无"]:
            return []

        # 模糊匹配 KG 中的罪名
        result = []
        kg_names = list(kg.keys())
        for name in raw_names:
            name = name.strip("\"' 　")
            if name in kg:
                result.append(name)
            else:
                # 模糊匹配：检查 KG 中是否包含该名称
                for kn in kg_names:
                    if name in kn or kn in name:
                        result.append(kn)
                        break

        return result[:3]
    except Exception:
        return []


# ===================== KG 查找 =====================

def kg_lookup(crime_names: list[str]) -> list[Document]:
    """根据罪名列表查找知识图谱，返回结构化 Document"""
    kg = _load_crime_kg()
    docs: list[Document] = []

    for name in crime_names:
        entry = kg.get(name)
        if not entry:
            continue

        # 构建结构化文本
        parts = [f"【{name}】"]
        if "类别" in entry:
            parts.append(f"类别：{entry['类别']}")
        if "概念与定义" in entry:
            parts.append(f"\n概念与定义：\n{entry['概念与定义'][:300]}")
        if "犯罪构成" in entry:
            parts.append(f"\n犯罪构成：\n{entry['犯罪构成'][:500]}")
        if "量刑处罚" in entry:
            parts.append(f"\n量刑处罚：\n{entry['量刑处罚'][:300]}")
        if "相关法条" in entry:
            parts.append(f"\n相关法条：\n{entry['相关法条'][:300]}")

        text = "\n".join(parts)
        docs.append(Document(
            page_content=text,
            metadata={
                "doc_type": "kg",
                "crime_name": name,
                "category": entry.get("类别", ""),
                "source_file": "犯罪知识图谱.txt",
            },
        ))

    return docs
