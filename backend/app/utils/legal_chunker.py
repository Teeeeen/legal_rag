"""法律文本专用分块器

针对中国法律条文和指导案例的结构化特征，定制分块策略：
1. 法律条文：以"条"为基本单位，保持条文完整性
2. 指导案例：按"裁判要旨/基本案情/裁判理由/裁判结果"等结构切分
3. 超长内容兜底：按款/项进行二级切分
4. 可选的上下文标头注入，提升 Embedding 质量
"""

import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ===================== 法律条文分块器 =====================

class LegalArticleSplitter(RecursiveCharacterTextSplitter):
    """法律条文分块器 — 以"条"为基本单位"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64, **kwargs):
        # 法律条文层级分隔符（优先级从高到低）
        separators = [
            r"\n第[一二三四五六七八九十百千]+编",     # 编
            r"\n第[一二三四五六七八九十百千]+章",     # 章
            r"\n第[一二三四五六七八九十百千]+节",     # 节
            r"\n第[一二三四五六七八九十\d]+条",       # 条（核心切分点）
            r"\n[一二三四五六七八九十]+、",            # 款
            r"\n（[一二三四五六七八九十]+）",          # 项
            r"\n\(\d+\)",                              # 数字项
            r"\n\d+\.",                                # 数字编号
            r"\n",                                     # 换行
        ]
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )


class LegalCaseSplitter(RecursiveCharacterTextSplitter):
    """指导案例分块器 — 按案例结构切分"""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128, **kwargs):
        # 指导案例的结构分隔符
        separators = [
            r"\n(?:裁判要旨|裁判要点)",
            r"\n(?:基本案情|案情简介|案件事实)",
            r"\n(?:裁判结果|判决结果)",
            r"\n(?:裁判理由|裁判说理|法院认为)",
            r"\n(?:关键词|相关法条)",
            r"\n(?:指导案例\d+号)",
            r"\n(?:一|二|三|四|五|六|七|八|九|十)、",
            r"\n\d+\.",
            r"\n",
        ]
        super().__init__(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            is_separator_regex=True,
            keep_separator=True,
            **kwargs,
        )


# ===================== 元数据提取 =====================

def extract_law_metadata(text: str, filename: str = "") -> dict:
    """从法律条文文本中提取元数据"""
    metadata = {
        "doc_type": "law",
        "source_file": filename,
        "law_name": "",
        "effective_date": "",
    }

    # 提取法律名称（通常在文件开头或标题行）
    name_patterns = [
        r"(中华人民共和国\S+?(?:法典|法|条例|规定|办法|决定|解释))",
        r"(《[^》]+》)",
        r"^(.+(?:法典|法|条例|规定|办法|决定|解释))",
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text[:500])
        if match:
            metadata["law_name"] = match.group(1).strip("《》")
            break

    if not metadata["law_name"] and filename:
        metadata["law_name"] = re.sub(r"\.\w+$", "", filename)

    # 提取生效日期
    date_patterns = [
        r"(\d{4}年\d{1,2}月\d{1,2}日)(?:起)?施行",
        r"自(\d{4}年\d{1,2}月\d{1,2}日)起施行",
        r"(\d{4}-\d{2}-\d{2})",
    ]
    for pattern in date_patterns:
        match = re.search(pattern, text[:2000])
        if match:
            metadata["effective_date"] = match.group(1)
            break

    return metadata


def extract_case_metadata(text: str, filename: str = "") -> dict:
    """从指导案例文本中提取元数据"""
    metadata = {
        "doc_type": "case",
        "source_file": filename,
        "case_number": "",
        "case_title": "",
        "guiding_number": "",
        "keywords": "",
    }

    # 提取指导案例编号
    guide_match = re.search(r"指导案例(\d+)号", text[:500])
    if guide_match:
        metadata["guiding_number"] = f"指导案例{guide_match.group(1)}号"

    # 提取案号
    case_match = re.search(r"[（(]\d{4}[）)][^\s]*\d+号", text[:1000])
    if case_match:
        metadata["case_number"] = case_match.group(0)

    # 提取关键词（用分号拼接为字符串，ChromaDB 不支持 list 值）
    kw_match = re.search(r"关键词[：:]\s*(.+?)(?:\n|$)", text[:1000])
    if kw_match:
        keywords = re.split(r"[；;,，\s]+", kw_match.group(1).strip())
        kw_list = [kw.strip() for kw in keywords if kw.strip()]
        metadata["keywords"] = "；".join(kw_list)

    if not metadata["case_title"] and filename:
        metadata["case_title"] = re.sub(r"\.\w+$", "", filename)

    return metadata


def enrich_chunk_metadata(chunk: Document, parent_metadata: dict, chunk_index: int) -> Document:
    """为分块添加详细的元数据"""
    text = chunk.page_content

    # 合并父文档元数据
    enriched = {**parent_metadata}
    enriched["chunk_index"] = chunk_index

    # 提取当前 chunk 所属的章节
    chapter_match = re.search(r"第[一二三四五六七八九十百千]+章\s*(.+?)(?:\n|$)", text)
    if chapter_match:
        enriched["chapter"] = chapter_match.group(0).strip()

    # 提取条号
    article_match = re.search(r"第([一二三四五六七八九十百千\d]+)条", text)
    if article_match:
        enriched["article_number"] = article_match.group(1)

    # 提取案例段落类型
    section_types = {
        "裁判要旨": "裁判要旨",
        "裁判要点": "裁判要旨",
        "基本案情": "基本案情",
        "案情简介": "基本案情",
        "裁判结果": "裁判结果",
        "裁判理由": "裁判理由",
        "法院认为": "裁判理由",
    }
    for keyword, section_type in section_types.items():
        if keyword in text[:50]:
            enriched["section_type"] = section_type
            break

    chunk.metadata = enriched
    return chunk


# ===================== 上下文标头注入 =====================

def add_contextual_header(chunk: Document, parent_metadata: dict) -> Document:
    """为分块内容添加结构化上下文标头，提升 Embedding 质量

    在 chunk 内容前插入 [法律名称: X | 章节: Y | 条号: 第Z条] 形式的标头。
    """
    parts = []

    law_name = parent_metadata.get("law_name", "")
    if law_name:
        parts.append(f"法律名称: {law_name}")

    chapter = chunk.metadata.get("chapter", "")
    if chapter:
        parts.append(f"章节: {chapter}")

    article = chunk.metadata.get("article_number", "")
    if article:
        parts.append(f"条号: 第{article}条")

    # 案例类元数据
    guiding = parent_metadata.get("guiding_number", "")
    if guiding:
        parts.append(f"案例: {guiding}")

    section_type = chunk.metadata.get("section_type", "")
    if section_type:
        parts.append(f"段落: {section_type}")

    if parts:
        header = "[" + " | ".join(parts) + "]\n"
        chunk.page_content = header + chunk.page_content

    return chunk


# ===================== 统一处理入口 =====================

def split_legal_document(
    text: str,
    doc_type: str = "law",
    filename: str = "",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """
    统一的法律文档分块入口

    Args:
        text: 文档全文
        doc_type: 文档类型 "law" | "case"
        filename: 源文件名
        chunk_size: 分块大小
        chunk_overlap: 分块重叠

    Returns:
        带有丰富元数据的 Document 列表
    """
    # 选择分块器
    if doc_type == "case":
        splitter = LegalCaseSplitter(chunk_size=chunk_size * 2, chunk_overlap=chunk_overlap * 2)
        parent_metadata = extract_case_metadata(text, filename)
    else:
        splitter = LegalArticleSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        parent_metadata = extract_law_metadata(text, filename)

    # 分块
    chunks = splitter.create_documents([text])

    # 元数据增强
    enriched_chunks = []
    for i, chunk in enumerate(chunks):
        enriched = enrich_chunk_metadata(chunk, parent_metadata, i)
        enriched_chunks.append(enriched)

    # 可选：注入上下文标头
    from app.config import settings
    if settings.CONTEXTUAL_CHUNKING:
        for chunk in enriched_chunks:
            add_contextual_header(chunk, parent_metadata)

    return enriched_chunks
