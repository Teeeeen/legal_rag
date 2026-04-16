"""元数据工具函数"""


def format_source_display(metadata: dict) -> str:
    """格式化来源显示文本"""
    doc_type = metadata.get("doc_type", "")

    if doc_type == "law":
        law_name = metadata.get("law_name", "未知法律")
        article = metadata.get("article_number", "")
        chapter = metadata.get("chapter", "")
        parts = [law_name]
        if chapter:
            parts.append(chapter)
        if article:
            parts.append(f"第{article}条")
        return " / ".join(parts)

    elif doc_type == "case":
        guiding = metadata.get("guiding_number", "")
        case_title = metadata.get("case_title", "")
        section = metadata.get("section_type", "")
        parts = []
        if guiding:
            parts.append(guiding)
        if case_title:
            parts.append(case_title)
        if section:
            parts.append(f"[{section}]")
        return " / ".join(parts) if parts else "指导案例"

    else:
        return metadata.get("source_file", "未知来源")
