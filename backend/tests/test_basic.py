"""基础测试"""

import pytest


def test_config_loads():
    """测试配置加载"""
    from app.config import settings
    assert settings.LLM_MODEL == "qwen3:8b"
    assert settings.EMBEDDING_MODEL == "bge-m3"
    assert settings.LAWS_COLLECTION == "laws"
    assert settings.CASES_COLLECTION == "cases"
    # 高级管线默认值
    assert settings.CONTEXTUAL_CHUNKING is True
    assert settings.HYDE_TEMPERATURE == 0.7
    assert settings.SELF_REFLECT_MAX_ITER == 1
    assert settings.DEFAULT_QUERY_TRANSFORM == "none"
    assert settings.DEFAULT_RERANK == "simple"
    assert settings.DEFAULT_GENERATION == "standard"


def test_legal_chunker_law():
    """测试法律条文分块"""
    from app.utils.legal_chunker import split_legal_document

    sample_law = """中华人民共和国民法典

第一编 总则

第一章 基本规定

第一条 为了保护民事主体的合法权益，调整民事关系，维护社会和经济秩序，适应中国特色社会主义发展要求，弘扬社会主义核心价值观，根据宪法，制定本法。

第二条 民法调整平等主体的自然人、法人和非法人组织之间的人身关系和财产关系。

第三条 民事主体的人身权利、财产权利以及其他合法权益受法律保护，任何组织或者个人不得侵犯。"""

    chunks = split_legal_document(sample_law, doc_type="law", filename="民法典.txt")
    assert len(chunks) > 0
    assert chunks[0].metadata["doc_type"] == "law"
    assert "民法典" in chunks[0].metadata.get("law_name", "")


def test_legal_chunker_case():
    """测试指导案例分块"""
    from app.utils.legal_chunker import split_legal_document

    sample_case = """指导案例24号

关键词：交通事故 损害赔偿

裁判要旨
交通事故的受害人没有过错，其体质状况不属于侵权责任法等法律规定的过错。

基本案情
原告荣宝英诉称，被告王阳驾驶车辆与原告发生交通事故。

裁判结果
法院判决被告承担全部赔偿责任。"""

    chunks = split_legal_document(sample_case, doc_type="case", filename="指导案例24号.txt")
    assert len(chunks) > 0
    assert chunks[0].metadata["doc_type"] == "case"


def test_metadata_format():
    """测试元数据格式化"""
    from app.utils.metadata import format_source_display

    law_meta = {"doc_type": "law", "law_name": "中华人民共和国民法典", "article_number": "595"}
    display = format_source_display(law_meta)
    assert "民法典" in display
    assert "595" in display

    case_meta = {"doc_type": "case", "guiding_number": "指导案例24号", "section_type": "裁判要旨"}
    display = format_source_display(case_meta)
    assert "指导案例24号" in display
