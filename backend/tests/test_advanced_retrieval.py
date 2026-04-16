"""高级检索功能测试"""

import pytest


def test_normalize_legal_terms_known_mappings():
    """测试法律术语规范化映射"""
    from app.services.query_rewriter import normalize_legal_terms

    assert "故意伤害" in normalize_legal_terms("打人怎么判刑")
    assert "盗窃" in normalize_legal_terms("偷东西被抓怎么办")
    assert "民间借贷纠纷" in normalize_legal_terms("借钱不还怎么起诉")
    assert "危险驾驶" in normalize_legal_terms("酒驾怎么处罚")
    assert "解除劳动合同" in normalize_legal_terms("被公司辞退")


def test_normalize_legal_terms_no_change():
    """测试不匹配时保持原文"""
    from app.services.query_rewriter import normalize_legal_terms

    original = "故意杀人罪的量刑标准"
    assert normalize_legal_terms(original) == original


def test_legal_term_map_coverage():
    """测试术语映射表至少有 15 个条目"""
    from app.services.query_rewriter import LEGAL_TERM_MAP

    assert len(LEGAL_TERM_MAP) >= 15


def test_contextual_header_law():
    """测试法律条文的上下文标头注入"""
    from langchain_core.documents import Document
    from app.utils.legal_chunker import add_contextual_header

    chunk = Document(
        page_content="第一条 为了保护民事主体的合法权益",
        metadata={"article_number": "一"},
    )
    parent_meta = {"law_name": "中华人民共和国民法典"}

    result = add_contextual_header(chunk, parent_meta)
    assert result.page_content.startswith("[")
    assert "法律名称: 中华人民共和国民法典" in result.page_content
    assert "条号: 第一条" in result.page_content


def test_contextual_header_case():
    """测试案例的上下文标头注入"""
    from langchain_core.documents import Document
    from app.utils.legal_chunker import add_contextual_header

    chunk = Document(
        page_content="交通事故的受害人没有过错",
        metadata={"section_type": "裁判要旨"},
    )
    parent_meta = {"guiding_number": "指导案例24号"}

    result = add_contextual_header(chunk, parent_meta)
    assert "案例: 指导案例24号" in result.page_content
    assert "段落: 裁判要旨" in result.page_content


def test_contextual_header_empty_metadata():
    """测试空元数据不添加标头"""
    from langchain_core.documents import Document
    from app.utils.legal_chunker import add_contextual_header

    chunk = Document(page_content="普通文本内容", metadata={})
    parent_meta = {}

    result = add_contextual_header(chunk, parent_meta)
    assert result.page_content == "普通文本内容"
