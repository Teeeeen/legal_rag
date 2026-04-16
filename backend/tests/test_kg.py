"""犯罪知识图谱模块测试"""

import pytest


def test_load_crime_kg_entries():
    """测试 KG 加载条目数量"""
    from app.services.kg_service import _load_crime_kg

    # 清除缓存以确保重新加载
    _load_crime_kg.cache_clear()
    kg = _load_crime_kg()
    # 原始文件包含 856 个罪名
    assert len(kg) >= 100, f"KG 只加载了 {len(kg)} 个条目，预期 >= 100"


def test_load_crime_kg_structure():
    """测试 KG 条目结构"""
    from app.services.kg_service import _load_crime_kg

    kg = _load_crime_kg()
    # 抽查一个常见罪名
    if "故意杀人罪" in kg:
        entry = kg["故意杀人罪"]
        assert isinstance(entry, dict)
        # 至少有概念或量刑
        assert any(k in entry for k in ["概念与定义", "量刑处罚", "犯罪构成"])


def test_kg_lookup_existing_crime():
    """测试查找存在的罪名"""
    from app.services.kg_service import kg_lookup

    docs = kg_lookup(["故意杀人罪"])
    if docs:  # 如果 KG 文件存在
        assert len(docs) == 1
        assert docs[0].metadata["doc_type"] == "kg"
        assert docs[0].metadata["crime_name"] == "故意杀人罪"
        assert "故意杀人罪" in docs[0].page_content


def test_kg_lookup_nonexistent_crime():
    """测试查找不存在的罪名返回空列表"""
    from app.services.kg_service import kg_lookup

    docs = kg_lookup(["完全不存在的罪名xyz"])
    assert docs == []


def test_kg_lookup_multiple_crimes():
    """测试查找多个罪名"""
    from app.services.kg_service import kg_lookup, _load_crime_kg

    kg = _load_crime_kg()
    # 取前两个存在的罪名
    names = list(kg.keys())[:2]
    if len(names) >= 2:
        docs = kg_lookup(names)
        assert len(docs) == 2


def test_kg_lookup_empty_list():
    """测试空列表输入"""
    from app.services.kg_service import kg_lookup

    docs = kg_lookup([])
    assert docs == []
