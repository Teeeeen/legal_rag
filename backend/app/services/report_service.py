"""性能报告生成、存储与查询服务"""

import json
import os
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.config import settings
from app.models.schemas import (
    BenchmarkResultV2,
    ReportMeta,
    ReportFull,
    ChatRecordMeta,
)


def _ensure_reports_dir() -> Path:
    """确保报告目录存在"""
    reports_dir = Path(settings.REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return reports_dir


def generate_report(benchmark: BenchmarkResultV2) -> ReportFull:
    """基于基准测试结果生成完整报告并保存到磁盘"""
    from app.services.kb_service import get_kb_stats

    report_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    created_at = datetime.now().isoformat()

    # 知识库统计
    try:
        kb_stats = get_kb_stats()
        kb_dict = kb_stats.model_dump()
    except Exception:
        kb_dict = {}

    # 轻量化亮点
    highlights = {
        "model": settings.LLM_MODEL,
        "embedding": settings.EMBEDDING_MODEL,
        "vector_db": "ChromaDB (嵌入式)",
        "deployment": "全本地部署，无需云端 API",
        "avg_latency_ms": benchmark.avg_latency_ms,
        "qps": benchmark.queries_per_second,
    }
    if benchmark.quality:
        highlights["avg_rouge_l"] = benchmark.quality.avg_rouge_l
        highlights["avg_retrieval_relevance"] = benchmark.quality.avg_retrieval_relevance
        highlights["avg_faithfulness"] = benchmark.quality.avg_faithfulness

    report = ReportFull(
        report_id=report_id,
        created_at=created_at,
        system_snapshot=benchmark.system_info,
        benchmark=benchmark.model_dump(),
        knowledge_base=kb_dict,
        lightweight_highlights=highlights,
    )

    # 保存到文件
    reports_dir = _ensure_reports_dir()
    filepath = reports_dir / f"{report_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report.model_dump(), f, ensure_ascii=False, indent=2)

    return report


def list_reports() -> list[ReportMeta]:
    """扫描报告目录，返回报告元信息列表（按时间倒序）"""
    reports_dir = _ensure_reports_dir()
    metas: list[ReportMeta] = []

    for filename in sorted(os.listdir(reports_dir), reverse=True):
        if not filename.endswith(".json"):
            continue
        filepath = reports_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            bench = data.get("benchmark", {})
            metas.append(ReportMeta(
                report_id=data.get("report_id", filename.replace(".json", "")),
                created_at=data.get("created_at", ""),
                total_queries=bench.get("total_queries", 0),
                avg_latency_ms=bench.get("avg_latency_ms", 0),
                queries_per_second=bench.get("queries_per_second", 0),
                has_quality=bench.get("quality") is not None,
            ))
        except Exception:
            continue

    return metas


def get_report_filepath(report_id: str) -> Path | None:
    """根据 report_id 返回报告文件路径，不存在返回 None"""
    reports_dir = _ensure_reports_dir()
    filepath = reports_dir / f"{report_id}.json"
    if filepath.exists():
        return filepath
    return None


# ===================== 问答性能记录 =====================

def _ensure_chat_records_dir() -> Path:
    """确保问答记录目录存在"""
    records_dir = Path(settings.CHAT_RECORDS_DIR)
    records_dir.mkdir(parents=True, exist_ok=True)
    return records_dir


def save_chat_record(record: dict) -> str:
    """保存一次问答的性能记录，返回 record_id"""
    record_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:6]
    record["record_id"] = record_id
    record["created_at"] = datetime.now().isoformat()

    records_dir = _ensure_chat_records_dir()
    filepath = records_dir / f"{record_id}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    return record_id


def list_chat_records() -> list[ChatRecordMeta]:
    """扫描问答记录目录，返回元信息列表（按时间倒序）"""
    records_dir = _ensure_chat_records_dir()
    metas: list[ChatRecordMeta] = []

    for filename in sorted(os.listdir(records_dir), reverse=True):
        if not filename.endswith(".json"):
            continue
        filepath = records_dir / filename
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            metas.append(ChatRecordMeta(
                record_id=data.get("record_id", filename.replace(".json", "")),
                created_at=data.get("created_at", ""),
                question=data.get("question", "")[:50],
                total_ms=data.get("metrics", {}).get("total_ms", 0),
                has_quality=data.get("quality") is not None,
            ))
        except Exception:
            continue

    return metas


def get_chat_record_filepath(record_id: str) -> Path | None:
    """根据 record_id 返回问答记录文件路径"""
    records_dir = _ensure_chat_records_dir()
    filepath = records_dir / f"{record_id}.json"
    if filepath.exists():
        return filepath
    return None
