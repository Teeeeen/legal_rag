"""问答 API"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import ChatRequest, ChatResponse, APIResponse
from app.services.rag_service import rag_query
from app.services.perf_service import get_system_info
from app.services.report_service import (
    save_chat_record,
    list_chat_records,
    get_chat_record_filepath,
)

router = APIRouter(prefix="/chat", tags=["问答"])


@router.post("", response_model=APIResponse)
async def chat(req: ChatRequest):
    """RAG 问答接口，支持系统监控和质量评估"""
    try:
        # 系统快照（问答前）
        system_before = None
        if req.monitor_system:
            system_before = get_system_info()

        # RAG 查询
        result: ChatResponse = await rag_query(
            question=req.question,
            use_rerank=req.use_rerank,
            use_query_rewrite=req.use_query_rewrite,
            top_k=req.top_k,
            collection=req.collection,
            query_transform=req.query_transform,
            rerank_strategy=req.rerank_strategy,
            generation_strategy=req.generation_strategy,
            use_kg=req.use_kg,
        )

        # 系统快照（问答后）
        system_after = None
        if req.monitor_system:
            system_after = get_system_info()

        # 质量评估
        quality = None
        if req.evaluate_quality:
            from app.services.quality_service import evaluate_single_query
            quality = await evaluate_single_query(
                query=req.question,
                answer=result.answer,
                sources=result.sources,
            )

        result.system_before = system_before
        result.system_after = system_after
        result.quality = quality

        return APIResponse(data=result.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"问答服务异常: {e}")


@router.post("/save-record", response_model=APIResponse)
async def save_record(record: dict):
    """保存一次问答的性能记录"""
    try:
        record_id = save_chat_record(record)
        return APIResponse(data={"record_id": record_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"保存记录失败: {e}")


@router.get("/records", response_model=APIResponse)
async def get_records():
    """列出问答性能记录"""
    metas = list_chat_records()
    return APIResponse(data=[m.model_dump() for m in metas])


@router.get("/records/{record_id}")
async def download_record(record_id: str):
    """下载问答性能记录 JSON"""
    filepath = get_chat_record_filepath(record_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"记录 {record_id} 不存在")
    return FileResponse(
        path=str(filepath),
        media_type="application/json",
        filename=f"chat_record_{record_id}.json",
    )
