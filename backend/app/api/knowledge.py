"""知识库管理 API"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from app.models.schemas import APIResponse
from app.services.kb_service import (
    upload_document,
    list_documents,
    delete_document,
    rebuild_index,
    get_kb_stats,
)

router = APIRouter(prefix="/knowledge", tags=["知识库"])


@router.post("/upload", response_model=APIResponse)
async def upload(
    file: UploadFile = File(...),
    doc_type: str | None = Form(None),
):
    """上传文档到知识库"""
    allowed_ext = {".txt", ".md", ".json"}
    filename = file.filename or "unknown.txt"
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_ext:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {ext}，支持: {allowed_ext}")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")

    result = await upload_document(filename, content, doc_type)
    return APIResponse(data=result.model_dump())


@router.get("/list", response_model=APIResponse)
async def list_files():
    """列出知识库文件"""
    files = list_documents()
    return APIResponse(data=[f.model_dump() for f in files])


@router.delete("/{filename}", response_model=APIResponse)
async def delete_file(filename: str):
    """删除知识库文件"""
    ok = delete_document(filename)
    if not ok:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")
    return APIResponse(message=f"已删除 {filename}，请重建索引以更新向量库")


@router.post("/rebuild", response_model=APIResponse)
async def rebuild():
    """重建知识库索引"""
    stats = await rebuild_index()
    return APIResponse(data=stats.model_dump())


@router.get("/stats", response_model=APIResponse)
async def stats():
    """知识库统计"""
    kb_stats = get_kb_stats()
    return APIResponse(data=kb_stats.model_dump())
