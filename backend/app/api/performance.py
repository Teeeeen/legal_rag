"""性能测试 API"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.models.schemas import APIResponse
from app.services.perf_service import get_system_info, run_benchmark
from app.services.report_service import generate_report, list_reports, get_report_filepath

router = APIRouter(prefix="/performance", tags=["性能监控"])


@router.get("/system", response_model=APIResponse)
async def system_info():
    """获取系统资源状态"""
    info = get_system_info()
    return APIResponse(data=info.model_dump())


@router.post("/bench", response_model=APIResponse)
async def benchmark(
    queries: list[str] | None = None,
    use_rerank: bool = True,
    evaluate_quality: bool = False,
):
    """运行基准测试，可选质量评估"""
    result = await run_benchmark(
        queries=queries,
        use_rerank=use_rerank,
        evaluate_quality=evaluate_quality,
    )
    return APIResponse(data=result.model_dump())


@router.post("/report", response_model=APIResponse)
async def create_report(
    queries: list[str] | None = None,
    use_rerank: bool = True,
    evaluate_quality: bool = True,
):
    """运行基准测试 + 质量评估，生成并保存完整报告"""
    bench_result = await run_benchmark(
        queries=queries,
        use_rerank=use_rerank,
        evaluate_quality=evaluate_quality,
    )
    report = generate_report(bench_result)
    return APIResponse(data=report.model_dump())


@router.get("/reports", response_model=APIResponse)
async def get_reports():
    """列出历史报告"""
    metas = list_reports()
    return APIResponse(data=[m.model_dump() for m in metas])


@router.get("/reports/{report_id}")
async def download_report(report_id: str):
    """下载指定报告 JSON 文件"""
    filepath = get_report_filepath(report_id)
    if filepath is None:
        raise HTTPException(status_code=404, detail=f"报告 {report_id} 不存在")
    return FileResponse(
        path=str(filepath),
        media_type="application/json",
        filename=f"report_{report_id}.json",
    )
