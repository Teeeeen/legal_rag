"""国家法律法规数据库爬虫

从 flk.npc.gov.cn 的公开 API 获取法律条文数据。

用法:
    cd backend
    python -m scripts.crawl_laws [--max 500] [--type law]

注意:
- 该爬虫仅抓取公开可访问的法律条文页面
- 请适度使用，遵守网站使用条款
- 默认限速：每次请求间隔 1 秒
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "laws" / "NPC_Crawled"

# flk.npc.gov.cn 的公开 API
LIST_API = "https://flk.npc.gov.cn/api/"
DETAIL_API = "https://flk.npc.gov.cn/api/detail"

# 法律类型映射
LAW_TYPES = {
    "flfg": "法律法规",
    "xzfg": "行政法规",
    "sfjs": "司法解释",
    "dfxfg": "地方性法规",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://flk.npc.gov.cn/",
}


def fetch_json(url: str, data: dict = None, retries: int = 3) -> dict:
    """发送 HTTP 请求并返回 JSON 响应"""
    for attempt in range(retries):
        try:
            if data:
                body = json.dumps(data).encode("utf-8")
                req = Request(url, data=body, headers={**HEADERS, "Content-Type": "application/json"})
            else:
                req = Request(url, headers=HEADERS)
            with urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (URLError, HTTPError, TimeoutError) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            print(f"  ✗ 请求失败 ({url}): {e}")
            return {}


def fetch_law_list(law_type: str = "flfg", page: int = 1, size: int = 10) -> dict:
    """获取法律列表"""
    data = {
        "type": law_type,
        "searchType": "title;vague",
        "sortTr": "f_bbrq_s;desc",
        "gbrqStart": "",
        "gbrqEnd": "",
        "sxrqStart": "",
        "sxrqEnd": "",
        "sort": "true",
        "page": page,
        "size": size,
        "title": "",
    }
    return fetch_json(LIST_API, data)


def fetch_law_detail(law_id: str) -> dict:
    """获取法律详情"""
    return fetch_json(f"{DETAIL_API}?id={law_id}")


def clean_html(html_text: str) -> str:
    """清理 HTML 标签，转为纯文本"""
    if not html_text:
        return ""
    # 替换常见 HTML 标签
    text = re.sub(r"<br\s*/?>", "\n", html_text)
    text = re.sub(r"<p[^>]*>", "\n", text)
    text = re.sub(r"</p>", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    # 清理 HTML 实体
    text = text.replace("&nbsp;", " ")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def crawl_laws(law_type: str = "flfg", max_count: int = 500, delay: float = 1.0):
    """爬取法律条文"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    type_name = LAW_TYPES.get(law_type, law_type)

    print(f"\n开始爬取 [{type_name}]...")

    page = 1
    size = 10
    total_saved = 0

    while total_saved < max_count:
        result = fetch_law_list(law_type, page, size)
        if not result:
            print(f"  第 {page} 页请求失败，停止")
            break

        data = result.get("result", {})
        records = data.get("data", [])
        total_available = data.get("totalSizes", 0)

        if not records:
            print(f"  第 {page} 页无数据，停止")
            break

        print(f"  第 {page} 页: {len(records)} 条 (共 {total_available} 条可用)")

        for record in records:
            if total_saved >= max_count:
                break

            law_id = record.get("id", "")
            title = record.get("title", "").strip()
            publish_date = record.get("publish", "")
            status = record.get("status", "")

            if not title or not law_id:
                continue

            # 获取详情
            detail = fetch_law_detail(law_id)
            time.sleep(delay)

            if not detail:
                continue

            detail_data = detail.get("result", {})
            body_list = detail_data.get("body", [])

            if not body_list:
                continue

            # 拼接所有正文内容
            full_text = ""
            for body_item in body_list:
                content = body_item.get("body", "")
                if content:
                    full_text += clean_html(content) + "\n\n"

            if len(full_text.strip()) < 100:
                continue

            # 构建输出文件
            safe_title = re.sub(r'[/<>:"|?*\\]', '_', title)[:80]
            filepath = OUTPUT_DIR / f"{safe_title}.txt"

            header = f"{title}\n"
            if publish_date:
                header += f"发布日期：{publish_date}\n"
            if status:
                header += f"状态：{status}\n"
            header += f"来源：国家法律法规数据库\n\n"

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(header + full_text.strip())

            total_saved += 1
            if total_saved % 10 == 0:
                print(f"  已保存 {total_saved} 部 {type_name}")

        page += 1

    print(f"  ✓ [{type_name}] 完成: {total_saved} 部")
    return total_saved


def main():
    parser = argparse.ArgumentParser(description="国家法律法规数据库爬虫")
    parser.add_argument("--max", type=int, default=500, help="每种类型最大爬取数量")
    parser.add_argument("--type", type=str, default="all",
                        choices=["all", "flfg", "xzfg", "sfjs", "dfxfg"],
                        help="法律类型 (all=全部)")
    parser.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")
    args = parser.parse_args()

    print("=" * 55)
    print("国家法律法规数据库爬虫")
    print(f"输出目录: {OUTPUT_DIR}")
    print("=" * 55)

    total = 0
    if args.type == "all":
        for lt in ["flfg", "xzfg", "sfjs"]:
            total += crawl_laws(lt, args.max, args.delay)
    else:
        total = crawl_laws(args.type, args.max, args.delay)

    print(f"\n{'=' * 55}")
    print(f"爬取完成: 共 {total} 部法律文档")
    print(f"保存位置: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
