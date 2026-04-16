"""知识库管理服务"""

import os
import json
from pathlib import Path
from langchain_core.documents import Document
from app.core.vectorstore import get_vectorstore, reset_store_cache
from app.core.embeddings import get_embeddings
from app.utils.legal_chunker import split_legal_document
from app.models.schemas import KnowledgeFileInfo, KnowledgeStats
from app.config import settings


# ===================== 文件读取 =====================

def _read_text_file(filepath: str) -> str:
    """读取文本文件"""
    encodings = ["utf-8", "gbk", "gb2312", "utf-16"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc) as f:
                return f.read()
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError(f"无法解码文件: {filepath}")


def _detect_doc_type(filepath: str) -> str:
    """根据路径判断文档类型"""
    path_lower = filepath.lower()
    if "case" in path_lower or "案例" in path_lower:
        return "case"
    return "law"


def _collect_files(dir_path: str, extensions: set[str]) -> list[str]:
    """递归收集目录下指定后缀的文件"""
    results = []
    if not os.path.isdir(dir_path):
        return results
    for root, _dirs, files in os.walk(dir_path):
        for fname in sorted(files):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in extensions:
                results.append(os.path.join(root, fname))
    return results


def _count_files(dir_path: str) -> int:
    """递归统计目录下的文件数"""
    count = 0
    if not os.path.isdir(dir_path):
        return 0
    for root, _dirs, files in os.walk(dir_path):
        count += sum(1 for f in files if not f.startswith("."))
    return count


# ===================== 知识库统计 =====================

def get_kb_stats() -> KnowledgeStats:
    """获取知识库统计信息"""
    stats = KnowledgeStats()
    for name in [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]:
        try:
            store = get_vectorstore(name)
            count = store._collection.count()
            if name == settings.LAWS_COLLECTION:
                stats.laws_chunks = count
            else:
                stats.cases_chunks = count
            stats.total_chunks += count
            stats.collections.append(name)
        except Exception:
            continue

    stats.total_files = _count_files(settings.LAWS_DIR) + _count_files(settings.CASES_DIR)
    return stats


# ===================== 上传文档 =====================

async def upload_document(
    filename: str,
    content: bytes,
    doc_type: str | None = None,
) -> KnowledgeFileInfo:
    """上传并索引一个文档"""
    # 确定类型和目标目录
    if doc_type is None:
        doc_type = _detect_doc_type(filename)
    target_dir = settings.CASES_DIR if doc_type == "case" else settings.LAWS_DIR
    os.makedirs(target_dir, exist_ok=True)

    # 保存文件
    filepath = os.path.join(target_dir, filename)
    with open(filepath, "wb") as f:
        f.write(content)

    # 读取并分块
    text = _read_text_file(filepath)
    chunks = split_legal_document(text, doc_type=doc_type, filename=filename)

    # 写入向量库
    collection_name = settings.CASES_COLLECTION if doc_type == "case" else settings.LAWS_COLLECTION
    store = get_vectorstore(collection_name)
    store.add_documents(chunks)

    return KnowledgeFileInfo(
        filename=filename,
        doc_type=doc_type,
        size_bytes=len(content),
        chunk_count=len(chunks),
    )


# ===================== 列出知识库文件 =====================

def list_documents() -> list[KnowledgeFileInfo]:
    """列出所有已上传的文档（递归子目录）"""
    files: list[KnowledgeFileInfo] = []
    for dir_path, dtype, exts in [
        (settings.LAWS_DIR, "law", {".txt", ".md"}),
        (settings.CASES_DIR, "case", {".txt", ".md", ".json"}),
    ]:
        for fpath in _collect_files(dir_path, exts):
            fname = os.path.relpath(fpath, dir_path)
            files.append(KnowledgeFileInfo(
                filename=fname,
                doc_type=dtype,
                size_bytes=os.path.getsize(fpath),
            ))
    return files


# ===================== 删除文档 =====================

def delete_document(filename: str) -> bool:
    """删除文档文件（需要重建索引才能从向量库移除）"""
    for dir_path in [settings.LAWS_DIR, settings.CASES_DIR]:
        fpath = os.path.join(dir_path, filename)
        if os.path.isfile(fpath):
            os.remove(fpath)
            return True
    return False


# ===================== 重建索引 =====================

async def rebuild_index() -> KnowledgeStats:
    """从 data/ 目录重建全部向量索引"""
    import chromadb

    # 清除现有数据
    reset_store_cache()
    client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
    for name in [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]:
        try:
            client.delete_collection(name)
        except Exception:
            pass

    # --- 法律条文 ---
    law_chunks: list[Document] = []
    for fpath in _collect_files(settings.LAWS_DIR, {".txt", ".md"}):
        try:
            text = _read_text_file(fpath)
            chunks = split_legal_document(text, doc_type="law", filename=os.path.basename(fpath))
            law_chunks.extend(chunks)
        except Exception:
            continue

    if law_chunks:
        reset_store_cache()
        store = get_vectorstore(settings.LAWS_COLLECTION)
        batch_size = 100
        for i in range(0, len(law_chunks), batch_size):
            store.add_documents(law_chunks[i : i + batch_size])

    # --- 案例 (JSONL) ---
    case_chunks: list[Document] = []
    case_count = 0
    max_cases = 500
    for fpath in _collect_files(settings.CASES_DIR, {".json"}):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if case_count >= max_cases:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = obj.get("A", "")
                    if not text or len(text) < 50:
                        continue
                    chunks = split_legal_document(
                        text, doc_type="case",
                        filename=f"{os.path.basename(fpath)}:案例{case_count + 1}",
                    )
                    case_chunks.extend(chunks)
                    case_count += 1
        except Exception:
            continue
        if case_count >= max_cases:
            break

    # 也导入 txt 格式案例
    for fpath in _collect_files(settings.CASES_DIR, {".txt", ".md"}):
        try:
            text = _read_text_file(fpath)
            chunks = split_legal_document(text, doc_type="case", filename=os.path.basename(fpath))
            case_chunks.extend(chunks)
        except Exception:
            continue

    if case_chunks:
        reset_store_cache()
        store = get_vectorstore(settings.CASES_COLLECTION)
        batch_size = 100
        for i in range(0, len(case_chunks), batch_size):
            store.add_documents(case_chunks[i : i + batch_size])

    reset_store_cache()
    return get_kb_stats()
