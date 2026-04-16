"""混合检索器 — BM25 稀疏检索 + 向量稠密检索 + RRF 融合"""

import jieba
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from app.core.vectorstore import get_vectorstore
from app.config import settings


class BM25ChineseRetriever(BaseRetriever):
    """基于 jieba 分词的 BM25 中文检索器"""

    documents: list[Document] = []
    tokenized_corpus: list[list[str]] = []
    bm25: BM25Okapi | None = None
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, documents: list[Document], k: int = 10, **kwargs):
        super().__init__(**kwargs)
        self.documents = documents
        self.k = k
        # jieba 分词构建语料库
        self.tokenized_corpus = [
            list(jieba.cut(doc.page_content)) for doc in documents
        ]
        if self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        if not self.bm25 or not self.documents:
            return []
        tokenized_query = list(jieba.cut(query))
        scores = self.bm25.get_scores(tokenized_query)
        # 获取 top-k 索引
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.k
        ]
        return [self.documents[i] for i in top_indices if scores[i] > 0]


class HybridRetriever(BaseRetriever):
    """混合检索器：BM25 + 向量检索 + RRF 融合"""

    bm25_retriever: BM25ChineseRetriever | None = None
    collection_names: list[str] = ["laws", "cases"]
    bm25_weight: float = settings.BM25_WEIGHT
    vector_weight: float = settings.VECTOR_WEIGHT
    k: int = settings.RETRIEVAL_TOP_K
    all_documents: list[Document] = []

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, collection_names: list[str] | None = None, **kwargs):
        super().__init__(**kwargs)
        if collection_names:
            self.collection_names = collection_names
        self._load_bm25_corpus()

    def _load_bm25_corpus(self):
        """从向量库加载文档构建 BM25 索引"""
        all_docs = []
        for name in self.collection_names:
            try:
                store = get_vectorstore(name)
                result = store._collection.get(include=["documents", "metadatas"])
                if result and result["documents"]:
                    for doc_text, meta in zip(
                        result["documents"], result["metadatas"] or [{}] * len(result["documents"])
                    ):
                        all_docs.append(Document(page_content=doc_text, metadata=meta or {}))
            except Exception:
                continue
        self.all_documents = all_docs
        if all_docs:
            self.bm25_retriever = BM25ChineseRetriever(documents=all_docs, k=self.k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        """RRF (Reciprocal Rank Fusion) 混合检索"""
        results_map: dict[str, dict] = {}  # content_hash -> {doc, rrf_score}
        rrf_k = 60  # RRF 常数

        # BM25 检索
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)
            for rank, doc in enumerate(bm25_results):
                key = hash(doc.page_content[:200])
                score = self.bm25_weight / (rrf_k + rank + 1)
                if key in results_map:
                    results_map[key]["score"] += score
                else:
                    results_map[key] = {"doc": doc, "score": score}

        # 向量检索
        for name in self.collection_names:
            try:
                vec_results = get_vectorstore(name).similarity_search(query, k=self.k)
                for rank, doc in enumerate(vec_results):
                    key = hash(doc.page_content[:200])
                    score = self.vector_weight / (rrf_k + rank + 1)
                    if key in results_map:
                        results_map[key]["score"] += score
                    else:
                        results_map[key] = {"doc": doc, "score": score}
            except Exception:
                continue

        # 按 RRF 分数排序
        sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results[: self.k]]

    def search_with_split_queries(
        self, bm25_query: str, vector_query: str
    ) -> list[Document]:
        """分离式检索：BM25 使用一个查询，向量检索使用另一个查询

        用于 HyDE 场景：bm25_query 为原始问题，vector_query 为假设文档。
        """
        results_map: dict[int, dict] = {}
        rrf_k = 60

        # BM25 检索（使用原始查询）
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(bm25_query)
            for rank, doc in enumerate(bm25_results):
                key = hash(doc.page_content[:200])
                score = self.bm25_weight / (rrf_k + rank + 1)
                if key in results_map:
                    results_map[key]["score"] += score
                else:
                    results_map[key] = {"doc": doc, "score": score}

        # 向量检索（使用假设文档 / HyDE 文本）
        for name in self.collection_names:
            try:
                vec_results = get_vectorstore(name).similarity_search(vector_query, k=self.k)
                for rank, doc in enumerate(vec_results):
                    key = hash(doc.page_content[:200])
                    score = self.vector_weight / (rrf_k + rank + 1)
                    if key in results_map:
                        results_map[key]["score"] += score
                    else:
                        results_map[key] = {"doc": doc, "score": score}
            except Exception:
                continue

        sorted_results = sorted(results_map.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_results[: self.k]]


def get_hybrid_retriever(
    collection_names: list[str] | None = None,
) -> HybridRetriever:
    """获取混合检索器"""
    names = collection_names or ["laws", "cases"]
    return HybridRetriever(collection_names=names)
