"""全局配置"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- 应用 ---
    APP_NAME: str = "法律 RAG 系统"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # --- Ollama / 模型 ---
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "qwen3:8b"
    LLM_TEMPERATURE: float = 0.3
    LLM_NUM_CTX: int = 8192
    LLM_TIMEOUT: int = 60
    EMBEDDING_MODEL: str = "bge-m3"

    # --- ChromaDB ---
    CHROMA_PERSIST_DIR: str = str(Path(__file__).resolve().parent.parent / "chroma_db")
    LAWS_COLLECTION: str = "laws"
    CASES_COLLECTION: str = "cases"

    # --- 检索 ---
    RETRIEVAL_TOP_K: int = 10
    BM25_WEIGHT: float = 0.5
    VECTOR_WEIGHT: float = 0.5
    RERANK_TOP_K: int = 5

    # --- 数据目录 ---
    DATA_DIR: str = str(Path(__file__).resolve().parent.parent / "data")
    LAWS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "laws")
    CASES_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "cases")
    REPORTS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "reports")
    CHAT_RECORDS_DIR: str = str(Path(__file__).resolve().parent.parent / "data" / "chat_records")

    # --- 高级 RAG 管线 ---
    CONTEXTUAL_CHUNKING: bool = True
    KG_DATA_PATH: str = str(
        Path(__file__).resolve().parent.parent / "data" / "laws" / "CrimeKG" / "犯罪知识图谱.txt"
    )
    HYDE_TEMPERATURE: float = 0.7
    SELF_REFLECT_MAX_ITER: int = 1
    DEFAULT_QUERY_TRANSFORM: str = "none"
    DEFAULT_RERANK: str = "simple"
    DEFAULT_GENERATION: str = "standard"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
