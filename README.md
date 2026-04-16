# 基于 LangChain 与本地大模型的轻量化 RAG 系统

> **Lightweight Legal RAG System** — 面向中国法律领域的本地化检索增强生成系统，可在消费级 GPU 上运行

<p align="center">
  <strong>Qwen3:8B + BGE-M3 + ChromaDB + LangChain + FastAPI + React</strong>
</p>

---

## 目录

- [项目概述](#项目概述)
- [核心特性](#核心特性)
- [系统架构](#系统架构)
- [法律领域 RAG 优化方案](#法律领域-rag-优化方案)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [后端详细说明](#后端详细说明)
  - [可插拔 RAG 管线架构](#可插拔-rag-管线架构)
  - [API 接口文档](#api-接口文档)
  - [配置参数](#配置参数)
- [前端详细说明](#前端详细说明)
  - [智能问答页面](#智能问答页面)
  - [知识库管理页面](#知识库管理页面)
  - [性能监控页面](#性能监控页面)
- [数据源](#数据源)
- [测试](#测试)
- [性能测试](#性能测试)
- [技术选型与决策](#技术选型与决策)
- [许可证](#许可证)

---

## 项目概述

本项目构建了一个**前后端分离**的轻量化 RAG（Retrieval-Augmented Generation）系统，以 **LangChain** 为编排框架，使用本地部署的 **Qwen3:8B** 作为生成模型、**BGE-M3** 作为 Embedding 模型，结合 **ChromaDB** 向量数据库，实现了从文档加载、智能分块、向量化存储到语义检索与生成的完整 RAG 流程。

系统选择**中国法律领域**作为垂直应用场景进行验证。法律文本具有高度结构化（编/章/节/条/款/项）、术语精确、条文间存在引用关系、用户查询常为口语化表述等显著特征，对 RAG 系统的分块策略、检索精度和生成忠实度提出了严格要求，是验证轻量化 RAG 系统能力的理想领域。系统针对这些领域特征实现了 **6 项专属 RAG 优化方案**（详见下方），也为将系统适配到其他垂直领域（医疗、金融、教育等）提供了可参考的方法论。

---

## 核心特性

- **本地化部署**：所有模型均通过 Ollama 在本地运行，无需依赖云端 API，实现"数据不出域"的隐私保护
- **法律领域深度优化**：6 项针对法律文本特征的专属 RAG 优化（结构化分块 / 上下文标头 / HyDE / 术语规范化 / 知识图谱 / 自我反思）
- **可插拔管线架构**：查询变换（5种）、重排序（3种）、生成（4种）四阶段均支持策略切换，前端可视化配置
- **前后端分离**：FastAPI 异步后端 + React 单页前端
- **性能与质量评估**：内置 CPU/内存实时监控、各阶段延迟分解、ROUGE/忠实度/相关性自动评估
- **多源数据整合**：整合 6 个开源法律数据集，共导入约 59,000 个文本分块

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                       Frontend (React + Vite)                    │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐                     │
│  │ 问答界面  │  │ 知识库管理│  │ 性能监控  │                     │
│  │ 策略配置  │  │ 上传/重建 │  │ 基准测试  │                     │
│  └─────┬────┘  └─────┬────┘  └─────┬─────┘                     │
│        └─────────────┼─────────────┘                            │
│                      │ HTTP / REST API                           │
├──────────────────────┼──────────────────────────────────────────┤
│                  Backend (FastAPI)                                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Pluggable RAG Pipeline                       │   │
│  │                                                           │   │
│  │  Stage 1: 查询变换                                        │   │
│  │    [无 | 多查询扩展 | HyDE | 子问题分解 | 多查询+HyDE]    │   │
│  │              ↓                                            │   │
│  │  Stage 2: 混合检索 (BM25 + 向量 + RRF 融合)               │   │
│  │    + Stage 2.5: 犯罪知识图谱查找 (可选，并行)              │   │
│  │              ↓                                            │   │
│  │  Stage 3: 重排序                                          │   │
│  │    [无 | 简单重排(Jaccard+元数据) | LLM重排]               │   │
│  │              ↓                                            │   │
│  │  Stage 4: 生成                                            │   │
│  │    [标准 | 链式推理CoT | 自我反思修正 | 结构化法律回答]      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ChromaDB  │  │Qwen3:8B   │  │BGE-M3    │  │犯罪知识图谱   │   │
│  │向量数据库│  │(Ollama)   │  │(Ollama)  │  │(内存缓存)    │   │
│  └──────────┘  └───────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 法律领域 RAG 优化方案

> 本节是本项目的核心价值所在。法律文本与通用文本有显著差异——高度结构化（编/章/节/条/款/项）、术语精确、条文间存在引用关系、用户查询常为口语化表述。主流 RAG 方案在面对这些特征时存在明确的短板，本项目针对每一项短板给出了专属解决方案。

### 1. 法律条文结构化分块 vs 主流固定窗口分块

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| 分块策略 | 固定 token 窗口（如 512 tokens），按字符数切割 | `LegalArticleSplitter`：以"条"为原子单位，按编/章/节/条/款/项层级切分 |
| 问题 | 一条法律条文可能被切成两半，上半段和下半段分别嵌入，语义破碎 | 以"第X条"为切分点，保证每个 chunk 包含完整的法律条文 |
| 案例处理 | 同上，案例的"案情"和"判决"可能混在一个 chunk 里 | `LegalCaseSplitter`：按"裁判要旨/基本案情/裁判理由/裁判结果"结构切分 |

**核心实现**：`backend/app/utils/legal_chunker.py`

```python
# 法律条文切分层级（优先级从高到低）
separators = [
    r"\n第[一二三四五六七八九十百千]+编",   # 编
    r"\n第[一二三四五六七八九十百千]+章",   # 章
    r"\n第[一二三四五六七八九十百千]+节",   # 节
    r"\n第[一二三四五六七八九十\d]+条",     # 条（核心切分点）
    r"\n[一二三四五六七八九十]+、",          # 款
    r"\n（[一二三四五六七八九十]+）",        # 项
]
```

### 2. 上下文标头注入 (Contextual Header) vs 裸文本嵌入

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| Chunk 内容 | 直接对原始文本做 Embedding | 在 chunk 前注入结构化标头：`[法律名称: 民法典 | 章节: 第九章 | 条号: 第595条]` |
| 问题 | 一段"当事人应当按照约定..."的文本脱离了"这是民法典第595条"的上下文，Embedding 无法捕捉其法律归属 | 标头让 Embedding 模型"看到"每段文本属于哪部法律、哪一章、第几条 |
| 灵感来源 | Anthropic 2024 年提出的 Contextual Retrieval，但原方案是让 LLM 为每个 chunk 生成上下文描述（成本高） | 本项目利用法律文本已有的结构化元数据（法律名/章节/条号），**零 LLM 成本**地构建上下文标头 |

**核心实现**：`backend/app/utils/legal_chunker.py` → `add_contextual_header()`

### 3. HyDE 假设文档检索 vs 直接查询嵌入

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| 向量检索输入 | 直接对用户问题做 Embedding，与文档 Embedding 比较 | HyDE：先让 LLM 生成一段假设性法律条文，用该条文做向量检索 |
| 问题 | 用户问"酒驾怎么判"→ 这是一个**问题**的 Embedding，与法律**条文**的 Embedding 分布差异大（query-document mismatch） | LLM 生成"根据刑法第一百三十三条之一，危险驾驶罪..."→ 这段假设条文与真实条文的 Embedding 分布一致 |
| 特别之处 | 通用 HyDE 对所有检索使用假设文档 | **分离式检索**：假设文档仅用于向量检索，原始问题仍用于 BM25（保留精确关键词匹配能力） |

**核心实现**：`backend/app/services/hyde.py` + `core/retriever.py` → `search_with_split_queries()`

### 4. 法律术语规范化 + 查询分解 vs 原始查询直接检索

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| 查询预处理 | 用户查询原样送入检索器 | 两步预处理：(1) 口语→术语映射 (2) 复杂问题分解为子问题 |
| 问题 | 用户说"打人怎么判"，但法律条文写的是"故意伤害罪"；用户说"酒驾"，法律写"危险驾驶" | 21 组口语→术语映射表 + LLM 子问题分解 |
| 法律领域特殊性 | 通用领域同义词问题不严重 | 法律领域口语和正式术语之间的鸿沟极大，"偷东西"vs"盗窃罪"、"老赖"vs"失信被执行人" |

**核心实现**：`backend/app/services/query_rewriter.py` → `normalize_legal_terms()` + `decompose_query()`

### 5. 犯罪知识图谱增强 (KAG) vs 纯向量检索

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| 知识来源 | 仅依赖向量数据库中的非结构化文本 | 向量检索 + 结构化犯罪知识图谱（455 个罪名的定义/构成要件/量刑/法条） |
| 问题 | 问"故意杀人罪的构成要件"时，向量检索可能返回民法典的"人身权利"相关条文（语义接近但领域错误） | KG 精确匹配罪名 → 返回该罪名的完整结构化知识（定义、四要件、量刑标准、相关法条）→ 合并到检索结果前列 |
| 实现方式 | 需要 Neo4j 等图数据库 | 轻量实现：将犯罪知识图谱文件解析为内存字典，LLM 提取问题中的罪名后精确查找 |

**核心实现**：`backend/app/services/kg_service.py`

### 6. 自我反思修正 vs 单次生成

| 维度 | 主流做法 | 本项目方案 |
|------|----------|------------|
| 生成流程 | LLM 单次生成回答，直接返回 | 生成 → 验证（引用的条文是否在参考资料中？概念是否准确？）→ 必要时修正重生成 |
| 问题 | LLM 可能"幻觉"出不存在的法条编号，如编造"民法典第999条" | 自我反思阶段检查回答中引用的法条是否确实存在于检索到的参考资料中 |
| 成本控制 | 多轮反思延迟高 | 限制最多 1 轮修正（增加约 3-5s），在准确性和响应速度之间取平衡 |

**核心实现**：`backend/app/services/self_reflect.py`

### 优化方案总结

| 优化方案 | 解决的法律领域特有问题 | 实现文件 |
|----------|------------------------|----------|
| 结构化分块 | 法律条文的编/章/节/条层级结构 | `legal_chunker.py` |
| 上下文标头注入 | chunk 脱离法律归属上下文 | `legal_chunker.py` |
| HyDE 假设文档 | 问题与条文的 Embedding 分布差异 | `hyde.py` + `retriever.py` |
| 法律术语规范化 | 口语与法律术语的鸿沟 | `query_rewriter.py` |
| 犯罪知识图谱增强 | 刑法领域需要结构化知识（构成要件/量刑） | `kg_service.py` |
| 自我反思修正 | LLM 幻觉法条编号的风险 | `self_reflect.py` |

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|----------|----------|
| CPU | 8 核 | 16 核 |
| RAM | 16GB | 32GB |
| GPU VRAM | 6GB (可 CPU 运行) | 8GB+ NVIDIA |
| 磁盘 | 20GB SSD | 50GB SSD |

### 软件依赖

- **Conda**（Miniconda / Anaconda）
- **Python** >= 3.10
- **Node.js** >= 18.0
- **Ollama** >= 0.3.0

---

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd Legal_rag
```

### 2. 安装 Ollama 并拉取模型

```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull qwen3:8b
ollama pull bge-m3
```

### 3. 启动 Ollama 服务

> **重要**：必须在启动后端之前确保 Ollama 服务正在运行，否则后端调用 LLM/Embedding 时会返回 500 错误。

```bash
ollama serve &
# 验证服务已启动
curl -s http://localhost:11434/api/tags | head -1
```

### 4. 启动后端

```bash
conda create -n Legal_rag python=3.10 -y
conda activate Legal_rag
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. 启动前端

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:5173（Vite 开发服务器自动将 `/api/*` 代理到 `http://localhost:8000`）。

### 6. 下载原始数据

> 原始数据集未包含在仓库中（体积大，且受各数据集自身许可约束）。请按以下方式手动获取并放置到对应目录：

```
backend/data/
├── laws/
│   ├── Chinese-Laws/          ← ModelScope dengcao/Chinese-Laws（177 部法律 TXT）
│   └── HF_Chinese_Laws/       ← HuggingFace twang2218/chinese-law-and-regulations
├── cases/
│   ├── CAIL2019-SCM/          ← GitHub thunlp/CAIL（test/train/valid.json）
│   └── CAIL2018_ALL_DATA.zip  ← https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip
└── qa/
    └── CrimeKgAssitant/       ← GitHub liuhuanyong/CrimeKgAssitant（含 data/qa_corpus.json 和 data/kg_crime.json）
```

各数据集的详细来源与链接见下方[数据源](#数据源)章节。

### 7. 数据预处理与导入

```bash
conda activate Legal_rag
cd backend
python -m scripts.prepare_datasets   # 格式转换：CAIL2018 ZIP→JSONL、CrimeKG→TXT、QA→参考答案
python -m scripts.import_data        # 分块并写入 ChromaDB 向量数据库
```

### 8. 运行测试

```bash
conda activate Legal_rag
cd backend
pytest tests/ -v
```

---

## 项目结构

```
Legal_rag/
├── README.md                             # 本文件
├── .gitignore
│
├── backend/
│   ├── requirements.txt                  # Python 依赖
│   ├── app/
│   │   ├── main.py                       # FastAPI 应用入口
│   │   ├── config.py                     # 全局配置（模型、数据库、检索、高级管线参数）
│   │   ├── api/                          # HTTP API 路由层
│   │   │   ├── chat.py                   # POST /api/chat — 问答接口
│   │   │   ├── knowledge.py             # /api/knowledge/* — 知识库管理
│   │   │   └── performance.py           # /api/performance/* — 性能监控
│   │   ├── core/                         # 核心组件
│   │   │   ├── llm.py                    # Qwen3:8B 初始化（ChatOllama）
│   │   │   ├── embeddings.py            # BGE-M3 初始化（OllamaEmbeddings）
│   │   │   ├── vectorstore.py           # ChromaDB 向量数据库（带实例缓存）
│   │   │   └── retriever.py             # 混合检索器（BM25 + 向量 + RRF 融合 + HyDE 分离检索）
│   │   ├── services/                     # 业务逻辑层
│   │   │   ├── pipeline.py              # 可插拔 RAG 管线编排器（策略枚举 + PipelineConfig）
│   │   │   ├── prompts.py               # 全部 Prompt 模板（标准/CoT/结构化/HyDE/反思/分解/KG）
│   │   │   ├── hyde.py                  # HyDE 假设文档生成
│   │   │   ├── self_reflect.py          # 自我反思与修正
│   │   │   ├── kg_service.py            # 犯罪知识图谱加载、实体提取与查找
│   │   │   ├── rag_service.py           # RAG 入口（向后兼容包装器）
│   │   │   ├── kb_service.py            # 知识库管理（上传/列表/删除/重建）
│   │   │   ├── reranker.py              # 重排序（Jaccard + 元数据加分 / LLM 批量评分）
│   │   │   ├── query_rewriter.py        # 查询重写 + 术语规范化 + 查询分解
│   │   │   ├── perf_service.py          # 性能基准测试
│   │   │   ├── quality_service.py       # 质量评估（ROUGE / 相关性 / 忠实度）
│   │   │   └── report_service.py        # 测试报告生成
│   │   ├── models/
│   │   │   └── schemas.py               # Pydantic 数据模型
│   │   └── utils/
│   │       ├── legal_chunker.py         # 法律文本专用分块器 + 上下文标头注入
│   │       └── metadata.py              # 元数据格式化工具
│   ├── data/
│   │   ├── laws/                         # 法律条文 + 犯罪知识图谱
│   │   ├── cases/                        # 案例数据（JSONL）
│   │   ├── qa/                           # QA 语料
│   │   └── reference/                    # 评估参考数据
│   ├── scripts/
│   │   ├── import_data.py               # 数据批量导入脚本
│   │   ├── prepare_datasets.py          # 数据格式转换脚本
│   │   └── crawl_laws.py                # 法律法规数据库爬虫（备用）
│   └── tests/
│       ├── test_basic.py                # 基础测试（配置、分块、元数据）
│       ├── test_pipeline.py             # 管线配置与策略枚举测试
│       ├── test_advanced_retrieval.py   # 高级检索功能测试（术语规范化、上下文标头）
│       ├── test_kg.py                   # 犯罪知识图谱模块测试
│       └── test_queries.txt             # 测试问题集
│
├── frontend/
│   ├── package.json                      # 依赖与脚本
│   ├── vite.config.js                    # Vite 构建配置 + API 代理
│   ├── index.html                        # HTML 入口
│   └── src/
│       ├── main.jsx                      # React 挂载入口
│       ├── App.jsx                       # 根组件（侧边栏导航 + 页面切换）
│       ├── pages/
│       │   ├── ChatPage.jsx             # 智能问答页面（管线策略配置面板）
│       │   ├── KnowledgePage.jsx        # 知识库管理页面
│       │   └── PerformancePage.jsx      # 性能监控页面
│       ├── services/
│       │   └── api.js                   # Axios API 封装
│       └── styles/
│           └── global.css               # 全局样式
│
└── docs/
    ├── final.md                          # 毕业论文终稿
    ├── mid_term.md                       # 中期检查报告
    ├── Thesis_Proposal.md               # 开题报告
    └── devlog.md                         # 开发问题记录与解决方案
```

---

## 后端详细说明

### 可插拔 RAG 管线架构

核心是 `pipeline.py` 中的 `RAGPipeline` 类，将 RAG 流程拆分为 4 个可独立配置的阶段：

```
用户提问 → PipelineConfig → RAGPipeline.execute()

  Stage 1: 查询变换 (QueryTransformStrategy)
    ├── none          — 不变换，直接使用原始问题
    ├── multi_query   — LLM 从 3 个角度改写查询（法律条文/司法解释/不同术语）
    ├── hyde          — 生成假设性法律条文用于向量检索（原始问题仍用于 BM25）
    ├── decompose     — 将复杂问题分解为多个法律子问题
    └── multi_query_hyde — 多查询 + HyDE 组合（两个 LLM 调用并行执行）

  Stage 2: 混合检索
    BM25 (jieba分词) + BGE-M3 向量检索 → RRF 融合
    + Stage 2.5: 犯罪知识图谱查找（可选，并行）

  Stage 3: 重排序 (RerankStrategy)
    ├── none    — 不重排
    ├── simple  — Jaccard 关键词重叠 + 法律名称/条号/案号元数据加分
    └── llm     — 先简单重排预筛选 8 篇候选，再 LLM 单次批量评分

  Stage 4: 生成 (GenerationStrategy)
    ├── standard          — 标准法律问答 Prompt
    ├── chain_of_thought  — 链式推理（识别问题→查找法律→分析要件→结论→注意事项）
    ├── self_reflect      — 标准生成 + 自我反思验证 + 必要时修正
    └── structured_legal  — 结构化输出（法律结论/适用法律/详细分析/注意事项）
```

**向后兼容**：旧版 API 参数自动映射：
- `use_rerank=false` → `rerank_strategy="none"`
- `use_query_rewrite=true` → `query_transform="multi_query"`

### API 接口文档

#### 问答接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | RAG 智能问答 |

**请求参数**：

```json
{
  "question": "故意杀人罪的量刑标准是什么？",
  "top_k": 5,
  "collection": "all",
  "query_transform": "hyde",
  "rerank_strategy": "simple",
  "generation_strategy": "chain_of_thought",
  "use_kg": true,
  "evaluate_quality": false,
  "monitor_system": false
}
```

| 参数 | 可选值 | 默认值 | 说明 |
|------|--------|--------|------|
| `question` | - | 必填 | 用户问题（1-2000字） |
| `query_transform` | `none` / `multi_query` / `hyde` / `decompose` / `multi_query_hyde` | `none` | 查询变换策略 |
| `rerank_strategy` | `none` / `simple` / `llm` | `simple` | 重排序策略 |
| `generation_strategy` | `standard` / `chain_of_thought` / `self_reflect` / `structured_legal` | `standard` | 生成策略 |
| `use_kg` | `true` / `false` | `false` | 犯罪知识图谱增强 |
| `collection` | `all` / `laws` / `cases` | `all` | 检索范围 |
| `evaluate_quality` | `true` / `false` | `false` | 是否评估回答质量 |
| `monitor_system` | `true` / `false` | `false` | 是否采集系统资源快照 |

**返回数据**：

```json
{
  "answer": "根据《中华人民共和国刑法》第二百三十二条...",
  "sources": [...],
  "metrics": {
    "query_rewrite_ms": 1500.0,
    "retrieval_ms": 320.5,
    "kg_lookup_ms": 2100.0,
    "rerank_ms": 15.2,
    "generation_ms": 3500.8,
    "self_reflect_ms": null,
    "total_ms": 7450.3,
    "was_corrected": false
  },
  "kg_entities": ["故意杀人罪"],
  "generation_strategy": "chain_of_thought",
  "pipeline_config": { ... }
}
```

#### 知识库管理

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/knowledge/upload` | 上传文档（支持 .txt .md .json） |
| GET | `/api/knowledge/list` | 列出已上传文件 |
| DELETE | `/api/knowledge/{filename}` | 删除指定文件 |
| POST | `/api/knowledge/rebuild` | 重建全部索引 |
| GET | `/api/knowledge/stats` | 知识库统计（文件数、分块数） |

#### 性能监控

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/performance/system` | CPU/内存实时状态 |
| POST | `/api/performance/bench` | 运行基准测试 |

### 配置参数

通过环境变量或 `.env` 文件覆盖默认配置：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL` | `qwen3:8b` | 生成模型 |
| `EMBEDDING_MODEL` | `bge-m3` | Embedding 模型 |
| `RETRIEVAL_TOP_K` | `10` | 检索返回文档数 |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | `0.5` / `0.5` | BM25 / 向量检索权重 |
| `CONTEXTUAL_CHUNKING` | `True` | 是否启用上下文标头注入 |
| `HYDE_TEMPERATURE` | `0.7` | HyDE 假设文档生成温度 |
| `SELF_REFLECT_MAX_ITER` | `1` | 自我反思最大修正轮次 |
| `DEFAULT_QUERY_TRANSFORM` | `none` | 默认查询变换策略 |
| `DEFAULT_RERANK` | `simple` | 默认重排序策略 |
| `DEFAULT_GENERATION` | `standard` | 默认生成策略 |

---

## 前端详细说明

### 前端依赖

| 包名 | 用途 |
|------|------|
| react / react-dom | UI 框架 |
| axios | HTTP 请求（超时 300s） |
| lucide-react | 图标库 |
| react-markdown | Markdown 渲染 |
| recharts | 性能监控图表 |
| vite / @vitejs/plugin-react | 构建工具 |

### API 调用封装

所有 API 调用通过 `src/services/api.js` 统一封装：

| 函数 | 说明 |
|------|------|
| `sendChat()` | 问答（含策略参数 `queryTransform`, `rerankStrategy`, `generationStrategy`, `useKg`） |
| `uploadDocument()` | 上传文档 |
| `listDocuments()` | 列出文件 |
| `deleteDocument()` | 删除文件 |
| `rebuildIndex()` | 重建索引 |
| `getKBStats()` | 知识库统计 |
| `getSystemInfo()` | 系统资源状态 |
| `runBenchmark()` | 基准测试 |

### 智能问答页面

对话式法律问答界面，核心功能：

- **对话式问答**：输入法律问题，AI 返回基于法律条文的专业回答（Markdown 渲染）
- **参考来源**：每条回答下方展示检索到的法律条文/案例来源卡片
- **策略配置面板**：通过下拉菜单配置 RAG 管线各阶段策略

```
┌─────────────────────────────────────────────────────────────────────┐
│ [检索配置]                                                          │
│   查询变换: [无▾ | 多查询扩展 | HyDE | 子问题分解 | 多查询+HyDE]    │
│   重排策略: [简单重排▾ | 无 | LLM重排]                              │
│   知识库:   [全部▾ | 法律条文 | 指导案例]                            │
│   [✓] KG增强                                                       │
│                                                                     │
│ [生成配置]                                                          │
│   生成策略: [标准▾ | 链式推理(CoT) | 自我修正 | 结构化法律回答]       │
│                                                                     │
│ | [✓] 性能监控  [  ] 质量评估                                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│              ┌───────────────────────┐                              │
│              │ 用户: 故意杀人罪的量刑？│                              │
│              └───────────────────────┘                              │
│  ┌──────────────────────────────────────────┐                      │
│  │ [CoT推理]  [已修正]                       │    ← 策略与修正徽章   │
│  │                                           │                      │
│  │ AI: ## 1. 识别法律问题                     │                      │
│  │     本问题涉及刑法中的故意杀人罪...         │                      │
│  │                                           │                      │
│  │ KG匹配: [故意杀人罪]                       │    ← KG 实体标签     │
│  │                                           │                      │
│  │ 参考来源 (5)                               │                      │
│  │ ┌ 刑法 / 第232条 ─────────────────┐       │                      │
│  │ │ 故意杀人的，处死刑、无期徒刑...   │       │                      │
│  │ └─────────────────────────────────┘       │                      │
│  │                                           │                      │
│  │ 查询变换 1500ms  检索 320ms  KG 2100ms    │    ← 各阶段耗时      │
│  │ 重排序 15ms  生成 3500ms  反思 2800ms      │                      │
│  │ 总计 10235ms                               │                      │
│  └──────────────────────────────────────────┘                      │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│ [请输入法律问题...                                    ] [发送]      │
└─────────────────────────────────────────────────────────────────────┘
```

#### 选项详解 — 检索配置

| 选项 | 可选值 | 说明 |
|------|--------|------|
| **查询变换** | 无变换 | 直接使用用户原始问题进行检索，速度最快 |
| | 多查询扩展 | LLM 将用户问题从 3 个角度改写（法律条文 / 司法解释 / 不同术语），扩大召回覆盖面。适合问题描述模糊时使用 |
| | HyDE | LLM 先生成一段假设性法律条文，用该条文做向量检索（解决"问题"与"条文"的语体不一致），原始问题仍用于 BM25 精确匹配。适合用户提问和法律条文表述差异大的场景 |
| | 子问题分解 | LLM 将复杂问题拆分为 2-4 个子问题（适用法律/构成要件/法律后果/术语规范化），分别检索后合并结果。适合一句话包含多个法律要点的复杂提问 |
| | 多查询+HyDE | 以上两种方式的组合，覆盖面最广但耗时最长（两个 LLM 调用并行执行） |
| **重排策略** | 无重排 | 直接使用 BM25+向量 RRF 融合的原始排序，零额外开销 |
| | 简单重排 | 基于 jieba 分词的 Jaccard 关键词重叠度 + 法律名称/条号/案号元数据加分，不消耗 LLM 资源，速度快且对法律条文检索效果好（**推荐默认**） |
| | LLM重排 | 先用简单重排预筛选候选文档，再由 LLM 对候选文档批量评分（单次调用），精度最高但增加约 8-15s 延迟 |
| **知识库** | 全部 | 同时检索法律条文和指导案例两个 Collection |
| | 法律条文 | 仅检索 laws Collection，适合问法条内容时使用 |
| | 指导案例 | 仅检索 cases Collection，适合找判例时使用 |
| **KG增强** | 开/关 | 开启后系统从问题中识别罪名，从犯罪知识图谱中查找该罪名的结构化知识（定义、构成要件、量刑标准、相关法条），合并到检索结果中。仅对刑法问题有效 |

#### 选项详解 — 生成配置

| 选项 | 说明 |
|------|------|
| **标准** | 直接根据检索到的参考资料生成法律回答，速度最快（推荐日常使用） |
| **链式推理(CoT)** | LLM 按"识别法律问题→查找适用法律→分析构成要件→得出结论→注意事项"五步推理后回答，回答更有条理但篇幅更长 |
| **自我修正** | 先用标准方式生成回答，然后 LLM 自我检查法条引用是否准确，有问题则修正重生成。回答上方显示"已修正"标签。增加约 3-8s 延迟 |
| **结构化法律回答** | 输出严格按"法律结论 / 适用法律 / 详细分析 / 注意事项"四段格式组织，适合需要正式法律意见格式的场景 |

#### 选项详解 — 评估选项

| 选项 | 说明 |
|------|------|
| **性能监控** | 开启后在问答前后分别采集系统 CPU/内存快照，实时显示资源状态。生成过程中在消息气泡内显示实时 CPU/内存数据 |
| **质量评估** | 开启后自动评估回答质量：ROUGE 指标（与参考答案的文本重叠度）、检索相关性（检索结果与问题的匹配度）、忠实度（回答是否忠实于检索到的参考资料） |

#### 回答区域显示元素

| 元素 | 含义 |
|------|------|
| **策略徽章**（紫色） | 标识使用了哪种非标准生成策略（CoT推理 / 自我修正 / 结构化） |
| **已修正徽章**（黄色） | 自我修正模式下，LLM 检查发现问题并重新生成后显示 |
| **KG匹配标签**（绿色） | 知识图谱匹配到的罪名实体，如 `[故意杀人罪]` |
| **参考来源卡片** | 检索到的法律条文/案例原文摘要，显示法律名称、条号等元数据 |
| **耗时指标栏** | 各阶段耗时：查询变换 / 检索 / KG / 重排序 / 生成 / 反思 / 总计（ms） |
| **性能&质量详情**（可展开） | 系统资源前后对比 + 质量评估得分（ROUGE-L / 检索相关性 / 忠实度） |

### 知识库管理页面

管理法律文档的上传、查看和索引重建：

- **统计概览**：4 张卡片展示文件总数、总分块数、法律分块数、案例分块数
- **上传文档**：选择类型（法律条文/指导案例）→ 选择文件 → 自动分块并写入向量库。支持 `.txt` `.md` `.json` 格式
- **文件列表**：显示已上传的文件名、类型标签、文件大小，支持删除
- **重建索引**：清除所有向量数据并从 `data/` 目录重新导入（带确认弹窗）

### 性能监控页面

系统资源监控和 RAG 基准测试：

- **实时资源监控**：CPU 使用率、内存使用率、已用/总内存，每 5 秒自动刷新
- **基准测试**：运行多条法律领域测试查询，展示汇总指标（平均延迟/QPS）、柱状图（检索/生成耗时分解）、详细结果列表
- **测试报告**：支持生成和下载完整性能报告（含质量评估）

---

## 数据源

### 法律条文 (`backend/data/laws/`)

| 数据集 | 来源 | 规模 | 说明 |
|--------|------|------|------|
| **Chinese-Laws** | [ModelScope dengcao/Chinese-Laws](https://www.modelscope.cn/datasets/dengcao/Chinese-Laws) | 177 部法律 (5.9MB) | 中国现行主要法律条文全文，TXT 格式 |
| **HF 法律法规库** | [HuggingFace twang2218/chinese-law-and-regulations](https://huggingface.co/datasets/twang2218/chinese-law-and-regulations) | 2,719 部法规 (44MB) | 法律、行政法规、地方性法规全文 |
| **犯罪知识图谱** | [GitHub liuhuanyong/CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant) | 455+ 罪名 (7.5MB) | 罪名定义、构成要件、量刑标准、相关法条 |

### 刑事案例 (`backend/data/cases/`)

| 数据集 | 来源 | 规模 | 说明 |
|--------|------|------|------|
| **CAIL2019-SCM** | [GitHub thunlp/CAIL](https://github.com/thunlp/CAIL) | 3 个文件 | 民事案例相似性匹配数据集 |
| **CAIL2018** | [CAIL2018 官方](https://cail.oss-cn-qingdao.aliyuncs.com/CAIL2018_ALL_DATA.zip) | 10,000 条 (已导入) / 154K+ 条 (可用) | 刑事法律文书 |

### 法律问答与评估 (`backend/data/qa/`, `backend/data/reference/`)

| 数据集 | 来源 | 规模 | 说明 |
|--------|------|------|------|
| **CrimeKgAssitant QA** | [GitHub liuhuanyong/CrimeKgAssitant](https://github.com/liuhuanyong/CrimeKgAssitant) | 203,459 条 | 真实法律咨询问答对 |
| **评估参考答案** | 从上述 QA 精选 | 195 条 | 按类别均匀采样，用于 ROUGE/相关性/忠实度评估 |

### 数据构成总览

系统经数据预处理与导入后，共生成约 **59,000** 个文本分块（chunk），存储于 ChromaDB 的两个 Collection 中：

| Collection | 数据来源 | 分块策略 | 分块参数 | 用途 |
|-----------|---------|---------|---------|------|
| `laws` | Chinese-Laws (177部) + HF法律法规库 (2,719部) + CrimeKG (455+罪名) | `LegalArticleSplitter`：按编/章/节/条层级切分 | chunk_size=512, overlap=64 | 法律条文检索 |
| `cases` | CAIL2019-SCM + CAIL2018 (10,000条) | `LegalCaseSplitter`：按裁判要旨/案情/理由/结果切分 | chunk_size=1024, overlap=128 | 案例检索 |

### 数据来源说明

所有数据集均来自公开的学术数据集或开源项目，通过 `scripts/prepare_datasets.py` 统一转换格式，再由 `scripts/import_data.py` 分块导入向量数据库：

| 数据集 | 原始来源 | 获取方式 | 原始格式 | 预处理 |
|--------|---------|---------|---------|--------|
| Chinese-Laws | ModelScope `dengcao/Chinese-Laws` | ModelScope 下载 | TXT（每部法律一个文件） | 直接导入，自动检测编码 (UTF-8/GBK) |
| HF 法律法规库 | HuggingFace `twang2218/chinese-law-and-regulations` | hf-mirror.com 镜像下载 | Markdown（每部法规一个文件） | 直接导入 |
| CrimeKG | GitHub `liuhuanyong/CrimeKgAssitant` | `prepare_datasets.py` 解析 | 按罪名分目录的结构化文本 | 解析罪名定义/构成要件/量刑/法条 → 合并为 TXT 导入 laws Collection + 内存 KG 字典 |
| CAIL2019-SCM | GitHub `thunlp/CAIL` | 仓库直接下载 | JSON（含 A/B/C 三案例 + 相似度标签） | 仅提取字段 A 作为案例文本 |
| CAIL2018 | CAIL 官方 Aliyun OSS | `prepare_datasets.py` 下载解压 | ZIP → 多层嵌套 JSON | 提取 `fact` + `meta`（accusation 等）→ JSONL |
| CrimeKgAssitant QA | GitHub `liuhuanyong/CrimeKgAssitant` | 同 CrimeKG | JSON 问答对 (203,459条) | 按类别均匀采样 195 条 → `reference_answers.json`，用于 ROUGE/忠实度评估 |

---

## 测试

系统包含 24 个单元测试用例，分布在 4 个测试文件中：

| 测试文件 | 覆盖范围 |
|---------|---------|
| `test_basic.py` | 配置加载、法律条文分块、案例分块、元数据格式化 |
| `test_pipeline.py` | PipelineConfig 默认值、策略枚举、向后兼容映射 |
| `test_advanced_retrieval.py` | 法律术语规范化、上下文标头注入 |
| `test_kg.py` | 知识图谱加载、罪名查找 |

```bash
cd backend && pytest tests/ -v
```

---

## 性能测试

系统内置性能测试模块，覆盖以下维度：

- **资源占用**：CPU/RAM 实时监控
- **响应速度**：端到端延迟、各阶段耗时分解（查询变换/检索/KG/重排序/生成/反思）
- **生成质量**：ROUGE、检索相关性、忠实度自动评估
- **吞吐量**：并发请求处理能力

### 各策略组合预期延迟

| 策略组合 | 查询变换 | 重排序 | 生成 | KG | 预期延迟 |
|---------|---------|--------|------|-----|---------|
| 快速 | 无 | 简单 | 标准 | 关 | 10-20s |
| 精确检索 | HyDE | 简单 | 标准 | 关 | 15-25s |
| 深度分析 | HyDE | 简单 | CoT | 开 | 25-40s |
| 最高质量 | 多查询+HyDE | LLM | 自我修正 | 开 | 40-80s |

---

## 技术选型与决策

| 组件 | 选择 | 理由 |
|------|------|------|
| 主模型 | Qwen3:8B (Ollama) | 中文能力强，8B 参数量适合 16GB 显存本地运行 |
| Embedding | BGE-M3 (Ollama) | 多语言多粒度，中文检索效果优秀 |
| 向量数据库 | ChromaDB | 轻量级、嵌入式、零配置部署、自带持久化和元数据过滤 |
| 后端框架 | FastAPI | 异步高性能、自动生成 OpenAPI 文档 |
| 前端框架 | React + Vite | 开发体验好、生态丰富、HMR 快速 |
| RAG 框架 | LangChain | 组件化设计、丰富的集成、生态成熟 |
| 稀疏检索 | BM25 (rank_bm25) | 精确匹配法律条文编号和关键词 |
| 中文分词 | jieba | 法律文本中文分词效果好 |

### 为什么不选其他方案

- **为什么不用 LlamaIndex**：LangChain 生态更成熟，自定义灵活度更高
- **为什么不用 Milvus/Weaviate**：ChromaDB 零配置部署，适合轻量化需求
- **为什么不用 FAISS**：ChromaDB 自带持久化和元数据过滤，更适合生产场景
- **为什么不用 API 模型**：本地部署保障数据隐私，法律数据敏感
- **为什么不用 Neo4j 图数据库**：犯罪知识图谱的查找需求是简单键值查找，内存字典 O(1) 即可满足

---

## 许可证

本项目仅供学习与研究使用，法律数据版权归原始数据提供方所有。
