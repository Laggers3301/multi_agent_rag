"""
全局配置：模型、嵌入、向量库、检索参数
"""

import os

# ==========================================
# LLM 配置 (Ollama 本地模型)
# ==========================================
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3:1.7b"  # 本地 Ollama 模型名称，可换成 qwen2.5:7b 等

# ==========================================
# 嵌入模型配置 (Sentence-Transformers 本地)
# ==========================================
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 轻量嵌入模型，本地自动下载

# ==========================================
# ChromaDB 向量库配置
# ==========================================
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CHROMA_COLLECTION_NAME = "research_docs"

# ==========================================
# RAG 参数
# ==========================================
CHUNK_SIZE = 500  # 文本分块大小
CHUNK_OVERLAP = 50  # 分块重叠大小
RETRIEVAL_TOP_K = 3  # 检索 Top-K 文档数（降低以提高精准度）
RELEVANCE_THRESHOLD = 0.3  # CRAG 相关性阈值（适配小知识库场景）

# ==========================================
# Agent 参数
# ==========================================
MAX_RESEARCH_ITERATIONS = 2  # 最大研究迭代轮次
MAX_REVISION_ROUNDS = 1  # 最大修订轮次（减少以降低延迟）
