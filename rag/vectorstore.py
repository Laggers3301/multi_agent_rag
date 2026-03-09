"""
ChromaDB 向量库管理
职责：管理文档的向量化存储与检索。
"""

import os

# 强制离线模式：模型权重已本地缓存，无需连接 HuggingFace
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from typing import List, Tuple
import chromadb
from chromadb.utils import embedding_functions
from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)


def get_collection():
    """获取或创建 ChromaDB 集合。"""
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 使用 SentenceTransformer 作为嵌入函数
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},  # 使用余弦相似度
    )
    return collection


def add_documents(chunks: List[str], metadata: dict = None):
    """将文本块存入向量库。"""
    collection = get_collection()
    ids = [f"doc_{i}_{hash(c) % 10000}" for i, c in enumerate(chunks)]
    metadatas = [metadata or {"source": "local"} for _ in chunks]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    print(f"   💾 已存入 {len(chunks)} 个文档块到 ChromaDB")


def query_documents(
    query: str, top_k: int = RETRIEVAL_TOP_K
) -> Tuple[List[str], List[float]]:
    """
    检索最相关的文档块。
    返回：(文档列表, 距离分数列表)
    距离越小 = 越相似（cosine distance）
    """
    collection = get_collection()

    if collection.count() == 0:
        return [], []

    results = collection.query(
        query_texts=[query], n_results=min(top_k, collection.count())
    )

    docs = results["documents"][0] if results["documents"] else []
    distances = results["distances"][0] if results["distances"] else []

    return docs, distances
