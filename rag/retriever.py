"""
CRAG (Corrective Retrieval-Augmented Generation) 检索器
职责：检索 → 质量评估 → 条件路由（通过 / Query Rewrite / Web 补充）
面试话术：CRAG 相比 Naive RAG 增加了检索质量评判环节，低质量时自动触发 Query Rewrite 或 Web Search 补充。
"""

from typing import List, Tuple
from config import RELEVANCE_THRESHOLD
from rag.vectorstore import query_documents


def evaluate_relevance(docs: List[str], distances: List[float]) -> float:
    """
    评估检索结果的相关性。
    将 ChromaDB 的 cosine distance 转换为相似度 (1 - distance)。
    返回平均相似度分数 [0, 1]。
    """
    if not distances:
        return 0.0
    similarities = [max(0, 1 - d) for d in distances]
    return sum(similarities) / len(similarities)


def rewrite_query(original_query: str) -> str:
    """
    Query Rewrite：当检索质量不达标时，对查询进行改写以提升召回。
    面试话术：这是 CRAG 闭环的关键一步。当 Grader 判定检索结果不相关时，
    我们不是直接返回低质量答案，而是改写 Query 重新检索。
    """
    from langchain_community.llms import Ollama
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

    prompt = f"""请将以下搜索查询改写为更精确、更具体的表述，以提高检索质量。
原始查询：{original_query}
改写后的查询（只返回改写结果，不要解释）："""

    rewritten = llm.invoke(prompt).strip()
    print(f"   🔄 [CRAG] Query Rewrite: '{original_query}' → '{rewritten[:80]}...'")
    return rewritten


def crag_retrieve(query: str, max_retries: int = 2) -> Tuple[List[str], float]:
    """
    CRAG 完整流程：
    1. 检索 → 2. 评估质量 → 3. 质量达标则返回，否则 Query Rewrite 后重试

    返回：(检索到的文档列表, 最终质量分数)
    """
    current_query = query

    for attempt in range(max_retries + 1):
        docs, distances = query_documents(current_query)
        quality = evaluate_relevance(docs, distances)

        if quality >= RELEVANCE_THRESHOLD or attempt >= max_retries:
            status = (
                "✅ 达标"
                if quality >= RELEVANCE_THRESHOLD
                else "⚠️ 未达标但已达最大重试"
            )
            print(f"   📎 [CRAG] 第 {attempt + 1} 轮检索，质量={quality:.2f} {status}")
            return docs, quality

        # 质量不达标，触发 Query Rewrite
        print(
            f"   📎 [CRAG] 第 {attempt + 1} 轮检索，质量={quality:.2f} < {RELEVANCE_THRESHOLD}，触发 Rewrite"
        )
        current_query = rewrite_query(current_query)

    return docs, quality
