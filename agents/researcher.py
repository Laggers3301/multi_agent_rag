"""
Researcher Agent：检索与摘要
职责：对每个子查询执行 CRAG 检索，产出研究笔记。
面试话术：Researcher 是 RAG 的核心执行者，配合 CRAG 闭环保障检索质量。
"""

from state import ResearchState


SUMMARIZER_PROMPT = """你是一个专业的研究助手。请根据以下检索到的参考文档，针对研究问题生成一段简洁且信息密集的研究笔记。

研究问题：{query}

参考文档：
{context}

要求：
1. 只提取与【研究问题】直接相关的核心信息，忽略无关内容
2. 用清晰的结构化格式呈现
3. 标注关键技术术语和公式
4. 不要讨论与研究问题无关的技术

研究笔记："""


def researcher_node(state: ResearchState) -> dict:
    """
    LangGraph Node：
    遍历 Planner 产出的 sub_queries，逐个执行 CRAG 检索 + LLM 摘要。
    """
    from langchain_community.llms import Ollama
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL
    from rag.retriever import crag_retrieve

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.2)

    all_docs = []
    all_notes = []
    total_quality = 0.0

    sub_queries = state.get("sub_queries", [state["query"]])

    for i, sq in enumerate(sub_queries, 1):
        print(f"\n[Researcher] 正在研究子问题 {i}/{len(sub_queries)}: {sq}")

        # CRAG 检索（含质量评估与 Query Rewrite）
        docs, quality_score = crag_retrieve(sq)
        total_quality += quality_score

        # 拼接检索文档上下文（仅使用 Top-K 中最相关的文档）
        context = "\n\n---\n\n".join(docs) if docs else "未检索到相关文档。"
        all_docs.extend(docs)

        # LLM 生成摘要笔记
        prompt = SUMMARIZER_PROMPT.format(query=sq, context=context)
        note = llm.invoke(prompt)
        all_notes.append(f"### 子问题 {i}: {sq}\n{note}")

        print(f"   已完成检索质量: {quality_score:.2f}, 摘要长度: {len(note)} 字符")

    avg_quality = total_quality / len(sub_queries) if sub_queries else 0.0
    print(f"\n[Researcher] 平均检索质量: {avg_quality:.2f}")

    return {
        "retrieved_docs": all_docs,
        "research_notes": all_notes,
        "retrieval_quality": avg_quality,
        "iteration": state.get("iteration", 0) + 1,
    }
