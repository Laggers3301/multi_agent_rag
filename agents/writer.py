"""
Writer Agent：报告生成
职责：将 Researcher 的研究笔记整合为一份结构化的研究报告。
"""

from state import ResearchState

WRITER_PROMPT = """你是一个专业的技术报告撰写者。请根据以下研究笔记，撰写一份完整、结构化的研究报告。

原始研究问题：{query}

研究笔记：
{notes}

{revision_instruction}

要求：
1. 报告应包含：概述、核心发现、技术细节、总结
2. 使用 Markdown 格式，层次清晰
3. 引用研究笔记中的关键数据与结论
4. 保持客观、专业的学术风格

研究报告："""


def writer_node(state: ResearchState) -> dict:
    """
    LangGraph Node：整合所有研究笔记，生成完整报告。
    """
    from langchain_community.llms import Ollama
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.4)

    notes_text = "\n\n".join(state.get("research_notes", []))

    # 如果是修订轮，附加审稿意见
    revision_instruction = ""
    if state.get("review_feedback") and state.get("revision_count", 0) > 0:
        revision_instruction = (
            f"审稿人在前一轮提出了以下修改建议，请在本轮报告中进行改进：\n"
            f"{state['review_feedback']}"
        )

    prompt = WRITER_PROMPT.format(
        query=state["query"],
        notes=notes_text,
        revision_instruction=revision_instruction,
    )

    report = llm.invoke(prompt)

    print(f"\n[Writer] 报告生成完毕，共 {len(report)} 字符")
    return {"draft_report": report}
