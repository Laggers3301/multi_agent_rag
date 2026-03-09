"""
Reviewer Agent：质量评估与路由决策
职责：审核 Writer 产出的报告，决定 accept（通过）/ revise（修订）/ research_more（需要更多研究）。
面试话术：Reviewer 是 CRAG 闭环的关键出口，它让 Agent 系统具备了自我迭代能力。
"""

from state import ResearchState
from config import MAX_RESEARCH_ITERATIONS, MAX_REVISION_ROUNDS

REVIEWER_PROMPT = """你是一个研究报告审稿人。请评估以下报告是否充分回答了研究问题。

原始研究问题：{query}

报告内容：
{report}

评估标准：
1. 报告是否回答了研究问题的核心要点
2. 技术描述是否基本准确
3. 内容结构是否清晰

重要规则：
- 只评估报告是否回答了【原始研究问题】本身，不要求报告讨论与问题无关的技术
- 如果报告已经较好地回答了问题，请直接给 "accept"
- 只有在报告明显遗漏了问题的核心要点时，才给 "revise"
- 只有在报告内容几乎为空时，才给 "research_more"

请严格按照以下 JSON 格式回复（不要输出其他内容）：
{{"score": 总分(1-10), "decision": "accept", "feedback": "简短评语"}}
"""


def reviewer_node(state: ResearchState) -> dict:
    """
    LangGraph Node：审核报告，产出决策与反馈。
    """
    import json
    from langchain_community.llms import Ollama
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

    prompt = REVIEWER_PROMPT.format(
        query=state["query"],
        report=state.get("draft_report", "(空报告)"),
    )

    response = llm.invoke(prompt)

    # 解析 JSON 决策
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(response[start:end])
        else:
            result = {
                "score": 7,
                "decision": "accept",
                "feedback": "无法解析审核结果，默认通过。",
            }
    except (json.JSONDecodeError, ValueError):
        result = {
            "score": 7,
            "decision": "accept",
            "feedback": "无法解析审核结果，默认通过。",
        }

    decision = result.get("decision", "accept")
    feedback = result.get("feedback", "")
    score = result.get("score", 7)

    # 规范化决策值
    if decision not in ("accept", "revise", "research_more"):
        decision = "accept"

    # 评分 >= 7 直接接受（小模型评分偏严，7分已经很好）
    if score >= 7:
        decision = "accept"

    # 强制终止条件：防止死循环
    current_revision = state.get("revision_count", 0)
    current_iteration = state.get("iteration", 0)

    if decision == "revise" and current_revision >= MAX_REVISION_ROUNDS:
        print(f"   ⚠️ 已达到最大修订轮次 ({MAX_REVISION_ROUNDS})，强制接受")
        decision = "accept"
    if decision == "research_more" and current_iteration >= MAX_RESEARCH_ITERATIONS:
        print(f"   ⚠️ 已达到最大研究轮次 ({MAX_RESEARCH_ITERATIONS})，强制接受")
        decision = "accept"

    print(f"\n[Reviewer] 评分: {score}/10 | 决策: {decision}")
    if feedback:
        print(f"   反馈: {feedback[:200]}")

    revision_count = current_revision + (1 if decision == "revise" else 0)

    return {
        "review_feedback": feedback,
        "review_decision": decision,
        "revision_count": revision_count,
    }
