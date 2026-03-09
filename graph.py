"""
LangGraph 状态图编排 v2 (核心文件)
升级点：
- 新增 WorkflowOptimizer 节点实现自适应动态路由
- 支持 DAG 任务依赖图驱动的研究流程
- 保留原有 Reviewer 条件路由（CRAG 检索质量评估）

拓扑结构 (v2)：

START → planner → researcher → optimizer → (条件路由)
                     ↑                         |
                     |          deep_research ──┘
                     |          expand ──→ planner (追加子任务)
                     |          converge ──→ writer → reviewer
                     |                                  |
                     +──── research_more ───────────────┘
                                                 revise → writer
                                                accept → END
"""

from langgraph.graph import StateGraph, END
from state import ResearchState
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from agents.reviewer import reviewer_node
from agents.workflow_optimizer import workflow_optimizer_node


def route_after_optimizer(state: ResearchState) -> str:
    """
    WorkflowOptimizer 之后的条件路由。
    根据 workflow_action 选择下一步：
    - "deep_research" → 回到 researcher 深挖
    - "expand" → 回到 planner 扩展子任务
    - "converge" → 进入 writer 写作收敛
    """
    action = state.get("workflow_action", "converge")

    if action == "deep_research":
        return "researcher"
    elif action == "expand":
        return "planner"
    else:
        return "writer"


def route_after_review(state: ResearchState) -> str:
    """
    Reviewer 之后的条件路由（保留原有 CRAG 逻辑）。
    """
    decision = state.get("review_decision", "accept")

    if decision == "accept":
        return "end"
    elif decision == "revise":
        return "writer"
    elif decision == "research_more":
        return "researcher"
    else:
        return "end"


def build_research_graph() -> StateGraph:
    """
    构建并编译 LangGraph v2 状态图。
    """
    workflow = StateGraph(ResearchState)

    # === 添加节点 ===
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("optimizer", workflow_optimizer_node)  # v2 新增
    workflow.add_node("writer", writer_node)
    workflow.add_node("reviewer", reviewer_node)

    # === 定义边 ===
    # 入口 → Planner → Researcher → Optimizer
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "optimizer")

    # Optimizer 条件路由: deep_research / expand / converge
    workflow.add_conditional_edges(
        "optimizer",
        route_after_optimizer,
        {
            "researcher": "researcher",  # 深挖
            "planner": "planner",  # 扩展
            "writer": "writer",  # 收敛
        },
    )

    # Writer → Reviewer
    workflow.add_edge("writer", "reviewer")

    # Reviewer 条件路由: accept / revise / research_more（保留原有）
    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "end": END,
            "writer": "writer",
            "researcher": "researcher",
        },
    )

    # === 编译 ===
    app = workflow.compile()
    return app
