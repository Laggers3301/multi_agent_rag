"""
LangGraph 状态定义：所有 Agent 共享的 TypedDict 状态字典。
升级版：支持 DAG 任务依赖图 + 自适应工作流路由历史。
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
import operator


class ResearchState(TypedDict):
    """
    Multi-Agent 研究系统的全局状态。
    LangGraph 中每个 Node 接收并返回该状态的部分更新。

    v2 升级：
    - task_dag: 带依赖关系的 DAG 任务图（借鉴 SOAN 自组织拓扑）
    - workflow_history: 自适应路由决策轨迹（借鉴 AFlow/EvoFlow）
    - workflow_action: 当前工作流优化器的决策
    """

    # 用户原始查询
    query: str

    # Planner 产出的子任务列表（保留兼容）
    sub_queries: List[str]

    # === DAG 任务依赖图（v2 新增）===
    # 格式: [{"id": "t1", "query": "...", "depends_on": []}, {"id": "t2", "query": "...", "depends_on": ["t1"]}]
    task_dag: List[Dict[str, Any]]

    # Researcher 检索到的参考文档（累积追加）
    retrieved_docs: Annotated[List[str], operator.add]

    # Researcher 对每个子任务的研究摘要（累积追加）
    research_notes: Annotated[List[str], operator.add]

    # Writer 生成的报告草稿
    draft_report: str

    # Reviewer 的评审意见
    review_feedback: str

    # Reviewer 决策：accept / revise / research_more
    review_decision: str

    # CRAG 检索质量分数（用于条件路由）
    retrieval_quality: float

    # 当前迭代轮次
    iteration: int

    # 修订轮次
    revision_count: int

    # === 自适应工作流路由（v2 新增）===
    # WorkflowOptimizer 的当前决策: "deep_research" / "expand" / "converge"
    workflow_action: str

    # 工作流路由决策历史轨迹（累积追加）
    workflow_history: Annotated[List[str], operator.add]

    # 已完成的 DAG 任务 ID 列表
    completed_tasks: List[str]
