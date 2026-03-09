"""
WorkflowOptimizer Agent（v2 新增）
借鉴 AFlow (MCTS 搜索优化) 和 EvoFlow (自进化工作流) 的核心思想：
在每一轮迭代后，根据当前研究质量评分，动态选择下一步策略。

三种候选动作：
  1. "deep_research" — 质量不足，需要深挖当前子任务
  2. "expand"        — 发现新的未覆盖维度，动态扩展子任务
  3. "converge"      — 质量达标，收敛进入写作阶段
"""

from state import ResearchState


def workflow_optimizer_node(state: ResearchState) -> dict:
    """
    自适应工作流路由器：根据多维评分选择最优动作。
    简化版 MCTS：对每个候选动作计算"价值得分"，选择最高的。
    """
    retrieval_quality = state.get("retrieval_quality", 0.0)
    iteration = state.get("iteration", 0)
    research_notes = state.get("research_notes", [])
    task_dag = state.get("task_dag", [])
    completed_tasks = state.get("completed_tasks", [])

    # === 计算多维评分 ===

    # 1. 覆盖率：已完成任务占 DAG 总任务的比例
    total_tasks = max(len(task_dag), 1)
    coverage = len(completed_tasks) / total_tasks

    # 2. 研究深度：平均每个子任务的研究笔记长度
    avg_note_length = 0
    if research_notes:
        avg_note_length = sum(len(n) for n in research_notes) / len(research_notes)
    depth_score = min(avg_note_length / 500.0, 1.0)  # 归一化到 [0, 1]

    # 3. 检索质量（来自 CRAG）
    quality_score = retrieval_quality

    # === 动作价值评估（简化版 MCTS / UCB）===
    scores = {
        "deep_research": 0.0,
        "expand": 0.0,
        "converge": 0.0,
    }

    # 深挖价值：质量低 & 覆盖率高时（已经搜了但不够深）
    scores["deep_research"] = (
        (1 - quality_score) * 0.5 + (1 - depth_score) * 0.3 + coverage * 0.2
    )

    # 扩展价值：覆盖率低时（还有未探索的方向）
    scores["expand"] = (
        (1 - coverage) * 0.6 + (1 - quality_score) * 0.2 + 0.2 * (iteration < 2)
    )

    # 收敛价值：质量高 & 覆盖率高时
    scores["converge"] = quality_score * 0.4 + coverage * 0.3 + depth_score * 0.3

    # 迭代惩罚：超过 3 轮强制收敛
    if iteration >= 3:
        scores["converge"] += 1.0

    # 选择最优动作
    action = max(scores, key=scores.get)

    print(f"\n[WorkflowOptimizer] 自适应路由决策（第 {iteration + 1} 轮）：")
    print(
        f"   ├─ 覆盖率: {coverage:.1%} | 质量: {quality_score:.3f} | 深度: {depth_score:.2f}"
    )
    print(
        f"   ├─ 动作评分: deep={scores['deep_research']:.3f} | expand={scores['expand']:.3f} | converge={scores['converge']:.3f}"
    )
    print(f"   └─ 决策: → {action.upper()}")

    history_entry = (
        f"[Optimizer] 第{iteration + 1}轮: "
        f"覆盖={coverage:.1%} 质量={quality_score:.3f} 深度={depth_score:.2f} "
        f"→ {action}"
    )

    return {
        "workflow_action": action,
        "iteration": iteration + 1,
        "workflow_history": [history_entry],
    }
