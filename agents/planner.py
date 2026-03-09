"""
Planner Agent v2：DAG 任务规划与依赖图生成
升级点（借鉴 SOAN 自组织拓扑）：
- 不再只输出扁平的子问题列表
- 输出带依赖关系的 DAG（有向无环图），支持并行/串行子任务调度
"""

import json
from state import ResearchState

PLANNER_PROMPT = """你是一个专业的研究规划师。你的任务是将用户的研究问题拆解为 2-5 个具体的、可独立检索的子任务，并标注它们之间的依赖关系。

规则：
1. 每个子任务应该足够具体，可以通过文档检索获得答案
2. 如果子任务 B 需要子任务 A 的结论才能开展，则 B 依赖 A
3. 没有依赖关系的子任务可以并行执行
4. 返回一个 JSON 列表，每项包含 id、query、depends_on 字段

用户的研究问题是：{query}

请直接返回 JSON 列表，格式如下（注意 depends_on 是 id 列表）：
[
  {{"id": "t1", "query": "子问题1（基础概念）", "depends_on": []}},
  {{"id": "t2", "query": "子问题2（依赖t1的深入分析）", "depends_on": ["t1"]}},
  {{"id": "t3", "query": "子问题3（独立方向）", "depends_on": []}}
]
"""


def planner_node(state: ResearchState) -> dict:
    """
    LangGraph Node：接收全局状态，产出 DAG 任务依赖图。
    同时向后兼容 sub_queries 字段。
    """
    from langchain_community.llms import Ollama
    from config import OLLAMA_MODEL, OLLAMA_BASE_URL

    llm = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.3)

    prompt = PLANNER_PROMPT.format(query=state["query"])
    response = llm.invoke(prompt)

    # 解析 LLM 返回的 DAG JSON
    task_dag = []
    sub_queries = []

    try:
        start = response.find("[")
        end = response.rfind("]") + 1
        if start != -1 and end > start:
            raw_dag = json.loads(response[start:end])
            # 验证格式
            for item in raw_dag:
                if isinstance(item, dict) and "id" in item and "query" in item:
                    task_dag.append(
                        {
                            "id": item["id"],
                            "query": item["query"],
                            "depends_on": item.get("depends_on", []),
                        }
                    )
                    sub_queries.append(item["query"])
    except (json.JSONDecodeError, ValueError):
        pass

    # 降级：如果解析失败，构造单节点 DAG
    if not task_dag:
        task_dag = [{"id": "t1", "query": state["query"], "depends_on": []}]
        sub_queries = [state["query"]]

    # 分析并行度
    independent = [t for t in task_dag if not t["depends_on"]]
    dependent = [t for t in task_dag if t["depends_on"]]

    print(f"\n📋 [Planner] 已生成 DAG 任务图（{len(task_dag)} 个节点）：")
    print(f"   ├─ 可并行任务: {len(independent)} 个")
    print(f"   └─ 串行依赖链: {len(dependent)} 个")
    for t in task_dag:
        deps = f" ← 依赖 {t['depends_on']}" if t["depends_on"] else " (独立)"
        print(f"   [{t['id']}] {t['query']}{deps}")

    current_iter = state.get("iteration", 0)
    current_rev = state.get("revision_count", 0)

    return {
        "sub_queries": sub_queries,
        "task_dag": task_dag,
        "iteration": current_iter,
        "revision_count": current_rev,
        "completed_tasks": [],
        "workflow_action": "",
        "workflow_history": [
            f"[Planner] 生成 DAG: {len(independent)} 并行 + {len(dependent)} 串行"
        ],
    }
