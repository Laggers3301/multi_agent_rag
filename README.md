# Agentic-RAG-Workflow: 基于 DAG 与自适应路由的多智能体调研引擎

本项目实现了一个高度灵活、支持复杂逻辑链任务的自动化 Multi-Agent 系统。借鉴 SOAN (自组织拓扑) 与 AFlow (MCTS 搜索优化) 等前沿论文思想，将传统 RAG 升级为闭环自进化工作流。

## 🌟 核心特性

- **DAG 任务编排 (Directed Acyclic Graph)**: Planner 节点产出带依赖关系的任务图，支持子任务的并行检索与串行逻辑推理，极大提升解决复杂问题的能力。
- **自适应工作流路由 (Workflow Optimizer)**: 模拟 MCTS 价值评分，根据检索质量、覆盖率及研究深度动态决策 [深挖/扩展/收敛] 路径。
- **CRAG (Corrective RAG) 闭环**: 内置 Grader 机制对检索相关度进行阈值判定。
- **结构化知识沉淀**: 自动生成 2.5K+ 字符的高质量 Markdown 技术调研报告。

## 🏗️ 项目结构

```text
multi_agent_rag/
├── README.md                # 本文件
├── requirements.txt         # 依赖配置 (langgraph, langchain, chromadb等)
├── config.py                # 全局配置 (模型选择、向量库路径)
├── state.py                 # LangGraph 状态定义 (TypedDict)
├── agents/                  # 智能体核心逻辑
│   ├── __init__.py
│   ├── planner.py           # 规划 Agent: DAG 任务拆解与依赖标记
│   ├── researcher.py        # 研究 Agent: 向量库检索 + 摘要提取
│   ├── writer.py            # 写作 Agent: 结构化报告生成
│   ├── reviewer.py          # 审核 Agent: 质量评估 + 逻辑闭环
│   └── workflow_optimizer.py # 优化 Agent: MCTS 启发式工作流决策 [NEW]
├── rag/                     # RAG 底层组件
│   ├── __init__.py
│   ├── loader.py            # 文档加载与分块 (Markdown/Text)
│   ├── vectorstore.py       # ChromaDB 向量库管理
│   └── retriever.py         # CRAG 检索器 (含 Grader 质量评估)
├── graph.py                 # LangGraph 状态图编排逻辑
├── main.py                  # 系统主入口
└── evaluate.py              # 性能评估脚本 (生成简历核心指标)
```

## 📈 实验表现 (MLA/MTP 主题测试)

- **检索增益**: 平均检索质量提升至 **0.468**，有效降低长尾知识幻觉率。
- **效率优势**: 逻辑链自动化规划将手动调研耗时由 172s 压缩至 **51s (↑3.3x)**。
- **决策鲁棒性**: 4 轮迭代内自动收敛，成功处理复杂循环与分支逻辑。

## 🛠️ 技术栈

- **LangGraph**: 核心状态机编排，支持循环与条件路由。
- **ChromaDB**: 本地轻量化向量数据库。
- **Ollama**: 本地大模型推理 (Qwen-2.5-7B/DeepSeek-R1-Distill)。
- **Corrective RAG**: 闭环纠偏检索机制。

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动研究任务
python main.py --query "介绍 DeepSeek-V3 的 MLA 注意力机制及推理优化"
```

## 🔧 未来规划
- [ ] 引入完整的 MCTS 搜索树扩展。
- [ ] 适配异步并行多 Researcher 调度。
