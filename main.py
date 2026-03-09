"""
Multi-Agent RAG Research System - 主入口
用法：
    python main.py --query "你的研究问题"
    python main.py --query "介绍 DeepSeek-V3 的 MLA 机制" --ingest ./docs/
"""

import argparse
import os
import time
import json
from graph import build_research_graph
from rag.loader import load_text_file, load_texts
from rag.vectorstore import add_documents, get_collection


# ==========================================
# 预置种子知识库（首次运行时自动灌入）
# 每个主题 2-3 条不同角度的描述，大幅提升检索召回率
# ==========================================
SEED_KNOWLEDGE = [
    # === Transformer 基础 ===
    "Transformer 是一种基于自注意力机制的深度学习架构，由 Vaswani 等人在 2017 年提出。核心组件包括 Multi-Head Attention、Feed-Forward Network、Layer Normalization 和 Residual Connection。",
    "Transformer 的 Encoder-Decoder 结构中，Encoder 使用双向自注意力，Decoder 使用因果掩码自注意力（Causal Attention）确保自回归生成。GPT 系列只使用 Decoder，BERT 只使用 Encoder。",
    # === MLA (DeepSeek) ===
    "MLA (Multi-head Latent Attention) 是 DeepSeek-V2/V3 提出的注意力变体。它通过低秩压缩将 KV Cache 压缩为潜变量 (Latent Vector)，在推理时大幅降低显存带宽压力。具体来说，MLA 先将输入投影到一个低维度的 c_KV 向量，然后通过上投影矩阵恢复出 K 和 V。",
    "MLA 相比传统 MHA 的核心优势是：MHA 需要为每个 Head 独立存储 Key 和 Value，KV Cache 随序列长度线性增长。而 MLA 将所有 Head 的 KV 共享压缩到一个低维潜变量中，显存占用降低为原来的 1/10。同时 MLA 保留了 RoPE 部分维度用于位置编码，其余维度做低秩压缩。",
    "MLA 与 MQA/GQA 的区别：MQA 让所有 Head 共享同一组 KV，信息损失大；GQA 将 Head 分组共享 KV；MLA 则先压缩到潜空间再恢复，表示能力更强。",
    # === LoRA / DoRA ===
    "LoRA (Low-Rank Adaptation) 通过在冻结的预训练权重旁添加低秩分解矩阵 A 和 B 来实现参数高效微调。微调时只更新 A 和 B，总参数量远小于全参微调。",
    "DoRA (Weight-Decomposed Low-Rank Adaptation) 进一步将权重分解为方向 (Direction) 矩阵和幅度 (Magnitude) 向量。相比 LoRA，DoRA 的梯度更新路径更接近全参微调，收敛更快。",
    "QLoRA 是 LoRA 的量化版本，将基座模型用 NF4 4-bit 量化存储以节约显存，仅在 LoRA 适配器部分使用全精度计算。",
    # === DPO / RLHF ===
    "DPO (Direct Preference Optimization) 通过数学推导将 RLHF 的奖励建模过程简化为直接在语言模型上优化偏好对。核心 Loss 为 -log(sigmoid(beta * (log(pi/ref)_chosen - log(pi/ref)_rejected)))。DPO 不需要单独训练 Reward Model。",
    "PPO (Proximal Policy Optimization) 是传统 RLHF 的主流算法，需要同时维护 Actor、Critic、Reward Model 和 Reference Model 四个模型，资源开销大。DPO 将其简化为只需 Policy 和 Reference 两个模型。",
    # === GRPO ===
    "GRPO (Group Relative Policy Optimization) 是 DeepSeek-R1 使用的强化学习方法。它摒弃了 Critic 模型，通过组内多个生成答案之间的相互比较来计算基线 (Baseline)，大幅降低了训练资源消耗。",
    "GRPO 与 PPO 的关键区别：PPO 依赖 Critic 模型估算 Value Function，而 GRPO 对同一个问题采样 G 个回答，以组内平均奖励作为 Baseline，省去了 Critic。GRPO 与 DPO 的区别：DPO 是离线学习偏好对，GRPO 是在线采样并即时优化。",
    # === Flash Attention ===
    "Flash Attention 通过 Tiling 分块策略，将 Attention 计算拆分为小块，在 GPU SRAM 上完成 softmax 计算后直接写回 HBM，避免了 O(N^2) 中间矩阵的显存占用。",
    "Flash Attention 的关键技术是 Online Softmax：不需要先计算完整 Attention Score 矩阵再做 softmax，而是在逐块读入的过程中用 Running Max 和 Running Sum 增量更新 softmax，实现了 O(N) 显存的注意力计算。",
    "Flash Attention 为什么快：传统 Attention 产生 N×N 的 Score 和 Probability 矩阵需要大量 HBM 读写（IO-bound），Flash Attention 通过 Tiling 让计算在 SRAM 内完成，将 HBM 读写从 O(N^2) 降到 O(N)。",
    # === vLLM / PagedAttention ===
    "vLLM 的核心技术 PagedAttention 借鉴了操作系统的虚拟内存管理机制，将 KV Cache 分割成固定大小的物理 Block，通过 Block Table 进行虚拟-物理映射，解决了显存碎片化问题。",
    "PagedAttention 的工作原理：每个请求维护一个 Block Table（类似页表），逻辑 Block 号映射到物理 Block 号。当 KV Cache 增长时按需分配新 Block，释放时归还物理 Block Pool。多个请求可通过 Copy-on-Write 共享已计算好的 Prefix KV Cache。",
    # === DeepSpeed ZeRO ===
    "DeepSpeed ZeRO 优化器通过三个级别分别切分优化器状态(ZeRO-1)、梯度(ZeRO-2)和模型参数(ZeRO-3)，在多卡环境中消除冗余存储。ZeRO-3 在需要参数时通过 All-Gather 通信获取，用完即释放。",
    "ZeRO-3 的通信机制：前向传播时，每层计算前先 All-Gather 拉取完整参数，计算完毕后立即释放非本地分片(Free)。反向传播时同理，先 Gather 参数计算梯度，然后 Reduce-Scatter 将梯度分散回各卡。",
    # === MoE ===
    "MoE (Mixture of Experts) 通过路由门控网络将不同的 Token 分发给不同的专家网络处理，在不增加推理计算量的情况下扩大模型容量。DeepSeek-V3 使用了 256 个路由专家和 1 个共享专家。",
    "MoE 的负载均衡问题：如果路由不均匀会导致部分 Expert 过载、部分闲置。解决方案包括 Auxiliary Loss（辅助损失）惩罚路由不均匀、Top-K 路由策略、Token Dropping/Padding，以及 DeepSeek 提出的无辅助损失的动态 Bias 调整。",
    # === RoPE ===
    "RoPE (Rotary Position Embedding) 将位置信息编码为旋转矩阵，通过对 Query 和 Key 向量进行旋转操作，使得注意力内积自然包含相对位置信息。其核心公式为 f(x,m) = x * e^(im*theta)。",
    "RoPE 的数学原理详解：将 d 维向量的相邻偶数/奇数维度组成 d/2 个 2D 子空间，每个子空间对应一个独立频率 theta_i = 10000^(-2i/d)。位置 m 的编码相当于在每个子空间内旋转角度 m*theta_i。内积时两个位置的旋转角度相减，自然得到相对位置编码。",
    "RoPE 的长上下文扩展方法：YaRN 通过对不同频率的 theta 施加不同的缩放因子，在不重新训练的情况下将上下文长度从 4K 扩展到 64K 甚至更长。",
    # === RAG / CRAG ===
    "RAG (Retrieval-Augmented Generation) 通过在生成前检索相关文档来增强大模型的回答质量。典型流程为：用户 Query → 向量化 → 向量库检索 Top-K → 拼接到 Prompt → LLM 生成答案。",
    "CRAG (Corrective RAG) 在 Naive RAG 基础上增加了检索质量评估（Grader）环节。工作流程：检索 → Grader 评估相关性 → 若不相关则触发 Query Rewrite 重新检索或启用 Web Search 补充；若相关则保留文档进入生成阶段。CRAG 显著降低了幻觉率。",
    # === LangGraph / LangChain ===
    "LangGraph 是 LangChain 生态中用于构建有状态多智能体系统的框架。它基于图论，将 Agent 行为建模为状态图的节点(Node)和边(Edge)，支持循环调用和条件路由。",
    "LangGraph 与 LangChain 的核心区别：LangChain 的 Chain 是线性的，适合简单顺序任务。LangGraph 引入了 StateGraph（有向图），支持条件分支、循环调用和状态持久化，适合复杂多步推理和 Agent 间协作。LangGraph 的核心概念包括 Node（处理节点）、Edge（连接边）、Conditional Edge（条件路由边）。",
    # === 小米 MiMO ===
    "小米 MiMO 模型通过 MTP (Multi-Token Prediction) 和 Hybrid Attention (5层SWA+1层GA) 实现了 7B 参数对标 72B 效果。MTP 通过多个预测头并行生成候选 Token，配合投机解码实现 2.6 倍推理加速。",
]


def initialize_seed_knowledge():
    """首次运行时将种子知识灌入向量库。"""
    collection = get_collection()
    if collection.count() == 0:
        print("首次运行：正在灌入种子知识库...")
        add_documents(SEED_KNOWLEDGE, metadata={"source": "seed_knowledge"})
        print(f"   已灌入 {len(SEED_KNOWLEDGE)} 条种子知识\n")
    else:
        print(f"向量库已有 {collection.count()} 条文档\n")


def ingest_local_docs(docs_dir: str):
    """灌入本地文档目录。"""
    if not os.path.isdir(docs_dir):
        print(f"目录不存在: {docs_dir}")
        return

    for filename in os.listdir(docs_dir):
        filepath = os.path.join(docs_dir, filename)
        if filename.endswith((".txt", ".md")):
            print(f"正在处理: {filename}")
            chunks = load_text_file(filepath)
            add_documents(chunks, metadata={"source": filename})


def run_research(query: str):
    """运行完整的 Multi-Agent 研究流程。"""
    print("=" * 60)
    print(f"启动 Multi-Agent Research System")
    print(f"研究问题: {query}")
    print("=" * 60)

    # 构建 LangGraph 状态图
    app = build_research_graph()

    # 初始状态
    initial_state = {
        "query": query,
        "sub_queries": [],
        "task_dag": [],
        "retrieved_docs": [],
        "research_notes": [],
        "draft_report": "",
        "review_feedback": "",
        "review_decision": "",
        "retrieval_quality": 0.0,
        "iteration": 0,
        "revision_count": 0,
        "workflow_action": "",
        "workflow_history": [],
        "completed_tasks": [],
    }

    # 执行并计时
    start_time = time.time()
    final_state = app.invoke(initial_state)
    elapsed = time.time() - start_time

    # 输出结果
    print("\n" + "=" * 60)
    print("最终研究报告 (节选)")
    print("=" * 60)
    report_text = final_state.get("draft_report", "(空)")
    try:
        print(report_text[:300] + "\n... [已自动截断，完整报告请查看输出目录文件] ...")
    except UnicodeEncodeError:
        safe_text = report_text[:300].encode("gbk", errors="replace").decode("gbk")
        print(
            safe_text + "\n... [已自动截断并替换特殊字符，完整报告请查看输出文件] ..."
        )

    # 输出指标摘要
    print("\n" + "=" * 60)
    print("运行指标摘要")
    print("=" * 60)
    metrics = {
        "总耗时(秒)": round(elapsed, 2),
        "研究迭代轮次": final_state.get("iteration", 0),
        "优化器动作轨迹": final_state.get("workflow_history", []),
        "DAG 任务完成率": f"{len(final_state.get('completed_tasks', []))} / {len(final_state.get('task_dag', [1]))}",
        "修订单次": final_state.get("revision_count", 0),
        "检索文档库命中": len(final_state.get("retrieved_docs", [])),
        "平均检索质量": round(final_state.get("retrieval_quality", 0), 3),
        "最终汇报体积": len(final_state.get("draft_report", "")),
    }
    for k, v in metrics.items():
        if isinstance(v, list) and k == "优化器动作轨迹":
            print(f"  {k}:")
            for action in v:
                print(f"      - {action}")
        else:
            print(f"  {k}: {v}")

    # 保存报告
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, "research_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_state.get("draft_report", ""))
    print(f"\n报告已保存至: {report_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"指标已保存至: {metrics_path}")

    return final_state, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent RAG Research System")
    parser.add_argument(
        "--query",
        type=str,
        default="介绍 DeepSeek-V3 的 MLA 注意力机制及其相比传统 MHA 的优势",
        help="研究问题",
    )
    parser.add_argument(
        "--ingest", type=str, default=None, help="要灌入的本地文档目录路径"
    )
    args = parser.parse_args()

    # 初始化种子知识库
    initialize_seed_knowledge()

    # 灌入本地文档（可选）
    if args.ingest:
        ingest_local_docs(args.ingest)

    # 运行研究
    run_research(args.query)
