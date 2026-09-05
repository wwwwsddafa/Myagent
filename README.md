# AI Agent 学习与实践仓库

> 从**手写 Agent 内核**到**框架化多智能体系统**的完整实践记录。
> 所有项目均为可运行代码，配套知识文档讲解实现原理、知识点与部署方式。

---

## 一、这个仓库在做什么

市面上讲 Agent 的资料大多停在"调 API 跑个 Demo"。这个仓库走的是另一条路：**先把 Agent 拆成零件，再一层层装回去**。

```
第 1 层  手搓零件    → 不用任何框架，自己写 LLM 客户端、工具执行器、三种推理范式
第 2 层  框架化      → 用 AutoGen / LangGraph 重写，理解框架替我们封装了什么
第 3 层  工程化      → RAG、结构化输出、可观测性、状态持久化、人在回路、质量审查闭环
```

顺着 `ch4 → notebooks → ch6` 的顺序读下来，能完整看到一个 Agent 系统是怎么从 200 行脚本长成可交付项目的。

---

## 二、学习路线

```
 ① 手写 Agent 框架 ............ src/ch4               先理解 Agent 的本质结构
            ↓
 ② 智能旅行助手 .............. notebooks/第一章       第一次把工具调用真正跑通
            ↓
 ③ 多智能体协作 .............. src/ch6/autoGen-*     从单 Agent 跨到团队分工
            ↓
 ④ 工作流与状态机 ............ src/ch6/langgraph/Pro1 建立 LangGraph 心智模型
            ↓
 ⑤ 综合项目 ................. src/ch6/langgraph/Pro2 前四步的能力全部落地
            ↓
 ⑥ 自研框架 ................. （规划中，暂不纳入本仓库）
```

---

## 三、项目速览

| # | 项目 | 目录 | 核心技术 | 状态 |
|---|---|---|---|---|
| ① | 手写 Agent 框架 | `src/ch4` | ReAct / Plan-and-Solve / Reflection、工具执行器、流式输出 | ✅ 可运行 |
| ② | 智能旅行助手 | `notebooks/第一章` | ReAct、wttr.in、Tavily | ✅ 可运行 |
| ③ | AutoGen 软件开发团队 | `src/ch6/autoGen-demo` | 4 角色团队、轮询调度、Streamlit | ✅ 可运行 |
| ④ | AutoGen 智能旅行助手 | `src/ch6/autoGen-test` | 5 Agent、自定义 Agent、持久化记忆、人在回路 | ✅ 可运行 |
| ⑤ | LangGraph 智能搜索助手 | `src/ch6/langgraph/Pro1` | StateGraph、Checkpoint、astream | ✅ 可运行 |
| ⑥ | LangGraph 智能研究助手 | `src/ch6/langgraph/Pro2` | RAG、条件路由、审查回环、Pydantic 结构化输出 | ✅ 可运行 |
| — | 自研 Agent 框架 | 规划中（未纳入本仓库） | 自研 core / tools 抽象 | 🚧 开发中 |

> ✅ 所有项目均已修复 `.env` 加载与路径问题，克隆后按 [00-环境准备](docs/00-环境准备与快速开始.md) 配置即可直接运行。

---

## 四、知识点索引

这是整个仓库的能力地图，面试复习可以直接按这张表过。

### Agent 核心范式

| 范式 | 关键词 | 实现位置 | 核心思路 |
|---|---|---|---|
| **ReAct** | Thought → Action → Observation | `src/ch4/react_agent.py` | 每步先推理再行动，用观察结果修正下一步 |
| **Plan-and-Solve** | 先规划后执行 | `src/ch4/plan_and_solve_*.py` | Planner 拆解步骤，Executor 逐步执行并回灌上下文 |
| **Reflection** | 执行 → 反思 → 优化 | `src/ch4/reflection.py` | 模型自我批判，反馈"无需改进"即收敛 |
| **Multi-Agent** | 角色分工 + 调度 | `src/ch6/autoGen-*` | 用 System Message 划边界，用调度机制保秩序 |
| **Workflow / Graph** | 状态机 | `src/ch6/langgraph/Pro2` | 把流程画成图，条件边分流、回环边重试 |

### 工程能力

| 能力 | 说明 | 出现位置 |
|---|---|---|
| 工具系统 | 注册中心 + 描述注入 + 生命周期管理 | `ch4/executor.py`、`Pro2/tools/` |
| RAG | 加载 → 切分 → 向量化 → 检索 | `Pro2/tools/knowledge.py` |
| 结构化输出 | Pydantic + OutputParser | `Pro2/agents/writer.py` |
| 状态持久化 | Checkpointer + thread_id | Pro1 / Pro2 |
| 记忆管理 | JSON 持久化 + 滑动窗口裁剪 | `autoGen-test/tools/memory_tool.py` |
| 人在回路 | 自定义 Agent 拦截每轮输出 | `autoGen-test/travel_assistant.py` |
| 可观测性 | 装饰器计时 + Callback 追踪 | `Pro2/mylogger.py`、`Pro2/mycallback.py` |
| 容错降级 | 解析失败降级、搜索失败降级 | 多处 |

---

## 五、文档目录

| 文档 | 内容 |
|---|---|
| [00-环境准备与快速开始](docs/00-环境准备与快速开始.md) | Python 版本、依赖安装、API Key 申请、`.env` 配置、已知坑位 |
| [01-手写 Agent 框架](docs/01-ch4-手写Agent框架.md) | 三种范式的原理与代码走读，Agent 的本质结构 |
| [02-智能旅行助手](docs/02-notebooks-智能旅行助手.md) | 工具调用的第一课，Observation 的颗粒度 |
| [03-AutoGen 多智能体协作](docs/03-autogen-多智能体协作.md) | 角色划分、轮询调度、记忆、人在回路 |
| [04-LangGraph 智能搜索助手](docs/04-langgraph-智能搜索助手.md) | State / Node / Edge / Checkpointer 心智模型 |
| [05-LangGraph 智能研究助手](docs/05-langgraph-智能研究助手.md) | RAG、条件路由、审查回环、可观测性，全仓库集大成 |

---

## 六、技术栈

| 类别 | 选型 |
|---|---|
| 语言 | Python 3.13 |
| Agent 框架 | AutoGen AgentChat 0.7.5、LangGraph 1.0.0a3、LangChain 0.3.28 |
| 模型接入 | OpenAI 兼容协议（ModelScope Qwen / 阿里 DashScope MaaS） |
| 向量库 | ChromaDB 1.5.9 |
| 搜索 | Tavily Search API、SerpApi、arXiv API |
| 其他 | Streamlit、python-dotenv、Pydantic、httpx |

---

## 七、关于 API Key

仓库**不包含任何密钥**。所有敏感配置走 `.env`，且已被 `.gitignore` 覆盖。
需要申请哪些 Key、免费额度多少、填在哪，见 [环境准备文档](docs/00-环境准备与快速开始.md)。
