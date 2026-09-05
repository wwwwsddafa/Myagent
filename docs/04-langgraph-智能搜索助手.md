# 04 · LangGraph 智能搜索助手（`src/ch6/langgraph/Pro1`）

> **这一章回答：LangGraph 的心智模型是什么？**
> 一个只有三个节点的线性流程，用来建立对状态机的直觉。

---

## 一、项目简介

```
用户输入："LangGraph 怎么用？"
    ↓
[understand]  理解意图，改写为精准搜索词  → "LangGraph tutorial StateGraph"
    ↓
[search]      调 Tavily 真实检索          → 综合答案 + Top5 结果
    ↓
[answer]      基于搜索结果生成带来源的答案
    ↓
  输出
```

相比 `ch4` 手写 ReAct，这里最大的变化是：**循环交给了框架**。不再是 `while` + 正则解析，而是把三个步骤声明为节点，用边连起来。

---

## 二、LangGraph 的四个核心概念

> **State 是骨架，Node 是函数，Edge 是流程，Checkpointer 是记忆。**

### 2.1 State（状态）

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话消息列表（自动追加）
    user_query: str                            # 用户查询
    search_query: str                          # 优化后的搜索词
    search_results: str                        # Tavily 搜索结果
    final_answer: str                          # 最终答案
    step: str                                  # 当前步骤
```

**知识点 ①：`add_messages` 是一个 Reducer**

这是 LangGraph 最容易让人困惑的地方，但理解了就通了：

```python
# 普通字段：新值直接覆盖旧值
state["search_results"] = "新结果"          # → 旧结果没了

# 带 Reducer 的字段：新值按规则合并
state["messages"] = [msg3]                  # → 实际变成 [msg1, msg2, msg3]
```

`Annotated[list, add_messages]` 的意思是：这个字段是 list 类型，合并时用 `add_messages` 函数处理（追加而非覆盖）。

```python
Annotated[类型, 合并函数]
         ↑        ↑
      声明类型   声明"新值怎么并入旧值"
```

**没有 Reducer 的字段就是覆盖语义。** 想要追加语义就得配 Reducer——对话历史必须追加，否则每轮都会丢失上文。

**知识点 ②：节点只返回"增量"**

```python
def tavily_search_node(state: SearchState) -> SearchState:
    ...
    return {
        "search_results": search_results,      # ← 只返回改动的字段
        "step": "searched",
        "messages": [AIMessage(content="✅ 搜索完成!")]
    }
```

节点不需要把整个 State 返回回去。返回什么字段，框架就更新什么字段，其余保持原样：

```
旧状态: {messages: [m1,m2], user_query: "...", search_results: "", step: "understood"}
节点返回: {search_results: "...", step: "searched", messages: [m3]}
新状态: {messages: [m1,m2,m3], user_query: "...", search_results: "...", step: "searched"}
                    ↑ add_messages 生效
```

这个设计让节点函数变得非常纯粹：**读 state，做一件事，返回改动。**

### 2.2 Node（节点）

节点就是普通 Python 函数，签名固定为 `(state) -> dict`：

```python
def understand_query_node(state: SearchState) -> SearchState:
    # 1. 从 state 取数据
    user_message = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg.content
            break

    # 2. 调 LLM
    response = llm.invoke([SystemMessage(content=understand_prompt)])

    # 3. 返回增量
    return {"user_query": response.content, "search_query": search_query, "step": "understood", ...}
```

### 2.3 Edge（边）

```python
workflow.add_edge(START, "understand")
workflow.add_edge("understand", "search")
workflow.add_edge("search", "answer")
workflow.add_edge("answer", END)
```

**知识点 ③：`add_edge` vs `add_conditional_edges`**

| 方法 | 语义 | 本项目 |
|---|---|---|
| `add_edge(A, B)` | 无条件跳转 | 全部使用 |
| `add_conditional_edges(A, 路由函数)` | 由函数返回值决定跳哪 | Pro2 才用 |

本项目刻意**没有条件判断**——线性流程是理解状态机最好的起点。Pro2 会在此基础上加条件边和回环边。

### 2.4 Checkpointer（记忆）

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
app.astream(initial_state, config=config)
```

**知识点 ④：`thread_id` 就是"会话 ID"**

Checkpointer 会在每个节点执行后保存状态快照。`thread_id` 决定这些快照属于哪个会话：

```
thread_id = "session-1"  →  独立的对话历史 A
thread_id = "session-2"  →  独立的对话历史 B
```

**相同 `thread_id` 的调用共享记忆，不同的互不干扰。** 这就是多轮对话的实现方式——不需要自己维护 history 列表，框架帮你存、帮你恢复。

`InMemorySaver` 是内存版（重启即失），生产环境可换成 `SqliteSaver` 或 `PostgresSaver` 做持久化。

---

## 三、两个实战技巧

### 3.1 搜索失败的降级设计

```python
def generate_answer_node(state):
    if state["step"] == "search_failed":
        fallback_prompt = f"""搜索API暂时不可用, 请基于您的知识回答用户的问题:
        请提供一个有用的回答, 并说明这是基于已有知识的回答。"""
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        return {"final_answer": response.content, "step": "completed", ...}
    ...
```

**知识点 ⑤：`step` 字段就是状态机的"状态码"**

用 `step` 记录执行到哪一步、是否成功，后续节点据此分支。这比在 state 里塞一堆布尔标志清晰得多。

**降级思路：搜索挂了不要整个崩掉，改用模型自有知识回答，并明确告知用户"这是基于已有知识的回答"。** 诚实标注来源比硬撑着编造更重要。

### 3.2 `astream` 异步流式

```python
async for output in app.astream(initial_state, config=config):
    for node_name, node_output in output.items():
        if "messages" in node_output and node_output["messages"]:
            latest_message = node_output["messages"][-1]
            if isinstance(latest_message, AIMessage):
                if node_name == "understand":
                    print(f"🤔 理解阶段: {latest_message.content}")
                elif node_name == "search":
                    print(f"🔍 搜索阶段: {latest_message.content}")
                elif node_name == "answer":
                    print(f"💡 最终回答:\n{latest_message.content}")
```

**知识点 ⑥：`invoke` vs `astream`**

| 方法 | 特点 | 适用 |
|---|---|---|
| `invoke()` | 阻塞，全部跑完才返回 | 简单调用、批处理 |
| `astream()` | 异步流式，**每完成一个节点就产出一次** | 需要实时反馈的长任务 |

Agent 任务动辄几十秒，用 `astream` 能让用户看到"正在理解 → 正在搜索 → 正在生成"的进度，**体验提升巨大**。

---

## 四、运行

```bash
cd src/ch6/langgraph/Pro1
python askAssistantAgent.py
```

> ⚠️ 记得先配好 `.env`（见环境文档坑 1）。这个文件的 LLM 客户端在**模块顶层**创建，拿不到 Key 会直接抛 `ValueError` ——这是有意的快速失败设计。

---

## 五、这一章的收获（总结）

1. **State 是骨架、Node 是函数、Edge 是流程、Checkpointer 是记忆**——这四句话就是 LangGraph 的全部。
2. **Reducer 决定字段是"覆盖"还是"追加"**，对话历史必须配 `add_messages`。
3. **节点只返回增量字段**，框架负责合并，这让节点函数极其纯粹。
4. **`thread_id` = 会话 ID**，多轮对话和会话隔离都靠它。
5. **用 `step` 字段做状态码**，让后续节点知道前面发生了什么、要不要降级。
6. **`astream` 让长任务可感知**，用户体验和 `invoke` 完全不是一个量级。
7. **状态机思维 vs 顺序编程**：把流程画成图之后，哪些步骤能并行、哪里要分支、哪里可能循环，一眼就看清楚了——这是 Chain 结构给不了的。
