# LangGraph 核心知识点笔记

> 基于智能搜索助手项目 (askAssistantAgent.py)

---

## 一、LangGraph 是什么？

LangGraph 是 LangChain 生态中的**状态机工作流框架**，用于构建有状态、多步骤的 AI 应用。

**核心思想：** 把复杂的 AI 任务拆分成多个节点（函数），通过图结构连接，按顺序或条件执行。

---

## 二、核心概念

### 1. State（状态）

**定义：** 工作流中所有节点共享的数据结构。

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class SearchState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表（自动追加）
    user_query: str                          # 用户查询
    search_query: str                        # 优化后的搜索词
    search_results: str                      # 搜索结果
    final_answer: str                        # 最终答案
    step: str                                # 当前步骤
```

**关键点：**

- 使用 `TypedDict` 定义类型安全的状态结构
- `Annotated[list, add_messages]` 让消息自动追加而非覆盖
- 每个节点返回**部分更新**，LangGraph 自动合并到全局状态

---

### 2. Node（节点）

**定义：** 工作流中的执行单元，本质是普通 Python 函数。

```python
def understand_query_node(state: SearchState) -> SearchState:
    """步骤1: 理解用户查询并生成搜索关键词"""
    # 1. 从状态中获取数据
    user_message = state["messages"][-1].content

    # 2. 执行业务逻辑（调用 LLM）
    response = llm.invoke([SystemMessage(content=prompt)])

    # 3. 返回状态更新（只返回需要修改的字段）
    return {
        "user_query": response.content,
        "search_query": extracted_query,
        "step": "understood",
        "messages": [AIMessage(content="我理解您的需求...")]
    }
```

**节点三要素：**

1. 接收 `state` 作为参数
2. 读取状态中的数据
3. 返回需要更新的状态字段（字典）

---

### 3. Edge（边）

**定义：** 连接节点的有向边，决定执行顺序。

```python
workflow.add_edge(START, "understand")     # 开始 → 理解
workflow.add_edge("understand", "search")  # 理解 → 搜索
workflow.add_edge("search", "answer")      # 搜索 → 回答
workflow.add_edge("answer", END)           # 回答 → 结束
```

**执行流程：**

```
START → understand → search → answer → END
```

---

### 4. StateGraph（状态图）

**定义：** 用于构建工作流的核心类。

```python
from langgraph.graph import StateGraph, START, END

# 1. 创建图，指定状态类型
workflow = StateGraph(SearchState)

# 2. 添加节点
workflow.add_node("understand", understand_query_node)
workflow.add_node("search", tavily_search_node)
workflow.add_node("answer", generate_answer_node)

# 3. 添加边（连接节点）
workflow.add_edge(START, "understand")
workflow.add_edge("understand", "search")
workflow.add_edge("search", "answer")
workflow.add_edge("answer", END)

# 4. 编译成可执行应用
app = workflow.compile(checkpointer=memory)
```

---

### 5. Checkpointer（检查点/记忆）

**定义：** 保存和恢复工作流状态的机制。

```python
from langgraph.checkpoint.memory import InMemorySaver

memory = InMemorySaver()  # 内存保存器
app = workflow.compile(checkpointer=memory)
```

**作用：**

- 保存每个节点执行后的状态快照
- 支持多轮对话（通过 `thread_id` 区分会话）
- 支持中断恢复

**使用方式：**

```python
config = {"configurable": {"thread_id": "session-1"}}

# 第1次调用
app.invoke({"messages": [HumanMessage("问题1")]}, config=config)

# 第2次调用（同一 thread_id，AI 记得历史）
app.invoke({"messages": [HumanMessage("问题2")]}, config=config)
```

---

## 三、项目架构解析

### 工作流流程图

```
用户输入问题
    ↓
┌─────────────────┐
│  understand节点  │  ← 理解用户意图，生成优化搜索词
└────────┬────────┘
         ↓
┌─────────────────┐
│   search节点    │  ← 调用 Tavily API 搜索信息
└────────┬────────┘
         ↓
┌─────────────────┐
│   answer节点    │  ← 基于搜索结果生成最终答案
└────────┬────────┘
         ↓
    返回给用户
```

### 数据流转

| 阶段       | 更新的字段                           | 说明                |
| ---------- | ------------------------------------ | ------------------- |
| 初始化     | `messages`                           | 用户输入            |
| understand | `user_query`, `search_query`, `step` | LLM 理解并优化查询  |
| search     | `search_results`, `step`             | Tavily API 返回结果 |
| answer     | `final_answer`, `step`               | LLM 综合生成答案    |

---

## 四、关键技术点

### 1. `Annotated[list, add_messages]`

**作用：** 定义消息的合并策略（Reducer）。

**对比：**

```python
# 普通字典：新值覆盖旧值
state = {"messages": [msg1, msg2]}
state["messages"] = [msg3]  # → [msg3]，msg1和msg2丢失

# 使用 add_messages：自动追加
state = {"messages": [msg1, msg2]}
# 节点返回 {"messages": [msg3]}
# → LangGraph 自动合并为 [msg1, msg2, msg3]
```

---

### 2. 状态的部分更新

节点只需返回需要修改的字段：

```python
def search_node(state: SearchState) -> SearchState:
    # 只返回 search_results 和 step，其他字段保持不变
    return {
        "search_results": "...",
        "step": "searched"
    }
```

LangGraph 会自动合并：

```python
旧状态: {messages: [...], user_query: "...", search_results: "", step: "understood"}
节点返回: {search_results: "...", step: "searched"}
新状态: {messages: [...], user_query: "...", search_results: "...", step: "searched"}
```

---

### 3. 异步执行

```python
async for output in app.astream(initial_state, config=config):
    for node_name, node_output in output.items():
        print(f"节点 {node_name} 输出: {node_output}")
```

**`astream` vs `invoke`：**
| 方法 | 特点 | 适用场景 |
|------|------|----------|
| `invoke()` | 阻塞，等待全部完成 | 简单调用 |
| `astream()` | 异步流式，逐步输出 | 实时反馈、长任务 |

---

### 4. 配置线程（多会话）

```python
config = {"configurable": {"thread_id": f"session-{count}"}}
```

- 每个 `thread_id` 对应独立的对话历史
- 相同 `thread_id` 的调用共享记忆
- 不同 `thread_id` 的调用互不干扰

---

## 五、完整代码模板

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

# 1. 定义状态
class MyState(TypedDict):
    messages: Annotated[list, add_messages]
    # 其他字段...

# 2. 定义节点函数
def node1(state: MyState) -> MyState:
    return {"messages": [...], "step": "done1"}

def node2(state: MyState) -> MyState:
    return {"messages": [...], "step": "done2"}

# 3. 构建工作流
workflow = StateGraph(MyState)
workflow.add_node("step1", node1)
workflow.add_node("step2", node2)
workflow.add_edge(START, "step1")
workflow.add_edge("step1", "step2")
workflow.add_edge("step2", END)

# 4. 编译
memory = InMemorySaver()
app = workflow.compile(checkpointer=memory)

# 5. 执行
result = app.invoke({
    "messages": [HumanMessage("用户输入")]
}, config={"configurable": {"thread_id": "session-1"}})
```

---

## 六、常见模式

### 1. 线性流程（本项目）

```
START → A → B → C → END
```

### 2. 条件分支（扩展）

```python
def route(state):
    if state["step"] == "success":
        return "success_node"
    else:
        return "fallback_node"

workflow.add_conditional_edges("check", route)
```

### 3. 循环流程

```
START → A → B → [条件判断] → A (循环) 或 END
```

---

## 七、核心 API 速查

| API                       | 作用         |
| ------------------------- | ------------ |
| `StateGraph(State)`       | 创建状态图   |
| `add_node(name, func)`    | 添加节点     |
| `add_edge(from, to)`      | 添加边       |
| `add_conditional_edges()` | 添加条件边   |
| `compile(checkpointer)`   | 编译工作流   |
| `invoke(input, config)`   | 同步执行     |
| `astream(input, config)`  | 异步流式执行 |
| `InMemorySaver()`         | 内存状态保存 |

---

## 八、学习建议

1. **先理解状态** → 状态是工作流的核心，所有节点共享
2. **再理解节点** → 节点就是普通函数，输入状态，返回更新
3. **最后理解图** → 图只是定义节点如何连接
4. **实践建议** → 从线性流程开始，再尝试条件分支和循环

---

_生成时间: 2026-08-31_
_基于项目: askAssistantAgent.py_
