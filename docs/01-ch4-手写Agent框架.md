# 01 · 手写 Agent 框架（`src/ch4`）

> **这一章回答一个问题：Agent 到底是什么？**
> 不用任何框架，把 LLM 客户端、工具执行器、三种推理范式全部手写一遍。

---

## 一、先看结论：Agent 的本质结构

写完这一章最大的收获是——**Agent 不是一种模型能力，而是一套工程结构**。它只由五块零件组成：

```
┌─────────────────────────────────────────────────┐
│                    Agent                        │
├─────────────────────────────────────────────────┤
│  1. Prompt 约束    → 规定模型"必须怎么输出"       │
│  2. 输出解析       → 从自然语言中抠出结构化指令    │
│  3. 工具调度       → 根据指令找到并执行对应函数    │
│  4. 循环控制       → 决定继续还是终止，防止跑飞    │
│  5. 状态记忆       → 把历史步骤喂回给模型          │
└─────────────────────────────────────────────────┘
```

**模型本身只负责"生成文本"，上面五件事全是代码做的。** 理解了这一点，再看 AutoGen、LangGraph 就不会觉得神秘——它们只是把这五块零件做了更好的封装。

---

## 二、目录结构

| 文件 | 职责 |
|---|---|
| `llm.py` | 通用 LLM 客户端，流式输出 |
| `search.py` | 搜索工具（SerpApi） |
| `executor.py` | 工具执行器 / 注册中心 |
| `react_agent.py` | ReAct 范式 |
| `plan_and_solve_planner.py` | 规划器 |
| `plan_and_solve_executor.py` | 执行器 |
| `plan_and_solve_PlanAndSolveAgent.py` | 上面两者的组装 |
| `reflection.py` | Reflection 范式 ⚠️ **未完成** |

---

## 三、零件一：LLM 客户端（`llm.py`）

### 3.1 设计要点

```python
class HelloAgentLLM:
    def __init__(self, model=None, api_key=None, base_url=None, timeout=None):
        self.model   = model   or os.environ.get("MODEL_NAME")
        api_key      = api_key or os.environ.get("OPENAI_API_KEY")
        base_url     = base_url or os.environ.get("OPENAI_BASE_URL")
        ...
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
```

**知识点 ①：OpenAI 兼容协议是事实标准**

关键在于 `base_url` 参数。只要一个厂商提供 OpenAI 兼容接口，改这一个参数就能换模型，**业务代码一行不用动**。本项目用的 ModelScope Qwen，换成 DashScope、月之暗面、本地 Ollama 都是改个 URL 的事。

这是这个项目最重要的工程决策之一：**不要把自己绑死在某一家模型上**。

### 3.2 流式输出

```python
response = self.client.chat.completions.create(
    model=self.model, messages=messages,
    temperature=temperature,
    stream=True,                      # ← 开启流式
)
collected_content = ""
for chunk in response:
    if not chunk.choices:
        continue
    content = chunk.choices[0].delta.content or ""
    if content:
        print(content, end="", flush=True)   # 逐字打印
        collected_content += content
return collected_content
```

**知识点 ②：流式返回的是"增量"不是"全量"**

开启 `stream=True` 后，API 返回的不是一条完整消息，而是一个**迭代器**，每次吐出一个 `chunk`。真正的文本在 `chunk.choices[0].delta.content` 里，而且**可能是 `None`**（最后几块通常为空，用来传 finish_reason）。

所以必须：
1. 判空（`or ""`）
2. 自己拼接（`collected_content += content`）

最终返回拼好的完整字符串。这样上层调用者不用关心是不是流式，**接口保持一致**。

**知识点 ③：`temperature=0` 的选择**

Agent 场景下默认设 0。因为 Agent 要的是**稳定可复现的推理和格式**，不是创意发挥。温度高了容易出现格式漂移，导致解析失败。

### 3.3 `.env` 加载

```python
env_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", ".env")
load_dotenv(env_path)
```

用 `__file__` 的绝对路径往上回溯三级找到仓库根，再拼 `data/.env`。这样**无论从哪个目录启动都能找到配置文件**。（对比之下，`autoGen-demo` 少回溯了一级，就出现了找不到 `.env` 的问题——见环境文档坑 1。）

---

## 四、零件二：工具系统

### 4.1 搜索工具（`search.py`）

```python
params = {
    "engine": "google", "q": query,
    "api_key": api_key,
    "gl": "cn",         # 国家代码
    "hl": "zh-cn",      # 语言代码
}
client = SerpApiClient(params)
results = client.get_dict()
```

**知识点 ④：工具返回值要"智能解析"，不能原样丢给模型**

这一段是整个工具设计里最值得学的：

```python
if "answer_box_list" in results:                    # 优先级 1：答案列表
    return "\n".join(results["answer_box_list"])
if "answer_box" in results and "answer" in ...:     # 优先级 2：直接答案
    return results["answer_box"]["answer"]
if "knowledge_graph" in results and ...:            # 优先级 3：知识图谱
    return results["knowledge_graph"]["description"]
if "organic_results" in results:                    # 优先级 4：降级取前 3 条摘要
    snippets = [f"[{i+1}] {res.get('title','')}\n{res.get('snippet','')}"
                for i, res in enumerate(results["organic_results"][:3])]
    return "\n\n".join(snippets)
```

**为什么要这么做？** 因为工具的返回值就是喂给模型的 **Observation**，而上下文窗口是有限的。Google 返回的原始 JSON 动辄几万 token，全塞进去既烧钱又稀释注意力。

**核心原则：工具返回的是"给模型看的信息"，不是"给人类看的原始数据"。** 要做截断、摘要、优先级排序。

### 4.2 工具执行器（`executor.py`）

```python
class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}   # {名称: {描述, 函数}}

    def register_tool(self, name, description, func):
        self.tools[name] = {"description": description, "func": func}

    def getTool(self, name):
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        return "\n".join([f"- {name}: {info['description']}"
                          for name, info in self.tools.items()])
```

**知识点 ⑤：工具描述是给模型看的"使用说明书"**

`getAvailableTools()` 生成的这段字符串会被拼进 Prompt。模型**完全靠这个描述来决定该不该用、用哪个工具**。

所以描述写得好不好，直接决定 Agent 的表现。对比：

```python
# ❌ 糟糕的描述
register_tool("Search", "搜索", search)

# ✅ 好的描述
register_tool("Search",
    "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。",
    search)
```

好的描述要说清**什么时候该用**。这比说清"它能干什么"更重要。

**知识点 ⑥：注册表模式**

工具和 Agent 之间是解耦的——Agent 只认工具名字符串，执行器负责名字到函数的映射。新增工具只需 `register_tool`，不用改 Agent 代码。这是后面所有框架（LangChain 的 `@tool`、AutoGen 的 `tools=[...]`）的共同底层思路。

---

## 五、范式一：ReAct（`react_agent.py`）

### 5.1 原理

ReAct = **Rea**soning + **Act**ing。模型不再一口气给出答案，而是循环执行：

```
Thought（思考）→ Action（行动）→ Observation（观察）→ Thought → ... → Finish
```

每一步都用上一步的观察结果来修正下一步决策，这就是"让模型学会用工具"的核心机制。

### 5.2 Prompt 设计

```
可用工具如下：
{tools}                                    ← getAvailableTools() 注入

请严格按照以下格式进行回应：
Thought: 你的思考过程...
Action: 你决定采取的行动，必须是以下格式之一：
    - `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
    - `Finish[最终答案]`:当你认为已经获得最终答案时。

Question: {question}
History: {history}                          ← 历史步骤回灌
```

三个变量缺一不可：`tools` 告诉它有什么、输出格式约束它怎么答、`history` 让它记得自己做过什么。

### 5.3 输出解析（关键难点）

```python
# 提取 Thought
thought_match = re.search(r"^Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL | re.MULTILINE)

# 提取 Action
action_match = re.search(r"^Action:\s*(.*?)$", text, re.DOTALL | re.MULTILINE)
```

**知识点 ⑦：这段正则为什么这么写**

| 片段 | 作用 |
|---|---|
| `^Thought:` | `MULTILINE` 让 `^` 匹配每行开头，而不是整个字符串开头 |
| `(.*?)` | **非贪婪**，匹配到最近的下一个标记就停 |
| `(?=\nAction:\|$)` | **正向断言**：后面必须是换行 + `Action:` 或字符串末尾，但**这段本身不包含进匹配结果** |
| `re.DOTALL` | 让 `.` 也能匹配换行符（思考内容往往是多行的） |

如果没有 `DOTALL`，多行的 Thought 只能匹配到第一行；如果贪婪匹配 `.*`，会一路吃到文本末尾，把 Action 也吞进去。

### 5.4 循环控制

```python
while current_step < self.max_steps:      # 步数上限，防跑飞
    prompt = REACT_PROMPT_TEMPLATE.format(...)
    response_text = self.llm_client.think(messages=messages)
    thought, action = self._parse_output(response_text)
    if not action:                        # 解析失败 → 终止
        print("⚠警告: 未能解析出有效的Action,流程终止。")
        break
    if action.startswith("Finish"):       # 终止条件 → 提取答案返回
        ...
    else:
        tool_name, tool_input = self._parse_action(action)
        observation = tool_function(tool_input)
        self.history.append(f"Action: {action}")            # ← 状态记忆
        self.history.append(f"Observation: {observation}")
```

**知识点 ⑧：Agent 循环的三道保险**

| 保险 | 代码 | 解决什么 |
|---|---|---|
| 步数上限 | `max_steps = 5` | 模型陷入死循环，无限烧 token |
| 解析失败即终止 | `if not action: break` | 模型输出格式漂移，继续跑只会更乱 |
| Finish 兜底清洗 | 正则失败时 `action.replace("Finish","")` | 模型想结束但格式不标准，硬取内容救回来 |

第三点特别能体现工程思维：**不要假设模型一定按格式输出**。所有来自 LLM 的内容都必须当成"不可信输入"处理。

---

## 六、范式二：Plan-and-Solve

与 ReAct 的区别：**先一次性把计划定好，再逐步执行**，而不是边做边想。

### 6.1 Planner

```python
# 要求模型输出：
```python
["步骤1", "步骤2", "步骤3", ...]
```

# 解析时：
plan_str = response_text.split("```python")[1].split("```")[0].strip()
plan = ast.literal_eval(plan_str)          # ← 注意是 literal_eval 不是 eval
```

**知识点 ⑨：为什么用 `ast.literal_eval` 而不是 `eval`**

`eval()` 会执行任意 Python 代码。模型输出是不可信的——万一它"幻觉"出一句 `os.system('rm -rf /')`，`eval` 就直接执行了。

`ast.literal_eval` 只解析**字面量**（字符串、数字、列表、字典、元组、布尔、None），遇到任何函数调用或表达式都会抛异常。这是处理 LLM 输出为数据结构时的**标准做法**。

外面再包一层 `try/except (ValueError, SyntaxError, IndexError)`，解析失败返回空列表，让上游决定是重试还是降级。

### 6.2 Executor

```python
for i, step in enumerate(plan):
    prompt = EXECUTOR_PROMPT_TEMPLATE.format(
        question=question, plan=plan,
        history=history if history else "无",   # ← 累积的历史
        current_step=step)
    response_text = self.llm_client.think(messages=messages)
    history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
```

**知识点 ⑩：`history` 的回灌是 Plan-and-Solve 的灵魂**

每执行一步，就把"这一步做了什么 + 得到了什么"追加进 `history`，再传给下一步。这样模型在处理第 3 步时，能看到第 1、2 步的结果，才能做连续推理。

同时注意 `plan=plan`（完整计划）也一直传着——让模型知道全局，不至于迷失在单步里。

### 6.3 两种范式怎么选

| 维度 | ReAct | Plan-and-Solve |
|---|---|---|
| 决策方式 | 边做边想，动态决策 | 先规划后执行 |
| 适合场景 | 路径不确定、需要实时信息 | 步骤明确、多步依赖 |
| 优点 | 灵活，能根据观察调整 | 稳定，不会跑偏 |
| 缺点 | 容易在中间步骤迷路 | **计划本身错了就全错** |
| Token 消耗 | 每步都要带完整历史 | 计划阶段一次性开销 |

**没有银弹，按场景选。** 这是这个项目最实用的结论之一。

---

## 七、范式三：Reflection ⚠️

**当前状态：不可运行。** `reflection.py` 有三个问题：

| 问题 | 位置 |
|---|---|
| `from llm import HelloAgentsLLM` → ImportError（真实类名是 `HelloAgentLLM`，少个 `s`） | 第 2 行 |
| `Memory` 类、`__init__`、`_get_llm_response` 三个函数体是 `...` 占位符 | 第 5、42、85 行 |
| `INITIAL_PROMPT_TEMPLATE`、`REFINE_PROMPT_TEMPLATE` 内容缺失 | 第 55、85 行 |

### 7.1 设计思路（从 `run()` 方法可读出）

```python
initial_code = 初始执行(task)                    # 1. 先干一遍
self.memory.add_record("execution", initial_code)

for i in range(self.max_iterations):             # 2. 迭代优化
    feedback = 反思(task, last_code)              #    a. 自我批判
    if "无需改进" in feedback:                    #    b. 收敛判断
        break
    refined_code = 优化(task, last_code, feedback)#    c. 根据反馈重写
    self.memory.add_record("execution", refined_code)
```

**核心思想：让模型批评自己的产出，再按批评意见重写。**

收敛条件是 `if "无需改进" in feedback`——用关键词判断模型是否满意。这是个简单但有效的做法，工程上比让模型输出 JSON 更省事（当然也更脆弱，模型可能一直不说"无需改进"，所以要配 `max_iterations` 上限）。

**知识点 ⑪：Memory 的作用**

Reflection 必须能看到自己**之前做过什么**，否则"反思"无从谈起。`Memory` 用 `add_record("execution"/"reflection", ...)` 分类记录轨迹，`get_trajectory()` 取完整历史。这比 ReAct 里简单的 `history: List[str]` 更进一步——**带了类型标签**。

---

## 八、运行

```bash
cd src/ch4
python llm.py                        # 测 LLM 客户端（写个快排）
python executor.py                   # 测工具执行器（搜"英伟达最新GPU"）
python react_agent.py                # 完整 ReAct："马斯克Optimus量产状态？中国供应链公司有哪些？"
python plan_and_solve_planner.py     # 完整 Plan-and-Solve：水果店卖苹果的算术题
```

`plan_and_solve_planner.py` 的 `__main__` 里直接演示了完整流程，很适合作为入口阅读。

---

## 九、这一章的收获（总结）

1. **Agent = Prompt 约束 + 输出解析 + 工具调度 + 循环控制 + 状态记忆**，模型只负责生成文本。
2. **工具返回值必须二次加工**——它是给模型看的 Observation，不是给人看的原始数据。
3. **工具描述要写清"什么时候用"**，这决定了模型选不选得对。
4. **所有 LLM 输出都当不可信输入处理**：解析要 try/except，执行要 `literal_eval` 而非 `eval`，格式异常要有兜底。
5. **范式之间没有银弹**：ReAct 灵活但易跑偏，Plan-and-Solve 稳定但怕计划错，Reflection 质量高但费 token。
6. **`base_url` 抽象 + `.env` 配置**是低成本换模型的关键，别把业务绑死在一家厂商上。
