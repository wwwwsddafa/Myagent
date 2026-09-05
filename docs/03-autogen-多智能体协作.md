# 03 · AutoGen 多智能体协作（`src/ch6/autoGen-*`）

> **这一章回答：从"一个 Agent 干活"到"一群 Agent 协作"，多出来的复杂度是什么？**
> 两个项目：任务驱动的软件开发团队 + 带记忆与人类反馈的旅行助手。

---

## 项目一：多智能体软件开发团队

`src/ch6/autoGen-demo/autoGen_software_development_agent.py`

### 1.1 它做了什么

让 4 个 AI 角色组成虚拟开发团队，端到端产出一个应用：

| 角色 | 职责 | 交接信号 |
|---|---|---|
| **ProductManager** | 需求分析、模块划分、技术选型、验收标准 | "工程师开始实现" |
| **Engineer** | 写完整可运行代码 | "请代码审查员检查" |
| **CodeReviewer** | 审查质量、安全、最佳实践 | "代码审查完成，请用户代理测试" |
| **UserProxy** | 代表用户验收 | "TERMINATE" |

任务是从零开发一个 **Streamlit 比特币行情应用**：实时价格、24h 涨跌幅与涨跌额、手动刷新、加载态与错误处理。

### 1.2 核心代码

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

team_chat = RoundRobinGroupChat(
    participants=[product_manager, engineer, code_reviewer, user_proxy],
    termination_condition=TextMentionTermination("TERMINATE"),
    max_turns=20,
)

result = await Console(team_chat.run_stream(task=task))
```

### 1.3 知识点

#### 知识点 ①：角色边界靠 System Message 划分

四个 Agent 用的是**同一个模型**，唯一的区别就是 System Message：

```python
system_message = """你是一位经验丰富的产品经理，专门负责软件产品的需求分析和项目规划。
你的核心职责包括:
1. **需求分析**: 深入理解用户需求，识别核心功能和边界条件
2. **技术规划**: 基于需求制定清晰的技术实现路径
...
当接到开发任务时，请按以下结构进行分析:
1. 需求理解与分析  2. 功能模块划分  3. 技术选型建议  4. 实现优先级排序  5. 验收标准定义
请简洁明了地回应，并在分析完成后说"工程师开始实现"。"""
```

注意两个设计：
- **职责列举**用编号，清晰无歧义
- **固定的交接暗号**（"工程师开始实现"）——即使底层是轮询调度，暗号让对话读起来像真实协作

**这是多 Agent 的第一原则：角色不是靠代码区分的，是靠 Prompt 区分的。**

#### 知识点 ②：轮询调度与"复读机"问题

`RoundRobinGroupChat` 是最简单的调度策略——按列表顺序挨个发言，不管内容。

**它的致命问题：Agent 容易互相复读。** A 说"方案不错"，B 说"同意 A 的观点"，C 说"我也认为"，几轮下来全是废话。

本项目用了三层防御：

| 防御 | 代码 | 作用 |
|---|---|---|
| 严格的输出约束 | "每次回复控制在3句话以内"、"只输出工具返回的关键信息，不要添加分析" | 压缩复读空间 |
| 交接暗号 | "完成后说XX" | 让每个 Agent 有明确出口 |
| 轮次上限 | `max_turns=20` | 最后兜底，防止无限空转 |

#### 知识点 ③：终止条件是必须的

```python
termination = TextMentionTermination("TERMINATE")
```

**没有终止条件的多 Agent 系统是不可用的**——模型不会主动说"我做完了"，它会一直礼貌地接话。

`TextMentionTermination` 是最朴素的检测（消息里出现了指定文本就停）。AutoGen 还提供 `MaxMessageTermination`（按轮数）、`StopMessageTermination`、`TimeoutTermination` 等，生产环境通常组合使用。

本项目第二个项目用的是 `##STOP##` 而不是 `TERMINATE`，注释里写明了原因：**避免与任务提示词中的文字冲突**。这是个很实在的细节——如果你的任务描述里恰好出现了"TERMINATE"，就会被误触发。

#### 知识点 ④：`model_info` 显式声明模型能力

```python
OpenAIChatCompletionClient(
    model=model, api_key=api_key, base_url=base_url,
    model_info={
        "vision": False,
        "function_calling": True,     # 支持工具调用
        "json_output": True,          # 支持 JSON 输出
        "family": "unknown",
    },
)
```

用第三方兼容模型时，AutoGen 无法自动探测它支持什么。这里手动声明，框架才能决定**要不要走 function calling 路径**。

> ⚠️ 运行时会有一条警告：`Missing required field 'structured_output' in ModelInfo`。无害，只影响解析输出为类的场景，需要时在字典里补 `"structured_output": False` 即可。

#### 知识点 ⑤：asyncio 是 Multi-Agent 的天然底座

```python
async def run_software_development_team():
    ...
    result = await Console(team_chat.run_stream(task=task))

if __name__ == "__main__":
    asyncio.run(run_software_development_team())
```

多个 Agent 要交替发言，本质是**协作式多任务**。`run_stream` 是异步生成器，每产出一个消息就 `await` 一次，这样才能边跑边打印（`Console` 实时展示对话过程），而不是等全部跑完才一次性输出。

---

## 项目二：智能旅行助手（记忆 + 人在回路）

`src/ch6/autoGen-test/travel_assistant.py`

### 2.1 五个 Agent

| Agent | 类型 | 职责 |
|---|---|---|
| `TravelPlanner` | AssistantAgent | 总协调，解析需求、派活 |
| `InfoResearcher` | AssistantAgent | **唯一带工具的**，负责联网查信息 |
| `ItineraryDesigner` | AssistantAgent | 输出格式化行程 |
| `MemoryManager` | AssistantAgent | 读写用户偏好与历史 |
| `HumanFeedbackAgent` | **自定义** | 拦截每轮输出，收集人类意见 |

**关键设计：只有 InfoResearcher 绑定工具。**

```python
AssistantAgent(name="InfoResearcher", model_client=..., system_message=...,
               tools=[get_weather, search_attractions, search_restaurants, search_travel_tips])
```

**知识点 ⑥：工具不要全员配发**

给所有 Agent 都挂工具会导致：
- 角色混乱（设计师跑去搜索）
- token 浪费（每个 Agent 的 Prompt 都要塞工具描述）
- 行为不可预测

**谁需要工具就给谁**，其余 Agent 只做纯语言处理。这是多 Agent 系统的性能与可控性优化点。

### 2.2 自定义 Agent：实现人在回路

这是整个仓库最有意思的一段代码。

```python
class HumanFeedbackAgent(BaseChatAgent):
    def __init__(self, name: str, memory: TravelMemory):
        super().__init__(name=name, description="人类反馈代理，接收用户对旅行计划的修改意见")
        self.memory = memory

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)                    # ① 声明产出消息类型

    async def on_messages(self, messages, cancellation_token) -> Response:
        last_content = ...
        self.memory.add_conversation("assistant", last_content)   # ② 先存档

        try:
            feedback = input(">>> 您的反馈: ").strip()              # ③ 阻塞等人类输入
        except EOFError:
            print("(非交互环境，自动确认)")
            feedback = "ok"

        if not feedback or feedback.lower() in ("ok", "确认", "好的"):
            feedback_text = "用户确认当前方案，请继续完善或输出最终计划。"
        elif feedback.upper() == "##STOP##":
            feedback_text = "##STOP##"
        else:
            feedback_text = f"用户反馈：{feedback}"

        self.memory.add_conversation("user", feedback_text)
        return Response(chat_message=TextMessage(content=feedback_text, source=self.name))

    async def on_reset(self, cancellation_token) -> None:
        self._feedback_count = 0
```

**知识点 ⑦：自定义 Agent 要实现三件套**

| 成员 | 作用 |
|---|---|
| `produced_message_types` | 声明这个 Agent 会产出什么类型的消息，框架据此做类型校验 |
| `on_messages()` | 核心逻辑，收到消息后怎么响应 |
| `on_reset()` | 重置内部状态 |

**知识点 ⑧：Human-in-the-loop 是低成本的质量闸门**

Agent 跑偏时，最省事的补救不是调 Prompt，而是**让人类在中途插一句**。这个 Agent 每轮输出后暂停，等人类说：
- `ok` → 继续
- 任意文字 → 当作修改意见传回团队
- `##STOP##` → 触发终止条件

**这是把"不可控的生成过程"变成"可控的人机协作过程"的最小改动。** 在客服、内容生成、代码生成这类容错率低的场景，人在回路几乎是标配。

**知识点 ⑨：`EOFError` 兜底**

```python
try:
    feedback = input(">>> 您的反馈: ").strip()
except EOFError:
    feedback = "ok"          # 非交互环境自动确认
```

在 CI、脚本、管道里运行时没有 stdin，`input()` 会抛 `EOFError`。捕获后自动确认，保证程序在非交互环境下也能跑完——**这是"脚本"和"可用工具"的分界线**。

### 2.3 记忆模块

`tools/memory_tool.py`

```python
class TravelMemory:
    def add_conversation(self, role: str, content: str):
        data = self._load()
        data["conversation_history"].append({
            "role": role, "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        data["conversation_history"] = data["conversation_history"][-40:]   # ← 滑动窗口
        self._save(data)

    def save_trip(self, destination, days, itinerary):
        data["past_trips"].append({...})
        data["past_trips"] = data["past_trips"][-5:]                        # ← 只留 5 次
        self._save(data)
```

**知识点 ⑩：记忆必须"可持久化 + 可裁剪"**

两个要求缺一不可：

| 要求 | 实现 | 不做的后果 |
|---|---|---|
| **可持久化** | JSON 文件落地，`_ensure_file()` 保证文件存在 | 程序一关记忆全丢，用户偏好每次重问 |
| **可裁剪** | 对话保留最近 40 条、行程保留最近 5 次 | 上下文无限膨胀，迟早撑爆窗口、拖垮速度 |

**滑动窗口是最简单有效的记忆策略**——新数据进、老数据出。生产场景可以升级为"摘要压缩"（把老对话总结成一段话）或"向量检索记忆"（按需召回），但思路一致。

**知识点 ⑪：JSON 读写的容错**

```python
def _load(self) -> dict:
    try:
        with open(self.memory_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {"user_preferences": {}, "conversation_history": [], "past_trips": []}
```

文件损坏或被手动编辑坏了，返回空结构而不是崩溃。**记忆丢了可以重建，程序崩了体验就没了。**

---

## 三、运行

```bash
# 先配好 .env（见 00-环境准备，注意坑 1 的路径问题）

cd src/ch6/autoGen-demo
python autoGen_software_development_agent.py

cd src/ch6/autoGen-test
python travel_assistant.py "帮我规划未来3天北京的行程，喜欢历史文化和美食"
```

`travel_assistant.py` 支持命令行传参，不传则进入交互式输入。

---

## 四、这一章的收获（总结）

1. **角色边界靠 System Message 划分，协作秩序靠调度机制保障。** 两者缺一不可。
2. **轮询调度会复读**，必须靠"严格的输出约束 + 交接暗号 + 轮次上限"三层防御压制。
3. **终止条件是刚需**，且要注意终止词不要和任务内容撞车。
4. **工具按需配发**，不要全员挂满——省 token、保角色清晰。
5. **记忆要能落地、能裁剪**，滑动窗口是最朴素的解法。
6. **人在回路是低成本兜底**：Agent 跑偏时，人插一句话比改十次 Prompt 更省事。
7. **asyncio 是 Multi-Agent 的底座**，协作式调度天然适合异步。
8. **自定义 Agent 只需实现三件套**：`produced_message_types` / `on_messages` / `on_reset`。
