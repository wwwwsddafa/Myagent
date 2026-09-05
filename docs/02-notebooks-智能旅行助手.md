# 02 · 智能旅行助手（`notebooks/第一章`）

> **这一章回答：工具调用真正跑通是什么感觉？**
> 两个 Notebook，从 API 探路到完整 ReAct 闭环。

---

## 一、项目简介

| 文件 | 作用 |
|---|---|
| `prepare_平台准备.ipynb` | 环境探路：分别测试天气 API、Tavily API、LLM API 能不能通 |
| `智能旅行助手.ipynb` | 完整实现：输入城市 → 查天气 → 按天气搜景点 → 输出推荐 |

**业务流程：**

```
用户输入："北京"
    ↓
[get_weather]  调用 wttr.in 查当前天气  → "北京当前天气：晴，气温 26 摄氏度"
    ↓
[get_attraction]  把天气当条件搜景点  → "晴天适合去的景点：故宫、颐和园..."
    ↓
[LLM 汇总]  输出最终推荐
```

注意这里的**工具链是串联依赖的**：景点搜索的输入依赖天气查询的输出。这是 Agent 比简单 RAG 强的地方——**下一步的输入来自上一步的结果**。

---

## 二、为什么先写 `prepare_平台准备.ipynb`

这个 Notebook 单独测了三个 API，看起来很基础，但**这一步省不得**。

```python
# 1. 测天气 API
response = requests.get("https://wttr.in/Beijing?format=j1")
print("状态码:", response.status_code)
print("响应内容:", response.text)        # ← 先看原始返回长什么样

# 2. 测 Tavily API
client = TavilyClient(api_key=key)
response = client.search("北京天气")
print(response)

# 3. 测 LLM API
client = OpenAI(api_key=..., base_url=...)
response = client.chat.completions.create(...)
print("LLM API连接成功，响应:", response.choices[0].message.content)
```

**知识点 ①：先探数据结构，再写解析逻辑**

每个 API 的返回结构都不一样，不打印出来看一眼就写解析，等于闭眼开车。`wttr.in` 的 `format=j1` 返回的是嵌套很深的 JSON（`data['current_condition'][0]['weatherDesc'][0]['value']`），只有看过原始输出才知道该怎么写取值路径。

**习惯：接任何新 API，第一件事是 `print(response)` 看全貌。**

---

## 三、工具函数设计

### 3.1 天气工具

```python
def get_weather(city: str) -> str:
    url = f"https://wttr.in/{city}?format=j1"
    try:
        response = requests.get(url)
        response.raise_for_status()                  # 非 2xx 直接抛异常
        data = response.json()

        current_condition = data['current_condition'][0]
        weather_desc = current_condition['weatherDesc'][0]['value']
        temp_c = current_condition['temp_C']

        return f"{city}当前天气:{weather_desc}，气温{temp_c}摄氏度"
    except requests.exceptions.RequestException as e:
        return f"错误:查询天气时遇到网络问题 - {e}"
    except (KeyError, IndexError) as e:
        return f"错误:解析天气数据失败，可能是城市名称无效 - {e}"
```

**知识点 ②：工具函数绝不抛异常，要返回"错误字符串"**

这一点和 `ch4` 的搜索工具一脉相承，是 Agent 工具设计的铁律：

```python
# ❌ 错误做法
def get_weather(city):
    return requests.get(url).json()['current_condition'][0]   # 网络挂了整个 Agent 崩溃

# ✅ 正确做法
def get_weather(city):
    try:
        ...
        return "北京当前天气:晴，气温26摄氏度"        # 成功 → 自然语言
    except ...:
        return "错误:查询天气时遇到网络问题 - ..."     # 失败 → 错误描述
```

**原因**：工具函数的返回值会成为 Observation 喂回给模型。如果返回的是**自然语言形式的错误**，模型能"看懂"并自行决策——比如换个城市名重试，或者告诉用户"天气服务暂时不可用"。如果直接抛异常，整个 Agent 循环就断了。

**让工具"可失败"，Agent 才"可恢复"。**

**知识点 ③：`raise_for_status()` 的必要性**

`requests.get()` 对 404、500 这类 HTTP 错误**不会主动抛异常**，只返回状态码。不调 `raise_for_status()` 的话，错误响应会被当成正常数据去 `.json()` 解析，报出一堆莫名其妙的 `KeyError`。

### 3.2 景点搜索工具

```python
def get_attraction(city: str, weather: str) -> str:
    query = f"'{city}' 在'{weather}'天气下最值得去的旅游景点推荐及理由"
    response = tavily.search(query=query, search_depth="basic", include_answer=True)

    if response.get("answer"):
        return response["answer"]          # ← 直接用 Tavily 整理好的答案
    # 否则自己格式化 results
```

**知识点 ④：Tavily 的 `include_answer=True` 是省 token 神器**

`include_answer=True` 会让 Tavily 用 AI 把搜索结果**总结成一段话**直接返回。相比自己拼接 5 条网页摘要，输入给模型的 token 少一个数量级，质量还更高。

**这就是为什么 Agent 场景推荐 Tavily 而不是直接爬 Google**——它返回的已经是"给模型看的信息"了。

---

## 四、Agent 主循环

```python
available_tools = {
    "get_weather": get_weather,
    "get_attraction": get_attraction,
}
```

```python
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：
Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下格式之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束
"""
```

**知识点 ⑤：工具签名要写进 Prompt**

注意这里是 `get_weather(city: str)` 而不是只写 `get_weather`。**带参数名和类型的函数签名，能让模型更准确地构造调用参数。**

**知识点 ⑥：主循环里的输出截断**

```python
llm_output = llm.generate(full_prompt, system_prompt=AGENT_SYSTEM_PROMPT)
match = re.search(r'(Thought:.*?Action:.*?)(?=\nThought:|$)', llm_output, re.DOTALL)
```

模型有时会一次性输出好几对 Thought-Action（"自作主张"把后面的步骤也规划了）。这里用正则**只取第一对**，保证每次循环只前进一步。

这是一个很典型的实战细节：**Prompt 里说了"每次只输出一对"，不代表模型一定照做，代码层面也要兜底。**

---

## 五、⚠️ 运行前必改

两个 Notebook 里都硬编码了原作者的路径：

```python
load_dotenv(r"c:\Python_Ai\agents\data\.env")   # ← 换成你自己的路径
```

改成相对路径即可：

```python
from dotenv import load_dotenv
load_dotenv("../../data/.env")
```

---

## 六、这一章的收获（总结）

1. **工具函数的返回值 = 喂给模型的 Observation**，它的颗粒度直接决定模型推理质量。返回原始 JSON 是浪费，返回加工过的自然语言才是正解。
2. **工具不能抛异常，要返回错误字符串**——这样模型才有机会自我恢复。
3. **接新 API 先 `print(response)` 看结构**，再写解析，不要凭猜。
4. **Tavily 的 `include_answer` 专为 Agent 设计**，比自己爬网页省事太多。
5. **Prompt 约束 + 代码兜底要同时做**：说了"每次只输出一对"，代码里还是要截断。
6. **踩到的坑**：多轮循环里 `prompt_history` 会不断膨胀，迟早撑爆上下文——这个问题在后面的 AutoGen 记忆裁剪和 LangGraph Checkpoint 里才被真正解决。
