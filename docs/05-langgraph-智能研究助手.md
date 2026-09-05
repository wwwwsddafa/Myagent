# 05 · LangGraph 智能研究助手（`src/ch6/langgraph/Pro2`）

> **全仓库集大成之作。** 前面四章学到的东西——ReAct 循环、多 Agent 分工、状态机、RAG、记忆——全部在这里落地。

---

## 一、它做了什么

输入一个问题，自动判断难度并分流，走完"检索 → 研究 → 分析 → 写作 → 审查"全流程，产出结构化研究报告，支持多轮追问。

```
                        ┌──────────────┐
                    →   │ simple_answer│──→ END          简单问题秒回
                    │   └──────────────┘
┌─────────┐   ┌─────────────┐
│  START  │──→│  classify   │                            问题分类
└─────────┘   └──────┬──────┘
                     │ research / report
              ┌──────▼───────┐
              │ tool_calling │                  LLM 决定调哪些工具
              └──────┬───────┘
                     │ 有工具调用？
          ┌──────────┴──────────┐
          ↓ Yes                 ↓ No
   ┌──────────────┐             │
   │tool_executor │             │        Tavily / arXiv / 知识库 / 存报告
   └──────┬───────┘             │
          └──────────┬──────────┘
              ┌──────▼───────┐
              │   collect    │                  汇总工具结果
              └──────┬───────┘
              ┌──────▼───────┐
              │  researcher  │◄─────────┐        整理研究笔记
              └──────┬───────┘          │
              ┌──────▼───────┐          │
              │   analyst    │          │        深度分析
              └──────┬───────┘          │
              ┌──────▼───────┐          │
              │    writer    │          │        Pydantic 结构化报告
              └──────┬───────┘          │
              ┌──────▼───────┐          │
              │    review    │──未通过──┘        质量审查回环（≤3 轮）
              └──────┬───────┘
                     │ 通过
                    END
```

**这张图里有三个 Pro1 没有的东西**：条件边（分流）、回环边（重试）、工具节点（自动执行）。

---

## 二、状态设计

```python
class ResearchState(TypedDict):
    # 对话
    messages: Annotated[list, add_messages]
    # 研究过程
    question: str
    question_type: str          # simple / research / report
    search_results: str
    knowledge_results: str
    research_notes: str
    analysis: str
    report: str
    # 控制
    iteration: int
    review_passed: bool
    review_feedback: str
    next_agent: str
```

**知识点 ①：状态要分"数据"和"控制"两类**

- **数据字段**：`search_results`、`research_notes`、`analysis`、`report` —— 沿流水线逐级加工传递
- **控制字段**：`iteration`、`review_passed`、`question_type` —— 决定图的走向

混在一起写也能跑，但分开之后状态机读起来清晰得多。**`iteration` 尤其关键**——它是防止死循环的计数器。

---

## 三、条件路由

### 3.1 按问题类型分流

```python
def route_by_type(state: ResearchState) -> str:
    if state.get("question_type") == "simple":
        return "simple_answer"
    return "tool_calling"

builder.add_conditional_edges("classify", route_by_type)
```

**知识点 ②：条件边就是一个"返回下一个节点名"的函数**

返回值必须是已注册的节点名（或 `END`）。就这么简单——**路由逻辑是纯函数，输入 state，输出字符串**，因此非常好测试。

**业务价值**：闲聊和简单问答直接返回，不走检索流程。**省钱、省时间。** 复杂问题才启动完整流水线。这是生产环境的必备优化。

### 3.2 按工具调用结果分流

```python
def route_after_tools(state: ResearchState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "collect"
```

**知识点 ③：`tool_calls` 是模型的"我要用工具"信号**

开启 function calling 后，模型如果决定用工具，返回的消息里会带 `tool_calls` 字段；如果认为不需要工具，就直接返回文本。

检查这个字段就能判断"模型想不想调工具"——**这也是 ReAct 的现代化实现**：不再靠正则解析 `Action: xxx[yyy]`，而是靠模型原生的结构化输出。稳定性提升巨大。

### 3.3 质量审查回环

```python
def route_after_review(state: ResearchState) -> str:
    if state.get("review_passed", False):
        return END
    return "researcher"                    # ← 回到研究节点重做
```

```python
def review_node(state: ResearchState) -> dict:
    iteration = state.get("iteration", 0)
    if iteration >= 3:
        return {"review_passed": True}     # ← 强制退出，防死循环
    response = llm.invoke([...审查报告质量。回复 PASS 或 REVISE。...])
    passed = "PASS" in response.content.upper()[:10]
    return {"review_passed": passed, "iteration": iteration + 1}
```

**知识点 ④：回环边 = 图结构相较于链结构的最大优势**

`Chain`（LCEL 管道）只能单向流动，一旦某一步质量不达标，没有任何"退回去重做"的机制。**图可以。**

**知识点 ⑤：任何回环都必须有强制退出条件**

```python
if iteration >= 3:
    return {"review_passed": True}         # 3 次还没过也放行
```

没有这一行的后果：模型一直判 REVISE，图无限循环，token 烧到破产。

**这是所有"自我迭代"类 Agent 的铁律：迭代上限 + 强制降级。** 宁可输出一个不够完美的报告，也不能让系统卡死。

---

## 四、工具生态

```python
# tools/__init__.py
ALL_TOOLS = [search_web, search_arxiv, search_knowledge_base, save_report, list_reports]
```

```python
builder.add_node("tool_executor", ToolNode(ALL_TOOLS))   # ← 框架内置节点
```

| 工具 | 数据源 | 用途 |
|---|---|---|
| `search_web` | Tavily | 联网搜索，`search_depth="advanced"` |
| `search_arxiv` | arXiv API | 学术论文检索 |
| `search_knowledge_base` | Chroma | 本地知识库 RAG 检索 |
| `save_report` | 本地文件 | 报告保存为 Markdown |
| `list_reports` | 本地文件 | 列举已存报告 |

**知识点 ⑥：`ToolNode` 把整个工具执行环节变成了一个节点**

对比 `ch4` 手写的 `ToolExecutor`（注册 → 查表 → 调用 → 拼 Observation），这里只需要一行 `ToolNode(ALL_TOOLS)`。框架帮你做了：工具描述的 Schema 生成、模型 tool_calls 的解析、并发执行、结果封装成 `ToolMessage`。

**这正是学习路径的价值**——手写过一遍，才知道框架省掉的是什么，出问题时也才能定位。

**知识点 ⑦：`@tool` 装饰器把函数变成工具**

```python
@tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索引擎搜索网络信息。当需要获取最新资讯、技术动态或公开信息时使用。
    query: 搜索关键词
    """
    ...
```

**函数的 docstring 就是工具描述。** 第一句是"什么时候用"，后面是参数说明——这直接决定模型选不选得对（对应 `ch4` 的知识点 ⑤）。

**类型注解 `query: str` 会被自动转成 JSON Schema**，模型据此构造参数。

---

## 五、RAG 知识库（核心）

`tools/knowledge.py`

### 5.1 完整链路

```python
def build_vector_store(docs_dir: str, save_path: str):
    # 1. 加载文档
    loader = DirectoryLoader(docs_dir, glob="**/*.md",
                             loader_cls=TextLoader,
                             loader_kwargs={"encoding": "utf-8"})
    docs = loader.load()

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3. 向量化 + 入库
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=save_path)
    return vector_store
```

### 5.2 知识点

#### 知识点 ⑧：为什么要切分（chunking）

**因为 Embedding 模型和上下文窗口都有长度限制。** 一篇 10 页的文档没法整体向量化——向量化会把长文本压缩成一个固定维度的向量，文本越长，语义越模糊，检索就越不准。

切成 500 字的小块，每块语义聚焦，检索精度高。

#### 知识点 ⑨：`chunk_overlap` 为什么不是 0

```
块1: [0 -------- 500]
块2:            [450 -------- 950]     ← 重叠 50 字
```

如果边界正好把一个完整的语义单元（比如一段论证）切成两半，两块各自都不完整，检索时可能都匹配不上。**重叠区保证边界处的信息不会丢失。**

经验值：overlap 取 chunk_size 的 10%~20%。

#### 知识点 ⑩：`RecursiveCharacterTextSplitter` 为什么是"递归"

它按 `["\n\n", "\n", " ", ""]` 的顺序**逐级尝试切分**：
1. 先按段落（`\n\n`）切
2. 段落太大，再按行（`\n`）切
3. 还太大，按空格切
4. 最后才按字符硬切

**这样能最大程度保留语义完整性**——不会把一句话从中间劈开。

#### 知识点 ⑪：自定义 Embeddings 适配器

项目用的 Embedding 服务没有现成的 LangChain 集成，于是自己写了一个：

```python
class MaaSEmbeddings(Embeddings):        # ← 继承抽象基类
    def __init__(self, model: str, api_key: str, base_url: str):
        self.base_url = base_url.rstrip("/")     # 去除尾部斜杠，防 URL 拼接出双斜杠

    def _embed(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        response = requests.post(url,
            headers={"Authorization": f"Bearer {self.api_key}", ...},
            json={"model": self.model, "input": texts}, timeout=60)
        response.raise_for_status()
        return [item["embedding"] for item in response.json()["data"]]

    def embed_documents(self, texts) -> list[list[float]]:   # 必须实现
        return self._embed(texts) if texts else []

    def embed_query(self, text) -> list[float]:              # 必须实现
        return self._embed([text])[0]
```

**继承 `langchain_core.embeddings.Embeddings` 并实现两个方法，就能接入任意向量服务。** 这是 LangChain 抽象层的核心价值：换向量库/换 Embedding 模型，业务代码不用动。

注意 `embed_documents`（批量入库用）和 `embed_query`（单条查询用）**是分开的**——有些服务对这两者的处理方式不同（比如查询时要加特殊前缀指令）。

#### 知识点 ⑫：Top-K 与单例 Retriever

```python
return vectorstore.as_retriever(search_kwargs={"k": 4})     # 返回 4 条最相关
```

```python
_retriever = None
def _get_retriever():
    global _retriever
    if _retriever is None:                 # ← 单例，避免重复加载向量库
        _retriever = get_retriever()
    return _retriever
```

**K 的取值是个权衡**：太小召回不足（漏掉关键信息），太大引入噪声（稀释注意力、浪费 token）。4 是常见起点，需要按实际效果调。

**单例是因为向量库加载开销大**，每次检索都重建会严重拖慢速度。

#### 知识点 ⑬：RAG 到底解决了什么

README 里总结得很到位，这里展开：

| 问题 | RAG 怎么解决 |
|---|---|
| **幻觉** | 模型的回答基于检索到的真实文档，而不是凭空生成 |
| **知识过时** | 更新知识库即可，不需要重新训练模型 |
| **数据安全** | 私有文档不出本地，只把检索片段送给模型 |

**核心流程：Retrieve（检索）→ Analyze（分析）→ Generate（生成）**。

关键在第二步：**把检索到的文档和原始问题一起打包进 Prompt，让模型"基于给定材料"回答**，而不是凭记忆。所以 README 里那句提醒很重要——"此时 AI 的推理是基于你给的材料，而不是它凭空编造"。

**知识点 ⑭：RAG 的瓶颈在检索侧，不在生成侧**

这是实践中最大的认知纠偏。**换一个更强的 LLM 对 RAG 效果的提升，远不如优化切分策略、换更好的 Embedding 模型、调对 Top-K。**

因为**检索错了，后面再强的模型也只能基于错误的材料生成**——垃圾进，垃圾出。

---

## 六、结构化输出

`agents/writer.py`

```python
class ResearchReport(BaseModel):
    title: str = Field(description="报告标题")
    summary: str = Field(description="一句话摘要")
    key_findings: list[str] = Field(description="关键发现列表")
    detailed_analysis: str = Field(description="详细分析内容")
    conclusion: str = Field(description="结论")
    references: list[str] = Field(description="参考来源")

parser = PydanticOutputParser(pydantic_object=ResearchReport)
chain = prompt | llm | parser
report: ResearchReport = chain.invoke({..., "format_instructions": parser.get_format_instructions()})
```

**知识点 ⑮：`Field(description=...)` 是写给模型看的**

`description` 有两个用途：生成 JSON Schema 约束模型输出，同时在出错时帮助模型自我纠正。**描述写得越具体，输出越稳定。**

**知识点 ⑯：LCEL 管道 `prompt | llm | parser`**

`|` 运算符把三个组件串成一条链：

```
输入 dict → ChatPromptTemplate 渲染 → ChatOpenAI 生成 → PydanticOutputParser 解析 → ResearchReport 对象
```

每个组件的输入输出类型自动对齐。**这是 LangChain 最优雅的设计**，一行代码表达完整数据流。

**知识点 ⑰：解析失败必须降级**

```python
try:
    report: ResearchReport = chain.invoke({...})
    return {"report": md_report}
except Exception as e:
    # 解析失败 → 降级为纯文本生成
    simple_prompt = ChatPromptTemplate.from_messages([...])
    chain = simple_prompt | llm
    result = chain.invoke({...})
    return {"report": result.content}
```

**结构化输出不是 100% 可靠的。** 模型可能输出格式不合规的 JSON，尤其是复杂嵌套结构时。必须有降级路径——宁可要一份格式普通的文本报告，也不要整个流程崩掉。

**这个原则贯穿全项目**：Reflection 的"无需改进"兜底、ReAct 的 Finish 清洗、搜索失败的降级，都是同一套思路。

---

## 七、可观测性

### 7.1 装饰器统计节点耗时

`mylogger.py`

```python
def log_node_execution(node_name: str):
    def decorator(func):
        def wrapper(state):
            logger.info(f"[{node_name}] 开始执行")
            start = time.time()
            result = func(state)                      # ← 执行原始函数
            elapsed = time.time() - start
            logger.info(f"[{node_name}] 完成, 耗时 {elapsed:.2f}s")
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, str):
                        logger.info(f"[{node_name}] 输出 {key}: {len(value)} 字符")
            return result
        return wrapper
    return decorator
```

用法：

```python
@log_node_execution("researcher_node")
def researcher_node(state: ResearchState) -> dict:
    ...
```

**知识点 ⑱：三层闭包装饰器**

```
log_node_execution("名字")  →  接收名字，返回装饰器
decorator(func)             →  接收函数，返回包装函数
wrapper(state)              →  真正被调用的函数：记日志 → 掐表 → 调原函数 → 记日志
```

**这是 AOP（面向切面编程）的经典实现**——不改业务代码，横切进日志和监控。对比 Java Spring 的 `@Around` 通知，思路完全一致。

用在 Agent 上的价值：**一眼看出哪个节点最慢、哪一步输出异常**（比如 `search_results: 0 字符` 说明搜索失败了）。

### 7.2 Callback 追踪 LLM 调用

`mycallback.py`

```python
class ResearchCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs): ...
    def on_llm_end(self, serialized, prompts, **kwargs): ...
    def on_tool_start(self, serialized, input_str, **kwargs): ...
    def on_tool_end(self, output, **kwargs): ...
    def on_chain_error(self, error, **kwargs): ...
```

```python
llm = ChatOpenAI(..., callbacks=[ResearchCallbackHandler()])
```

**知识点 ⑲：装饰器 vs Callback 的分工**

| 手段 | 粒度 | 管什么 |
|---|---|---|
| 装饰器 | 节点级 | 我们自己写的函数：耗时、输出规模 |
| Callback | 框架内部 | LLM 请求/响应、工具调用等**框架接管的部分** |

**两者互补**：装饰器管"我的代码"，Callback 管"框架的内部行为"。合起来才是完整的可观测性。

**没有可观测性的 Agent 就是黑盒**——出问题时只能靠猜。这一点在生产环境是致命的。

---

## 八、运行

```bash
cd src/ch6/langgraph/Pro2

# 首次运行：构建向量库（会自动读取 data/knowledge/*.md）
python tools/knowledge.py

# 交互式
python main.py

# 单次查询
python main.py "LangGraph 和 AutoGen 该怎么选"

# 单元测试
python -m unittest discover tests
```

`.env` 需要 7 个配置项（LLM 三项、Embedding 三项、Tavily 一项），模板见 [环境准备文档](00-环境准备与快速开始.md#52-位置二srcch6langgraphpro2env)。

### 已知的小瑕疵

`route_after_tools` 判断为 No 时直接跳到 `collect`，此时 `search_results` 可能为空，researcher 拿不到外部信息。真实场景中可以在这里加一个"强制检索"的兜底分支。

---

## 九、这一章的收获（总结）

1. **图比链更贴合真实业务**：条件边让"简单问题秒回、复杂问题深挖"，回环边让质量可迭代——这是 Chain 结构做不到的。
2. **任何回环都必须有强制退出**（`iteration >= 3`），宁可输出不完美的结果也不能卡死。
3. **RAG 的瓶颈在检索侧**：切分策略、Embedding 质量、Top-K 参数，比换更强的 LLM 更影响最终答案。
4. **自定义 Embeddings 只需继承基类实现两个方法**，抽象层的价值就在于此。
5. **结构化输出必须配降级**——`PydanticOutputParser` 不是 100% 可靠的。
6. **可观测性 = 装饰器（管我的代码）+ Callback（管框架内部）**，缺了它就是黑盒。
7. **生产级 Agent 的四个标准**：状态可持久化、输出有 Schema、失败有降级、过程有日志。这四点决定了 Demo 能不能真正上线。
8. 到这里，ReAct、Plan-and-Solve、Multi-Agent 三种范式被整合进了同一个工程——**Agent 开发的完整方法论成型了**。
