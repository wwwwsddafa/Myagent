"""分析 Agent：负责深度分析"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from state import ResearchState
from _env import load_project_env
import os

# 加载 .env 配置（自动定位本项目根目录）
load_project_env()

from mylogger import log_node_execution
from mycallback import ResearchCallbackHandler


@log_node_execution("analyst_node")
def analyst_node(state: ResearchState) -> dict:
    """分析节点：深度分析研究结果"""
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        callbacks=[ResearchCallbackHandler()],
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个分析专家。基于研究笔记的内容，进行深度分析：
                    1. 提炼核心观点
                    2. 对比不同来源的信息
                    3. 发现趋势和模式
                    4. 给出专业见解

                    输出结构化的分析报告。""",
            ),
            (
                "human",
                """问题: {question}

                    研究笔记:
                    {notes}

                    请进行深度分析。""",
            ),
        ]
    )
    # LCEL 管道：prompt | llm | parser 链式调用(创建了一个处理链 (Chain)，将"提示词模板"和"大模型"连接起来。)
    chain = prompt | llm
    # 执行管道
    #                      第一个参数是一个{}, 里面有上面prompt中的需要的变量
    response = chain.invoke(
        {
            "question": state["question"],
            "notes": state.get("research_notes", "无研究笔记"),
        }
    )
    return {"analysis": response.content}


# ============================================================
# 测试analyst_node() 函数代码
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("开始测试 analyst_node()")
    print("=" * 70)
    # --------------------------------------------------------
    # 1. 构造测试数据
    # --------------------------------------------------------
    test_state: ResearchState = {
        "messages": [],
        "question": (
            "目前人工智能 Agent 的主要架构模式有哪些？" "它们分别适用于什么场景？"
        ),
        "question_type": "research",
        "research_notes": """
                # 研究笔记：人工智能 Agent 架构模式分析

                ## 1. ReAct 架构

                ReAct（Reasoning + Acting）通过思考、行动和观察的循环，
                让大语言模型在推理过程中动态调用外部工具。

                适合需要实时获取外部信息、调用搜索引擎、API 或其他工具，
                并且任务执行路径不固定的场景。

                ## 2. Plan-and-Execute 架构

                Plan-and-Execute 首先由规划器生成完整的任务计划，
                然后按照计划逐步执行。

                适合长周期、多步骤的复杂任务，例如项目规划、
                复杂数据分析和长文档生成。

                ## 3. Multi-Agent 架构

                Multi-Agent 将复杂任务拆分给多个具有不同角色和专业能力的 Agent，
                多个 Agent 通过协作共同完成任务。

                适合软件开发、复杂研究、跨领域问题以及需要专业分工的任务。

                ## 4. Workflow / Graph Agent

                Workflow 或 Graph Agent 使用预定义的工作流或状态图控制执行过程。

                例如 LangGraph 可以通过节点、边和 State 管理 Agent 的执行过程。

                这种方式具有较强的可控性和稳定性，
                适合企业业务流程、审批流程、客服和自动化任务。

                ## 5. 初步结论

                不同 Agent 架构并不存在绝对的优劣。

                ReAct 更强调动态决策和工具调用；
                Plan-and-Execute 更适合复杂的多步骤任务；
                Multi-Agent 更适合需要专业分工的复杂任务；
                Workflow / Graph 更强调确定性、可控性和稳定性。

                实际项目中也可以组合使用多种架构。
                """,
        # analyst_node() 当前不会读取这个字段，
        # 这里只是为了保持 ResearchState 的完整结构。
        "analysis": "",
        "sources": [],
        "search_results": [],
        "report": "",
    }
    # --------------------------------------------------------
    # 2. 打印测试输入
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("测试输入")
    print("=" * 70)
    print("\n研究问题：")
    print(test_state["question"])
    print("\n问题类型：")
    print(test_state["question_type"])
    print("\n研究笔记：")
    print(test_state["research_notes"])
    # --------------------------------------------------------
    # 3. 调用 analyst_node()
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("开始执行 analyst_node()")
    print("=" * 70)
    try:
        result = analyst_node(test_state)
        print("\n✅ analyst_node() 执行成功")
    except Exception as e:
        print("\n❌ analyst_node() 执行失败")
        print(f"异常类型：{type(e).__name__}")
        print(f"异常信息：{e}")
        raise
    # --------------------------------------------------------
    # 4. 检查返回结果类型
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("检查返回结果")
    print("=" * 70)
    print("\n返回结果类型：")
    print(type(result))
    assert isinstance(result, dict), f"返回结果应该是 dict，实际是 {type(result)}"
    print("✅ 返回结果类型正确")
    # --------------------------------------------------------
    # 6. 检查 analysis
    # --------------------------------------------------------
    analysis = result["analysis"]
    print("✅ ", analysis)
