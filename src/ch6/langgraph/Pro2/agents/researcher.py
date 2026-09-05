"""研究型 Agent：负责 Web Search + RAG Knowledge Base + LLM 的组合研究代理"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from state import ResearchState
from _env import load_project_env
import os

# 加载 .env 配置（自动定位本项目根目录）
load_project_env()

from mylogger import log_node_execution
from mycallback import ResearchCallbackHandler


@log_node_execution("researcher_node")
def researcher_node(state: ResearchState) -> dict:
    """研究节点：搜索并整理信息"""
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
    # 如果之前有搜索结果或知识库结果，综合整理
    existing_info = ""
    if state.get("search_results"):
        existing_info += f"网络搜索结果:\n{state['search_results']}\n\n"
    if state.get("knowledge_results"):
        existing_info += f"知识库检索结果:\n{state['knowledge_results']}\n\n"
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个研究专家。你的任务是：
                    1. 分析用户的问题
                    2. 整理已有信息
                    3. 总结关键发现
                    4. 指出信息缺口（如果有的话）

                    请用中文输出研究笔记，包含：
                    - 关键发现（要点列表）
                    - 信息来源评估
                    - 需要补充的信息（如有）""",
            ),
            (
                "human",
                """问题: {question}

                    已有信息:
                    {info}

                    请整理研究笔记。""",
            ),
        ]
    )
    # LCEL 管道：prompt | llm | parser 链式调用
    chain = prompt | llm
    # 执行管道
    #                      第一个参数是一个{}, 里面有上面prompt中的需要的变量
    response = chain.invoke({"question": state["question"], "info": existing_info})
    return {"research_notes": response.content}


"""
测试 researcher_node() 函数

功能：
1. 构造 ResearchState
2. 调用 researcher_node()
3. 输出研究笔记
"""


if __name__ == "__main__":
    """测试 researcher_node"""
    # ============================================================
    # 1. 构造测试状态
    # ============================================================
    state = {
        # 对话历史
        "messages": [],
        # 原始问题
        "question": "目前人工智能 Agent 的主要架构模式有哪些？它们分别适用于什么场景？",
        # 问题类型
        "question_type": "research",
        # 网络搜索结果
        "search_results": """
                            搜索结果 1：
                            ReAct 是一种经典的 Agent 架构，通过 Thought、Action、Observation
                            的循环让大语言模型进行推理并调用外部工具。
                            搜索结果 2：
                            当前 Agent 架构逐渐从单 Agent 发展到 Multi-Agent。
                            Multi-Agent 系统可以通过多个具有不同职责的 Agent 协同完成复杂任务。
                            搜索结果 3：
                            LangGraph 等框架支持基于状态图构建复杂 Agent，
                            可以通过节点、边和状态管理 Agent 的执行流程。
                            """,
        # 知识库检索结果
        "knowledge_results": """
                            知识库资料：
                            1. ReAct：
                            将推理和行动结合起来，Agent 根据当前状态决定下一步行动。
                            2. Plan-and-Execute：
                            Agent 首先生成完整计划，然后逐步执行计划中的任务。
                            3. Multi-Agent：
                            将复杂任务拆分给多个具有不同角色的 Agent，
                            通过 Agent 之间的协作完成任务。
                            4. Workflow / Graph Agent：
                            使用预定义的工作流或状态图控制 Agent 的执行过程，
                            适合流程比较明确、需要较强可控性的业务场景。
                            """,
        # 研究笔记
        "research_notes": "",
        # 分析结果
        "analysis": "",
        # 最终报告
        "report": "",
        # 当前迭代次数
        "iteration": 0,
        # 质量审查
        "review_passed": False,
        # 审查反馈
        "review_feedback": "",
        # 下一个 Agent
        "next_agent": "",
    }

    # ============================================================
    # 2. 调用 researcher_node
    # ============================================================
    print("=" * 70)
    print("开始测试 researcher_node()")
    print("=" * 70)
    print(f"\n研究问题：{state['question']}")
    print(f"问题类型：{state['question_type']}")
    try:
        result = researcher_node(state)
        # ========================================================
        # 3. 检查返回结果
        # ========================================================
        print("\n" + "=" * 70)
        print("researcher_node() 执行成功")
        print("=" * 70)
        print("\n返回结果类型：")
        print(type(result))
        print("\n返回结果：")
        print(result)
        print("*" * 70)
        # ========================================================
        # 4. 检查 research_notes
        # ========================================================
        if "research_notes" in result:
            print("\n" + "=" * 70)
            print("研究笔记")
            print("=" * 70)
            print(result["research_notes"])
        else:
            print("\n❌ 错误：返回结果中没有 research_notes")
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ researcher_node() 执行失败")
        print("=" * 70)
        print(f"异常类型：{type(e).__name__}")
        print(f"异常信息：{e}")
        # 开发阶段建议打印完整 traceback
        import traceback