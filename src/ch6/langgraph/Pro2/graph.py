"""LangGraph 工作流定义"""

from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from state import ResearchState
from agents.researcher import researcher_node
from agents.analyst import analyst_node
from agents.writer import writer_node
from tools import ALL_TOOLS  # 导入所有的工具

import os

from _env import load_project_env
from mylogger import log_node_execution

# 加载 .env 配置（自动定位本项目根目录）
load_project_env()


# ===== 节点函数 =====
@log_node_execution("classify_question")
def classify_question(state: ResearchState) -> dict:
    """问题分类"""
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    last_msg = state["messages"][-1]
    question = last_msg.content if hasattr(last_msg, "content") else state["question"]
    response = llm.invoke(
        [
            SystemMessage(content="""根据问题判断类型，只回复一个词：
                                - "simple" — 闲聊、常识、定义
                                - "research" — 需搜索分析的问题
                                - "report" — 需完整报告的深度问题"""),
            HumanMessage(content=f"问题: {question}"),
        ]
    )
    q_type = response.content.strip().lower()
    if q_type not in ("simple", "research", "report"):
        q_type = "research"
    return {"question": question, "question_type": q_type}


@log_node_execution("simple_answer")
def simple_answer(state: ResearchState) -> dict:
    """直接回答(simple类型的问题)"""
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    response = llm.invoke(state["messages"])  # messages是历史消息列表
    return {"messages": [response]}


@log_node_execution("tool_calling_node")
def tool_calling_node(state: ResearchState) -> dict:
    """工具调用: 调用工具来丰富信息，用于研究类型的问题"""
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    # 工具绑定
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    response = llm_with_tools.invoke(
        [
            SystemMessage(content="你是搜索专家。使用提供的工具搜索以下问题的信息。"),
            HumanMessage(content=state["question"]),
        ]
    )
    return {"messages": [response]}


@log_node_execution("collect_results")
def collect_results(state: ResearchState) -> dict:
    """收集工具搜索的结果"""
    results = []
    for msg in state["messages"]:
        # ToolMessage 有 content
        if hasattr(msg, "name") and msg.content:
            results.append(f"[{msg.name}] {msg.content[:500]}")
    combined = "\n\n".join(results)
    return {"search_results": combined}


@log_node_execution("review_node")
def review_node(state: ResearchState) -> dict:
    """质量审查"""
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    iteration = state.get("iteration", 0)
    if iteration >= 3:
        return {"review_passed": True}  # 3次以上通过审查
    response = llm.invoke(
        [
            SystemMessage(content="""审查报告质量。回复 PASS 或 REVISE。"""),
            HumanMessage(
                content=f"问题: {state.get('question', '')}\n\n报告:\n{state.get('report', '')[:2000]}"
            ),
        ]
    )
    passed = "PASS" in response.content.upper()[:10]
    return {"review_passed": passed, "iteration": iteration + 1}


@log_node_execution("route_by_type")
def route_by_type(state: ResearchState) -> str:
    """根据问题类型路由到不同的节点"""
    if state.get("question_type") == "simple":
        return "simple_answer"
    return "tool_calling"


@log_node_execution("route_after_tools")
def route_after_tools(state: ResearchState) -> str:
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_executor"
    return "collect"


@log_node_execution("route_after_review")
def route_after_review(state: ResearchState) -> str:
    """根据审查结果路由到不同的节点"""
    if state.get("review_passed", False):
        return END
    return "researcher"


# ===== 构建图 =====
def build_graph():
    builder = StateGraph(ResearchState)
    # 节点
    builder.add_node("classify", classify_question)
    builder.add_node("simple_answer", simple_answer)
    builder.add_node("tool_calling", tool_calling_node)
    builder.add_node("tool_executor", ToolNode(ALL_TOOLS))
    builder.add_node("collect", collect_results)
    builder.add_node("researcher", researcher_node)
    builder.add_node("analyst", analyst_node)
    builder.add_node("writer", writer_node)
    builder.add_node("review", review_node)

    # 边
    builder.add_edge(START, "classify")  # 从开始节点到分类节点
    builder.add_conditional_edges("classify", route_by_type)  # 分类节点根据问题类型路由
    builder.add_edge("simple_answer", END)  # 简单问题直接回答

    builder.add_conditional_edges(
        "tool_calling", route_after_tools
    )  # 工具调用节点根据是否有工具调用路由

    builder.add_edge("tool_executor", "collect")  # 工具执行节点到收集节点
    builder.add_edge("collect", "researcher")  # 收集节点到研究节点
    builder.add_edge("researcher", "analyst")  # 研究节点到分析师节点
    builder.add_edge("analyst", "writer")  # 分析师节点到写作者节点
    builder.add_edge("writer", "review")  # 写作者节点到审查节点
    builder.add_conditional_edges(
        "review", route_after_review
    )  # 审查节点根据是否通过路由

    # 编译
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)
    return graph
