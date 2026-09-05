"""
智能搜索助手 - 基于 LangGraph + Tavily API 的真实搜索系统
1. 理解用户查询
2. 使用Tavily API真实搜索信息
3. 生成基于搜索结果的回答
"""

import asyncio
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from tavily import TavilyClient

from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv

# 加载环境变量：从当前文件所在目录向上逐层查找 data/.env 或 .env
def _load_env():
    """从当前文件向上逐层查找 data/.env 或 .env 并加载（兼容任意运行目录）"""
    current_file = os.path.abspath(__file__)
    dir_to_check = os.path.dirname(current_file)

    while True:
        candidates = [
            os.path.join(dir_to_check, "data", ".env"),
            os.path.join(dir_to_check, ".env"),
        ]
        for path in candidates:
            if os.path.exists(path):
                load_dotenv(path, override=True)
                print(f"✅ 已加载环境变量: {path}")
                return True
        parent = os.path.dirname(dir_to_check)
        if parent == dir_to_check:  # 已到达文件系统根
            break
        dir_to_check = parent

    load_dotenv()
    print("⚠️ 未找到 .env 文件，尝试从系统环境变量读取")
    return False

_load_env()


# 定义状态结构
class SearchState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话消息列表，add_messages是Reducer ：是一个函数，定义"如何把新值合并到旧值中"。
    user_query: str  # 用户查询
    search_query: str  # 优化后的搜索查询
    search_results: str  # Tavily搜索结果
    final_answer: str  # 最终答案
    step: str  # 当前步骤


# 初始化模型和Tavily客户端
api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
model = os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_NAME", "gpt-4o-mini")

if not api_key:
    raise ValueError("未找到 API Key，请设置 LLM_API_KEY 或 OPENAI_API_KEY 环境变量")

llm = ChatOpenAI(
    model=model,
    api_key=api_key,
    base_url=base_url,
    temperature=0.7,
)

# 初始化Tavily客户端
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def understand_query_node(state: SearchState) -> SearchState:
    """步骤1: 理解用户查询并生成搜索关键词"""
    # 获取最新的用户消息
    user_message = ""
    for msg in reversed(state["messages"]):  # reversed(): 从最后(最新)消息开始遍历
        if isinstance(msg, HumanMessage):  # 判断是否是用户消息
            user_message = msg.content
            break
    understand_prompt = f"""分析用户的查询: {user_message}

        请完成两个任务:
        1. 简洁地总结用户想要了解什么
        2. 生成适合搜索的关键词 (中英文均可, 要精准)

        请严格按照以下格式输出:
        理解: [用户需求总结]
        搜索词: [精准搜索关键词]"""
    response = llm.invoke([SystemMessage(content=understand_prompt)])
    # 提取搜索关键词
    response_text = response.content
    search_query = user_message  # 默认使用原始查询
    if "搜索词:" in response_text:
        search_query = response_text.split("搜索词:")[1].strip()
    elif "搜索关键词:" in response_text:
        search_query = response_text.split("搜索关键词:")[1].strip()
    return {
        "user_query": response.content,  # 系统显示用
        "search_query": search_query,  # 优化后的搜索查询
        "step": "understood",  # 当前步骤是"understood", 表示理解了用户需求
        "messages": [AIMessage(content=f"我理解您的需求: {response.content}")],
    }


def tavily_search_node(state: SearchState) -> SearchState:
    """步骤2: 使用Tavily API进行真实搜索"""
    search_query = state["search_query"]
    try:

        print(f"🔍 正在使用的搜索词: {search_query}")
        # 调用Tavily搜索API
        response = tavily_client.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,  # 包含综合答案
            include_raw_content=False,  # 不包含原始内容
            max_results=5,  # 最多返回5个结果
        )

        # 处理搜索结果
        search_results = ""
        # 优先使用Tavily的综合答案
        if response.get("answer"):
            search_results = f"综合答案:\n{response['answer']}\n\n"
        # 添加具体的搜索结果
        if response.get("results"):
            search_results += "相关信息:\n"
            for i, result in enumerate(response["results"][:3], 1):
                title = result.get("title", "")
                content = result.get("content", "")
                url = result.get("url", "")
                search_results += f"{i}. {title}\n{content}\n来源: {url}\n\n"
        if not search_results:
            search_results = "抱歉, 没有找到相关信息。"
        return {
            "search_results": search_results,
            "step": "searched",
            "messages": [
                AIMessage(content="✅ 搜索完成! 找到了相关信息, 正在为您整理答案...")
            ],
        }
    except Exception as e:
        error_msg = f"搜索时发生错误: {str(e)}"
        print(f"❌ {error_msg}")
        return {
            "search_results": f"搜索失败: {error_msg}",
            "step": "search_failed",
            "messages": [
                AIMessage(content="❌ 搜索遇到问题, 我将基于已有知识为您回答")
            ],
        }


def generate_answer_node(state: SearchState) -> SearchState:
    """步骤3: 基于搜索结果生成最终答案"""
    # 检查是否有搜索结果
    if state["step"] == "search_failed":
        # 如果搜索失败, 基于LLM知识回答
        fallback_prompt = f"""搜索API暂时不可用, 请基于您的知识回答用户的问题:

        用户问题: {state['user_query']}

        请提供一个有用的回答, 并说明这是基于已有知识的回答。"""
        response = llm.invoke([SystemMessage(content=fallback_prompt)])
        return {
            "final_answer": response.content,
            "step": "completed",
            "messages": [AIMessage(content=response.content)],
        }

    # 基于Tavily搜索结果用llm重新规范生成答案
    answer_prompt = f"""基于以下搜索结果为用户提供完整、准确的答案:

        用户问题: {state['user_query']}

        搜索结果:
        {state['search_results']}

        请要求:
        1. 综合搜索结果, 提供准确、有用的回答
        2. 如果涉及技术问题, 提供具体的解决方案或代码
        3. 引用重要信息的来源
        4. 回答要结构清晰, 易于理解
        5. 如果搜索结果不存在, 请说明并提供依我知识建议"""
    response = llm.invoke([SystemMessage(content=answer_prompt)])
    return {
        "final_answer": response.content,
        "step": "completed",
        "messages": [AIMessage(content=response.content)],
    }


# 构建搜索工作流
def create_search_assistant():
    workflow = StateGraph(SearchState)
    # 添加三个节点
    workflow.add_node("understand", understand_query_node)
    workflow.add_node("search", tavily_search_node)
    workflow.add_node("answer", generate_answer_node)
    # 设置线性流程: 这里没有条件判断
    workflow.add_edge(START, "understand")
    workflow.add_edge("understand", "search")
    workflow.add_edge("search", "answer")
    workflow.add_edge("answer", END)
    # 编程图
    memory = InMemorySaver()  # 内存保存器, 用于存储状态，让 AI 在"多轮对话间"有记忆
    app = workflow.compile(checkpointer=memory)  # 编译工作流, 指定状态保存器
    return app


async def main():
    """主函数: 运行智能搜索助手"""
    # 检查API密钥
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ 错误: 请在.env文件中配置TAVILY_API_KEY")
        return
    app = create_search_assistant()
    print("🔍 智能搜索助手启动!")
    print("我会使用Tavily API为您搜索最新、最准确的信息")
    print("支持查询类型: 新闻、技术、知识问答等")
    print("输入 'quit' 退出\n")
    session_count = 0  # 会话计数器
    while True:
        user_input = input("👉 您想了解什么? ").strip()  # 获取用户输入, 移除首尾空格
        if user_input.lower() in ("退出", "q", "quit", "exit"):
            print("感谢使用! 再见! 👋")
            break
        if not user_input:
            continue
        session_count += 1
        config = {"configurable": {"thread_id": f"search-session-{session_count}"}}
        # 初始化状态
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "user_query": "",
            "search_query": "",
            "search_results": "",
            "final_answer": "",
            "step": "start",
        }
        try:
            print("\n" + "=" * 60)
            # 执行工作流
            async for output in app.astream(initial_state, config=config):
                for node_name, node_output in output.items():
                    if "messages" in node_output and node_output["messages"]:
                        latest_message = node_output["messages"][-1]
                        # 取最后一条消息(最新消息)
                        if isinstance(latest_message, AIMessage):
                            if node_name == "understand":
                                print(f"🤔 理解阶段: {latest_message.content}")
                            elif node_name == "search":
                                print(f"🔍 搜索阶段: {latest_message.content}")
                            elif node_name == "answer":
                                print(f"💡 最终回答:\n{latest_message.content}")

            print("\n" + "=" * 60 + "\n")
        except Exception as e:
            print(f"❌ 发生错误: {e}")
            print("请重新输入您的问题。\n")


if __name__ == "__main__":
    # print(llm)

    # 测试 1 understand_query_node(state: SearchState)
    # state = {
    #     "messages": [HumanMessage(content="你好, 我想知道transformer是什么")],
    #     "user_query": "",
    #     "search_query": "",
    #     "search_results": "",
    #     "final_answer": "",
    #     "step": "understood",
    # }
    # print(understand_query_node(state))

    # 测试2 tavily_search_node(state: SearchState)
    # state = {
    #     "messages": [HumanMessage(content="你好, 我想知道transformer是什么")],
    #     "user_query": "",
    #     "search_query": "Transformer 架构原理, Transformer 模型详解, What is Transformer architecture",
    #     "search_results": "",
    #     "final_answer": "",
    #     "step": "understood",
    # }
    # print(tavily_search_node(state))

    # 测试3 generate_answer_node(state: SearchState)
    # state = {
    #     "messages": [HumanMessage(content="你好, 我想知道transformer是什么")],
    #     "user_query": "transformer是什么",
    #     "search_query": "Transformer 架构原理, Transformer 模型详解, What is Transformer architecture",
    #     "search_results": "综合答案: \nTransformer 是一种基于 self-attention 模型的序列到序列模型, 它在 NLP 领域中取得了显著的成功。",
    #     "final_answer": "",
    #     "step": "understood",
    # }
    # print(generate_answer_node(state))

    # 以异步方式运行主函数
    asyncio.run(main())