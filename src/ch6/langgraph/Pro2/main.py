"""智能研究助手 - 主程序"""

from graph import build_graph
import sys


def interactive_mode(graph):
    """交互式对话模式"""
    print("=" * 50)
    print("智能研究助手 v1.0")
    print("输入问题开始研究，输入 'quit' 退出")
    print("=" * 50)
    thread_id = "session-1"
    config = {"configurable": {"thread_id": thread_id}}
    while True:
        user_input = input("\n您: ").strip()  # 表示用户输入的问题
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q", "退出", "end"):
            print("再见!")
            break
        # 调用图
        result = graph.invoke({"messages": [("user", user_input)]}, config=config)
        # 输出结果
        print("\n" + "=" * 40)
        if result.get("report"):
            print("研究报告:")
            print(result["report"])
        else:
            # 简单回答:   messages
            for msg in result["messages"]:
                if hasattr(msg, "content") and msg.content:
                    # 只输出最后的 AI 回复
                    pass
            last_ai = None
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                    last_ai = msg.content
            if last_ai:
                print(f"AI: {last_ai}")
        print("=" * 40)


def single_query(graph, question: str):
    """单次查询模式"""
    config = {"configurable": {"thread_id": "single"}}  # graph以单线程模式运行
    result = graph.invoke(
        {"messages": [("user", question)]}, config=config
    )  #  {"messages":[("user",question)]} 就是图中传递的state状态
    if result.get("report"):
        print(result.get("report"))
    else:
        for msg in result["messages"]:
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                print(msg.content)


# 允许通过主程序运行传参数来完成指定任务，也可以通过交互式方式连续对话来完成任务
if __name__ == "__main__":
    graph = build_graph()
    if len(sys.argv) > 1:  # python main.py transformers是什么 RAG是什么
        query = " ".join(sys.argv[1:])
        single_query(graph, query)
    else:
        interactive_mode(graph)
