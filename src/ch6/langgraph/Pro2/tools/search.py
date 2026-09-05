"""搜索工具"""

from langchain_core.tools import tool
from _env import load_project_env

load_project_env()


@tool
def search_web(query: str) -> str:
    """使用 Tavily 搜索引擎搜索网络信息。当需要获取最新资讯、技术动态或公开信息时使用。
    query: 搜索关键词
    """
    try:
        import os
        from tavily import TavilyClient

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "搜索失败: 未配置 TAVILY_API_KEY"
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(
            query=query, search_depth="advanced", max_results=5, include_answer=True
        )
        results = response.get("results", [])  # results节点是一个列表，有很多条目的数据
        if not results:
            return "未找到相关结果"
        output = ""
        # Tavily 的 AI 综合答案
        answer = response.get("answer")  # answer节点是一个字符串，包含综合答案
        if answer:
            output += f"综合答案:\n{answer}\n\n"
        # 搜索结果
        output += "相关搜索结果:\n"
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            score = r.get("score", 0)

            output += f"{i}. {title}\n"
            output += f"   URL: {url}\n"
            output += f"   相关性: {score:.2f}\n"
            output += f"   摘要: {content[:300]}\n\n"
        return output

    except Exception as e:
        return f"搜索失败: {e}"


# 装饰器 @tool 用于定义一个*工具函数*  ->给langgraph使用, 用于在图中调用，langgraph会注册这个工具( name,description,func )
@tool
def search_arxiv(query: str) -> str:
    """搜索 arXiv 学术论文。当需要查找学术论文、研究成果时使用。
    query: 论文搜索关键词"""
    try:
        import httpx

        url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 5,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, params=params)
        # 简单解析 XML 结果
        entries = resp.text.split("<entry>")[1:]  # 跳过头部
        results = []
        for entry in entries[:5]:
            title = entry.split("<title>")[1].split("</title>")[0].strip()
            summary = (
                entry.split("<summary>")[1].split("</summary>")[0].strip()
            )  # 论文的摘要
            results.append(f"- {title}\n  摘要: {summary[:200]}")
        return "\n\n".join(results) if results else "未找到相关论文"
    except Exception as e:
        return f"搜索失败: {e}"


if __name__ == "__main__":
    # result = search_arxiv.invoke({"query": "transformer"})
    # print(result)
    # print("**" * 60)
    # print(search_arxiv)
    # print(search_web)
    result = search_web.invoke({"query": "最新比特币价格"})
    print(result)
    print("**" * 60)