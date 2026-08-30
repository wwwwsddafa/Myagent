from serpapi import SerpApiClient
import os

from dotenv import load_dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", ".env")
load_dotenv(env_path)
def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误:SERPAPI_API_KEY 未在 .env 文件中配置。"
        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn",  # 语言代码
        }
        client = SerpApiClient(params)
        results = client.get_dict()

        # 智能解析:优先寻找最直接的答案
        if "answer_box_list" in results:  # 检查是否有Google的答案摘要框
            return "\n".join(results["answer_box_list"])
        if (
            "answer_box" in results and "answer" in results["answer_box"]
        ):  # 检查是否有答案摘要框
            return results["answer_box"]["answer"]
        if (
            "knowledge_graph" in results and "description" in results["knowledge_graph"]
        ):  # 检查是否有知识图谱描述
            return results["knowledge_graph"]["description"]

        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，退而求其次，返回前三个常规搜索结果的摘要，这种"智能解析"能为LLM提供质量更高的信息输入
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"
    except Exception as e:
        return f"搜索时发生错误: {e}"


if __name__ == "__main__":
    print(search("华为最新手机"))