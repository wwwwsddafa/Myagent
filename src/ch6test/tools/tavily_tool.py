import os
from tavily import TavilyClient
from dotenv import load_dotenv

# 加载环境变量（支持从项目根目录的 data/.env 加载）
_env_loaded = False


def _ensure_env():
    global _env_loaded
    if not _env_loaded:
        # 从当前文件所在位置向上查找 data/.env
        current = os.path.abspath(__file__)
        possible_paths = []
        # 向上遍历3层目录查找 data/.env
        for _ in range(4):
            current = os.path.dirname(current)
            possible_paths.append(os.path.join(current, "data", ".env"))
        # 也尝试当前工作目录
        possible_paths.append(os.path.join(os.getcwd(), "data", ".env"))

        for path in possible_paths:
            if os.path.exists(path):
                load_dotenv(path)
                _env_loaded = True
                return
        _env_loaded = True


def _get_client() -> TavilyClient:
    """获取 Tavily 客户端实例"""
    _ensure_env()
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("未配置 TAVILY_API_KEY 环境变量")
    return TavilyClient(api_key=api_key)


def search_attractions(city: str, weather: str = "", days: int = 3) -> str:
    """
    搜索指定城市的旅游景点推荐。

    Args:
        city: 城市名称
        weather: 当前天气状况，用于推荐适合的景点类型
        days: 旅行天数

    Returns:
        景点推荐信息
    """
    try:
        tavily = _get_client()

        query = f"{city} 最值得去的旅游景点推荐 必去景点"
        if weather:
            query += f" {weather}天气适合去的景点"

        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=5,
        )

        if response.get("answer"):
            return f"【{city}景点推荐】\n{response['answer']}"

        results = []
        for r in response.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:200]}...")

        if not results:
            return f"未找到{city}的景点推荐信息。"

        return f"【{city}景点推荐】\n" + "\n".join(results)

    except ValueError as e:
        return f"景点搜索配置错误: {e}"
    except Exception as e:
        return f"景点搜索失败: {e}"


def search_restaurants(city: str, preference: str = "") -> str:
    """
    搜索指定城市的美食餐厅推荐。

    Args:
        city: 城市名称
        preference: 饮食偏好，如"川菜"、"火锅"、"清淡"等

    Returns:
        餐厅推荐信息
    """
    try:
        tavily = _get_client()

        query = f"{city} 美食推荐 必吃餐厅"
        if preference:
            query += f" {preference}"

        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=5,
        )

        if response.get("answer"):
            return f"【{city}美食推荐】\n{response['answer']}"

        results = []
        for r in response.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:200]}...")

        if not results:
            return f"未找到{city}的美食推荐信息。"

        return f"【{city}美食推荐】\n" + "\n".join(results)

    except ValueError as e:
        return f"美食搜索配置错误: {e}"
    except Exception as e:
        return f"美食搜索失败: {e}"


def search_travel_tips(city: str) -> str:
    """
    搜索旅行小贴士和注意事项。

    Args:
        city: 城市名称

    Returns:
        旅行贴士信息
    """
    try:
        tavily = _get_client()

        query = f"{city} 旅游攻略 注意事项 实用贴士"

        response = tavily.search(
            query=query,
            search_depth="basic",
            include_answer=True,
            max_results=3,
        )

        if response.get("answer"):
            return f"【{city}旅行贴士】\n{response['answer']}"

        results = []
        for r in response.get("results", []):
            results.append(f"- {r['title']}: {r['content'][:200]}...")

        if not results:
            return f"未找到{city}的旅行贴士。"

        return f"【{city}旅行贴士】\n" + "\n".join(results)

    except ValueError as e:
        return f"贴士搜索配置错误: {e}"
    except Exception as e:
        return f"贴士搜索失败: {e}"


if __name__ == "__main__":
    print(search_attractions("北京", weather="晴天"))