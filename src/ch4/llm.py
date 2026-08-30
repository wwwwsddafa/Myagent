import os
from openai import OpenAI
from dotenv import load_dotenv  # 用于加载环境变量
from typing import List, Dict  # 用于类型提示

# import sys
# from dotenv import load_dotenv

# 获取当前文件的父目录的父目录（即项目根目录），并添加到sys.path，以后上线需要将agents模块发布，这里就不要了
# project_root = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# 加载 .env 文件中的环境变量
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", ".env")
load_dotenv(env_path)
class HelloAgentLLM:
    """
    定制的LLM客户端。
    它用于调用任何兼容openai接口的服务，**并默认使用流式响应。**
    """

    def __init__(
            self,
            model: str = None,
            api_key: str = None,
            base_url: str = None,
            timeout: int = None,
    ):
        """
        初始化客户端。优先使用传入参数，如果未提供，则从环境变量加载。
        """
        self.model = model or os.environ.get("MODEL_NAME")
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        timeout = timeout or int(os.environ.get("LLM_TIMEOUT", 60))

        if not all([self.model, api_key, base_url]):
            raise ValueError("模型id、api密钥和服务地址必须被提供或在.env文件中定义。")

        # 创建openai客户端实例
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """
        调用大语言模型进行思考，并返回其响应。
        """
        print(f"🔍 正在调用 [{self.model}] 模型...")
        try:
            # 调用openai API发送请求
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                stream=True,  # 开启流式响应
            )

            # 处理流式响应
            print("✅ 大语言模型响应成功:")
            collected_content = ""
            for chunk in response:  # 遍历流式响应的每个块(把回复拆成若干块，每次输出几个块，实现流式响应)
                if not chunk.choices:  # 如果块中没有选择，跳过
                    continue
                content = chunk.choices[0].delta.content or ""
                if content:
                    # 从块中提取内容，逐打印生成字符
                    print(content, end="", flush=True)
                    collected_content += content
            print()  # 在流式输出结束后换行
            return collected_content
        except Exception as e:
            print(f"❌ 调用LLM API时发生错误: {e}")
            return ""


# --- 客户端使用示例 ---
# 这个main只有在当前脚本作为主程序运行时才会执行
if __name__ == "__main__":
    try:
        llmClient = HelloAgentLLM()
        exampleMessages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that writes Python code.",
            },
            {"role": "user", "content": "写一个快速排序算法。"},
        ]
        print("----- 调用LLM -----")
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print("\n\n---- 完整模型响应 ----")
            print(responseText)
    except ValueError as e:
        print(e)