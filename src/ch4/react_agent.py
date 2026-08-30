from llm import HelloAgentLLM
from executor import ToolExecutor
from search import search

REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
    - `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
    - `Finish[最终答案]`:当你认为已经获得最终答案时。
当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题：
Question: {question}
History: {history}
"""


class ReActAgent:
    def __init__(
            self,
            llm_client: HelloAgentLLM,
            tool_executor: ToolExecutor,
            max_steps: int = 5,
    ):
        self.llm_client = llm_client  # 初始化大语言模型客户端
        self.tool_executor = tool_executor  # 初始化工具执行器
        self.max_steps = max_steps  # 最大迭代次数
        self.history = []  # 初始化空的历史记录列表

    # 解析LLM输出，此方法为私有方法，仅用于内部使用，它负责从LLM的响应中分离出Thought和Action两个部分。
    def _parse_output(self, text: str):
        """解析LLM的输出，提取Thought和Action。"""
        # 导入了 Python 的正则表达式模块 re
        import re

        # Thought: 匹配到 Action: 或文本末尾
        # \s* 匹配任意数量的空白字符
        # .*? 匹配任意数量的任意字符，非贪婪匹配
        # (?=nAction:) 正向断言，后面必须是 \nAction: 或文本末尾，但这个东西不包含在当前匹配结果中
        # 所以它针对的结构是：
        # # Thought: 我需要先查询北京的天气。\n然后根据天气情况判断是否适合跑步。我喜欢，不喜欢交谈\n
        # # Action: search_weather[北京]
        thought_match = re.search(
            r"^Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL | re.MULTILINE
        )  # DOTALL 标志允许匹配换行符
        # Action: 匹配到文本末尾
        action_match = re.search(r"^Action:\s*(.*?)$", text, re.DOTALL | re.MULTILINE)
        # 为什么它这个 group(1) 是 1，这个称为捕获组？
        # 因为正则表达式中只有一对括号,而括号就是用来定义"捕获组"的
        # (.*?)         → 第1对括号! 捕获就是这个括号的内容
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    # 解析Action字符串，此方法为私有方法，仅用于内部使用，它负责从Action字符串中提取工具名称和输入。
    def _parse_action(self, action_text: str):
        """解析Action字符串，提取工具名称和输入。
        action_text: 例如: calculator[123 + 456]
                      show[]
                      search[北京\n天气很好]
        """
        import re

        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        if match:
#             (\w+)	第1捕获组: 工具名(字母/数字/下划线)
#             (.*)	第2捕获组: 方括号内的任意内容(包括换行)
            return match.group(1), match.group(2)
        return None, None

    # 类的入口方法
    def run(self, question: str):
        """运行ReAct智能体来回答一个问题。"""
        import re

        self.history = []  # 每次运行时重置历史记录
        current_step = 0

        # 循环次数限定为最大迭代次数
        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            tools_desc = self.tool_executor.getAvailableTools()

            history_str = "\n".join(self.history)

            # 1. 格式化提示词
            # 初始化系统提示词
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc, question=question, history=history_str
            )
            # 2. 调用LLM进行思考
            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误:LLM未能返回有效响应。")
                break
            # 3. 如果LLM有响应，则解析LLM的输出
            thought, action = self._parse_output(response_text)
            if thought:
                print(f"思考: {thought}")
            if not action:
                print("⚠警告: 未能解析出有效的Action,流程终止。")
                break
            # 4. 根据解析出的Action来执行具体的操作
            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束循环。注意：关于为什么是"Finish",请看上面的系统提示词，它已经规范了LLM的输出格式。
                # - `Finish[最终答案]`:当你认为已经获得最终答案时。
                # 例如：Finish[12*34]
                matchOK = re.match(r"Finish\[(.*)\]", action, re.DOTALL)
                if matchOK:
                    # 如果匹配成功，则提取最终答案并结束循环
                    final_answer = matchOK.group(1)  # (.*)捕获到的第一组的内容
                    print(f"✅最终答案:{final_answer}")
                    return final_answer
                else:
                    # 如果 LLM 没有遵循标准格式输出，做兜底处理
                    print(
                        f"⚠ 警告: 虽然包含 Finish,但格式不标准,提取失败。原始输出为: {action}"
                    )
                    # 为了程序不崩溃，你可以直接把 action 当作答案返回，或者抛出具体错误
                    final_answer = action.replace("Finish", "").strip()
                    # 尝试简单粗暴清洗一下
                    print(f"🔧 强制返回清洗后的结果: {final_answer}")
                    return final_answer
            else:
                # 不是"Finish"指令，则Action中就是工具调用指令
                tool_name, tool_input = self._parse_action(action)
                if not tool_name or not tool_input:
                    # 无效Action格式和输入，则继续下一次循环
                    continue
                # 有效Action格式和输入
                print(f"🔧 行动: {tool_name}[{tool_input}]")
                tool_function = self.tool_executor.getTool(tool_name)
                if not tool_function:
                    print(f"⚠警告: 未找到工具 {tool_name}.")
                    observation = f"警告: 未找到工具 {tool_name}."
                else:
                    observation = tool_function(tool_input)  # 调用工具函数，并获取观察结果
                    print(f"🔍观察: {observation}")

                # 更新历史记录
                self.history.append(f"Action: {action}")
                self.history.append(f"Observation: {observation}")

        # 注意：这里代表循环结束，说明任务完成或最大迭代次数已到。
        print(f"⚠已达到最大迭代次数({self.max_steps}),流程终止。")
        return None


if __name__ == "__main__":
    llm = HelloAgentLLM()
    tool_executor = ToolExecutor()
    search_desc = "这是一个网页搜索引擎，当你需要回答关于时事，事实以及在你的知识库中找不到的信息时，应使用此工具"
    tool_executor.register_tool("Search", search_desc, search)

    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)

    # 给一个问题
    question = "马斯克的Optimus机器人目前的量产状态?它的中国供应链公司有哪些?"
    agent.run(question)