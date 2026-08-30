from typing import List, Dict, Any, Optional
from llm import HelloAgentsLLM


class Memory:
    ...  # 类体在截图中被折叠（第6~50行），内容不可见；从下方注释可知其包含 add_record / get_last_execution / get_trajectory 等方法


# 初始提示词
INITIAL_PROMPT_TEMPLATE = """
...（第55~61行在截图中被折叠，模板内容不可见）
"""

# 反思提示词
REFLECT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下:
{tools}

请严格按照以下格式进行回应:

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一:
- `{{tool_name}}[{{tool_input}}]`:调用一个可用工具。
- `Finish[最终答案]`:当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

现在，请开始解决以下问题:
Question: {question}
History: {history}
"""

# 优化提示词
REFINE_PROMPT_TEMPLATE = """
...（第85~98行在截图中被折叠，模板内容不可见）
"""


# 反思智能体
class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        ...  # 方法体在截图中被折叠（第104~106行），内容不可见

    def run(self, task: str):
        print(f"\n---  开始处理任务  ---\n任务: {task}")

        # --- 1. 初始执行 ---
        print("\n---  正在进行初始尝试  ---")
        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)  # 调用llm获取初始代码
        self.memory.add_record("execution", initial_code)  # 记录初始执行结果

        # --- 2. 迭代循环:反思与优化 ---
        for i in range(self.max_iterations):
            print(f"\n---  第 {i+1}/{self.max_iterations} 轮迭代  ---")

            # a. 反思
            print("\n-> 正在进行反思...")
            last_code = self.memory.get_last_execution()  # 取最近一次的执行结果
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task, code=last_code
            )  # 格式化反思提示
            feedback = self._get_llm_response(reflect_prompt)  # 调用llm获取反思反馈
            self.memory.add_record("reflection", feedback)  # 记录反思反馈

            # b. 检查是否需要停止
            if "无需改进" in feedback:
                print("\n✅ 反思认为代码已无需改进，任务完成。")
                break

            # c. 优化
            print("\n-> 正在进行优化...")
            # 格式化优化提示
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task, last_code_attempt=last_code, feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()  # 跳出循环后，获取最终的执行结果
        print(f"\n---  任务完成  ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        ...  # 方法体在截图中被折叠（第149~152行），内容不可见


if __name__ == "__main__":
    # memory = Memory()
    # memory.add_record("execution", "def add(a, b): return a + b")
    # memory.add_record("reflection", "这是一条反馈,这个代码可以正确工作，但不好")

    # print(memory.get_last_execution())
    # print("=====" * 10)
    # print(memory.get_trajectory())
    # memory.add_record("execution", "def add2(a, b): return a + b")
    # memory.add_record("reflection", "这是一条反馈,这个代码更好了")

    # print("=====" * 10)
    # print(memory.get_trajectory())

    llm_client = HelloAgentsLLM()
    reflection_agent = ReflectionAgent(llm_client, 7)
    reflection_agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")
