PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划，```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""

# 假定  HelloAgentsLLM 类已经定义好
import ast  # ast是Python的标准库，用于安全地执行字符串，将其转换为Python表达式


class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)
        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]
        print("--- 正在生成计划 ---")
        # 使用流式输出来获取完整的计划
        response_text = self.llm_client.think(messages=messages) or ""
        print(f"✅ 计划已生成:\n{response_text}")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            # 上面response_text的格式是: ```python
            #                     ["步骤1", "步骤2", "步骤3", ...]
            #                 ```
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)  # 此时它将 字符串转换成了一个列表对象
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


import os
import ast
from dotenv import load_dotenv
from typing import List, Dict
from llm import HelloAgentLLM

load_dotenv()

if __name__ == "__main__":
    try:
        llm_client = HelloAgentLLM()
        
        # 测试完整的 Plan-And-Solve 流程
        from plan_and_solve_PlanAndSolveAgent import PlanAndSolveAgent
        agent = PlanAndSolveAgent(llm_client)
        
        question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果?"
        
        # 运行完整的规划+执行流程，获取最终答案
        final_answer = agent.run(question)
        
        print(f"\n{'='*50}")
        print(f"最终结果: {final_answer}")
        print(f"{'='*50}")

    except ValueError as e:
        print(e)