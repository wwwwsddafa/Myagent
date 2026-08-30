EXECUTOR_PROMPT_TEMPLATE = """
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
"""


class Executor:
    def __init__(self, llm_client):
        """
        初始化执行器。
        
        Args:
            llm_client: 大语言模型客户端实例，用于调用 AI 模型
        """
        self.llm_client = llm_client

    def execute(self, question: str, plan: list[str]) -> str:
        """
        根据计划，逐步执行并解决问题。
        
        Args:
            question: 原始问题
            plan: 由规划器生成的步骤列表
            
        Returns:
            str: 最终答案
        """
        # 检查计划是否为空
        if not plan:
            print("\n--- 执行终止 --- \n计划为空，无法执行。")
            return ""
        
        history = ""  # 用于存储历史步骤和结果的字符串
        response_text = ""  # 初始化响应变量，防止空计划时未定义
        
        print("\n--- 正在执行计划 ---")
        
        for i, step in enumerate(plan):
            print(f"\n-> 正在执行步骤 {i+1}/{len(plan)}: {step}")
            
            # 构建提示词，将问题、计划、历史和当前步骤填入模板
            prompt = EXECUTOR_PROMPT_TEMPLATE.format(
                question=question,
                plan=plan,
                history=history if history else "无",  # 如果是第一步，则历史为空
                current_step=step
            )
            
            messages = [{"role": "user", "content": prompt}]
            
            # 调用大语言模型获取当前步骤的执行结果
            response_text = self.llm_client.think(messages=messages) or ""
            
            # 更新历史记录，将当前步骤和结果追加到历史中，为下一步做准备
            history += f"步骤 {i+1}: {step}\n结果: {response_text}\n\n"
            
            print(f"✅ 步骤 {i+1} 已完成，结果: {response_text}")

        # 循环结束后，最后一步的响应就是最终答案
        final_answer = response_text
        return final_answer