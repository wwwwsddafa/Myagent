import re
from typing import List, Dict, Any, Optional

from llm import HelloAgentLLM


class Memory:
    """
    反思智能体的记忆模块。

    职责：按类型（execution / reflection）记录每一次执行结果与反思反馈，
    形成一条完整的「执行轨迹」，供后续的反思与优化步骤读取。

    之所以要带类型标签，而不是简单地用一个字符串列表存放历史，
    是因为反思环节需要快速区分「哪一条是代码、哪一条是评审意见」。
    """

    def __init__(self):
        # 每条记录形如：{"type": "execution" | "reflection", "content": "..."}
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        添加一条记忆记录。

        :param record_type: 记录类型，固定为 "execution"（执行结果）或 "reflection"（反思反馈）
        :param content: 记录内容
        """
        if record_type not in ("execution", "reflection"):
            raise ValueError(
                f"不支持的记录类型: {record_type}，仅支持 'execution' 或 'reflection'"
            )
        self.records.append({"type": record_type, "content": content})

    def get_trajectory(self) -> str:
        """获取格式化的完整执行轨迹，用于调试或回灌给模型。"""
        if not self.records:
            return "暂无历史记录"

        lines = []
        for i, record in enumerate(self.records, 1):
            tag = "执行" if record["type"] == "execution" else "反思"
            lines.append(f"[{i}][{tag}] {record['content']}")
        return "\n".join(lines)

    def get_last_execution(self) -> str:
        """获取最近一次的执行结果（即最新一版代码）。没有则返回空字符串。"""
        for record in reversed(self.records):
            if record["type"] == "execution":
                return record["content"]
        return ""

    def get_last_reflection(self) -> str:
        """获取最近一次的反思反馈。没有则返回空字符串。"""
        for record in reversed(self.records):
            if record["type"] == "reflection":
                return record["content"]
        return ""

    def get_execution_count(self) -> int:
        """统计目前一共执行了多少轮（用于判断是否真的发生了优化）。"""
        return sum(1 for r in self.records if r["type"] == "execution")

    def clear(self):
        """清空所有记忆。"""
        self.records = []


# 初始提示词：让模型产出第一版代码
INITIAL_PROMPT_TEMPLATE = """
你是一位资深的 Python 工程师。请为以下任务编写代码。

任务:
{task}

要求:
1. 只输出代码本身，不要输出任何解释、说明文字或 Markdown 代码块标记。
2. 代码必须可以直接运行，并包含必要的异常处理。
3. 如果需要演示用法，请写在 if __name__ == "__main__": 代码块中。

请开始编写:
"""

# 反思提示词：让模型以评审专家的身份批判上一版代码
REFLECT_PROMPT_TEMPLATE = """
你是一位严格的代码评审专家。请审查下面这段代码，判断它是否完美地完成了任务。

原始任务:
{task}

待审查的代码:
```python
{code}
```

请从以下四个维度进行审查:
1. **正确性**: 是否完全满足任务要求?边界条件是否处理正确?
2. **健壮性**: 是否有必要的异常处理?是否存在潜在的运行错误?
3. **可读性**: 命名是否清晰?是否有必要的注释?
4. **效率**: 算法复杂度是否合理?是否存在明显的性能问题?

输出规则(必须严格遵守):
- 如果代码**存在任何**可以改进的地方，请用简洁的中文逐条列出具体的改进建议。
- 如果代码**已经完美**、没有任何可改进之处，请**只输出**"无需改进"这四个字，
  不要输出任何其他内容。

请开始审查:
"""

# 优化提示词：让模型根据评审意见重写代码
REFINE_PROMPT_TEMPLATE = """
你是一位资深的 Python 工程师。请根据评审意见，重写下面这段代码。

原始任务:
{task}

上一版代码:
```python
{last_code_attempt}
```

评审意见:
{feedback}

要求:
1. 严格针对评审意见逐条修正。
2. 只输出修正后的完整代码本身，不要输出任何解释、说明文字或 Markdown 代码块标记。
3. 确保代码可以直接运行。

请输出修正后的代码:
"""


# 反思智能体
class ReflectionAgent:
    """
    反思（Reflection）智能体。

    工作流程：
        初始执行 → 【反思 → 优化】× N 轮 → 产出最终代码

    与 ReAct 的区别在于：ReAct 的每一轮是「面向外部工具」的（调用工具获取新信息），
    而 Reflection 的每一轮是「面向自身产出」的（批判并重写自己的上一版结果）。
    """

    def __init__(self, llm_client, max_iterations=3):
        """
        :param llm_client: 大语言模型客户端实例
        :param max_iterations: 最大反思迭代轮数，防止模型一直不满意导致无限循环
        """
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

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
        else:
            # for-else: 循环正常走完（未 break）说明始终没达到"无需改进"
            print(
                f"\n⚠️ 已达到最大迭代次数({self.max_iterations})，"
                "代码仍未通过全部审查，将返回当前最新版本。"
            )

        final_code = self.memory.get_last_execution()  # 跳出循环后，获取最终的执行结果
        print(f"\n---  任务完成  ---\n最终生成的代码:\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        """
        调用大模型并获取响应。

        这里额外做了一件事：清洗模型输出中的 Markdown 代码块标记。
        模型经常会把代码包在 ```python ... ``` 里，如果不清洗，
        这些标记会随着每一轮迭代不断累积，最终产出根本无法执行的代码。
        """
        messages = [{"role": "user", "content": prompt}]
        response_text = self.llm_client.think(messages=messages)
        return self._extract_code(response_text)

    @staticmethod
    def _extract_code(text: str) -> str:
        """从模型的原始输出中提取纯代码，去掉 Markdown 代码块围栏。"""
        if not text:
            return ""

        cleaned = text.strip()

        # 情况一：整段输出被 ```python ... ``` 包裹
        match = re.match(r"^```(?:python|py)?\s*\n(.*?)\n?```$", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 情况二：输出中夹着解释文字，取第一个代码块
        match = re.search(r"```(?:python|py)?\s*\n(.*?)\n?```", cleaned, re.DOTALL)
        if match:
            return match.group(1).strip()

        # 情况三：没有围栏，原样返回
        return cleaned


if __name__ == "__main__":
    # --- 1. 先单独验证 Memory 模块 ---
    memory = Memory()
    memory.add_record("execution", "def add(a, b): return a + b")
    memory.add_record("reflection", "这是一条反馈,这个代码可以正确工作，但不好")

    print(memory.get_last_execution())
    print("=====" * 10)
    print(memory.get_trajectory())
    memory.add_record("execution", "def add2(a, b): return a + b")
    memory.add_record("reflection", "这是一条反馈,这个代码更好了")

    print("=====" * 10)
    print(memory.get_trajectory())

    # --- 2. 再跑完整的反思流程 ---
    llm_client = HelloAgentLLM()
    reflection_agent = ReflectionAgent(llm_client, 7)
    reflection_agent.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")
