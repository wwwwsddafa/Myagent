"""State 定义 — 贯穿整个工作流的核心数据结构"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages  # 用于添加消息到状态中的装饰器

#字典
class ResearchState(TypedDict):
    """研究助手的全局状态"""

    # 对话
    messages: Annotated[list, add_messages]  # 消息历史

    # 研究过程
    question: str  # 原始问题
    question_type: str  # 问题类型: simple / research / report
    search_results: str  # 搜索结果
    knowledge_results: str  # 知识库检索结果
    research_notes: str  # 研究笔记
    analysis: str  # 分析结果
    report: str  # 最终报告

    # 控制
    iteration: int  # 当前迭代次数
    review_passed: bool  # 质量审查是否通过
    review_feedback: str  # 审查反馈
    next_agent: str  # 下一个 Agent




   