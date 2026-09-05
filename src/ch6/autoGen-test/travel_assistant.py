"""
基于 AutoGen 的智能旅行助手
功能：规划未来 n 天某地的旅行计划
特性：
  1. Tavily 搜索工具（景点、美食、攻略）
  2. 天气查询工具
  3. 记忆功能（用户偏好、对话历史、旅行记录）
  4. 人类反馈机制（每轮输出后可提出修改意见）
  5. 多 Agent 协作
"""

import os
import sys
import asyncio
import traceback
from typing import Sequence

from dotenv import load_dotenv

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import (
    ChatMessage,
    TextMessage,
    MultiModalMessage,
)
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken

# 自定义工具
from tools import (
    get_weather,
    search_attractions,
    search_restaurants,
    search_travel_tips,
    TravelMemory,
)

# 加载环境变量：从当前文件所在目录向上逐层查找 data/.env 或 .env
def _load_env():
    """从当前文件向上逐层查找 data/.env 或 .env 并加载（兼容任意运行目录）"""
    current_file = os.path.abspath(__file__)
    dir_to_check = os.path.dirname(current_file)

    while True:
        candidates = [
            os.path.join(dir_to_check, "data", ".env"),
            os.path.join(dir_to_check, ".env"),
        ]
        for path in candidates:
            if os.path.exists(path):
                load_dotenv(path, override=True)
                print(f"✅ 已加载环境变量: {path}")
                return True
        parent = os.path.dirname(dir_to_check)
        if parent == dir_to_check:  # 已到达文件系统根
            break
        dir_to_check = parent

    load_dotenv()
    print("⚠️ 未找到 .env 文件，尝试从系统环境变量读取")
    return False

_load_env()


# =============================================================================
# 1. 模型客户端
# =============================================================================

def create_model_client() -> OpenAIChatCompletionClient:
    """创建 OpenAI 兼容的模型客户端"""
    # 支持多种环境变量命名方式
    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("LLM_MODEL_ID") or os.getenv("MODEL_NAME", "gpt-4o")

    if not api_key:
        raise ValueError("未找到 API Key，请设置 LLM_API_KEY 或 OPENAI_API_KEY 环境变量")

    return OpenAIChatCompletionClient(
        model=model,
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )


# =============================================================================
# 2. 人类反馈代理（自定义 Agent）
# =============================================================================

class HumanFeedbackAgent(BaseChatAgent):
    """
    人类反馈代理：每轮输出后暂停，等待人类输入反馈。
    支持：确认(ok)、修改意见、终止(TERMINATE)
    """

    def __init__(self, name: str, memory: TravelMemory):
        super().__init__(name=name, description="人类反馈代理，接收用户对旅行计划的修改意见")
        self.memory = memory
        self._feedback_count = 0

    @property
    def produced_message_types(self) -> Sequence[type[ChatMessage]]:
        return (TextMessage,)

    async def on_messages(
        self, messages: Sequence[ChatMessage], cancellation_token: CancellationToken
    ) -> Response:
        # 获取上一条消息内容
        last_content = ""
        if messages:
            last = messages[-1]
            if isinstance(last, TextMessage):
                last_content = last.content
            elif isinstance(last, MultiModalMessage):
                last_content = str(last.content)

        # 保存到记忆
        if last_content:
            self.memory.add_conversation("assistant", last_content)

        # 显示并请求反馈
        print("\n" + "-" * 40)
        print("【反馈】ok=确认  ##STOP##=结束  其他=修改意见")
        print("-" * 40)

        # 尝试获取人类反馈（支持交互式和非交互式环境）
        try:
            feedback = input(">>> 您的反馈: ").strip()
        except EOFError:
            # 非交互环境，自动确认
            print("(非交互环境，自动确认)")
            feedback = "ok"

        # 处理反馈
        if not feedback or feedback.lower() in ("ok", "确认", "好的"):
            feedback_text = "用户确认当前方案，请继续完善或输出最终计划。"
        elif feedback.upper() == "##STOP##":
            feedback_text = "##STOP##"
        else:
            feedback_text = f"用户反馈：{feedback}"
            self._feedback_count += 1

        # 保存用户反馈到记忆
        self.memory.add_conversation("user", feedback_text)

        return Response(
            chat_message=TextMessage(content=feedback_text, source=self.name)
        )

    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        self._feedback_count = 0


# =============================================================================
# 3. 记忆管理代理
# =============================================================================

def create_memory_agent(model_client, memory: TravelMemory) -> AssistantAgent:
    """创建记忆管理代理"""
    system_message = """你是记忆管理员。

职责：记录用户偏好，提供历史上下文。

规则：
- 记忆信息：{memory_context}
- 如果用户提到偏好，调用 update_preference 记录
- 每次回复不超过2句话，只输出关键记忆信息
- 不要重复其他Agent的内容
"""
    # 动态填充记忆上下文
    memory_context = (
        f"{memory.get_preferences()}\n\n"
        f"{memory.get_past_trips()}\n\n"
        f"近期对话:\n{memory.get_conversation_history(limit=5)}"
    )

    return AssistantAgent(
        name="MemoryManager",
        model_client=model_client,
        system_message=system_message.format(memory_context=memory_context),
        tools=[
            memory.update_preference,
            memory.add_conversation,
            memory.get_preferences,
            memory.get_conversation_history,
        ],
    )


# =============================================================================
# 4. 信息研究代理（调用搜索和天气工具）
# =============================================================================

def create_research_agent(model_client) -> AssistantAgent:
    """创建信息研究代理，负责调用工具收集旅行信息"""
    system_message = """你是旅行信息研究员。

收到规划师指令后，立即按顺序调用工具：
1. get_weather(城市, n天) → 获取天气
2. search_attractions(城市, n天) → 获取景点
3. search_restaurants(城市) → 获取美食
4. search_travel_tips(城市) → 获取贴士

输出原则：
- 只输出工具返回的关键信息，不要添加分析或建议
- 用编号列表，每条信息一行，总计不超过15行
- 不要重复规划师的话，不要做"需求确认"
"""
    return AssistantAgent(
        name="InfoResearcher",
        model_client=model_client,
        system_message=system_message,
        tools=[
            get_weather,
            search_attractions,
            search_restaurants,
            search_travel_tips,
        ],
    )


# =============================================================================
# 5. 行程设计代理
# =============================================================================

def create_designer_agent(model_client) -> AssistantAgent:
    """创建行程设计代理，负责生成详细的旅行计划"""
    system_message = """你是行程设计师。根据研究员提供的信息，直接输出行程。

输出格式：
# 目的地 N天旅行计划
  天气概况: xxx

## Day 1 (日期 星期x) | 天气: xx
  上午: 景点+简介
  中午: 餐厅+推荐菜
  下午: 景点+简介
  晚上: 活动+推荐

## Day 2 ...
(以此类推)

## 贴士
  - 交通/注意事项

规则：
- 直接输出行程，不要加"思考过程"、"分析"等开场白
- 每天4个时段，每个时段1行
- 相近景点安排在同一天
"""
    return AssistantAgent(
        name="ItineraryDesigner",
        model_client=model_client,
        system_message=system_message,
    )


# =============================================================================
# 6. 主协调代理（旅行规划师）
# =============================================================================

def create_planner_agent(model_client) -> AssistantAgent:
    """创建旅行规划师代理，负责协调整个流程"""
    system_message = """你是旅行主规划师。

职责：解析用户需求 → 交给研究员 → 交给设计师 → 确认结果。

规则：
- 用户没说明天数则默认3天，没说明偏好则默认"经典综合"
- 收到需求后，简洁地确认目的地和天数（1句话），然后直接说"请研究员收集信息"
- 不要问用户问题，不要输出表格，不要长篇大论
- 每次回复控制在3句话以内
"""
    return AssistantAgent(
        name="TravelPlanner",
        model_client=model_client,
        system_message=system_message,
    )


# =============================================================================
# 7. 主运行函数
# =============================================================================

async def run_travel_assistant():
    """运行智能旅行助手"""
    print("🌍 欢迎使用智能旅行助手！")
    print("=" * 60)

    # 初始化
    print("🔧 正在初始化...")
    model_client = create_model_client()
    memory = TravelMemory()

    # 创建 Agent
    planner = create_planner_agent(model_client)
    researcher = create_research_agent(model_client)
    designer = create_designer_agent(model_client)
    memory_agent = create_memory_agent(model_client, memory)
    human = HumanFeedbackAgent(name="HumanFeedback", memory=memory)

    # 终止条件：提到 ##STOP## 时结束（避免与任务提示中的文字冲突）
    termination = TextMentionTermination("##STOP##")

    # 创建轮询团队
    team = RoundRobinGroupChat(
        participants=[planner, researcher, designer, memory_agent, human],
        termination_condition=termination,
        max_turns=30,
    )

    # 获取用户初始需求（支持命令行参数或交互式输入）
    if len(sys.argv) > 1:
        # 从命令行参数获取任务
        user_task = " ".join(sys.argv[1:])
        print(f"\n📋 从命令行接收任务: {user_task}")
    else:
        print("\n请告诉我您的旅行计划：")
        print('  例如："我计划未来3天去北京旅行，喜欢历史文化和美食"')
        try:
            user_task = input(">>> ").strip()
        except EOFError:
            # 非交互环境，使用默认任务
            user_task = ""

    if not user_task:
        user_task = "帮我规划未来3天北京的旅行计划，喜欢历史文化和美食"
        print(f"使用默认任务: {user_task}")

    # 保存用户输入到记忆
    memory.add_conversation("user", user_task)

    # 构建完整任务提示
    task = f"""用户旅行需求: {user_task}

协作流程：
1. TravelPlanner: 确认目的地和天数
2. InfoResearcher: 调用工具获取天气、景点、美食、贴士
3. ItineraryDesigner: 输出格式化行程
4. MemoryManager: 记录偏好
5. HumanFeedback: 收集反馈
"""

    print("\n🚀 启动旅行规划团队...")
    print("=" * 60)

    try:
        result = await Console(team.run_stream(task=task))

        print("\n" + "=" * 60)
        print("✅ 旅行规划完成！")

    except Exception as e:
        print(f"\n❌ 运行错误: {e}")
        traceback.print_exc()
        return None


# =============================================================================
# 8. 入口
# =============================================================================

if __name__ == "__main__":
    try:
        asyncio.run(run_travel_assistant())
    except KeyboardInterrupt:
        print("\n\n👋 用户中断，再见！")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        traceback.print_exc()