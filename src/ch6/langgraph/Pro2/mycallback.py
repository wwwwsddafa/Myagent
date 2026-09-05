"""自定义回调：跟踪 Agent LLM 执行过程"""

from langchain_core.callbacks import BaseCallbackHandler
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s -[%(name)s]- %(levelname)s - %(message)s"
)
logger = logging.getLogger("research_assistant")


class ResearchCallbackHandler(BaseCallbackHandler):
    """研究助手项目的llm的回调处理函数"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        """记录LLM开始执行"""
        logger.info(f"[LLM] 开始执行，输入参数: {prompts}")

    def on_llm_end(self, serialized, prompts, **kwargs):
        """记录LLM开始执行"""
        logger.info(
            f"[LLM] 结束，输入参数: {prompts}, 输出参数: {kwargs.get('output', 'none')}"
        )

    def on_tool_start(self, serialized, input_str, **kwargs):
        logger.info(
            f"[工具调用] {serialized.get('name', 'unknown')}: {input_str[:100]}"
        )

    def on_tool_end(self, output, **kwargs):
        logger.info(f"[工具结果] {str(output)[:100]}")

    def on_chain_error(self, error, **kwargs):
        logger.info(f"[chain错误] {error}")
