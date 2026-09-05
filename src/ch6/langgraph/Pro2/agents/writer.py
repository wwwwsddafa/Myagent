"""写作 Agent：负责生成最终报告"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from state import ResearchState
from _env import load_project_env
import os

# 加载 .env 配置（自动定位本项目根目录）
load_project_env()

from mylogger import log_node_execution


class ResearchReport(BaseModel):
    """研究报告结构"""

    title: str = Field(description="报告标题")
    summary: str = Field(description="一句话摘要")
    key_findings: list[str] = Field(description="关键发现列表")
    detailed_analysis: str = Field(description="详细分析内容")
    conclusion: str = Field(description="结论")
    references: list[str] = Field(description="参考来源")


@log_node_execution("writer_node")
def writer_node(state: ResearchState) -> dict:
    """写作节点：生成结构化报告"""
    # llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # 使用 ModelScope + Qwen 模型
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL_ID", "Qwen/Qwen3.5-35B-A3B"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/"),
        timeout=int(os.getenv("LLM_TIMEOUT", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    #        输出解析器，它会去了解 ResearchReport的结构，生成对应的 JSON 字符串
    parser = PydanticOutputParser(pydantic_object=ResearchReport)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """你是一个技术写作专家。基于研究和分析结果，撰写一份专业的报告。

                    {format_instructions}

                    要求：
                    - 语言简洁专业
                    - 观点有据可依
                    - 结构清晰完整""",
            ),
            (
                "human",
                """问题: {question}

                    研究笔记:
                    {notes}

                    分析结果:
                    {analysis}

                    请撰写报告。""",
            ),
        ]
    )
    chain = prompt | llm | parser
    try:
        report: ResearchReport = chain.invoke(
            {
                "question": state["question"],
                "notes": state.get("research_notes", ""),
                "analysis": state.get("analysis", ""),
                "format_instructions": parser.get_format_instructions(),  # 格式指令
            }
        )
        # 根据 report的结构，生成最终的 markdown格式的报告
        md_report = f"""# {report.title}

                        > {report.summary}

                        ## 关键发现

                        {chr(10).join(f'- {f}' for f in report.key_findings)}

                        ## 详细分析

                        {report.detailed_analysis}

                        ## 结论

                        {report.conclusion}

                        ## 参考来源

                        {chr(10).join(f'- {r}' for r in report.references)}
                        """
        return {"report": md_report}
    except Exception as e:
        # 解析失败时，使用纯文本回退
        simple_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是写作专家。基于以下信息写一份简洁的研究报告。"),
                (
                    "human",
                    "问题: {question}\n\n研究笔记:\n{notes}\n\n分析:\n{analysis}",
                ),
            ]
        )
        # 降级处理到简单的纯文本生成，而不是markdown生成
        chain = simple_prompt | llm
        result = chain.invoke(
            {
                "question": state["question"],
                "notes": state.get("research_notes", ""),
                "analysis": state.get("analysis", ""),
            }
        )
        return {"report": result.content}


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":
    """测试 writer_node()"""

    # 构造测试状态
    test_state: ResearchState = {
        "messages": [],
        "question": "什么是 RAG？RAG 在企业知识库问答系统中有哪些优势？",
        "question_type": "research",
        # 模拟 researcher_node 产生的研究笔记
        "research_notes": """
                        研究笔记：

                        1. RAG（Retrieval-Augmented Generation）即检索增强生成，
                        是一种将信息检索与大语言模型生成能力结合的技术。

                        2. RAG 的基本流程通常包括：
                        - 用户提出问题
                        - 将问题转换为向量
                        - 从向量数据库中检索相关文档
                        - 将检索结果与用户问题一起提交给大语言模型
                        - 大语言模型基于检索内容生成答案

                        3. RAG 可以降低大语言模型产生幻觉的概率，
                        因为模型可以参考外部知识库中的真实资料。

                        4. RAG 不需要重新训练大语言模型，
                        因此相比微调模型，更新企业知识库更加方便。

                        5. RAG 特别适合企业内部知识库、客服系统、
                        技术文档问答、法律文档检索等场景。

                        6. RAG 的效果高度依赖检索质量。
                        如果召回的文档与问题无关，即使大语言模型能力很强，
                        最终答案也可能不准确。
                        """,
        # 模拟 analyst_node 产生的分析结果
        "analysis": """
                    分析结果：

                    RAG 的核心价值在于解决大语言模型知识时效性和专业领域知识不足的问题。
                    传统大语言模型主要依赖训练阶段获得的知识，当企业内部知识发生变化时，
                    重新训练模型成本较高。

                    RAG 则将企业知识存储在外部知识库中，在用户提问时动态检索相关内容，
                    再将检索结果作为上下文提供给大语言模型。因此，只需要更新知识库，
                    通常就可以让系统获得新的知识。

                    从企业应用角度来看，RAG 具有三个主要优势：

                    第一，知识更新成本较低。企业可以直接更新文档和向量数据库，
                    不需要频繁重新训练大语言模型。

                    第二，可解释性相对较好。系统可以保存检索到的文档，
                    并将其作为回答的参考来源。

                    第三，适合企业私有知识场景。企业可以将内部技术文档、
                    产品资料、制度文件等构建为知识库。

                    但是 RAG 也存在一定局限性。例如：
                    - 文档切分不合理会影响检索效果；
                    - 向量模型质量会影响语义匹配；
                    - Top-K 参数设置不合理可能导致召回不足或噪声过多；
                    - 大模型仍然可能错误理解检索结果。

                    因此，一个高质量的 RAG 系统通常需要同时优化：
                    文档处理、Embedding 模型、向量检索、重排序以及大语言模型生成。
                    """,
        # writer_node 最终会写入这个字段
        "report": "",
    }

    print("=" * 80)
    print("开始测试 writer_node()")
    print("=" * 80)

    # 调用写作节点
    result = writer_node(test_state)

    print("\n" + "=" * 80)
    print(result["report"])
