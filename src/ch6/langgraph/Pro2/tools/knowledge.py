"""本地知识库工具（RAG）"""

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.embeddings import Embeddings  # 定义向量嵌入模型的接口
import requests  # 用于发送 HTTP 请求的库

from dotenv import load_dotenv
import sys
from langchain_core.tools import tool

# 获取当前文件的父目录的父目录（即项目根目录）, 并添加到sys.path, 以后上线需要将myAgents模块发布，这里就不要了
# 1. 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 2. 判断项目根目录是否在sys.path中，若不在则添加
if project_root not in sys.path:
    sys.path.insert(0, project_root)
print("project_root:", project_root)
env_file = os.path.join(project_root, ".env")  # 找到.env文件路径
print("env_file:", env_file)
print("env_exists:", os.path.exists(env_file))
# 3. 加载.env文件
load_dotenv(env_file)

model = os.getenv("EMBED_MODEL_NAME")
api_key = os.getenv("EMBED_API_KEY")
base_url = os.getenv("EMBED_BASE_URL")

print("model:", model)
print("base_url:", base_url)


def build_vector_store(docs_dir: str, save_path: str):
    """从文档目录构建向量存储"""
    # 1. 加载指定目录docs_dir下所有的文档
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print()
    print("加载到文档数量:", len(docs))
    for doc in docs:
        print("文件:", doc.metadata.get("source"))
        print("内容长度:", len(doc.page_content))
    if not docs:
        raise ValueError(f"知识库目录为空: {docs_dir}")
    # ==========================================================
    # 2. 文档切分
    # RecursiveCharacterTextSplitter是一种基于字符的文档切分器，它会将文档内容按字符数进行切分，
    # 它可以处理包含Markdown格式的文档，也可以处理普通文本文档。
    # ==========================================================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个文档块的最大字符数
        chunk_overlap=50,  # 相邻文档块之间的重叠字符数
    )
    chunks = splitter.split_documents(docs)
    print("切分后文档块数量:", len(chunks))
    print("切分后前三个文档块示例:")
    for i, chunk in enumerate(chunks[:2], 1):
        print(
            f"Chunk {i}:",
            len(chunk.page_content),
            "字符",
            # chunk.page_content[:500],
        )
        # print("==" * 70)
    # ==========================================================
    # 3. 创建 Embedding 模型
    # ==========================================================
    embeddings = get_embeddings()
    # ==========================================================
    # 5. 创建 Chroma 向量存储
    # ==========================================================
    print()
    print("正在创建 Chroma 向量数据库...")
    from langchain_community.vectorstores import Chroma

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=save_path,
    )
    print("Chroma 向量数据库创建完成,文档向量数量:", vector_store._collection.count())
    return vector_store


def get_embeddings():
    """创建 MaaS Embedding 模型"""
    model = os.getenv("EMBED_MODEL_NAME")
    api_key = os.getenv("EMBED_API_KEY")
    base_url = os.getenv("EMBED_BASE_URL")
    if not model:
        raise ValueError("缺少 EMBED_MODEL_NAME")
    if not api_key:
        raise ValueError("缺少 EMBED_API_KEY")
    if not base_url:
        raise ValueError("缺少 EMBED_BASE_URL")
    return MaaSEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


class MaaSEmbeddings(Embeddings):
    """阿里云 dashscope MaaS Embedding 模型适配器"""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        """调用 MaaS Embedding API"""
        url = f"{self.base_url}/embeddings"  # 构建 API URL
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "input": texts,
            },
            timeout=60,
        )
        response.raise_for_status()  # 检查响应状态码是否正常
        data = response.json()  # 解析 JSON 响应
        return [item["embedding"] for item in data["data"]]

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """批量生成文档向量"""
        if not texts:
            return []
        return self._embed(texts)

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """生成查询向量"""
        return self._embed([text])[0]


def get_retriever(
    docs_dir: str = "./data/knowledge",
    save_path: str = "./data/vector_store",
):
    """获取检索器"""
    embeddings = get_embeddings()
    try:
        from langchain_community.vectorstores import Chroma

        vectorstore = Chroma(
            persist_directory=save_path,
            embedding_function=embeddings,
        )
        count = vectorstore._collection.count()
        print(f"发现已有 Chroma 数据，共 {count} 条")
        if count == 0:
            raise ValueError("Chroma 向量数据库为空")
    except Exception as e:
        print(
            "加载已有向量数据库失败:",
            e,
        )
        print("重新构建向量数据库...")
        vectorstore = build_vector_store(
            docs_dir,
            save_path,
        )
    return vectorstore.as_retriever(
        search_kwargs={
            "k": 4,  # 返回 4 个最相关的文档
        }
    )


# ==============================================================
# 全局 Retriever
# ==============================================================
_retriever = None


def _get_retriever():
    """获取全局检索器, 单例 模式"""
    global _retriever  # 全局变量
    if _retriever is None:
        _retriever = get_retriever()
    return _retriever


# ==============================================================
# Agent Tool
# ==============================================================
@tool
def search_knowledge_base(query: str) -> str:
    """从本地知识库中检索相关信息。当需要查询已有文档、内部资料、历史记录时使用。
    query: 待检索关键词
    """
    retriever = _get_retriever()
    results = retriever.invoke(query)  # 到本地知识库中执行检索
    if not results:
        return "未找到相关文档"
    # 格式化输出
    formatted_results = [
        f"{i + 1}, {doc.metadata['source']}, {doc.metadata}:. {doc.page_content}"
        for i, doc in enumerate(results)  # 循环results，每个doc是一个Document对象
    ]
    return "\n***********************\n".join(formatted_results)


if __name__ == "__main__":
    print("=" * 60)
    print("开始构建本地知识库")
    print("=" * 60)
    build_vector_store(
        "./data/knowledge",
        "./data/vector_store",
    )

    # embeddings = get_embeddings()
    # # result = embeddings.embed_query("你好")
    # # print(result, len(result))  #  text-embedding-v3: 1024维

    # result = embeddings.embed_documents(
    #     ["你好,你是哪个模型", "transformers是什么", "什么是词嵌入向量"]
    # )
    # print(result, len(result))  #  3 个文档向量，每个向量 1024 维

    print()
    print("=" * 60)
    print("开始测试检索")
    print("=" * 60)
    query = "transformers"
    results = search_knowledge_base.invoke({"query": query})
    print()
    print("检索结果:")
    print(results)
