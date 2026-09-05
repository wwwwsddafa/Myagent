"""知识库工具测试: search_knowledge_base / build_vector_store / MaaSEmbeddings"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# 确保 tools 模块可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.knowledge import (
    search_knowledge_base,
    build_vector_store,
    get_embeddings,
    MaaSEmbeddings,
    get_retriever,
)


class TestMaaSEmbeddings(unittest.TestCase):
    """测试 MaaS Embedding 模型适配器"""

    def setUp(self):
        self.embeddings = MaaSEmbeddings(
            model="test-model",
            api_key="test-api-key",
            base_url="https://test.example.com/api/v1",
        )

    def test_init_strips_trailing_slash(self):
        """基础 URL 应去除尾部斜杠"""
        emb = MaaSEmbeddings(
            model="m", api_key="k", base_url="https://test.com/api/v1/"
        )
        self.assertEqual(emb.base_url, "https://test.com/api/v1")

    @patch("tools.knowledge.requests.post")
    def test_embed_query(self, mock_post):
        """单条查询向量化"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3]}]
        }
        mock_post.return_value = mock_response

        result = self.embeddings.embed_query("测试文本")
        self.assertEqual(result, [0.1, 0.2, 0.3])
        mock_post.assert_called_once()
        print(f"\n[embed_query] 结果维度: {len(result)}")

    @patch("tools.knowledge.requests.post")
    def test_embed_documents(self, mock_post):
        """批量文档向量化"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }
        mock_post.return_value = mock_response

        result = self.embeddings.embed_documents(["文本1", "文本2"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [0.1, 0.2, 0.3])
        print(f"\n[embed_documents] 文档数: {len(result)}, 维度: {len(result[0])}")

    @patch("tools.knowledge.requests.post")
    def test_embed_documents_empty(self, mock_post):
        """空列表应返回空列表"""
        result = self.embeddings.embed_documents([])
        self.assertEqual(result, [])
        mock_post.assert_not_called()


class TestSearchKnowledgeBase(unittest.TestCase):
    """测试知识库检索"""

    @patch("tools.knowledge._get_retriever")
    def test_search_knowledge_base_with_results(self, mock_get_retriever):
        """有结果时返回格式化内容"""
        mock_doc1 = MagicMock()
        mock_doc1.metadata = {"source": "doc1.md"}
        mock_doc1.page_content = "这是文档1的内容。"
        mock_doc2 = MagicMock()
        mock_doc2.metadata = {"source": "doc2.md"}
        mock_doc2.page_content = "这是文档2的内容。"

        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [mock_doc1, mock_doc2]
        mock_get_retriever.return_value = mock_retriever

        result = search_knowledge_base.invoke({"query": "测试查询"})
        self.assertIn("doc1.md", result)
        self.assertIn("doc2.md", result)
        self.assertIn("***********************", result)
        print(f"\n[search_knowledge_base 有结果] {result[:200]}...")

    @patch("tools.knowledge._get_retriever")
    def test_search_knowledge_base_no_results(self, mock_get_retriever):
        """无结果时返回提示"""
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = []
        mock_get_retriever.return_value = mock_retriever

        result = search_knowledge_base.invoke({"query": "不存在的查询"})
        self.assertEqual(result, "未找到相关文档")
        print(f"\n[search_knowledge_base 无结果] {result}")


class TestBuildVectorStore(unittest.TestCase):
    """测试向量存储构建"""

    @patch("tools.knowledge.get_embeddings")
    @patch("langchain_community.vectorstores.Chroma")
    def test_build_vector_store(self, mock_chroma, mock_get_embeddings):
        """构建向量存储基本流程"""
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.count.return_value = 5
        mock_chroma.from_documents.return_value = mock_vectorstore

        result = build_vector_store(
            docs_dir="./data/knowledge",
            save_path="./data/vector_store",
        )
        mock_chroma.from_documents.assert_called_once()
        self.assertEqual(result, mock_vectorstore)
        print(f"\n[build_vector_store] 构建成功，文档向量数: 5")


class TestGetRetriever(unittest.TestCase):
    """测试检索器获取"""

    @patch("tools.knowledge.get_embeddings")
    @patch("langchain_community.vectorstores.Chroma")
    def test_get_retriever_existing(self, mock_chroma, mock_get_embeddings):
        """已有向量数据库时直接加载"""
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_vectorstore = MagicMock()
        mock_vectorstore._collection.count.return_value = 10
        mock_chroma.return_value = mock_vectorstore

        retriever = get_retriever(
            docs_dir="./data/knowledge",
            save_path="./data/vector_store",
        )
        self.assertIsNotNone(retriever)
        print(f"\n[get_retriever] 加载已有数据库，文档数: 10")


if __name__ == "__main__":
    unittest.main(verbosity=2)