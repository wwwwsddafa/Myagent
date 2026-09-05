"""搜索工具测试: search_web / search_arxiv"""

import os
import sys
import unittest

# 确保 tools 模块可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search import search_web, search_arxiv


class TestSearchWeb(unittest.TestCase):
    """测试 Tavily 网络搜索"""

    def test_search_web_success(self):
        """正常搜索应返回结果"""
        result = search_web.invoke({"query": "Python LangGraph"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertNotIn("搜索失败", result)
        self.assertNotIn("未配置", result)
        print(f"\n[search_web 结果] {result[:200]}...")

    def test_search_web_returns_structured(self):
        """结果应包含标题和URL"""
        result = search_web.invoke({"query": "OpenAI GPT"})
        self.assertIn("相关搜索结果:", result)
        print(f"\n[search_web 结构化] 结果长度: {len(result)} 字符")

    def test_search_web_chinese(self):
        """中文搜索"""
        result = search_web.invoke({"query": "人工智能最新进展"})
        self.assertIsInstance(result, str)
        self.assertNotIn("搜索失败", result)
        print(f"\n[search_web 中文] {result[:200]}...")

    def test_search_web_empty_query(self):
        """空查询也应返回结果（Tavily 会处理）"""
        result = search_web.invoke({"query": ""})
        self.assertIsInstance(result, str)
        print(f"\n[search_web 空查询] {result[:200]}...")


class TestSearchArxiv(unittest.TestCase):
    """测试 arXiv 学术搜索"""

    def test_search_arxiv_success(self):
        """正常搜索学术论文"""
        result = search_arxiv.invoke({"query": "transformer"})
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 10)
        self.assertNotIn("搜索失败", result)
        print(f"\n[search_arxiv 结果] {result[:300]}...")

    def test_search_arxiv_returns_title_and_summary(self):
        """结果应包含标题和摘要"""
        result = search_arxiv.invoke({"query": "attention mechanism"})
        self.assertIn("摘要:", result)
        print(f"\n[search_arxiv 结构化] 结果长度: {len(result)} 字符")

    def test_search_arxiv_no_results(self):
        """无结果查询"""
        result = search_arxiv.invoke({"query": "xyznonexistentfoobar123456"})
        self.assertIsInstance(result, str)
        print(f"\n[search_arxiv 无结果] {result}")


if __name__ == "__main__":
    unittest.main(verbosity=2)