"""文件操作工具测试: save_report / list_reports"""

import os
import sys
import tempfile
import shutil
import unittest
from unittest.mock import patch

# 确保 tools 模块可以被导入
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.file_ops import save_report, list_reports


class TestSaveReport(unittest.TestCase):
    """测试报告保存功能"""

    def setUp(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.report_dir = os.path.join(self.temp_dir, "data", "reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def tearDown(self):
        """每个测试后清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_report_basic(self):
        """基本保存功能"""
        content = "# 测试报告\n\n这是测试内容。"
        result = save_report.invoke({"filename": "test_basic", "content": content})
        self.assertIn("报告已保存", result)
        self.assertIn("test_basic.md", result)
        print(f"\n[save_report 基本] {result}")

    def test_save_report_without_md_extension(self):
        """不带 .md 后缀自动添加"""
        content = "报告内容"
        result = save_report.invoke({"filename": "my_report", "content": content})
        self.assertIn("my_report.md", result)
        print(f"\n[save_report 自动加后缀] {result}")

    def test_save_report_with_md_extension(self):
        """已带 .md 后缀不会重复添加"""
        content = "报告内容"
        result = save_report.invoke({"filename": "my_report.md", "content": content})
        self.assertIn("my_report.md", result)
        self.assertNotIn(".md.md", result)
        print(f"\n[save_report 已有后缀] {result}")

    def test_save_report_content_length(self):
        """验证保存的内容长度正确"""
        content = "A" * 1000
        result = save_report.invoke({"filename": "long_report", "content": content})
        self.assertIn("1000 字符", result)
        print(f"\n[save_report 内容长度] {result}")

    def test_save_report_empty_content(self):
        """空内容也能保存"""
        result = save_report.invoke({"filename": "empty_report", "content": ""})
        self.assertIn("报告已保存", result)
        self.assertIn("0 字符", result)
        print(f"\n[save_report 空内容] {result}")


class TestListReports(unittest.TestCase):
    """测试报告列表功能"""

    def setUp(self):
        """每个测试前创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """每个测试后清理临时目录"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch("tools.file_ops.os.path.exists", return_value=False)
    def test_list_reports_no_directory(self, mock_exists):
        """无报告目录时返回提示"""
        result = list_reports.invoke({})
        self.assertEqual(result, "暂无报告")
        print(f"\n[list_reports 无目录] {result}")

    @patch("tools.file_ops.os.path.exists", return_value=True)
    @patch("tools.file_ops.os.listdir", return_value=[])
    def test_list_reports_empty_directory(self, mock_listdir, mock_exists):
        """空目录时返回提示"""
        result = list_reports.invoke({})
        self.assertEqual(result, "暂无报告")
        print(f"\n[list_reports 空目录] {result}")

    @patch("tools.file_ops.os.path.exists", return_value=True)
    @patch("tools.file_ops.os.listdir")
    @patch("tools.file_ops.os.path.getsize")
    def test_list_reports_with_files(self, mock_getsize, mock_listdir, mock_exists):
        """有报告文件时正确列出"""
        mock_listdir.return_value = ["report1.md", "report2.md"]
        mock_getsize.side_effect = [100, 200]

        result = list_reports.invoke({})
        self.assertIn("已保存的报告", result)
        self.assertIn("report1.md", result)
        self.assertIn("report2.md", result)
        print(f"\n[list_reports 有文件] {result}")


if __name__ == "__main__":
    unittest.main(verbosity=2)