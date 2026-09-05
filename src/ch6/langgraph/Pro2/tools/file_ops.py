"""文件操作工具"""

import os
from langchain_core.tools import tool


@tool
def save_report(filename: str, content: str) -> str:
    """将报告内容保存为 Markdown 文件。
    filename: 文件名 (不含路径, 自动保存到 ./data/reports/)
    content: 报告内容 (Markdown 格式) """
    try:
        report_dir = "./data/reports"  # ./data/reports 代表主程序main所在的目录
        os.makedirs(report_dir, exist_ok=True)  # 确保目录存在
        filepath = os.path.join(report_dir, filename)
        if not filepath.endswith(".md"):
            filepath += ".md"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return f"报告已保存: {filepath} ({len(content)} 字符)"
    except Exception as e:
        return f"保存报告内容失败: {str(e)}"


@tool
def list_reports() -> str:
    """列出所有已保存的报告文件"""
    report_dir = "./data/reports"
    if not os.path.exists(report_dir):
        return "暂无报告"
    files = os.listdir(report_dir)  # 列出目录下的所有文件
    if not files:
        return "暂无报告"
    result = "已保存的报告:\n"
    for f in sorted(files):
        size = os.path.getsize(os.path.join(report_dir, f))
        result += f"  - {f} ({size} bytes)\n"
    return result


if __name__ == "__main__":
    result = save_report.invoke(
        {"filename": "test_report2", "content": "这是一个测试报告2。"}
    )
    print(result)

    result = list_reports.invoke({})
    print(result)