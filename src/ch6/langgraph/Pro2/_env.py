"""统一的 .env 加载器（Pro2 项目专用）。

作用：无论从哪个目录运行 Pro2 下的模块，都能自动定位到本项目的 .env 文件
（位于 src/ch6/langgraph/Pro2/.env），而不依赖"当前工作目录恰好是 Pro2"。

原理：从本文件所在目录开始，逐层向上查找包含 .env 的目录并加载。
"""

import os

from dotenv import load_dotenv


def load_project_env() -> None:
    """从当前文件向上查找 .env 并加载，找不到时回退到基于 cwd 的默认加载。"""
    cur = os.path.dirname(os.path.abspath(__file__))
    while True:
        env_path = os.path.join(cur, ".env")
        if os.path.exists(env_path):
            load_dotenv(env_path, override=True)
            return
        parent = os.path.dirname(cur)
        if parent == cur:  # 到达文件系统根
            break
        cur = parent
    # 兜底：基于当前工作目录的默认加载
    load_dotenv()


load_project_env()
