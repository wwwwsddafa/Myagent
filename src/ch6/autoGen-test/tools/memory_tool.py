import json
import os
from datetime import datetime
from typing import Any


class TravelMemory:
    """
    旅行助手的记忆管理器。
    负责存储和检索用户偏好、对话历史和旅行计划。
    """

    def __init__(self, memory_file: str = None):
        if memory_file is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            memory_file = os.path.join(base_dir, "memory", "travel_memory.json")

        self.memory_file = memory_file
        self._ensure_file()

    def _ensure_file(self):
        """确保记忆文件存在"""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        if not os.path.exists(self.memory_file):
            self._save({
                "user_preferences": {},
                "conversation_history": [],
                "past_trips": [],
            })

    def _load(self) -> dict:
        """加载记忆数据"""
        try:
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return {
                "user_preferences": {},
                "conversation_history": [],
                "past_trips": [],
            }

    def _save(self, data: dict):
        """保存记忆数据"""
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add_conversation(self, role: str, content: str):
        """添加对话记录"""
        data = self._load()
        data["conversation_history"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        })
        # 只保留最近20轮对话
        data["conversation_history"] = data["conversation_history"][-40:]
        self._save(data)

    def get_conversation_history(self, limit: int = 10) -> str:
        """获取格式化的对话历史"""
        data = self._load()
        history = data["conversation_history"][-limit * 2:]
        lines = []
        for entry in history:
            role = "用户" if entry["role"] == "user" else "助手"
            lines.append(f"[{role}] {entry['content'][:200]}")
        return "\n".join(lines) if lines else "暂无历史对话"

    def update_preference(self, key: str, value: Any):
        """更新用户偏好"""
        data = self._load()
        if "user_preferences" not in data:
            data["user_preferences"] = {}
        data["user_preferences"][key] = value
        self._save(data)

    def get_preferences(self) -> str:
        """获取用户偏好摘要"""
        data = self._load()
        prefs = data.get("user_preferences", {})
        if not prefs:
            return "暂无用户偏好记录"
        lines = ["【用户偏好】"]
        for k, v in prefs.items():
            lines.append(f"  {k}: {v}")
        return "\n".join(lines)

    def save_trip(self, destination: str, days: int, itinerary: str):
        """保存旅行计划"""
        data = self._load()
        data["past_trips"].append({
            "destination": destination,
            "days": days,
            "itinerary": itinerary,
            "date": datetime.now().isoformat(),
        })
        # 只保留最近5次旅行
        data["past_trips"] = data["past_trips"][-5:]
        self._save(data)

    def get_past_trips(self) -> str:
        """获取历史旅行记录"""
        data = self._load()
        trips = data.get("past_trips", [])
        if not trips:
            return "暂无历史旅行记录"
        lines = ["【历史旅行】"]
        for t in trips:
            lines.append(f"  - {t['destination']} ({t['days']}天) - {t['date'][:10]}")
        return "\n".join(lines)

    def clear(self):
        """清空所有记忆"""
        self._save({
            "user_preferences": {},
            "conversation_history": [],
            "past_trips": [],
        })


if __name__ == "__main__":
    mem = TravelMemory()
    mem.update_preference("喜欢的活动", "户外、历史")
    mem.add_conversation("user", "我想去北京玩3天")
    print(mem.get_preferences())
    print(mem.get_conversation_history())