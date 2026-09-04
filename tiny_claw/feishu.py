from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass
from typing import Any

from .schema import Message, ROLE_USER


@dataclass
class ApprovalResult:
    allowed: bool
    reason: str


class ApprovalManager:
    def __init__(self) -> None:
        self._pending: dict[str, tuple[threading.Event, ApprovalResult | None]] = {}
        self._lock = threading.RLock()

    def wait_for_approval(self, task_id: str, tool_name: str, arguments: str, reporter: Any = None) -> tuple[bool, str]:
        event = threading.Event()
        with self._lock:
            self._pending[task_id] = (event, None)
        notice = f"⚠️ 高危操作审批请求\n工具: {tool_name}\n参数: {arguments}\n任务 ID: {task_id}\n请回复 approve {task_id} 或 reject {task_id}。"
        if reporter:
            reporter.send_msg(notice)
        else:
            print(notice)
        event.wait()
        with self._lock:
            _, result = self._pending.pop(task_id, (event, ApprovalResult(False, "审批已失效")))
        result = result or ApprovalResult(False, "审批已失效")
        return result.allowed, result.reason

    def resolve_approval(self, task_id: str, allowed: bool, reason: str) -> None:
        with self._lock:
            entry = self._pending.get(task_id)
            if entry is None:
                return
            event, _ = entry
            self._pending[task_id] = (event, ApprovalResult(allowed, reason))
            event.set()


GLOBAL_APPROVAL_MANAGER = ApprovalManager()


def is_dangerous_command(tool_name: str, arguments: str) -> bool:
    if tool_name not in {"bash", "write_file", "edit_file"}:
        return False
    if tool_name != "bash":
        return False
    return any(re.search(pattern, arguments) for pattern in (r"rm\s+-r", r"sudo\s+", r"drop\s+", r">.*\.go"))


class FeishuReporter:
    def __init__(self, client: Any, chat_id: str) -> None:
        self.client = client
        self.chat_id = chat_id

    def send_msg(self, text: str) -> None:
        if self.client is None:
            return
        try:
            self.client.send_text(self.chat_id, text)
        except AttributeError as exc:
            raise RuntimeError("飞书客户端需要提供 send_text(chat_id, text) 方法") from exc

    def on_thinking(self) -> None:
        self.send_msg("🤔 模型正在慢思考 (Thinking)...")

    def on_tool_call(self, tool_name: str, arguments: str) -> None:
        self.send_msg(f"🛠️ 正在执行工具：{tool_name}\n参数：{arguments}")

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        self.send_msg((f"⚠️ 执行报错 ({tool_name})：\n{result}" if is_error else f"✅ 执行成功 ({tool_name})"))

    def on_message(self, content: str) -> None:
        self.send_msg(content)


class FeishuBot:
    """飞书事件处理库；具体 SDK 客户端通过依赖注入，便于离线测试。"""

    def __init__(self, agent_engine: Any, session: Any, client: Any = None) -> None:
        if client is None and (not os.getenv("FEISHU_APP_ID") or not os.getenv("FEISHU_APP_SECRET")):
            raise RuntimeError("请设置 FEISHU_APP_ID 和 FEISHU_APP_SECRET，或传入 client")
        self.engine = agent_engine
        self.session = session
        self.client = client

    def handle_message(self, chat_id: str, content: str) -> None:
        content = content.strip()
        if content.startswith("approve "):
            GLOBAL_APPROVAL_MANAGER.resolve_approval(content[8:].strip(), True, "人类管理员已批准操作")
            return
        if content.startswith("reject "):
            GLOBAL_APPROVAL_MANAGER.resolve_approval(content[7:].strip(), False, "人类管理员认为该操作存在极高风险，已拒绝")
            return
        reporter = FeishuReporter(self.client, chat_id)
        self.session.append(Message(ROLE_USER, content))
        self.engine.run(self.session, reporter)

    def get_event_dispatcher(self) -> Any:
        return self.handle_message
