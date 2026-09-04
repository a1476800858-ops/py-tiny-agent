from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .schema import Message, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER

log = logging.getLogger(__name__)


@dataclass
class Session:
    id: str
    work_dir: str
    created_at: float
    updated_at: float
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_cny: float = 0.0
    _history: list[Message] = field(default_factory=list, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def new(cls, session_id: str, work_dir: str) -> "Session":
        import time

        now = time.time()
        return cls(session_id, work_dir, now, now)

    def append(self, *messages: Message) -> None:
        import time

        with self._lock:
            self._history.extend(messages)
            self.updated_at = time.time()

    def get_working_memory(self, limit: int) -> list[Message]:
        with self._lock:
            if limit <= 0 or len(self._history) <= limit:
                return list(self._history)
            result = list(self._history[-limit:])
        while result and result[0].role == ROLE_USER and result[0].tool_call_id:
            result.pop(0)
        return result

    def record_usage(self, prompt: int, completion: int, cost: float) -> None:
        with self._lock:
            self.total_prompt_tokens += prompt
            self.total_completion_tokens += completion
            self.total_cost_cny += cost


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()

    def get_or_create(self, session_id: str, work_dir: str) -> Session:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = Session.new(session_id, work_dir)
            return self._sessions[session_id]


GLOBAL_SESSION_MANAGER = SessionManager()


class Compactor:
    def __init__(self, max_chars: int, retain_last_messages: int) -> None:
        self.max_chars = max_chars
        self.retain_last_messages = retain_last_messages

    @staticmethod
    def _estimate(messages: list[Message]) -> int:
        return sum(
            len(message.content)
            + sum(len(call.name) + len(call.arguments_json()) for call in message.tool_calls)
            for message in messages
        )

    def compact(self, messages: list[Message]) -> list[Message]:
        current_length = self._estimate(messages)
        if current_length < self.max_chars:
            return messages
        log.warning("[Compactor] 内存告警：上下文长度 %d 超过阈值 %d，触发压缩", current_length, self.max_chars)
        protected_start = max(0, len(messages) - self.retain_last_messages)
        result: list[Message] = []
        for index, message in enumerate(messages):
            if message.role == ROLE_SYSTEM:
                result.append(message)
                continue
            copied = Message(
                role=message.role,
                content=message.content,
                tool_calls=list(message.tool_calls),
                tool_call_id=message.tool_call_id,
                usage=message.usage,
            )
            recent = index >= protected_start
            if message.role == ROLE_USER and message.tool_call_id:
                if not recent and len(message.content) > 200:
                    copied.content = f"...[为了节省内存，早期的工具输出已被系统强制清理。原始长度: {len(message.content)} 字节]..."
                elif recent and len(message.content) > 1000:
                    copied.content = (
                        message.content[:500]
                        + f"\n\n...[内容过长，中间 {len(message.content) - 1000} 字节已被系统截断]...\n\n"
                        + message.content[-500:]
                    )
            elif message.role == ROLE_ASSISTANT and message.content and not recent and len(message.content) > 200:
                copied.content = "...[早期的推理思考过程已折叠]..."
            result.append(copied)
        log.info("[Compactor] 压缩完成：%d -> %d 字符", current_length, self._estimate(result))
        return result


@dataclass
class Skill:
    name: str
    description: str
    body: str


class SkillLoader:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = Path(work_dir)

    @staticmethod
    def parse(content: str) -> Skill:
        skill = Skill("Unknown Skill", "No description provided.", content)
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) == 3:
                skill.body = parts[2].strip()
                for line in parts[1].splitlines():
                    key, separator, value = line.strip().partition(":")
                    if separator and key == "name":
                        skill.name = value.strip()
                    elif separator and key == "description":
                        skill.description = value.strip()
        return skill

    def load_all(self) -> str:
        base = self.work_dir / ".claw" / "skills"
        if not base.exists():
            return ""
        chunks = [
            "\n### 可用专业技能 (Agent Skills)\n",
            "以下是你拥有的标准化外挂技能，请在符合 description 描述的场景下严格遵循其正文指令：\n\n",
        ]
        for path in sorted(base.rglob("SKILL.md")):
            try:
                skill = self.parse(path.read_text(encoding="utf-8"))
            except OSError:
                continue
            chunks.extend(
                [
                    f"#### 技能名称: {skill.name}\n",
                    f"**触发条件**: {skill.description}\n\n",
                    "**执行指南**:\n",
                    skill.body,
                    "\n\n---\n",
                ]
            )
        result = "".join(chunks)
        return result if len(result) >= 50 else ""


class PromptComposer:
    def __init__(self, work_dir: str, plan_mode: bool) -> None:
        self.work_dir = Path(work_dir)
        self.plan_mode = plan_mode
        self.skill_loader = SkillLoader(work_dir)

    def build(self) -> Message:
        parts = [
            """# 核心身份
你名叫 py-tiny-claw，一个由驾驭工程驱动的研发助手。
你能通过系统提供的内置工具，创建、读取、修改和执行工作区中的代码。

# 核心纪律 (CRITICAL)
1. 如需检查文件是否存在，请使用 bash 的 ls 或 test -f，而不是对目录使用 read_file。
2. 创建新文件时，务必使用 write_file，并同时提供 path 和 content 参数。
3. 编辑文件前务必先读取现有文件，以理解上下文。
4. 无论何时你需要写代码或创建文件，都要直接使用 write_file 工具。
5. 遇到工具执行报错时，仔细阅读 stderr，尝试自己修正命令并重试。
6. 始终用中文回复，以便传达你的进展和想法。
"""
        ]
        if self.plan_mode:
            parts.append(
                """
# 长程任务与状态外部化强制规范 (Plan Mode: ON)
本模式下必须将架构思路和执行进度持久化到 PLAN.md 与 TODO.md。
收到新指令后先用 bash 检查两个文件；不存在时先用 write_file 创建 PLAN.md 和 TODO.md，存在时先读取并从第一个未完成任务继续。
每完成一个子任务，立即将 TODO.md 对应复选框改为 [x]；遇到报错或迷失时重新读取 TODO.md。
"""
            )
        agents = self.work_dir / "AGENTS.md"
        if agents.is_file():
            try:
                parts.append("\n# 项目专属指南 (来自 AGENTS.md)\n```markdown\n")
                parts.append(agents.read_text(encoding="utf-8"))
                parts.append("\n```\n")
            except OSError:
                pass
        parts.append(self.skill_loader.load_all())
        return Message(ROLE_SYSTEM, "".join(parts))


class RecoveryManager:
    def analyze_and_inject(self, tool_name: str, raw_error: str) -> str:
        lower = raw_error.lower()
        hint = ""
        if tool_name == "edit_file":
            if "在文件中未找到 old_text" in raw_error or "找不到该代码片段" in raw_error:
                hint = "你提供的 old_text 与文件当前内容不一致。请先使用 `read_file` 获取最新内容后再编辑。"
            elif "匹配到了多处" in raw_error or "提供更多上下文" in raw_error:
                hint = "old_text 不够具体，请增加上下相邻代码以确保唯一性。"
        elif tool_name in {"read_file", "write_file"}:
            if "no such file or directory" in lower or "文件不存在" in raw_error:
                hint = "路径似乎不正确，请先使用 `bash` 的 ls 或 find 查找正确路径。"
            elif "permission denied" in lower or "权限" in raw_error:
                hint = "你没有权限操作该文件，请检查工作区限制。"
        elif tool_name == "bash":
            if "command not found" in lower:
                hint = "系统中未安装该命令，请寻找替代命令。"
            elif "超时" in raw_error or "deadlineexceeded" in lower:
                hint = "命令执行超时；常驻服务应转入后台，不要阻塞主线程。"
            elif "syntax error" in lower:
                hint = "Bash 语法错误，请检查引号转义和特殊字符。"
        return raw_error if not hint else f"{raw_error}\n\n[系统救援指南]: {hint}"
