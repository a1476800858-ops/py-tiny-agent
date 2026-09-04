from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Protocol

from .observability import Span, start_span
from .schema import ToolCall, ToolDefinition, ToolResult

log = logging.getLogger(__name__)


class ToolError(RuntimeError):
    pass


def parse_args(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    try:
        value = json.loads(arguments.decode() if isinstance(arguments, bytes) else arguments)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ToolError(f"参数解析失败: {exc}") from exc
    if not isinstance(value, dict):
        raise ToolError("参数解析失败: 参数必须是 JSON 对象")
    return value


def resolve_workspace_path(work_dir: str | Path, relative_path: str) -> Path:
    base = Path(work_dir).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ToolError("路径必须是工作区内的相对路径")
    resolved = (base / candidate).resolve(strict=False)
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ToolError("路径越界：只能操作工作区及其子目录中的文件") from exc
    return resolved


class BaseTool(Protocol):
    def name(self) -> str: ...
    def definition(self) -> ToolDefinition: ...
    def execute(self, arguments: Any) -> str: ...


class ReadFileTool:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir

    def name(self) -> str:
        return "read_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name(), "读取指定路径的文件内容。请提供相对工作区的路径。", {"type": "object", "properties": {"path": {"type": "string", "description": "要读取的文件路径"}}, "required": ["path"]})

    def execute(self, arguments: Any) -> str:
        args = parse_args(arguments)
        path = resolve_workspace_path(self.work_dir, str(args.get("path", "")))
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"打开文件失败: {exc}") from exc
        if len(content) > 8000:
            return content[:8000] + "\n\n...[由于内容过长，已被系统截断至前 8000 字节]..."
        return content


class WriteFileTool:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir

    def name(self) -> str:
        return "write_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name(), "创建或覆盖写入一个文件。如果目录不存在会自动创建。", {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]})

    def execute(self, arguments: Any) -> str:
        args = parse_args(arguments)
        path = resolve_workspace_path(self.work_dir, str(args.get("path", "")))
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(args.get("content", "")), encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"写入文件失败: {exc}") from exc
        return f"成功将内容写入到文件: {args.get('path', '')}"


def fuzzy_replace(original: str, old: str, new: str) -> str:
    count = original.count(old)
    if count == 1:
        return original.replace(old, new, 1)
    if count > 1:
        raise ToolError(f"old_text 匹配到了 {count} 处，请提供更多的上下文代码以确保唯一性")
    content = original.replace("\r\n", "\n")
    normalized_old = old.replace("\r\n", "\n")
    count = content.count(normalized_old)
    if count == 1:
        return content.replace(normalized_old, new, 1)
    trimmed = normalized_old.strip()
    if trimmed:
        count = content.count(trimmed)
        if count == 1:
            return content.replace(trimmed, new, 1)
    content_lines = content.split("\n")
    old_lines = [line.strip() for line in normalized_old.strip().split("\n")]
    if not old_lines or len(content_lines) < len(old_lines):
        raise ToolError("找不到该代码片段")
    matches = []
    for index in range(len(content_lines) - len(old_lines) + 1):
        if all(content_lines[index + offset].strip() == line for offset, line in enumerate(old_lines)):
            matches.append(index)
    if not matches:
        raise ToolError("在文件中未找到 old_text，请大模型先调用 read_file 仔细确认文件内容和缩进")
    if len(matches) > 1:
        raise ToolError(f"模糊匹配到了 {len(matches)} 处相似代码，请提供更多上下行代码以精确定位")
    start = matches[0]
    return "\n".join(content_lines[:start] + [new] + content_lines[start + len(old_lines):])


class EditFileTool:
    def __init__(self, work_dir: str) -> None:
        self.work_dir = work_dir

    def name(self) -> str:
        return "edit_file"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name(), "对现有文件进行局部字符串替换，请提供足够上下文确保唯一。", {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]})

    def execute(self, arguments: Any) -> str:
        args = parse_args(arguments)
        path = resolve_workspace_path(self.work_dir, str(args.get("path", "")))
        try:
            original = path.read_text(encoding="utf-8")
            updated = fuzzy_replace(original, str(args.get("old_text", "")), str(args.get("new_text", "")))
            path.write_text(updated, encoding="utf-8")
        except ToolError:
            raise
        except OSError as exc:
            raise ToolError(f"读取或写回文件失败，请确认路径是否正确: {exc}") from exc
        return f"✅ 成功修改文件: {args.get('path', '')}"


class BashTool:
    def __init__(self, work_dir: str, timeout: float = 30.0) -> None:
        self.work_dir = work_dir
        self.timeout = timeout

    def name(self) -> str:
        return "bash"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name(), "在当前工作区执行任意的 bash 命令，返回 stdout 和 stderr。", {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]})

    def execute(self, arguments: Any) -> str:
        args = parse_args(arguments)
        bash = shutil.which("bash")
        if not bash:
            raise ToolError("系统中未找到 bash，无法执行 bash 工具")
        try:
            completed = subprocess.run([bash, "-lc", str(args.get("command", ""))], cwd=self.work_dir, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return output + f"\n[警告: 命令执行超时({int(self.timeout)}s)，已被系统强制终止。]"
        output = (completed.stdout or "") + (completed.stderr or "")
        if completed.returncode != 0:
            return f"执行报错: exit code {completed.returncode}\n输出:\n{output}"
        if not output:
            return "命令执行成功，无终端输出。"
        return output if len(output) <= 8000 else output[:8000] + "\n\n...[终端输出过长，已截断至前 8000 字节]..."


Middleware = Callable[[ToolCall], tuple[bool, str]]


class Registry:
    def __init__(self) -> None:
        import threading

        self._tools: dict[str, BaseTool] = {}
        self._middlewares: list[Middleware] = []
        self._lock = threading.RLock()

    def use(self, middleware: Middleware) -> None:
        with self._lock:
            self._middlewares.append(middleware)

    def register(self, tool: BaseTool) -> None:
        with self._lock:
            self._tools[tool.name()] = tool
        log.info("[Registry] 成功挂载工具: %s", tool.name())

    def get_available_tools(self) -> list[ToolDefinition]:
        with self._lock:
            return [tool.definition() for tool in self._tools.values()]

    def execute(self, call: ToolCall, parent_span: Span | None = None) -> ToolResult:
        span = start_span(parent_span, "Tool.Execute")
        span.add_attribute("tool_name", call.name)
        span.add_attribute("arguments", call.arguments_json())
        try:
            with self._lock:
                tool = self._tools.get(call.name)
                middlewares = list(self._middlewares)
            if tool is None:
                return ToolResult(call.id, f"Error: 系统中不存在名为 '{call.name}' 的工具。", True)
            for middleware in middlewares:
                allowed, reason = middleware(call)
                if not allowed:
                    span.add_attribute("intercepted", True)
                    return ToolResult(call.id, f"执行被系统拦截。原因: {reason}", True)
            try:
                output = tool.execute(call.arguments)
            except Exception as exc:
                output = f"Error executing {call.name}: {exc}"
                return ToolResult(call.id, output, True)
            span.add_attribute("output_preview", output[:100])
            return ToolResult(call.id, output, False)
        finally:
            span.end()


class SubagentTool:
    def __init__(self, runner: Any, read_only_registry: Registry, reporter: Any = None) -> None:
        self.runner = runner
        self.read_only_registry = read_only_registry
        self.reporter = reporter

    def name(self) -> str:
        return "spawn_subagent"

    def definition(self) -> ToolDefinition:
        return ToolDefinition(self.name(), "派出一个用于深度探索的只读子智能体。", {"type": "object", "properties": {"task_prompt": {"type": "string"}}, "required": ["task_prompt"]})

    def execute(self, arguments: Any) -> str:
        args = parse_args(arguments)
        try:
            summary = self.runner.run_sub(str(args.get("task_prompt", "")), self.read_only_registry, self.reporter)
            return f"【子智能体探索报告】：\n{summary}"
        except Exception as exc:
            return f"子智能体执行失败: {exc}"
