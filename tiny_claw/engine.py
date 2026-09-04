from __future__ import annotations

import hashlib
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

from .context import Compactor, PromptComposer, RecoveryManager, Session
from .observability import Span, export_trace_to_file, start_span
from .schema import Message, ToolCall, ToolResult, ROLE_ASSISTANT, ROLE_USER
from .tools import Registry

log = logging.getLogger(__name__)


class Reporter(Protocol):
    def on_thinking(self) -> None: ...
    def on_tool_call(self, tool_name: str, arguments: str) -> None: ...
    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None: ...
    def on_message(self, content: str) -> None: ...


class TerminalReporter:
    def on_thinking(self) -> None:
        print("\n[🤔 思考中] 模型正在推理...")

    def on_tool_call(self, tool_name: str, arguments: str) -> None:
        display = arguments.replace("\n", "\\n").replace("\r", "\\r")
        if len(display) > 150:
            display = display[:150] + "... (已截断)"
        print(f"[🛠️ 调用工具] {tool_name}\n   参数: {display}")

    def on_tool_result(self, tool_name: str, result: str, is_error: bool) -> None:
        status = "❌ 执行失败" if is_error else "✅ 执行成功"
        print(f"[{status}] {tool_name}")
        if is_error and result:
            print(f"   错误: {result}")

    def on_message(self, content: str) -> None:
        if content:
            print(f"\n🤖 Agent 回复:\n{content}\n")


class ReminderInjector:
    def __init__(self) -> None:
        self._failures: dict[str, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _fingerprint(call: ToolCall) -> str:
        return hashlib.md5((call.name + call.arguments_json()).encode()).hexdigest()

    def check_and_inject(self, call: ToolCall | None, result: ToolResult | None) -> Message | None:
        if call is None or result is None:
            return None
        with self._lock:
            if not result.is_error:
                self._failures.clear()
                return None
            fingerprint = self._fingerprint(call)
            self._failures[fingerprint] = self._failures.get(fingerprint, 0) + 1
            count = self._failures[fingerprint]
        if count < 3:
            return None
        return Message(
            ROLE_USER,
            f"""[SYSTEM REMINDER 警告]
你似乎陷入了死循环。你刚刚连续 {count} 次使用相同的参数调用了 '{call.name}' 工具，并且都失败了。
请立即停止这种无效的重试！你需要停止猜测参数、彻底改变策略；如果无法解决，请直接向用户说明需要人工帮助。""",
        )


class AgentEngine:
    def __init__(self, provider: Any, registry: Registry, enable_thinking: bool = False, plan_mode: bool = True) -> None:
        self.provider = provider
        self.registry = registry
        self.enable_thinking = enable_thinking
        self.plan_mode = plan_mode
        self.compactor = Compactor(200_000, 6)
        self.recovery = RecoveryManager()
        self.injector = ReminderInjector()

    def _run_tool(self, call: ToolCall, reporter: Reporter | None, span: Span) -> tuple[ToolResult, str]:
        if reporter:
            reporter.on_tool_call(call.name, call.arguments_json())
        result = self.registry.execute(call, span)
        output = result.output
        if result.is_error:
            output = self.recovery.analyze_and_inject(call.name, output)
        if reporter:
            reporter.on_tool_result(call.name, output[:200] + ("... (已截断)" if len(output) > 200 else ""), result.is_error)
        return result, output

    def run(self, session: Session, reporter: Reporter | None = None) -> None:
        log.info("[Engine] 唤醒会话 [%s]，工作区: %s (PlanMode: %s)", session.id, session.work_dir, self.plan_mode)
        root = start_span(None, "Agent.Run")
        root.add_attribute("SessionID", session.id)
        root.add_attribute("WorkDir", session.work_dir)
        try:
            system_message = PromptComposer(session.work_dir, self.plan_mode).build()
            turn = 0
            while True:
                turn += 1
                turn_span = start_span(root, f"Turn-{turn}")
                try:
                    available = self.registry.get_available_tools()
                    working = session.get_working_memory(20)
                    if working and working[0].role != ROLE_USER:
                        working.insert(0, Message(ROLE_USER, "[系统占位符] 这是为了保持上下文连贯性而注入的断点标记。请继续执行你刚才的任务。"))
                    context = self.compactor.compact([system_message, *working])
                    turn_span.add_attribute("context_message_count", len(context))
                    thinking_content = ""
                    if self.enable_thinking:
                        if reporter:
                            reporter.on_thinking()
                        thinking_span = start_span(turn_span, "LLM.Thinking")
                        try:
                            thinking = self.provider.generate(context, [])
                            thinking_content = thinking.content
                        except Exception as exc:
                            raise RuntimeError(f"Thinking 阶段失败: {exc}") from exc
                        finally:
                            thinking_span.end()
                        if thinking_content:
                            context = [*context, thinking]
                    action_span = start_span(turn_span, "LLM.Action")
                    try:
                        try:
                            action = self.provider.generate(context, available)
                        except Exception as exc:
                            raise RuntimeError(f"Action 阶段失败: {exc}") from exc
                    finally:
                        action_span.end()
                    final_content = (thinking_content + "\n" + action.content).strip()
                    session.append(Message(ROLE_ASSISTANT, final_content, list(action.tool_calls), usage=action.usage))
                    if action.content and reporter:
                        reporter.on_message(action.content)
                    if not action.tool_calls:
                        break
                    results: list[tuple[ToolResult, str] | None] = [None] * len(action.tool_calls)
                    with ThreadPoolExecutor(max_workers=len(action.tool_calls)) as pool:
                        futures = [pool.submit(self._run_tool, call, reporter, turn_span) for call in action.tool_calls]
                        for index, future in enumerate(futures):
                            results[index] = future.result()
                    observations = [
                        Message(ROLE_USER, output, tool_call_id=call.id)
                        for call, result_pair in zip(action.tool_calls, results)
                        if result_pair is not None
                        for output in [result_pair[1]]
                    ]
                    session.append(*observations)
                    first_result, _ = results[0] if results[0] is not None else (None, "")
                    reminder = self.injector.check_and_inject(action.tool_calls[0], first_result)
                    if reminder:
                        session.append(reminder)
                finally:
                    turn_span.end()
        finally:
            root.end()
            try:
                export_trace_to_file(root, session.work_dir, session.id)
            except OSError:
                log.exception("[Tracing] 无法导出 Trace")

    def run_sub(self, task_prompt: str, read_only_registry: Registry, reporter: Reporter | None = None) -> str:
        history = [
            Message(ROLE_USER, "你是一个专门负责深度探索的探路者。必须使用内置只读工具搜集确切信息，找到答案后输出纯文本汇报。"),
            Message(ROLE_USER, task_prompt),
        ]
        for _ in range(10):
            response = self.provider.generate(self.compactor.compact(history), read_only_registry.get_available_tools())
            history.append(response)
            if not response.tool_calls:
                return response.content
            outputs: list[str] = [""] * len(response.tool_calls)
            with ThreadPoolExecutor(max_workers=len(response.tool_calls)) as pool:
                futures = []
                for call in response.tool_calls:
                    if reporter:
                        reporter.on_tool_call(f"[Subagent] {call.name}", call.arguments_json())
                    futures.append(pool.submit(read_only_registry.execute, call))
                for index, future in enumerate(futures):
                    result = future.result()
                    outputs[index] = self.recovery.analyze_and_inject(response.tool_calls[index].name, result.output) if result.is_error else result.output
                    if reporter:
                        reporter.on_tool_result(f"[Subagent] {response.tool_calls[index].name}", outputs[index][:200], result.is_error)
            history.extend(Message(ROLE_USER, output, tool_call_id=call.id) for call, output in zip(response.tool_calls, outputs))
        raise RuntimeError("子智能体探索过于深入，超过 10 轮被强制召回，请主 Agent 给它更明确的指令")
