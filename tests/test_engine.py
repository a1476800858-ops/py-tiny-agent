import tempfile
import unittest

from tiny_claw.context import Session
from tiny_claw.engine import AgentEngine, ReminderInjector
from tiny_claw.providers import MockProvider
from tiny_claw.schema import Message, ToolCall, ToolResult, ROLE_ASSISTANT
from tiny_claw.tools import ReadFileTool, Registry, WriteFileTool


class EngineTest(unittest.TestCase):
    def test_engine_runs_tool_then_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            provider = MockProvider([
                Message(ROLE_ASSISTANT, "", [ToolCall("1", "write_file", {"path": "result.txt", "content": "ok"})]),
                Message(ROLE_ASSISTANT, "已经完成"),
            ])
            registry = Registry()
            registry.register(WriteFileTool(directory))
            session = Session.new("s", directory)
            session.append(Message("user", "写文件"))
            AgentEngine(provider, registry, plan_mode=False).run(session)
            with open(f"{directory}/result.txt", encoding="utf-8") as result_file:
                self.assertEqual(result_file.read(), "ok")
            self.assertEqual(len(provider.calls), 2)

    def test_reminder_after_three_identical_failures(self) -> None:
        injector = ReminderInjector()
        call = ToolCall("1", "read_file", {"path": "missing"})
        result = ToolResult("1", "失败", True)
        self.assertIsNone(injector.check_and_inject(call, result))
        self.assertIsNone(injector.check_and_inject(call, result))
        self.assertIn("死循环", injector.check_and_inject(call, result).content)

    def test_subagent_uses_read_only_registry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with open(f"{directory}/note.txt", "w", encoding="utf-8") as file:
                file.write("事实")
            provider = MockProvider([
                Message(ROLE_ASSISTANT, "", [ToolCall("1", "read_file", {"path": "note.txt"})]),
                Message(ROLE_ASSISTANT, "报告：事实"),
            ])
            read_only = Registry()
            read_only.register(ReadFileTool(directory))
            summary = AgentEngine(provider, read_only, plan_mode=False).run_sub("读取 note.txt", read_only)
            self.assertEqual(summary, "报告：事实")
