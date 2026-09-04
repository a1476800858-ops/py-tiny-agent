import tempfile
import unittest
from pathlib import Path

from tiny_claw.context import Compactor, PromptComposer, RecoveryManager, Session
from tiny_claw.schema import Message, ToolCall, ROLE_ASSISTANT, ROLE_SYSTEM, ROLE_USER


class ContextTest(unittest.TestCase):
    def test_session_drops_orphan_tool_result(self) -> None:
        session = Session.new("s", ".")
        session.append(Message(ROLE_USER, "question"), Message(ROLE_ASSISTANT, "action", [ToolCall("1", "x", {})]), Message(ROLE_USER, "observation", tool_call_id="1"), Message(ROLE_USER, "next"))
        memory = session.get_working_memory(2)
        self.assertEqual([item.content for item in memory], ["next"])

    def test_compactor_preserves_system_and_tool_calls(self) -> None:
        compactor = Compactor(10, 1)
        call = ToolCall("1", "bash", {"command": "x"})
        messages = [Message(ROLE_SYSTEM, "system"), Message(ROLE_ASSISTANT, "x" * 300, [call]), Message(ROLE_USER, "y" * 1200, tool_call_id="1")]
        result = compactor.compact(messages)
        self.assertEqual(result[0].content, "system")
        self.assertEqual(result[1].tool_calls[0].name, "bash")
        self.assertIn("内容过长", result[2].content)

    def test_recovery_hint(self) -> None:
        result = RecoveryManager().analyze_and_inject("edit_file", "在文件中未找到 old_text")
        self.assertIn("read_file", result)

    def test_prompt_loads_agents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "AGENTS.md").write_text("只允许测试", encoding="utf-8")
            prompt = PromptComposer(directory, False).build()
            self.assertIn("只允许测试", prompt.content)
