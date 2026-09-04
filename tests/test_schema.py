import json
import unittest

from tiny_claw.schema import Message, ToolCall, Usage, ROLE_ASSISTANT


class SchemaTest(unittest.TestCase):
    def test_round_trip(self) -> None:
        original = Message(ROLE_ASSISTANT, "完成", [ToolCall("1", "read_file", {"path": "a.txt"})], usage=Usage(3, 4))
        restored = Message.from_dict(json.loads(json.dumps(original.to_dict(), ensure_ascii=False)))
        self.assertEqual(restored.role, original.role)
        self.assertEqual(restored.tool_calls[0].arguments["path"], "a.txt")
        self.assertEqual(restored.usage.prompt_tokens, 3)
