import tempfile
import unittest
from pathlib import Path

from tiny_claw.schema import ToolCall
from tiny_claw.tools import EditFileTool, ReadFileTool, Registry, ToolError, WriteFileTool, fuzzy_replace, resolve_workspace_path


class ToolsTest(unittest.TestCase):
    def test_fuzzy_replace_levels(self) -> None:
        self.assertEqual(fuzzy_replace("a\r\nb", "a\nb", "c"), "c")
        self.assertEqual(fuzzy_replace("  a\n  b\n", "a\nb", "c"), "c\n")
        self.assertEqual(fuzzy_replace("x\n  a\n  b\ny", "a\nb", "c"), "x\nc\ny")

    def test_fuzzy_replace_rejects_ambiguous(self) -> None:
        with self.assertRaises(ToolError):
            fuzzy_replace("a\na", "a", "b")

    def test_file_tools_and_workspace_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            WriteFileTool(directory).execute({"path": "nested/a.txt", "content": "hello"})
            self.assertEqual(ReadFileTool(directory).execute({"path": "nested/a.txt"}), "hello")
            EditFileTool(directory).execute({"path": "nested/a.txt", "old_text": "hello", "new_text": "world"})
            self.assertEqual(Path(directory, "nested/a.txt").read_text(), "world")
            with self.assertRaises(ToolError):
                resolve_workspace_path(directory, "../outside.txt")

    def test_registry_middleware_and_unknown_tool(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry()
            registry.register(ReadFileTool(directory))
            registry.use(lambda _call: (False, "测试拦截"))
            blocked = registry.execute(ToolCall("1", "read_file", {"path": "x"}))
            self.assertTrue(blocked.is_error)
            self.assertIn("测试拦截", blocked.output)
            unknown = Registry().execute(ToolCall("2", "missing", {}))
            self.assertTrue(unknown.is_error)
