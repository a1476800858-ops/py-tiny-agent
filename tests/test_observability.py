import tempfile
import unittest
from pathlib import Path

from tiny_claw.context import Session
from tiny_claw.observability import CostTracker, export_trace_to_file, start_span
from tiny_claw.providers import MockProvider
from tiny_claw.schema import Message, Usage, ROLE_ASSISTANT


class ObservabilityTest(unittest.TestCase):
    def test_cost_and_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            session = Session.new("s", directory)
            provider = MockProvider([Message(ROLE_ASSISTANT, "ok", usage=Usage(100, 50))])
            CostTracker(provider, "glm-4.5-air", session).generate([], [])
            self.assertAlmostEqual(session.total_cost_cny, 0.0000225)
            root = start_span(None, "root")
            start_span(root, "child").end()
            root.end()
            path = export_trace_to_file(root, directory, "s")
            self.assertTrue(Path(path).exists())
