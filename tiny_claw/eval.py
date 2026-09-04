from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .context import Session
from .engine import AgentEngine
from .observability import CostTracker
from .providers import ZhipuOpenAIProvider
from .schema import Message, ROLE_USER
from .tools import BashTool, EditFileTool, ReadFileTool, Registry, WriteFileTool

log = logging.getLogger(__name__)


@dataclass
class TestCase:
    id: str
    name: str
    setup_script: str
    task_prompt: str
    validate_script: str
    max_turns: int = 0


@dataclass
class TestResult:
    test_case_id: str
    passed: bool
    total_cost_cny: float = 0.0
    duration_ms: int = 0
    error_msg: str = ""


def _run_script(script: str, cwd: Path) -> tuple[bool, str]:
    bash = shutil.which("bash")
    if not bash:
        return False, "系统中未找到 bash，无法执行评测脚本"
    try:
        completed = subprocess.run([bash, "-lc", script], cwd=cwd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode == 0, output


class BenchmarkRunner:
    def __init__(self, model_name: str = "glm-4.5-air") -> None:
        self.model_name = model_name

    def run_suite(self, testcases: list[TestCase]) -> list[TestResult]:
        results = [self.run_single_test(testcase) for testcase in testcases]
        passed = sum(result.passed for result in results)
        total = sum(result.total_cost_cny for result in results)
        print("\n================ 🏆 跑分终极报告 ================")
        print(f"总用例数: {len(results)} | 成功数: {passed} | 成功率: {(passed / len(results) * 100 if results else 0):.2f}%")
        print(f"总消耗成本: ¥{total:.6f}")
        return results

    def run_single_test(self, testcase: TestCase) -> TestResult:
        started = time.perf_counter()
        root = Path.cwd() / "workspace" / f"{testcase.id}_{time.time_ns()}"
        root.mkdir(parents=True, exist_ok=True)
        if testcase.setup_script:
            ok, output = _run_script(testcase.setup_script, root)
            if not ok:
                return TestResult(testcase.id, False, error_msg=f"靶机 Setup 失败: {output}")
        try:
            provider = ZhipuOpenAIProvider(self.model_name)
            session = Session.new(testcase.id, str(root))
            tracked = CostTracker(provider, self.model_name, session)
            registry = Registry()
            for tool in (ReadFileTool(str(root)), WriteFileTool(str(root)), BashTool(str(root)), EditFileTool(str(root))):
                registry.register(tool)
            session.append(Message(ROLE_USER, testcase.task_prompt))
            AgentEngine(tracked, registry, enable_thinking=False, plan_mode=False).run(session)
        except Exception as exc:
            return TestResult(testcase.id, False, session.total_cost_cny if "session" in locals() else 0.0, error_msg=f"Agent 崩溃: {exc}")
        ok, output = _run_script(testcase.validate_script, root)
        duration = int((time.perf_counter() - started) * 1000)
        if not ok:
            return TestResult(testcase.id, False, session.total_cost_cny, duration, f"验证脚本执行失败: {output}")
        return TestResult(testcase.id, True, session.total_cost_cny, duration)


DEFAULT_TESTCASES = [
    TestCase("test_001_edit", "测试模糊替换工具的准确性", "echo '{\"name\": \"tiny-claw\", \"version\": \"v1.0.0\"}' > config.json", "当前目录下有一个 config.json。请你使用 edit_file 工具，将其中的 version 从 v1.0.0 改为 v2.0.0。不要做其他多余操作。", "grep '\"version\": \"v2.0.0\"' config.json"),
    TestCase("test_002_code_gen", "测试代码阅读与创建新文件的综合能力", "printf 'package math\\n\\nfunc Multiply(a, b int) int {\\n\\treturn a * b\\n}\\n' > math.go", "当前目录下有一个 math.go。请阅读它并写一个规范的 math_test.go 测试 Multiply。", "grep 'func TestMultiply' math_test.go"),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="运行 tiny-claw 自动化评测")
    parser.add_argument("--model", default="glm-4.5-air")
    args = parser.parse_args(argv)
    if not os.getenv("ZHIPU_API_KEY"):
        parser.error("请先设置 ZHIPU_API_KEY 环境变量进行跑分测试")
    BenchmarkRunner(args.model).run_suite(DEFAULT_TESTCASES)
    return 0
