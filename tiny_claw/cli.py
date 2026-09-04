from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from .context import GLOBAL_SESSION_MANAGER
from .engine import AgentEngine, TerminalReporter
from .observability import CostTracker
from .providers import ProviderError, ZhipuClaudeProvider, ZhipuOpenAIProvider
from .schema import Message, ROLE_USER
from .tools import BashTool, EditFileTool, ReadFileTool, Registry, WriteFileTool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="py-tiny-claw Agent CLI")
    parser.add_argument("--prompt", required=True, help="要交给 Agent 执行的任务描述")
    parser.add_argument("--dir", default=".", dest="work_dir", help="Agent 工作区目录")
    parser.add_argument("--session", default="cli_default_session", help="会话 ID")
    parser.add_argument("--model", default="glm-4.5-air")
    parser.add_argument("--provider", choices=("openai", "claude"), default="openai")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--plan-mode", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    work_dir = str(Path(args.work_dir).resolve())
    try:
        provider = ZhipuOpenAIProvider(args.model) if args.provider == "openai" else ZhipuClaudeProvider(args.model)
    except ProviderError as exc:
        print(f"启动失败: {exc}")
        return 1
    session = GLOBAL_SESSION_MANAGER.get_or_create(args.session, work_dir)
    tracked = CostTracker(provider, args.model, session)
    registry = Registry()
    for tool in (ReadFileTool(work_dir), WriteFileTool(work_dir), BashTool(work_dir), EditFileTool(work_dir)):
        registry.register(tool)
    session.append(Message(ROLE_USER, args.prompt))
    print("=" * 50)
    print(f"🚀 启动 py-tiny-claw CLI 引擎...\n📁 锁定工作区: {work_dir}\n🎯 收到任务: {args.prompt}")
    try:
        AgentEngine(tracked, registry, args.thinking, args.plan_mode).run(session, TerminalReporter())
    except Exception as exc:
        print(f"\n💥 引擎运行崩溃: {exc}")
        return 1
    print(f"✨ 任务圆满结束。\n💰 Session 累计消耗: ¥{session.total_cost_cny:.6f} | Token: Input {session.total_prompt_tokens}, Output {session.total_completion_tokens}")
    return 0
