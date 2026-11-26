"""带有本地命令和文件操作工具的 AgentScope 2.x 智能体。"""

import argparse
import asyncio
import shutil
from pathlib import Path
from typing import AsyncGenerator

from agentscope.agent import Agent
from agentscope.event import (
    AgentEvent,
    ConfirmResult,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from agentscope.middleware import AgenticMemoryMiddleware
from agentscope.message import Msg, UserMsg
from agentscope.tool import Bash, Read, Toolkit, Write
from model_config import build_model

MEMORY_WORKDIR = "./agent_memory"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行选项。"""
    parser = argparse.ArgumentParser(description="带记忆的本地文件助手")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="启动前清空本地记忆",
    )
    return parser.parse_args(argv)


def _reset_memory_directory() -> None:
    """仅清理本助手的记忆目录，避免误删其他路径。"""
    memory_path = Path(MEMORY_WORKDIR)
    if memory_path.is_symlink():
        raise RuntimeError(f"拒绝清理符号链接记忆目录：{memory_path}")
    if memory_path.exists():
        shutil.rmtree(memory_path)


async def reply_stream_with_confirmation(
    agent: Agent,
    inputs: Msg,
) -> AsyncGenerator[AgentEvent | Msg, None]:
    """Stream a reply and request approval before running pending tools."""
    current_input: Msg | UserConfirmResultEvent = inputs

    while True:
        pending_confirmations: list[RequireUserConfirmEvent] = []
        final_message: Msg | None = None

        async for event in agent.reply_stream(
            current_input,
            yield_final_msg=True,
        ):
            if isinstance(event, RequireUserConfirmEvent):
                pending_confirmations.append(event)
                yield event
            elif isinstance(event, RequireExternalExecutionEvent):
                tool_names = ", ".join(
                    tool_call.name for tool_call in event.tool_calls
                )
                raise RuntimeError(
                    "当前程序没有外部工具执行器，无法执行："
                    f"{tool_names}"
                )
            elif isinstance(event, Msg):
                final_message = event
            else:
                yield event

        if not pending_confirmations:
            if final_message is None:
                raise RuntimeError("智能体结束时没有返回最终消息。")
            yield final_message
            return

        confirm_results: list[ConfirmResult] = []
        for pending in pending_confirmations:
            for tool_call in pending.tool_calls:
                print(
                    f"\n工具请求：{tool_call.name}\n"
                    f"参数：{tool_call.input}"
                )
                answer = await asyncio.to_thread(
                    input,
                    "是否允许执行？输入 y 确认，其他输入拒绝：[y/N] ",
                )
                confirm_results.append(
                    ConfirmResult(
                        confirmed=answer.strip().lower() in {"y", "yes"},
                        tool_call=tool_call,
                    ),
                )

        current_input = UserConfirmResultEvent(
            reply_id=pending_confirmations[0].reply_id,
            confirm_results=confirm_results,
        )


async def main(reset_memory: bool = False) -> None:
    """运行文件助手的交互式对话。"""
    if reset_memory:
        _reset_memory_directory()

    toolkit = Toolkit(tools=[Bash(), Read(), Write()])
    agent = Agent(
        name="文件助手",
        system_prompt=(
            "你是一个中文本地文件助手，可以在本地运行命令、读取文件和写入文件。"
            "请根据用户的需求自主决定是否调用以及调用哪个工具；"
            "请留意用户每次明确提供或更正的姓名、称呼和长期偏好，"
            "立即按记忆规范保存或更新，并在后续相关对话中自然引用；"
            "不要猜测用户未明确提供的信息。完成任务后用中文简洁回答。"
        ),
        model=build_model(),
        toolkit=toolkit,
        middlewares=[
            AgenticMemoryMiddleware(workdir=MEMORY_WORKDIR),
        ],
    )

    print("=== 文件助手 ===")
    print("可用能力：运行本地命令、读取文件、写入文件。输入 quit、exit 或 退出结束对话。")

    while True:
        try:
            user_input: str = input("你：").strip()
        except EOFError:
            print("\n输入已结束，对话退出。")
            break

        if user_input.lower() in {"quit", "exit", "退出"}:
            print("文件助手：再见。")
            break
        if not user_input:
            continue

        reply: Msg | None = None
        async for event in reply_stream_with_confirmation(
            agent,
            UserMsg(name="user", content=user_input),
        ):
            if isinstance(event, Msg):
                reply = event

        if reply is None:
            raise RuntimeError("智能体结束时没有返回最终消息。")
        print(f"文件助手：{reply.get_text_content()}")


if __name__ == "__main__":
    asyncio.run(main(reset_memory=parse_args().reset))