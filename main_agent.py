"""带有本地命令和文件操作工具的 AgentScope 2.x 智能体。

本模块同时提供可复用的异步流式回复函数、记忆目录管理、
命令行参数解析和交互式聊天入口。
"""

import argparse  # argparse 负责把命令行参数转换为结构化对象。
import asyncio  # asyncio 提供异步事件循环和非阻塞等待能力。
import shutil  # shutil 提供递归删除目录等文件系统操作。
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
    """解析命令行选项。

    argv 为可选参数列表，主要用于测试或在其他 Python 函数中调用时传入参数；
    若不传入，argparse 会读取真实的 sys.argv。
    """
    parser = argparse.ArgumentParser(description="带记忆的本地文件助手")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="启动前清空本地记忆",
    )
    return parser.parse_args(argv)


def _reset_memory_directory() -> None:
    """仅清理本助手的记忆目录，避免误删其他路径。

    shutil.rmtree 会递归删除目录及其中的全部文件；这里先拒绝符号链接，
    再仅根据 MEMORY_WORKDIR 构造明确删除目标，降低误删风险。
    """
    memory_path = Path(MEMORY_WORKDIR)
    if memory_path.is_symlink():
        raise RuntimeError(f"拒绝清理符号链接记忆目录：{memory_path}")
    if memory_path.exists():
        shutil.rmtree(memory_path)


async def reply_stream_with_confirmation(
    agent: Agent,
    inputs: Msg,
) -> AsyncGenerator[AgentEvent | Msg, None]:
    """流式回复并在工具执行前逐项请求用户确认。

    返回类型使用异步生成器：调用方可一边接收事件，一边渲染。
    用户拒绝某项工具后，对应工具不会进入已确认结果列表。
    """
    # current_input 是下一轮 reply_stream 的输入；首次为普通用户消息，
    # 后续确认轮次替换为 UserConfirmResultEvent。
    current_input: Msg | UserConfirmResultEvent = inputs

    while True:
        # pending_confirmations 收集本轮模型提出的所有确认需求；
        # final_message 保存 yield_final_msg=True 时产生的最终文本消息。
        pending_confirmations: list[RequireUserConfirmEvent] = []
        final_message: Msg | None = None

        async for event in agent.reply_stream(
            current_input,
            yield_final_msg=True,
        ):
            if isinstance(event, RequireUserConfirmEvent):
                # 先保存确认事件，又转发给调用方，以便上层打印或渲染状态。
                pending_confirmations.append(event)
                yield event
            elif isinstance(event, RequireExternalExecutionEvent):
                # ExternalExecution 表示工具需要宿主进程自行执行；
                # 当前示例没有实现该执行器，因此明确报错而不是静默忽略。
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
                # 文本增量、工具调用和工具结果等普通事件原样向上传递。
                yield event

        if not pending_confirmations:
            if final_message is None:
                raise RuntimeError("智能体结束时没有返回最终消息。")
            # 事件流自然结束时返回最终消息，然后 return 终止生成器。
            yield final_message
            return

        # 对每个待确认工具逐一询问；答案不区分大小写，并以 y/yes 开头等价。
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