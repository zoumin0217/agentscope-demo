"""展示 AgentScope 2.x 智能体的流式事件。"""

import asyncio

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.tool import Bash, Read, Toolkit, Write

from main_agent import build_model, reply_stream_with_confirmation


async def main() -> None:
    """运行一次工具任务并打印关键流式事件。"""
    agent = Agent(
        name="文件助手",
        system_prompt=(
            "你是一个中文本地文件助手，可以运行本地命令、读取文件和写入文件。"
            "请根据用户需求自主选择工具，并用中文简洁回答。"
        ),
        model=build_model(),
        toolkit=Toolkit(tools=[Bash(), Read(), Write()]),
    )

    print("=== 文件助手流式事件演示 ===")
    print("任务：运行 dir 或 ls 查看当前目录的文件。\n")

    async for event in reply_stream_with_confirmation(
        agent,
        UserMsg(
            name="user",
            content="请运行 dir 或 ls 查看当前目录有哪些文件，并简要列出。",
        ),
    ):
        event_name = type(event).__name__
        if "TextBlockDelta" in event_name and hasattr(event, "delta"):
            print(event.delta, end="", flush=True)
        elif "ToolCall" in event_name or "ToolResult" in event_name:
            print(f"\n[{event_name}]")

    print()


if __name__ == "__main__":
    asyncio.run(main())
