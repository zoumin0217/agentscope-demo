"""展示 AgentScope 2.x 智能体的流式事件。

流式输出可让用户在模型生成文本或调用工具时看到中间过程。
本示例复用 main_agent.py 中的确认逻辑，并演示事件类型如何被识别。
"""

import asyncio

from agentscope.agent import Agent
from agentscope.message import UserMsg
# Bash/Read/Write 分别提供终端命令、文件读取和文件写入能力；
# Toolkit 负责把多个工具注册到同一个智能体。
from agentscope.tool import Bash, Read, Toolkit, Write

# build_model 创建模型，reply_stream_with_confirmation 在敏感工具调用前请求确认。
from main_agent import build_model, reply_stream_with_confirmation


async def main() -> None:
    """运行一次工具任务并打印关键流式事件。"""
    # Toolkit 接收工具实例列表；智能体会根据模型的工具调用指令选择合适工具。
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

    # async for 遍历异步生成器：每产生一个事件就立即执行一次循环体，
    # 所以文本增量可以实时显示，而无需等待整段回复完成。
    async for event in reply_stream_with_confirmation(
        agent,
        UserMsg(
            name="user",
            content="请运行 dir 或 ls 查看当前目录有哪些文件，并简要列出。",
        ),
    ):
        # type(event).__name__ 提取事件对象的类名，用于判断属于文本增量还是工具事件。
        event_name = type(event).__name__
        if "TextBlockDelta" in event_name and hasattr(event, "delta"):
            print(event.delta, end="", flush=True)
        elif "ToolCall" in event_name or "ToolResult" in event_name:
            print(f"\n[{event_name}]")

    print()


if __name__ == "__main__":
    asyncio.run(main())
