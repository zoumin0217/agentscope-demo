"""AgentScope 2.x 最小对话示例。"""

import asyncio

from agentscope.agent import Agent
from agentscope.message import UserMsg
from model_config import build_model


async def main() -> None:
    """创建智能体并发送一条问候消息。"""
    agent = Agent(
        name="小助",
        system_prompt="你是一个友好、简洁的中文 AI 助手。",
        model=build_model(),
    )
    reply = await agent.reply(
        UserMsg(name="用户", content="介绍下你自己"),
    )
    print(reply.content)


if __name__ == "__main__":
    asyncio.run(main())
