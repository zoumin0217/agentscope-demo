"""AgentScope 2.x 最小对话示例。"""

import asyncio
import os

from dotenv import load_dotenv

from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential, OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import DeepSeekChatModel, OpenAIChatModel

load_dotenv()


def build_model() -> DeepSeekChatModel | OpenAIChatModel:
    """根据环境变量创建 DeepSeek 或 OpenAI 对话模型。"""
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        return DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=deepseek_api_key),
            model="deepseek-chat",
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIChatModel(
            credential=OpenAICredential(
                api_key=openai_api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            ),
            model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        )

    raise ValueError(
        "未配置 API Key，请在项目根目录的 .env 文件中设置 "
        "DEEPSEEK_API_KEY 或 OPENAI_API_KEY。"
    )


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
