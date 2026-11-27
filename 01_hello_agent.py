"""AgentScope 2.x 最小对话示例。

本文件刻意保持最短：演示如何创建一个智能体、发送一条用户消息，
并等待模型生成文本回复。它适合初学者先理解 AgentScope 最基础的调用链。
"""

# asyncio 是 Python 标准库模块，用于运行异步函数；Agent.reply 是异步方法，
# 因此不能直接调用，需通过 asyncio.run() 在事件循环中等待其完成。
import asyncio

# Agent 是具备提示词、模型和可选工具的对话主体。
from agentscope.agent import Agent
# UserMsg 表示一次来自用户的消息；name 主要用于区分对话角色和显示来源。
from agentscope.message import UserMsg
# build_model 根据环境变量和本机 Ollama 选择具体模型提供商。
from model_config import build_model


async def main() -> None:
    """创建智能体并发送一条问候消息。

    函数以 async 定义，表示内部可以使用 await 暂停等待而不阻塞其他异步任务。
    -> None 表示该函数执行完成后不主动返回业务数据。
    """
    # 在 Agent 中集中指定名称、系统提示词、模型和工具等运行配置。
    # system_prompt 会影响模型如何理解自己的角色和回答约束。
    agent = Agent(
        name="小助",
        system_prompt="你是一个友好、简洁的中文 AI 助手。",
        model=build_model(),
    )
    # UserMsg 中的 content 是本轮真正发送给模型的文本。
    # await 会等待 agent.reply 产生完整答复，而不是立即返回协程对象。
    reply = await agent.reply(
        UserMsg(name="用户", content="介绍下你自己"),
    )
    # print 将回复内容块转换为文本并输出到终端。
    print(reply.content)


# Python 直接执行脚本时会设置 __name__ 为 "__main__"；
# 当该文件被其他模块导入时不会进入此分支，这可以避免自动发送网络请求。
if __name__ == "__main__":
    asyncio.run(main())
