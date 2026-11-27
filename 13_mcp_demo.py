"""使用 AgentScope MCPClient 接入本机 stdio 文件系统服务。"""

import asyncio
import os
import shutil
from typing import Any

from agentscope.agent import Agent
from agentscope.mcp import MCPClient
from agentscope.message import UserMsg
from agentscope.tool import Toolkit
from model_config import build_model


def build_filesystem_mcp_config() -> dict[str, Any]:
    """构造适用于当前操作系统的本机文件系统 MCP 配置。

    MCP stdio 模式通过标准输入输出启动子进程并持续通信。
    配置中的 command 和 args 会告诉 AgentScope 如何启动该子进程。
    """
    if shutil.which("npx") is None:
        raise RuntimeError(
            "当前 Python 进程的 PATH 中找不到 npx。"
            "请安装 Node.js（包含 npm/npx），然后重启 VS Code 或终端。",
        )

    if os.name == "nt":
        # Windows 的 npx 通常是 .cmd 文件，CreateProcess 不能直接启动它；
        # 通过 cmd.exe /c 让 Windows shell 按 PATH 和 PATHEXT 解析 npx.cmd。
        return {
            "type": "stdio_mcp",
            "command": os.environ.get("COMSPEC", "cmd.exe"),
            "args": [
                "/d",
                "/s",
                "/c",
                "npx -y @modelcontextprotocol/server-filesystem ./",
            ],
        }

    return {
        "type": "stdio_mcp",
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./"],
    }


async def main() -> None:
    """连接本机文件系统 MCP，并让智能体列举当前目录。"""
    # stdio MCP 需要在会话期间保持同一个子进程和连接，因此启用有状态模式；
    # 这样才能显式 connect()，并在任务结束后 close()。
    client = MCPClient(
        name="filesystem",
        mcp_config=build_filesystem_mcp_config(),
        is_stateful=True,
    )

    # AgentScope 2.x 的 connect() 是异步方法；先连接，再把客户端交给 Toolkit。
    await client.connect()
    try:
        toolkit = Toolkit(mcps=[client])
        agent = Agent(
            name="文件助手",
            system_prompt="你是一个中文文件助手。请使用可用工具准确完成文件相关任务。",
            model=build_model(),
            toolkit=toolkit,
        )
        reply = await agent.reply(
            UserMsg(
                name="用户",
                content="请列举当前目录中的文件和文件夹。",
            ),
        )
        print(reply.content)
    finally:
        # 即使创建模型或执行任务时发生异常，也关闭 MCP 子进程和连接。
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
