"""使用 AgentScope MCPClient 接入 Playwright MCP 的交互式示例。"""

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from agentscope.agent import Agent
from agentscope.mcp import MCPClient
from agentscope.message import Msg, UserMsg
from agentscope.tool import Toolkit
from main_agent import reply_stream_with_confirmation
from model_config import build_model


DEMO_TASKS = {
    "1": (
        "网页截图",
        "打开 https://example.com，确认视口为 1280x720，"
        "截取当前视口并保存到 mcp_outputs/。"
        "截图时不要指定文件名，让 Playwright MCP 自动保存到输出目录，"
        "最后告诉我截图的保存路径。",
        ("--viewport-size", "1280x720"),
    ),
    "2": (
        "搜索 GitHub 仓库",
        "打开 https://github.com，使用 GitHub 搜索功能搜索 "
        "microsoft/playwright-mcp，然后打开结果列表中的第一个仓库结果。",
        (),
    ),
    "3": (
        "iPhone 15 模拟",
        "在 iPhone 15 设备模拟环境下打开 https://www.example.com，"
        "截取当前页面并保存到 mcp_outputs/。"
        "截图时不要指定文件名，让 Playwright MCP 自动保存到输出目录，"
        "最后告诉我截图的保存路径。",
        ("--device", "iPhone 15"),
    ),
}


def choose_task() -> tuple[str, str, tuple[str, ...]] | None:
    """显示示例任务菜单并返回所选任务。

    返回三元组依次是界面标题、发送给智能体的提示词和额外 MCP 启动参数。
    用户选择退出时返回 None。
    """
    print("请选择一个 Playwright MCP 演示任务：")
    for key, (title, _, _) in DEMO_TASKS.items():
        print(f"{key}. {title}")
    print("q. 退出")

    choice = input("请输入选项：").strip().lower()
    if choice in {"q", "quit", "exit"}:
        return None
    if choice not in DEMO_TASKS:
        raise ValueError(f"无效选项：{choice!r}，请选择 1、2、3 或 q。")

    title, prompt, extra_args = DEMO_TASKS[choice]
    return title, prompt, extra_args


def build_playwright_mcp_config(
    extra_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    """创建适用于当前操作系统的 Playwright MCP stdio 配置。"""
    if shutil.which("npx") is None:
        raise RuntimeError(
            "当前 Python 进程的 PATH 中找不到 npx。"
            "请安装 Node.js 20 或更高版本，然后重启 VS Code 或终端。",
        )

    Path("mcp_outputs").mkdir(parents=True, exist_ok=True)
    server_args = [
        "-y",
        "@playwright/mcp@latest",
        "--output-dir",
        "./mcp_outputs",
        *extra_args,
    ]

    if os.name == "nt":
        # Windows 的 npx 通常是 .cmd 文件，需通过 cmd.exe /c 启动。
        command = os.environ.get("COMSPEC", "cmd.exe")
        args = [
            "/d",
            "/s",
            "/c",
            subprocess.list2cmdline(["npx", *server_args]),
        ]
    else:
        command = "npx"
        args = server_args

    return {
        "type": "stdio_mcp",
        "command": command,
        "args": args,
    }


async def run_task(
    title: str,
    prompt: str,
    server_args: tuple[str, ...],
) -> None:
    """连接 Playwright MCP 并让浏览器助手执行所选任务。"""
    client = MCPClient(
        name="playwright",
        mcp_config=build_playwright_mcp_config(server_args),
        # stdio_mcp 是有状态连接；必须设置为 True 才能 connect()。
        is_stateful=True,
    )

    try:
        # AgentScope 2.0.12 的 connect() 是异步方法，必须 await。
        await client.connect()
        # 先 connect() 再构造 Toolkit，确保 MCP 工具已完成初始化。
        toolkit = Toolkit(mcps=[client])
        agent = Agent(
            name="浏览器助手",
            system_prompt=(
                "你是一个中文浏览器助手。请使用 Playwright MCP 提供的浏览器工具"
                "完成网页导航、交互和截图任务。操作前查看页面状态；"
                "保存截图时如用户没有指定文件名，不要传 filename，"
                "以便服务器将截图放入配置的输出目录。"
            ),
            model=build_model(),
            toolkit=toolkit,
        )

        print(f"\n正在执行：{title}")
        final_reply: Msg | None = None
        async for event in reply_stream_with_confirmation(
            agent,
            UserMsg(name="用户", content=prompt),
        ):
            if isinstance(event, Msg):
                final_reply = event

        if final_reply is None:
            raise RuntimeError("浏览器助手结束时没有返回最终答复。")
        print(final_reply.get_text_content())
    finally:
        # 即使模型请求或浏览器操作失败，也关闭 MCP 子进程与连接。
        await client.close()


async def main() -> None:
    """选择并执行一个 Playwright MCP 演示任务。"""
    selection = choose_task()
    if selection is None:
        return

    await run_task(*selection)


if __name__ == "__main__":
    asyncio.run(main())
