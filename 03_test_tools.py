"""基础导入和模型配置测试。"""

import asyncio
import runpy
from pathlib import Path
from typing import Any

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_SCRIPTS = (
    "01_hello_agent.py",
    "02_stream_events.py",
    "main_agent.py",
)
MODEL_BUILDER_SCRIPTS = (
    "01_hello_agent.py",
    "main_agent.py",
)


def load_script(script_name: str) -> dict[str, Any]:
    """加载脚本但不运行其命令行入口。"""
    return runpy.run_path(
        str(PROJECT_ROOT / script_name),
        run_name=f"_pytest_{Path(script_name).stem}",
    )


@pytest.mark.parametrize("script_name", EXAMPLE_SCRIPTS)
def test_example_scripts_import(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认各示例脚本的模块级导入可以正常完成。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))

    load_script(script_name)


@pytest.mark.parametrize("script_name", MODEL_BUILDER_SCRIPTS)
def test_build_model_without_api_key_exits(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """没有配置 API Key 时，模型配置应明确退出。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    module = load_script(script_name)

    with pytest.raises(SystemExit, match="未配置 API Key"):
        module["build_model"]()


@pytest.mark.skip(reason="需要有效 API Key 和可用网络，避免测试时调用真实模型服务")
def test_real_model_response() -> None:
    """可在配置真实 API 后手动启用的在线模型冒烟测试。"""
    from agentscope.agent import Agent
    from agentscope.message import UserMsg

    module = load_script("main_agent.py")
    agent = Agent(
        name="测试助手",
        system_prompt="请简短回复。",
        model=module["build_model"](),
    )
    reply = asyncio.run(
        agent.reply(UserMsg(name="pytest", content="回复：测试成功")),
    )

    assert reply.get_text_content().strip()
