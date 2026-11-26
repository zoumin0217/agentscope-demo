"""基础导入和模型配置测试。"""

import asyncio
import runpy
from contextlib import nullcontext
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


def refuse_ollama_connection(*args: Any, **kwargs: Any) -> None:
    """模拟默认 Ollama 服务未启动。"""
    raise OSError("Ollama is not running")


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
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    monkeypatch.setenv("OLLAMA_HOST", "")
    monkeypatch.setenv("OLLAMA_MODEL", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    module = load_script(script_name)
    namespace = module["build_model"].__globals__
    monkeypatch.setattr(
        namespace["socket"],
        "create_connection",
        refuse_ollama_connection,
    )

    with pytest.raises(SystemExit, match="未找到可用模型配置"):
        module["build_model"]()


@pytest.mark.parametrize("script_name", MODEL_BUILDER_SCRIPTS)
def test_build_model_provider_priority(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认各厂商按指定顺序选择，并使用标准 credential。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    for key in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_MODEL",
        "DASHSCOPE_API_KEY",
        "OLLAMA_HOST",
        "OLLAMA_MODEL",
        "DEEPSEEK_API_KEY",
    ):
        monkeypatch.setenv(key, "")
    module = load_script(script_name)
    namespace = module["build_model"].__globals__

    def factory(provider: str) -> Any:
        def create(**kwargs: Any) -> tuple[str, dict[str, Any]]:
            return provider, kwargs

        return create

    for provider in ("OpenAI", "DashScope", "Ollama", "DeepSeek"):
        monkeypatch.setitem(
            namespace,
            f"{provider}ChatModel",
            factory(provider),
        )

    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "dashscope-key")
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    result = module["build_model"]()
    assert result[0] == "OpenAI"
    assert type(result[1]["credential"]).__name__ == "OpenAICredential"
    assert result[1]["model"] == "gpt-4o-mini"

    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = module["build_model"]()
    assert result[0] == "DashScope"
    assert type(result[1]["credential"]).__name__ == "DashScopeCredential"
    assert result[1]["model"] == "qwen-plus"

    monkeypatch.setenv("DASHSCOPE_API_KEY", "")
    result = module["build_model"]()
    assert result[0] == "Ollama"
    assert type(result[1]["credential"]).__name__ == "OllamaCredential"
    assert result[1]["credential"].host == "http://localhost:11434"
    assert result[1]["model"] == "llama3.2"

    monkeypatch.setenv("OLLAMA_HOST", "")
    monkeypatch.setattr(
        namespace["socket"],
        "create_connection",
        lambda *args, **kwargs: nullcontext(),
    )
    result = module["build_model"]()
    assert result[0] == "Ollama"

    monkeypatch.setattr(
        namespace["socket"],
        "create_connection",
        refuse_ollama_connection,
    )
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    result = module["build_model"]()
    assert result[0] == "DeepSeek"
    assert type(result[1]["credential"]).__name__ == "DeepSeekCredential"
    assert result[1]["model"] == "deepseek-chat"


@pytest.mark.parametrize("reset_requested", [False, True])
def test_main_memory_middleware_and_optional_reset(
    reset_requested: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认记忆 middleware 已接入，且只有 --reset 会清空记忆目录。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    monkeypatch.chdir(tmp_path)
    memory_file = tmp_path / "agent_memory" / "Memory" / "user.md"
    memory_file.parent.mkdir(parents=True)
    memory_file.write_text("姓名：小明", encoding="utf-8")
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=true", encoding="utf-8")
    module = load_script("main_agent.py")
    namespace = module["main"].__globals__
    agent_kwargs: dict[str, Any] = {}

    class FakeAgent:
        def __init__(self, **kwargs: Any) -> None:
            agent_kwargs.update(kwargs)

    monkeypatch.setitem(namespace, "Agent", FakeAgent)
    monkeypatch.setitem(namespace, "build_model", lambda: object())
    monkeypatch.setattr("builtins.input", lambda prompt: "quit")
    args = module["parse_args"](["--reset"] if reset_requested else [])

    asyncio.run(module["main"](reset_memory=args.reset))

    assert memory_file.exists() is not reset_requested
    assert env_file.read_text(encoding="utf-8") == "KEEP_ME=true"
    assert len(agent_kwargs["middlewares"]) == 1
    middleware = agent_kwargs["middlewares"][0]
    assert type(middleware).__name__ == "AgenticMemoryMiddleware"
    assert middleware._workdir == "./agent_memory"
    assert "姓名" in agent_kwargs["system_prompt"]
    assert "偏好" in agent_kwargs["system_prompt"]


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
