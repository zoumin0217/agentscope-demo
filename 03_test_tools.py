"""基础导入和模型配置测试。"""

import asyncio
import runpy
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest
from agentscope.message import UserMsg


PROJECT_ROOT = Path(__file__).resolve().parent
EXAMPLE_SCRIPTS = (
    "01_hello_agent.py",
    "02_stream_events.py",
    "11_team_agent.py",
    "12_review_team.py",
    "main_agent.py",
    "14_gradio_ui.py",
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


def test_gradio_chat_returns_only_final_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认 Gradio 只返回最终答复文本，并传入当前轮 UserMsg。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    module = load_script("14_gradio_ui.py")
    namespace = module["chat"].__globals__
    received: list[Any] = []

    class FakeAgent:
        async def reply_stream(
            self,
            inputs: Any,
            yield_final_msg: bool,
        ) -> Any:
            received.append(inputs)
            yield UserMsg(name="assistant", content="处理完成")

    monkeypatch.setitem(namespace, "agent", FakeAgent())

    result = asyncio.run(module["chat"]("读取文件", []))

    assert result == "处理完成"
    assert received[0].name == "user"
    assert received[0].get_text_content() == "读取文件"


def test_gradio_chat_confirms_tools_and_hides_internal_events(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """确认工具事件先显示必要参数，再通过用户答复继续执行。"""
    from agentscope.event import RequireUserConfirmEvent, UserConfirmResultEvent
    from agentscope.message import ToolCallBlock

    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    module = load_script("14_gradio_ui.py")
    namespace = module["chat"].__globals__
    tool_call = ToolCallBlock(
        id="call-1",
        name="Read",
        input='{"file_path":"README.md"}',
    )
    received: list[Any] = []

    class FakeAgent:
        async def reply_stream(
            self,
            inputs: Any,
            yield_final_msg: bool,
        ) -> Any:
            received.append(inputs)
            if isinstance(inputs, UserConfirmResultEvent):
                assert inputs.confirm_results[0].confirmed
                yield UserMsg(name="assistant", content="读取完成")
            else:
                yield RequireUserConfirmEvent(
                    reply_id="reply-1",
                    tool_calls=[tool_call],
                )

    monkeypatch.setitem(namespace, "agent", FakeAgent())

    prompt = asyncio.run(module["chat"]("读取 README", []))
    result = asyncio.run(module["chat"]("y", []))

    assert "Read" in prompt
    assert "README.md" in prompt
    assert "reply-1" not in prompt
    assert result == "读取完成"
    assert isinstance(received[1], UserConfirmResultEvent)


def test_gradio_chat_logs_errors_without_exposing_traceback(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """确认异常细节只进入服务端日志，前端收到友好提示。"""
    monkeypatch.syspath_prepend(str(PROJECT_ROOT))
    module = load_script("14_gradio_ui.py")
    namespace = module["chat"].__globals__

    class BrokenAgent:
        async def reply_stream(
            self,
            inputs: Any,
            yield_final_msg: bool,
        ) -> Any:
            raise RuntimeError("internal traceback detail")
            yield

    monkeypatch.setitem(namespace, "agent", BrokenAgent())

    result = asyncio.run(module["chat"]("触发错误", []))

    assert "抱歉" in result
    assert "internal traceback detail" not in result
    assert "internal traceback detail" in caplog.text


def test_review_team_returns_leader_summary() -> None:
    """确认需求以 UserMsg 交给团队，最终返回组长消息。"""
    from types import SimpleNamespace

    from agentscope.event import ReplyStartEvent
    from agentscope.message import Msg, TextBlock

    module = load_script("12_review_team.py")
    received: list[Any] = []
    summary = Msg(
        name="评审组长",
        role="assistant",
        content=[TextBlock(text="最终评审汇总")],
    )

    class FakePipeline:
        leader = SimpleNamespace(
            name="评审组长",
            state=SimpleNamespace(
                get_reply_message=lambda name: (
                    summary if name == "评审组长" else None
                ),
            ),
        )

        async def reply_stream(self, inputs: Any) -> Any:
            received.append(inputs)
            yield ReplyStartEvent(
                session_id="session",
                reply_id="reply",
                name="评审组长",
            )

    result = asyncio.run(
        module["review_requirement"]("实现导出功能", FakePipeline()),
    )

    assert result == "最终评审汇总"
    assert isinstance(received[0], Msg)
    assert received[0].role == "user"
    assert received[0].get_text_content() == "实现导出功能"


def test_team_agent_returns_leader_reply() -> None:
    """确认 TeamPipeline 输入到达领队，且最终只取领队答复。"""
    from types import SimpleNamespace

    from agentscope.event import ReplyStartEvent
    from agentscope.message import Msg, TextBlock

    module = load_script("11_team_agent.py")
    received: list[Any] = []
    summary = Msg(
        name="领队",
        role="assistant",
        content=[TextBlock(text="团队最终稿")],
    )

    class FakePipeline:
        leader = SimpleNamespace(
            name="领队",
            state=SimpleNamespace(
                get_reply_message=lambda name: (
                    summary if name == "领队" else None
                ),
            ),
        )

        async def reply_stream(self, inputs: Any) -> Any:
            received.append(inputs)
            for name in ("领队", "调研", "写作", "评审"):
                yield ReplyStartEvent(
                    session_id="session",
                    reply_id=f"reply-{name}",
                    name=name,
                )

    result = asyncio.run(
        module["run_team_task"]("写一段介绍", FakePipeline()),
    )

    assert result == "团队最终稿"
    assert received[0].role == "user"
    assert received[0].get_text_content() == "写一段介绍"


def test_team_agent_rejects_missing_member_calls() -> None:
    """避免没有调用全部成员时仍返回看似成功的汇总。"""
    from types import SimpleNamespace

    from agentscope.event import ReplyStartEvent
    from agentscope.message import Msg, TextBlock

    module = load_script("11_team_agent.py")
    summary = Msg(
        name="领队",
        role="assistant",
        content=[TextBlock(text="未完成评审的内容")],
    )

    class FakePipeline:
        leader = SimpleNamespace(
            name="领队",
            state=SimpleNamespace(
                get_reply_message=lambda name: (
                    summary if name == "领队" else None
                ),
            ),
        )

        async def reply_stream(self, inputs: Any) -> Any:
            yield ReplyStartEvent(
                session_id="session",
                reply_id="reply",
                name="领队",
            )

    with pytest.raises(RuntimeError, match="缺少"):
        asyncio.run(
            module["run_team_task"]("写一段介绍", FakePipeline()),
        )


def test_team_agent_rejects_empty_leader_reply() -> None:
    """避免把真实模型的空最终消息误报为成功。"""
    from types import SimpleNamespace

    from agentscope.event import ReplyStartEvent
    from agentscope.message import Msg

    module = load_script("11_team_agent.py")
    empty_reply = Msg(name="领队", role="assistant", content=[])

    class FakePipeline:
        leader = SimpleNamespace(
            name="领队",
            state=SimpleNamespace(
                get_reply_message=lambda name: (
                    empty_reply if name == "领队" else None
                ),
            ),
        )

        async def reply_stream(self, inputs: Any) -> Any:
            for name in ("领队", "调研", "写作", "评审"):
                yield ReplyStartEvent(
                    session_id="session",
                    reply_id=f"reply-{name}",
                    name=name,
                )

    with pytest.raises(RuntimeError, match="最终答复为空"):
        asyncio.run(
            module["run_team_task"]("写一段介绍", FakePipeline()),
        )


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
