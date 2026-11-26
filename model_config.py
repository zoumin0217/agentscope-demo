"""共享的 AgentScope 模型选择配置。"""

import os
import socket

from dotenv import load_dotenv

from agentscope.credential import (
    DashScopeCredential,
    DeepSeekCredential,
    OllamaCredential,
    OpenAICredential,
)
from agentscope.model import (
    ChatModelBase,
    DashScopeChatModel,
    DeepSeekChatModel,
    OllamaChatModel,
    OpenAIChatModel,
)

load_dotenv()


def build_model() -> ChatModelBase:
    """按优先级自动选择可用厂商并创建对话模型。"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return OpenAIChatModel(
            credential=OpenAICredential(
                api_key=openai_api_key,
                base_url=os.getenv("OPENAI_BASE_URL") or None,
            ),
            model=os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
        )

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if dashscope_api_key:
        return DashScopeChatModel(
            credential=DashScopeCredential(api_key=dashscope_api_key),
            model="qwen-plus",
        )

    # 显式配置 OLLAMA_HOST 时按该地址连接；否则探测本机默认端口。
    ollama_host = os.getenv("OLLAMA_HOST", "").strip()
    if ollama_host:
        if "://" not in ollama_host:
            ollama_host = f"http://{ollama_host}"
    else:
        try:
            with socket.create_connection(("localhost", 11434), timeout=0.5):
                ollama_host = "http://localhost:11434"
        except OSError:
            ollama_host = ""

    if ollama_host:
        return OllamaChatModel(
            credential=OllamaCredential(host=ollama_host.rstrip("/")),
            model=os.getenv("OLLAMA_MODEL") or "llama3.2",
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        return DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=deepseek_api_key),
            model="deepseek-chat",
        )

    raise SystemExit(
        "未找到可用模型配置。请在项目根目录的 .env 文件中设置 "
        "OPENAI_API_KEY、DASHSCOPE_API_KEY 或 DEEPSEEK_API_KEY，"
        "或启动本地 Ollama（默认 localhost:11434），"
        "也可配置 OLLAMA_HOST。可参考 .env.example。"
    )
