"""共享的 AgentScope 模型选择配置。

各示例都可以调用 build_model() 创建模型对象，而不必重复编写厂商判断。
函数优先读取环境变量中的云模型配置，再尝试本机 Ollama，最后才报错退出。
"""

import os  # os 用于读取操作系统环境变量和创建提供商凭证配置。
import socket  # socket 用于探测本机 11434 端口是否存在 Ollama 服务。

from dotenv import load_dotenv  # load_dotenv 从 .env 文件加载键值对到环境变量。

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

load_dotenv()  # 模块导入时立即加载项目根目录或上层目录中的 .env 文件。


def build_model() -> ChatModelBase:
    """按优先级自动选择可用厂商并创建对话模型。

    优先级为 OpenAI、DashScope、Ollama、DeepSeek。
    所选凭证和模型字符串会传入 AgentScope 对应的 ChatModel 子类。
    """
    # os.getenv 找不到变量时返回 None；这里把空字符串和 None 都视为未配置。
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        # OPENAI_BASE_URL 可为空；或 None 表示使用 SDK 默认 API 地址。
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
