# AgentScope Demo

这是一个基于 **AgentScope 2.x** 的 Python 智能体示例工程，演示如何：

- 按优先级自动选择 OpenAI、DashScope、Ollama 或 DeepSeek 模型
- 使用流式响应接收文本增量和工具事件
- 为本地命令、文件读取和文件写入工具添加人工确认
- 在 Windows 环境中运行交互式文件助手

## 项目结构

```text
agentscope-demo/
├── 01_hello_agent.py       # 最小对话示例
├── 02_stream_events.py     # 流式事件与工具调用演示
├── main_agent.py           # 文件助手及工具确认的完整示例
├── model_config.py         # 共享的多厂商模型选择
├── agent_memory/           # 运行时生成的长期记忆（已加入 .gitignore）
├── 03_test_tools.py        # pytest 基础测试
├── .env.example            # API Key 和模型配置模板
├── requirements.txt        # Python 运行依赖
├── notes.txt               # 项目备注
└── .gitignore              # 忽略虚拟环境、环境变量和缓存
```

## 示例说明

### 1. 最小对话：`01_hello_agent.py`

创建一个名为“小助”的中文智能体，向模型发送“介绍下你自己”，并打印完整回复。

模型选择规则（两份示例的 `build_model()` 一致）：

1. `OPENAI_API_KEY`：使用 `OpenAIChatModel`，默认模型为 `gpt-4o-mini`
2. `DASHSCOPE_API_KEY`：使用 `DashScopeChatModel(model="qwen-plus")`
3. 配置 `OLLAMA_HOST`，或探测到本机 `localhost:11434` 可连接：使用 `OllamaChatModel`，默认模型为 `llama3.2`
4. `DEEPSEEK_API_KEY`：使用 `DeepSeekChatModel(model="deepseek-chat")`
5. 都不可用时提示配置 `.env` 或启动 Ollama

### 2. 流式事件：`02_stream_events.py`

演示流式文本输出和工具事件识别：

- `TextBlockDelta`：文本增量，逐段输出
- `ToolCall`：智能体发起工具调用
- `ToolResult`：工具调用返回结果

示例任务会提示智能体查看当前目录中的文件。

### 3. 完整文件助手：`main_agent.py`

提供一个可以持续交互的本地文件助手，支持：

- `Bash()`：运行本地命令
- `Read()`：读取文件
- `Write()`：写入文件

助手使用 `AgenticMemoryMiddleware` 保存用户明确提供的姓名、称呼和长期偏好，并在后续相关对话中检索引用。记忆存放在 `agent_memory/`，已通过 `.gitignore` 排除。

智能体在执行工具前会逐项请求人工确认；输入 `y` 或 `yes` 允许执行，其他输入拒绝。交互中输入 `quit`、`exit` 或 `退出` 可结束程序。

## 环境要求

- Windows 10/11
- Python 3.10 或更高版本
- 可访问所选云模型服务的网络环境，或已启动的本地 Ollama
- 按优先级配置 OpenAI、DashScope、Ollama 或 DeepSeek 中任一种

> 本工程中的 `.venv` 环境已验证 AgentScope 版本为 `2.0.12`，并安装了 `python-dotenv`。

## 安装与运行

### 1. 创建并激活虚拟环境

PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

依赖文件锁定了当前工程使用的 AgentScope 和 `python-dotenv` 版本，并声明 pytest 测试依赖。

### 2. 配置环境变量

复制模板：

```powershell
Copy-Item .env.example .env
```

按需要在 `.env` 中配置以下任一方式：

#### 使用 OpenAI

```dotenv
OPENAI_API_KEY=你的OpenAI密钥
OPENAI_BASE_URL=
OPENAI_MODEL=gpt-4o-mini
```

兼容接口可按服务商文档填写 `OPENAI_BASE_URL` 和 `OPENAI_MODEL`。OpenAI credential 的标准用法为：

```python
OpenAIChatModel(
    credential=OpenAICredential(api_key=api_key, base_url=base_url),
    model=model_name,
)
```

#### 使用 DashScope

```dotenv
DASHSCOPE_API_KEY=你的百炼API密钥
```

标准用法：

```python
DashScopeChatModel(
    credential=DashScopeCredential(api_key=api_key),
    model="qwen-plus",
)
```

#### 使用本地 Ollama

默认探测 `localhost:11434`，也可以显式配置服务地址和模型：

```dotenv
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

标准用法：

```python
OllamaChatModel(
    credential=OllamaCredential(host=ollama_host),
    model=ollama_model,
)
```

#### 使用 DeepSeek

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
```

标准用法：

```python
DeepSeekChatModel(
    credential=DeepSeekCredential(api_key=api_key),
    model="deepseek-chat",
)
```

> 不要提交 `.env`。项目的 `.gitignore` 已包含该文件。

### 3. 运行示例

运行最小对话示例：

```powershell
python 01_hello_agent.py
```

运行流式事件示例：

```powershell
python 02_stream_events.py
```

运行交互式文件助手：

```powershell
python main_agent.py
```

需要从空记忆开始时，在启动前传入 `--reset`：

```powershell
python main_agent.py --reset
```

该选项只会清除 `agent_memory/`；不传入时会保留已有记忆。

### 4. 运行测试

```powershell
python -m pytest 03_test_tools.py
```

测试会检查示例脚本能否导入，以及未配置 API Key 时 `build_model()` 是否以 `SystemExit` 明确退出。真实模型请求测试默认使用 `pytest.mark.skip` 跳过，不会访问网络或产生 API 费用。

## 配置说明

| 环境变量 | 必需性 | 说明 |
| --- | --- | --- |
| `OPENAI_API_KEY` | 可选 | 优先级最高；使用 OpenAI/兼容接口 |
| `OPENAI_BASE_URL` | 可选 | OpenAI 兼容接口地址；官方接口可留空 |
| `OPENAI_MODEL` | 可选 | OpenAI 模型名称；默认 `gpt-4o-mini` |
| `DASHSCOPE_API_KEY` | 可选 | OpenAI key 未设置时使用 DashScope `qwen-plus` |
| `OLLAMA_HOST` | 可选 | 配置后使用该 Ollama 地址；未设置则探测 `localhost:11434` |
| `OLLAMA_MODEL` | 可选 | Ollama 模型名称；默认 `llama3.2` |
| `DEEPSEEK_API_KEY` | 可选 | 前三种方式不可用时使用 `deepseek-chat` |

## 工具确认机制

`main_agent.py` 中的 `reply_stream_with_confirmation()` 会：

1. 调用 `agent.reply_stream()` 获取流式事件
2. 收集待确认的工具调用
3. 打印工具名称和参数
4. 读取终端中的确认结果
5. 将 `ConfirmResult` 回传给智能体，继续或结束执行

如果智能体请求外部执行器处理的工具，而当前示例没有注册执行器，程序会明确抛出 `RuntimeError`。

## 安全提示

文件和命令工具会在本机生效。运行前请仔细检查工具请求中的参数：

- 不要在没有确认的情况下执行高风险命令
- 不要让智能体读取、写入或删除敏感目录中的文件
- 不要在代码、提交或聊天记录中暴露 API Key
- 建议先在测试目录或隔离环境中试用

## 常见问题

### 提示未找到可用模型配置

确认项目根目录存在 `.env`，并按优先级配置至少一种云服务 API Key，或启动本地 Ollama：

```dotenv
OPENAI_API_KEY=...
DASHSCOPE_API_KEY=...
DEEPSEEK_API_KEY=...
```

本地模型默认检测 `localhost:11434`，也可填写 `OLLAMA_HOST`。详细配置参见 `.env.example`。

### 可以使用现有虚拟环境吗？

可以：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 为什么运行流式示例时没有看到工具事件？

只有智能体实际决定调用工具时才会产生工具相关事件。示例任务要求查看目录，通常会触发 `Bash()`，但模型行为可能因提示和模型实现而变化。
