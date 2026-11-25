# AgentScope Demo

这是一个基于 **AgentScope 2.x** 的 Python 智能体示例工程，演示如何：

- 创建支持 DeepSeek 或 OpenAI 兼容接口的中文智能体
- 使用流式响应接收文本增量和工具事件
- 为本地命令、文件读取和文件写入工具添加人工确认
- 在 Windows 环境中运行交互式文件助手

## 项目结构

```text
agentscope-demo/
├── 01_hello_agent.py       # 最小对话示例
├── 02_stream_events.py     # 流式事件与工具调用演示
├── main_agent.py           # 文件助手及工具确认的完整示例
├── .env.example            # API Key 和模型配置模板
├── requirements.txt        # Python 运行依赖
├── notes.txt               # 项目备注
└── .gitignore              # 忽略虚拟环境、环境变量和缓存
```

## 示例说明

### 1. 最小对话：`01_hello_agent.py`

创建一个名为“小助”的中文智能体，向模型发送“介绍下你自己”，并打印完整回复。

模型选择规则：

1. 配置了 `DEEPSEEK_API_KEY` 时使用 `deepseek-chat`
2. 否则配置了 `OPENAI_API_KEY` 时使用兼容 OpenAI 的接口
3. 两者都未配置时抛出配置错误

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

智能体在执行工具前会逐项请求人工确认；输入 `y` 或 `yes` 允许执行，其他输入拒绝。交互中输入 `quit`、`exit` 或 `退出` 可结束程序。

## 环境要求

- Windows 10/11
- Python 3.10 或更高版本
- 可访问所选模型服务的网络环境
- DeepSeek API Key 或 OpenAI 兼容服务的 API Key

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

依赖文件锁定了当前工程使用的 AgentScope 和 `python-dotenv` 版本。

### 2. 配置环境变量

复制模板：

```powershell
Copy-Item .env.example .env
```

在 `.env` 中二选一配置：

#### 使用 DeepSeek

```dotenv
DEEPSEEK_API_KEY=你的DeepSeek密钥
OPENAI_API_KEY=
OPENAI_BASE_URL=
OPENAI_MODEL=
```

#### 使用 OpenAI 或兼容接口

```dotenv
DEEPSEEK_API_KEY=
OPENAI_API_KEY=你的API密钥
OPENAI_BASE_URL=https://your-api-provider.example/v1
OPENAI_MODEL=模型名称
```

对于 OpenAI 官方接口，可以将 `OPENAI_BASE_URL` 留空，并在 `OPENAI_MODEL` 中指定模型；未填写时默认使用 `gpt-4o-mini`。

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

## 配置说明

| 环境变量 | 必需性 | 说明 |
| --- | --- | --- |
| `DEEPSEEK_API_KEY` | 二选一 | 配置后优先使用 DeepSeek 模型 |
| `OPENAI_API_KEY` | 二选一 | 未配置 DeepSeek 时使用 OpenAI/兼容接口 |
| `OPENAI_BASE_URL` | 可选 | OpenAI 兼容接口地址；官方接口可留空 |
| `OPENAI_MODEL` | 可选 | 兼容接口或官方模型名称；默认 `gpt-4o-mini` |

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

### 提示未配置 API Key

确认项目根目录存在 `.env`，且至少配置了以下之一：

```dotenv
DEEPSEEK_API_KEY=...
```

或：

```dotenv
OPENAI_API_KEY=...
```

### 可以使用现有虚拟环境吗？

可以：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 为什么运行流式示例时没有看到工具事件？

只有智能体实际决定调用工具时才会产生工具相关事件。示例任务要求查看目录，通常会触发 `Bash()`，但模型行为可能因提示和模型实现而变化。
