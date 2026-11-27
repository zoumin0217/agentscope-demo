"""基于 Gradio 的中文本地文件助手界面。

该模块维护全局 Agent 实例与待确认工具状态，把异步智能体事件转换为
适合 Web 聊天界面展示的字符串，并确保异常不会把后端 traceback 暴露给用户。
"""

import logging  # logging 将异常和运行信息写入服务端日志。
from typing import Any

import gradio as gr

from agentscope.agent import Agent
from agentscope.event import (
    ConfirmResult,
    RequireExternalExecutionEvent,
    RequireUserConfirmEvent,
    UserConfirmResultEvent,
)
from agentscope.message import Msg, ToolCallBlock, UserMsg
from agentscope.middleware import AgenticMemoryMiddleware
from agentscope.tool import Bash, Read, Toolkit, Write
from main_agent import MEMORY_WORKDIR, build_model

logger = logging.getLogger(__name__)

agent: Agent | None = None
_pending_reply_id: str | None = None
_pending_tool_calls: list[ToolCallBlock] = []
_pending_confirm_results: list[ConfirmResult] = []


def build_agent() -> Agent:
    """创建与命令行助手一致的模型、工具和记忆中间件。"""
    toolkit = Toolkit(tools=[Bash(), Read(), Write()])
    return Agent(
        name="文件助手",
        system_prompt=(
            "你是一个中文本地文件助手，可以在本地运行命令、读取文件和写入文件。"
            "请根据用户的需求自主决定是否调用以及调用哪个工具；"
            "请留意用户每次明确提供或更正的姓名、称呼和长期偏好，"
            "立即按记忆规范保存或更新，并在后续相关对话中自然引用；"
            "不要猜测用户未明确提供的信息。完成任务后用中文简洁回答。"
        ),
        model=build_model(),
        toolkit=toolkit,
        middlewares=[
            AgenticMemoryMiddleware(workdir=MEMORY_WORKDIR),
        ],
    )


def _clear_pending_confirmation() -> None:
    """清除尚未处理的工具确认状态。"""
    global _pending_reply_id
    _pending_reply_id = None
    _pending_tool_calls.clear()
    _pending_confirm_results.clear()


def _confirmation_prompt() -> str:
    """只向界面展示当前待确认工具及其必要参数。"""
    tool_call = _pending_tool_calls[0]
    details = tool_call.input
    if len(details) > 500:
        details = f"{details[:500]}…"
    return (
        f"需要确认工具调用：{tool_call.name}\n"
        f"参数：{details}\n\n"
        "请回复 y/yes/是 允许，或 n/no/否 拒绝。"
    )


def _queue_confirmations(
    events: list[RequireUserConfirmEvent],
) -> None:
    """缓存确认事件供下一次 Gradio 消息处理。"""
    global _pending_reply_id
    if not events:
        raise RuntimeError("没有可供确认的工具调用。")

    reply_id = events[0].reply_id
    calls = [
        tool_call
        for event in events
        for tool_call in event.tool_calls
    ]
    if not calls or any(event.reply_id != reply_id for event in events):
        raise RuntimeError("工具确认事件状态不完整。")

    _pending_reply_id = reply_id
    _pending_tool_calls.extend(calls)


async def _run_agent(
    inputs: Msg | UserConfirmResultEvent,
) -> str:
    """执行 Agent 回复，只将最终文本或确认提示发给前端。"""
    if agent is None:
        raise RuntimeError("Agent 尚未初始化。")

    confirmation_events: list[RequireUserConfirmEvent] = []
    final_reply: Msg | None = None
    async for event in agent.reply_stream(inputs, yield_final_msg=True):
        if isinstance(event, RequireUserConfirmEvent):
            confirmation_events.append(event)
        elif isinstance(event, RequireExternalExecutionEvent):
            raise RuntimeError("当前 Web 界面不支持外部工具执行器。")
        elif isinstance(event, Msg):
            final_reply = event

    if confirmation_events:
        _queue_confirmations(confirmation_events)
        return _confirmation_prompt()
    if final_reply is None:
        raise RuntimeError("Agent 没有返回最终消息。")

    # AgentScope 的 content 是内容块列表；Gradio 聊天消息使用纯文本。
    return final_reply.get_text_content()


def _confirmation_result(message: str) -> bool | None:
    """解析确认/拒绝文本；非确认内容不继续执行工具。"""
    answer = message.strip().lower()
    if answer in {"y", "yes", "是", "确认", "允许"}:
        return True
    if answer in {"n", "no", "否", "拒绝"}:
        return False
    return None


async def chat(message: str, history: list[dict[str, Any]]) -> str:
    """处理聊天输入，并在需要时通过下一轮消息确认工具调用。"""
    del history  # 对话上下文由 Agent 自身维护。
    if agent is None:
        return "助手尚未初始化，请重启 Web 服务后再试。"
    if not message.strip():
        return "请输入消息后再发送。"

    try:
        if _pending_tool_calls:
            confirmed = _confirmation_result(message)
            if confirmed is None:
                return _confirmation_prompt()

            if _pending_reply_id is None:
                raise RuntimeError("待确认工具缺少回复标识。")
            tool_call = _pending_tool_calls.pop(0)
            _pending_confirm_results.append(
                ConfirmResult(
                    confirmed=confirmed,
                    tool_call=tool_call,
                ),
            )
            if _pending_tool_calls:
                return _confirmation_prompt()

            confirmation = UserConfirmResultEvent(
                reply_id=_pending_reply_id,
                confirm_results=list(_pending_confirm_results),
            )
            _clear_pending_confirmation()
            return await _run_agent(confirmation)

        return await _run_agent(
            UserMsg(name="user", content=message),
        )
    except SystemExit:
        logger.exception("Gradio 助手的模型配置退出。")
        _clear_pending_confirmation()
        return "模型配置暂不可用，请检查 .env 配置或本地模型服务。"
    except Exception:
        logger.exception("Gradio 助手处理请求失败。")
        _clear_pending_confirmation()
        return "抱歉，刚才处理请求时遇到问题。请稍后重试；详细诊断已记录在服务端日志。"


def main() -> None:
    """启动仅监听本机地址的 Gradio 界面。"""
    global agent
    logging.basicConfig(level=logging.INFO)
    try:
        agent = build_agent()
    except SystemExit as exc:
        logger.error("助手无法启动：%s", exc)
        raise
    except Exception:
        logger.exception("初始化 Gradio 助手失败。")
        raise SystemExit("助手初始化失败，请检查服务端日志。") from None

    gr.ChatInterface(
        chat,
        title="本地文件助手",
        description="执行本地工具前会先在对话中请求确认。",
        concurrency_limit=1,
        api_visibility="private",
    ).launch(server_name="127.0.0.1", show_error=False)


if __name__ == "__main__":
    main()
