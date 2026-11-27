"""基于 AgentScope TeamPipeline 的多智能体协作示例。

领队负责拆分任务并调用调研、写作、评审三个成员；
最终答复通过领队的 AgentState 检索，而不是简单拼接成员的中间消息。
"""

import argparse
import asyncio

from agentscope.agent import Agent
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg, UserMsg
from agentscope.pipeline import TeamMember, TeamPipeline

from model_config import build_model

DEFAULT_TASK = "帮我产出一篇 200 字的“AI 智能体入门”介绍"


def build_team() -> TeamPipeline:
    """构造领队和分工明确的团队成员。"""
    # 领队明确要求分派工作并整合意见，避免只由自己直接作答。
    leader = Agent(
        name="领队",
        system_prompt=(
            "你是团队领队，负责拆分需求、分派任务并汇总最终结果。"
            "你必须按调研、写作、评审的顺序逐一调用三位队员，每次只调用一位，"
            "等待其答复后再调用下一位，不可并发分派、跳过成员或独自作答。"
            "调研提供事实和要点，写作根据调研结果形成初稿，"
            "评审检查初稿的准确性、完整性和表达。"
            "收到全部三位队员的答复后，必须输出非空的最终成果；"
            "不要编造调研依据，也不要只结束在工具调用。"
        ),
        model=build_model(),
    )

    # 每个成员拥有独立职责，TeamPipeline 会将其描述提供给领队用于分派。
    members = [
        TeamMember(
            agent=Agent(
                name="调研",
                system_prompt=(
                    "你负责根据任务搜集和整理可靠的事实、背景与关键要点。"
                    "区分确定信息与不确定信息，不要编造来源或数据。"
                ),
                model=build_model(),
            ),
            description="搜集可靠事实、背景信息和写作要点。",
        ),
        TeamMember(
            agent=Agent(
                name="写作",
                system_prompt=(
                    "你负责根据领队提供的任务和调研结果撰写清晰、准确的初稿。"
                    "遵循指定的主题、篇幅和受众要求，不添加未经支持的事实。"
                ),
                model=build_model(),
            ),
            description="根据需求和调研结果撰写结构清楚的初稿。",
        ),
        TeamMember(
            agent=Agent(
                name="评审",
                system_prompt=(
                    "你负责审查初稿的事实准确性、完整性、逻辑和表达。"
                    "指出具体问题并给出可执行的修改建议，不要重写成无关主题。"
                ),
                model=build_model(),
            ),
            description="检查初稿质量并提出有依据的修改建议。",
        ),
    ]

    # TeamPipeline 负责执行领队的成员分派，并将成员答复交回领队。
    return TeamPipeline(
        leader=leader,
        members=members,
        reset_members=True,
    )


async def run_team_task(
    task: str,
    pipeline: TeamPipeline | None = None,
) -> str:
    """运行一次团队任务，并返回领队的最终答复。"""
    if not task.strip():
        raise ValueError("任务内容不能为空。")

    team = pipeline if pipeline is not None else build_team()
    announced_agents: set[str] = set()

    # 用户消息只交给领队；pipeline 内部负责调用队员和收集结果。
    async for event in team.reply_stream(
        UserMsg(name="user", content=task),
    ):
        # 只输出成员开始工作的状态，不把中间推理内容当作最终结果。
        if (
            isinstance(event, ReplyStartEvent)
            and event.name not in announced_agents
        ):
            print(f"团队成员开始处理：{event.name}", flush=True)
            announced_agents.add(event.name)

    required_agents = ("调研", "写作", "评审")
    missing_agents = [
        name for name in required_agents if name not in announced_agents
    ]
    if missing_agents:
        raise RuntimeError(
            f"团队流程未完成全部成员调用，缺少：{'、'.join(missing_agents)}。",
        )

    # TeamPipeline 输出事件流；最终 Msg 保存在领队当前 AgentState 中。
    final_reply: Msg | None = team.leader.state.get_reply_message(
        team.leader.name,
    )
    if final_reply is None:
        raise RuntimeError("团队流程结束时没有生成领队的最终答复。")

    final_text = final_reply.get_text_content().strip()
    if not final_text:
        raise RuntimeError(
            "团队流程结束时领队的最终答复为空；"
            "请确认所选模型支持工具调用并能生成文本答复。",
        )

    return final_text


async def main(task: str | None = None) -> None:
    """接收用户任务，执行团队协作并展示最终答复。"""
    if task is None:
        task = input(
            f"请输入任务（直接回车使用默认任务：{DEFAULT_TASK}）：",
        ).strip()
    if not task:
        task = DEFAULT_TASK

    # 独立封装团队运行逻辑，既便于 CLI 使用，也便于离线测试。
    final_answer = await run_team_task(task)
    print("\n=== 团队最终结果 ===")
    print(final_answer, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="运行多智能体团队任务")
    parser.add_argument(
        "--task",
        help="直接指定任务；省略时进入交互输入",
    )
    arguments = parser.parse_args()
    asyncio.run(main(arguments.task))
