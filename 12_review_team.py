"""基于 AgentScope TeamPipeline 的代码评审小团队示例。"""

import asyncio

from agentscope.agent import Agent
from agentscope.event import ReplyStartEvent
from agentscope.message import Msg, UserMsg
from agentscope.pipeline import TeamMember, TeamPipeline

from model_config import build_model


def build_review_team() -> TeamPipeline:
    """创建负责分派、并行分析和最终汇总的评审团队。"""
    # 组长使用独立 Agent，明确要求调用每位成员并给出结构化汇总。
    leader = Agent(
        name="评审组长",
        system_prompt=(
            "你是代码评审组长，负责把用户需求拆分并分派给三位队员。"
            "每次评审都必须分别调用需求分析师、代码审查员和测试方案师，"
            "再结合他们的结果汇总。"
            "最终用中文输出：需求理解、主要风险/审查意见、测试建议、待澄清问题。"
            "用户只提供需求文字而没有代码时，不得编造具体代码缺陷或文件位置。"
        ),
        model=build_model(),
    )

    # 每位队员只聚焦一个评审维度，避免职责重叠并让组长易于汇总。
    members = [
        TeamMember(
            agent=Agent(
                name="需求分析师",
                system_prompt=(
                    "你是需求分析师。提炼用户目标、范围、验收标准和约束；"
                    "指出含糊之处及需要澄清的问题，不要臆造实现细节。"
                ),
                model=build_model(),
            ),
            description="拆解需求目标、范围、验收标准、约束和待澄清问题。",
        ),
        TeamMember(
            agent=Agent(
                name="代码审查员",
                system_prompt=(
                    "你是资深代码审查员。根据给出的需求或代码识别正确性、"
                    "安全性、兼容性和可维护性风险；按严重程度说明依据。"
                    "没有提供代码时，只指出需求层面的风险，不声称发现了代码缺陷。"
                ),
                model=build_model(),
            ),
            description="检查正确性、安全性、兼容性和可维护性风险，并说明依据。",
        ),
        TeamMember(
            agent=Agent(
                name="测试方案师",
                system_prompt=(
                    "你是测试方案师。为需求设计可执行的测试场景，覆盖正常路径、"
                    "边界条件、异常处理和回归风险，并给出预期结果。"
                ),
                model=build_model(),
            ),
            description="设计覆盖正常、边界、异常和回归场景的测试方案。",
        ),
    ]

    # TeamPipeline 把队员作为可分派任务的对象，并把队员答复交回组长。
    return TeamPipeline(
        leader=leader,
        members=members,
        reset_members=True,
    )


async def review_requirement(
    requirement: str,
    pipeline: TeamPipeline | None = None,
) -> str:
    """将一段需求交给团队评审，并返回组长的最终汇总。"""
    if not requirement.strip():
        raise ValueError("需求内容不能为空。")

    team = pipeline or build_review_team()
    seen_agents: set[str] = set()

    # 输入只发送给组长；TeamPipeline 会处理内部的成员分派和结果回传。
    async for event in team.reply_stream(
        UserMsg(name="user", content=requirement),
    ):
        # 只显示成员开始工作的状态，不把内部推理或中间消息暴露到终端。
        if isinstance(event, ReplyStartEvent) and event.name not in seen_agents:
            print(f"团队成员开始评审：{event.name}")
            seen_agents.add(event.name)

    # TeamPipeline 只输出事件流，最终 Msg 保存在组长当前的 AgentState 中。
    final_reply: Msg | None = team.leader.state.get_reply_message(
        team.leader.name,
    )
    if final_reply is None:
        raise RuntimeError("评审团队结束时没有生成组长汇总。")

    return final_reply.get_text_content()


async def main() -> None:
    """读取一段用户需求，执行一次团队评审并打印汇总。"""
    requirement = input("请输入需要评审的需求：").strip()
    if not requirement:
        raise SystemExit("需求内容不能为空。")

    # 在入口处创建团队，便于复用 review_requirement 进行自动化测试。
    summary = await review_requirement(requirement)
    print("\n=== 代码评审小团队汇总 ===")
    print(summary)


if __name__ == "__main__":
    asyncio.run(main())
