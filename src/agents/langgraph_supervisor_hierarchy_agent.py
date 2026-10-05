from langchain.agents import create_agent
from langgraph_supervisor import create_supervisor

from agents.langgraph_supervisor_agent import add, multiply, web_search
from core import get_model, settings

model = get_model(settings.DEFAULT_MODEL)


def workflow(chosen_model):
    math_agent = create_agent(
        model=chosen_model,
        tools=[add, multiply],
        name="sub-agent-math_expert",  # 将此图节点标记为子 Agent
        system_prompt="You are a math expert. Always use one tool at a time.",
    ).with_config(tags=["skip_stream"])

    research_agent = (
        create_supervisor(
            [math_agent],
            model=chosen_model,
            tools=[web_search],
            prompt="You are a world class researcher with access to web search. Do not do any math, you have a math expert for that. ",
            supervisor_name="supervisor-research_expert",  # 将此图节点标记为数学 Agent 的主管
        )
        .compile(name="sub-agent-research_expert")  # 将此图节点标记为主主管的子 Agent
        .with_config(tags=["skip_stream"])
    )  # 界面忽略子 Agent 的流式 token

    # 创建主管 Agent 工作流
    return create_supervisor(
        [research_agent],
        model=chosen_model,
        prompt=(
            "You are a team supervisor managing a research expert with math capabilities."
            "For current events, use research_agent. "
        ),
        add_handoff_back_messages=True,
        # 界面要求此值为 True，以便明确识别控制权何时交还
        output_mode="full_history",  # 否则重新加载对话时，不会包含子 Agent 的消息
    )  # 主管 Agent 的默认名称为 "supervisor"。


langgraph_supervisor_hierarchy_agent = workflow(model).compile()
