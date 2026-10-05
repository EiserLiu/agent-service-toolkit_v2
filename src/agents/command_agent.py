import random
from typing import Literal

from langchain_core.messages import AIMessage
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.types import Command


class AgentState(MessagesState, total=False):
    """`total=False` 来自 PEP 589 规范。

    文档：https://typing.readthedocs.io/en/latest/spec/typeddict.html#totality
    """


# 定义节点


def node_a(state: AgentState) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["a", "b"])
    goto: Literal["node_b", "node_c"]
    # 这里替代了条件边函数的作用
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # 注意：Command 可以同时更新图状态并指定下一个节点
    return Command(
        # 这里更新状态
        update={"messages": [AIMessage(content=f"Hello {value}")]},
        # 这里替代了边的作用
        goto=goto,
    )


def node_b(state: AgentState):
    print("Called B")
    return {"messages": [AIMessage(content="Hello B")]}


def node_c(state: AgentState):
    print("Called C")
    return {"messages": [AIMessage(content="Hello C")]}


builder = StateGraph(AgentState)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# 注意：节点 A、B、C 之间没有显式定义边！

command_agent = builder.compile()
