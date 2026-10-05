import json
from typing import Any
from unittest.mock import patch

import pytest
from ag_ui.core.events import Event
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.types import interrupt
from pydantic import TypeAdapter

from core import settings
from schema.models import FakeModelName

FAKE_RESPONSE = "The answer is 42"

event_adapter: TypeAdapter[Event] = TypeAdapter(Event)

# 记录 Agent 实际收到的 configurable，供 forwardedProps 测试使用
captured_configurable: dict[str, Any] = {}


async def call_model(state: MessagesState, config: RunnableConfig) -> MessagesState:
    captured_configurable.update(config["configurable"])
    model = FakeListChatModel(responses=[FAKE_RESPONSE])
    response = await model.ainvoke(state["messages"])
    return {"messages": [response]}


model_graph = StateGraph(MessagesState)
model_graph.add_node("model", call_model)
model_graph.set_entry_point("model")
model_graph.add_edge("model", END)
model_agent = model_graph.compile(checkpointer=MemorySaver())


async def ask_color(state: MessagesState) -> MessagesState:
    answer = interrupt("What is your favorite color?")
    return {"messages": [AIMessage(content=f"Your favorite color is {answer}")]}


interrupt_graph = StateGraph(MessagesState)
interrupt_graph.add_node("ask", ask_color)
interrupt_graph.set_entry_point("ask")
interrupt_graph.add_edge("ask", END)
interrupt_agent = interrupt_graph.compile(checkpointer=MemorySaver())


@pytest.fixture(autouse=True)
def _reset_captured_configurable():
    """使用 model_agent 的测试都会写入此共享字典，因此每次测试前先重置。"""
    captured_configurable.clear()
    yield


@pytest.fixture
def allow_fake_model(monkeypatch):
    """使 FakeModelName.FAKE 能通过 AVAILABLE_MODELS 允许列表检查。"""
    monkeypatch.setattr(settings, "AVAILABLE_MODELS", {FakeModelName.FAKE})


@pytest.fixture
def mock_agui_agent():
    def agent_lookup(agent_id: str):
        agents = {"model-agent": model_agent, "interrupt-agent": interrupt_agent}
        try:
            return agents[agent_id]
        except KeyError:
            raise KeyError(agent_id)

    with patch("service.agui.get_agent", side_effect=agent_lookup):
        yield


def run_input(thread_id: str = "test-thread", **overrides: Any) -> dict[str, Any]:
    body = {
        "threadId": thread_id,
        "runId": "test-run",
        "messages": [{"id": "msg-1", "role": "user", "content": "Hello"}],
        "tools": [],
        "context": [],
        "state": {},
        "forwardedProps": {},
    }
    body.update(overrides)
    return body


def collect_events(test_client, path: str, body: dict[str, Any]) -> list[dict[str, Any]]:
    """向 AG-UI 接口发送 POST 请求，并将 SSE 响应解析为事件字典。"""
    with test_client.stream("POST", path, json=body) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = []
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
        return events


def test_agui_stream_lifecycle(mock_agui_agent, test_client) -> None:
    """基本运行应输出符合 AG-UI 协议的事件流。"""
    events = collect_events(test_client, "/agui/model-agent/run", run_input())

    # 每个事件都必须能解析为有效的 AG-UI 事件，以验证协议兼容性。
    for event in events:
        event_adapter.validate_python(event)

    types = [e["type"] for e in events]
    assert types[0] == "RUN_STARTED"
    assert types[-1] == "RUN_FINISHED"

    # 将 token 拼接为模型响应
    text = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
    assert text == FAKE_RESPONSE

    # 最终消息快照包含本轮对话
    snapshots = [e for e in events if e["type"] == "MESSAGES_SNAPSHOT"]
    assert snapshots
    roles = [(m["role"], m.get("content")) for m in snapshots[-1]["messages"]]
    assert ("user", "Hello") in roles
    assert ("assistant", FAKE_RESPONSE) in roles


def test_agui_no_raw_events(mock_agui_agent, test_client) -> None:
    """RAW 透传事件会暴露服务端内部信息，因此应被过滤。"""
    events = collect_events(test_client, "/agui/model-agent/run", run_input())
    assert all(e["type"] != "RAW" for e in events)


def test_agui_default_agent_route(mock_agui_agent, test_client) -> None:
    """POST /agui/run 应回退到默认 Agent。"""
    with patch("service.agui.get_agent", return_value=model_agent) as mock_get_agent:
        events = collect_events(test_client, "/agui/run", run_input())
    from agents import DEFAULT_AGENT

    mock_get_agent.assert_called_once_with(DEFAULT_AGENT)
    assert events[-1]["type"] == "RUN_FINISHED"


def test_agui_unknown_agent(mock_agui_agent, test_client) -> None:
    response = test_client.post("/agui/no-such-agent/run", json=run_input())
    assert response.status_code == 404


def test_agui_configurable_passthrough(mock_agui_agent, allow_fake_model, test_client) -> None:
    """forwardedProps.configurable 的值应传递到 Agent 的 configurable。"""
    body = run_input(
        thread_id="passthrough-thread",
        forwardedProps={"configurable": {"model": "fake", "user_id": "user-123"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)
    assert captured_configurable.get("model") == "fake"
    assert captured_configurable.get("user_id") == "user-123"


def test_agui_records_thread_metadata(mock_agui_agent, test_client) -> None:
    """AG-UI 运行应记录 user_id 和 agent_id，使会话能被 /threads 列出。"""
    body = run_input(
        thread_id="metadata-thread",
        forwardedProps={"configurable": {"user_id": "user-123"}},
    )
    collect_events(test_client, "/agui/model-agent/run", body)

    tup = model_agent.checkpointer.get_tuple(
        RunnableConfig(configurable={"thread_id": "metadata-thread"})
    )
    assert tup is not None
    assert tup.metadata["user_id"] == "user-123"
    assert tup.metadata["agent_id"] == "model-agent"


def test_agui_generates_user_id_when_missing(mock_agui_agent, test_client) -> None:
    """客户端未提供 user_id 时仍会记录一个，与 /invoke 的行为一致。"""
    collect_events(test_client, "/agui/model-agent/run", run_input(thread_id="anon-thread"))

    tup = model_agent.checkpointer.get_tuple(
        RunnableConfig(configurable={"thread_id": "anon-thread"})
    )
    assert tup is not None
    assert tup.metadata["user_id"]
    assert captured_configurable.get("user_id") == tup.metadata["user_id"]


def test_agui_configurable_reserved_keys(mock_agui_agent, test_client) -> None:
    body = run_input(forwardedProps={"configurable": {"thread_id": "hijack"}})
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 422
    assert "reserved" in response.json()["detail"]


def test_agui_configurable_wrong_type(mock_agui_agent, test_client) -> None:
    body = run_input(forwardedProps={"configurable": "not-a-dict"})
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 422


def test_agui_configurable_model_not_available(mock_agui_agent, test_client) -> None:
    """不在 AVAILABLE_MODELS 允许列表中的模型，应在运行开始前被拒绝。"""
    body = run_input(
        thread_id="model-not-available-thread",
        forwardedProps={"configurable": {"model": "not-a-real-model"}},
    )
    response = test_client.post("/agui/model-agent/run", json=body)
    assert response.status_code == 400
    assert "not available" in response.json()["detail"]


def test_agui_interrupt_and_resume(mock_agui_agent, test_client) -> None:
    """中断应以 on_interrupt 自定义事件呈现，并且能够恢复。"""
    thread_id = "interrupt-thread"
    events = collect_events(test_client, "/agui/interrupt-agent/run", run_input(thread_id))
    assert events[-1]["type"] == "RUN_FINISHED"
    interrupts = [e for e in events if e["type"] == "CUSTOM" and e["name"] == "on_interrupt"]
    assert len(interrupts) == 1
    assert interrupts[0]["value"] == "What is your favorite color?"

    # 使用回答恢复本次运行
    resume_body = run_input(thread_id, messages=[], forwardedProps={"command": {"resume": "blue"}})
    events = collect_events(test_client, "/agui/interrupt-agent/run", resume_body)
    assert events[-1]["type"] == "RUN_FINISHED"
    snapshots = [e for e in events if e["type"] == "MESSAGES_SNAPSHOT"]
    contents = [m.get("content") for m in snapshots[-1]["messages"]]
    assert "Your favorite color is blue" in contents


def test_agui_auth(mock_settings, mock_agui_agent, test_client) -> None:
    """AG-UI 接口应使用与服务其他接口相同的 Bearer 认证。"""
    from pydantic import SecretStr

    mock_settings.AUTH_SECRET = SecretStr("test-secret")
    response = test_client.post("/agui/model-agent/run", json=run_input())
    assert response.status_code == 401

    response = test_client.post(
        "/agui/model-agent/run",
        json=run_input(),
        headers={"Authorization": "Bearer wrong-secret"},
    )
    assert response.status_code == 401

    response = test_client.post(
        "/agui/model-agent/run",
        json=run_input(),
        headers={"Authorization": "Bearer test-secret"},
    )
    assert response.status_code == 200
