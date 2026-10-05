from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, Mock

import pytest
from streamlit.testing.v1 import AppTest

from client import AgentClientError
from schema import ChatHistory, ChatMessage, ThreadSummary, UserThreads
from schema.models import OpenAIModelName


def test_app_simple_non_streaming(mock_agent_client):
    """测试整个应用的正常流程。"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    WELCOME_START = "Hello! I'm an AI agent. Ask me anything!"
    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[0].markdown[0].value.startswith(WELCOME_START)

    at.sidebar.toggle[0].set_value(False)  # 关闭流式输出
    at.chat_input[0].set_value(PROMPT).run()
    print(at)
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE
    assert not at.exception


def test_app_settings(mock_agent_client):
    """测试整个应用的正常流程。"""
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "1234"
    at.run()

    PROMPT = "Know any jokes?"
    RESPONSE = "Sure! Here's a joke:"

    mock_agent_client.ainvoke = AsyncMock(
        return_value=ChatMessage(type="ai", content=RESPONSE),
    )

    at.sidebar.toggle[0].set_value(False)  # 关闭流式输出
    assert at.sidebar.selectbox[0].value == "gpt-5-nano"
    assert mock_agent_client.agent == "test-agent"
    at.sidebar.selectbox[0].set_value("gpt-5-mini")
    at.sidebar.selectbox[1].set_value("chatbot")
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    # 基础检查
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == RESPONSE

    # 检查参数是否与设置一致
    assert mock_agent_client.agent == "chatbot"
    mock_agent_client.ainvoke.assert_called_with(
        message=PROMPT,
        model=OpenAIModelName.GPT_5_MINI,
        thread_id=at.session_state.thread_id,
        user_id="1234",
    )
    assert not at.exception


def test_app_thread_id_history(mock_agent_client):
    """测试是否生成 thread_id。"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # 重置并设置 thread_id
    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["thread_id"] = "1234"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    # URL 中没有指定 Agent，因此使用客户端选中的 Agent 读取历史。
    mock_agent_client.get_history.assert_called_with(thread_id="1234", agent="test-agent")
    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].avatar == "assistant"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_resume_with_agent_param(mock_agent_client):
    """URL 中的 ?agent= 参数应使历史恢复使用对应 Agent 的图。"""

    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["thread_id"] = "1234"
    at.query_params["agent"] = "chatbot"
    HISTORY = [
        ChatMessage(type="human", content="What is the weather?"),
        ChatMessage(type="ai", content="The weather is sunny."),
    ]
    mock_agent_client.get_history.return_value = ChatHistory(messages=HISTORY)
    at.run()
    print(at)
    assert at.session_state.thread_id == "1234"
    # 通过 URL 中指定的 Agent 获取历史，而不是默认 Agent。
    mock_agent_client.get_history.assert_called_with(thread_id="1234", agent="chatbot")
    assert at.chat_message[0].markdown[0].value == "What is the weather?"
    assert at.chat_message[1].markdown[0].value == "The weather is sunny."
    assert not at.exception


def test_app_feedback(mock_agent_client):
    """TODO：尚未找到与 st.feedback 交互的方法。"""

    pass


@pytest.mark.asyncio
async def test_app_streaming(mock_agent_client):
    """测试启用流式输出的应用，包括工具消息。"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # 设置模拟流式响应
    PROMPT = "What is 6 * 7?"
    ai_with_tool = ChatMessage(
        type="ai",
        content="",
        tool_calls=[{"name": "calculator", "id": "test_call_id", "args": {"expression": "6 * 7"}}],
    )
    tool_message = ChatMessage(type="tool", content="42", tool_call_id="test_call_id")
    final_ai_message = ChatMessage(type="ai", content="The answer is 42")

    messages = [ai_with_tool, tool_message, final_ai_message]

    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)  # 启用流式输出
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "user"
    assert at.chat_message[0].markdown[0].value == PROMPT
    response = at.chat_message[1]
    tool_status = response.status[0]
    assert response.avatar == "assistant"
    assert tool_status.label == "🛠️ Tool Call: calculator"
    assert tool_status.icon == ":material/check:"
    assert tool_status.markdown[0].value == "Input:"
    assert tool_status.json[0].value == '{"expression": "6 * 7"}'
    assert tool_status.markdown[1].value == "Output:"
    assert tool_status.markdown[2].value == "42"
    assert response.markdown[-1].value == "The answer is 42"
    assert not at.exception


@pytest.mark.asyncio
async def test_app_init_error(mock_agent_client):
    """测试 Agent 初始化出错时的应用行为。"""
    at = AppTest.from_file("../../src/streamlit_app.py").run()

    # 设置模拟流式响应
    PROMPT = "What is 6 * 7?"
    mock_agent_client.astream.side_effect = AgentClientError("Error connecting to agent")

    at.toggle[0].set_value(True)  # 启用流式输出
    at.chat_input[0].set_value(PROMPT).run()
    print(at)

    assert at.chat_message[0].avatar == "assistant"
    assert at.chat_message[1].avatar == "user"
    assert at.chat_message[1].markdown[0].value == PROMPT
    assert at.error[0].value == "Error generating response: Error connecting to agent"
    assert not at.exception


def test_app_new_chat_btn(mock_agent_client):
    at = AppTest.from_file("../../src/streamlit_app.py").run()
    thread_id_a = at.session_state.thread_id

    at.sidebar.button[0].click().run()

    assert at.session_state.thread_id != thread_id_a
    assert not at.exception


@pytest.fixture
def multi_agent_messages():
    """为多 Agent 测试提供可复用消息的测试夹具。"""
    from schema import ChatMessage

    # 工具 1
    tool_1 = ChatMessage(
        type="ai",
        content="Starting tool 1...",
        tool_calls=[{"name": "do_work_1", "id": "tool-1", "args": {"my-arg": "value"}}],
    )
    tool_1_result = ChatMessage(type="tool", content="Tool 1 complete", tool_call_id="tool-1")

    # 工具 2
    tool_2 = ChatMessage(
        type="ai",
        content="Starting tool 2...",
        tool_calls=[{"name": "do_work_2", "id": "tool-2", "args": {"my-arg-2": "value"}}],
    )
    tool_2_result = ChatMessage(type="tool", content="Tool 2 complete", tool_call_id="tool-2")

    # 将控制权交给 Agent A
    transfer_a = ChatMessage(
        type="ai",
        content="Transferring to agent A...",
        tool_calls=[
            {"name": "transfer_to_agent_a", "id": "transfer-a", "args": {"task": "task_1"}}
        ],
    )
    transfer_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_a",
        tool_call_id="transfer-a",
    )

    # Agent A 将控制权交给子 Agent B
    transfer_b_from_a = ChatMessage(
        type="ai",
        content="Agent A delegating to agent B...",
        tool_calls=[
            {"name": "transfer_to_agent_b", "id": "transfer-a-b", "args": {"sub_task": "task_2"}}
        ],
    )
    transfer_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_b",
        tool_call_id="transfer-a-b",
    )

    # Agent B 将控制权交还给 A
    transfer_back_b = ChatMessage(
        type="ai",
        content="Agent B finished.",
        tool_calls=[
            {"name": "transfer_back_to_agent_a", "id": "back-b-a", "args": {"result": "result_2"}}
        ],
    )
    transfer_back_b_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_agent_a",
        tool_call_id="back-b-a",
    )

    # Agent A 将控制权交还给主管 Agent
    transfer_back_a = ChatMessage(
        type="ai",
        content="Agent A finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-a-super",
                "args": {"result": "result_1"},
            }
        ],
    )
    transfer_back_a_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-a-super",
    )

    # 主管继续执行，并将控制权交给 Agent C（与 A 同级）
    supervisor_continues = ChatMessage(
        type="ai",
        content="Now transferring to agent C...",
        tool_calls=[
            {"name": "transfer_to_agent_c", "id": "transfer-c", "args": {"task": "task_3"}}
        ],
    )
    transfer_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred via transfer_to_agent_c",
        tool_call_id="transfer-c",
    )

    # Agent C 交还控制权
    transfer_back_c = ChatMessage(
        type="ai",
        content="Agent C finished.",
        tool_calls=[
            {
                "name": "transfer_back_to_supervisor",
                "id": "back-c-super",
                "args": {"result": "result_3"},
            }
        ],
    )
    transfer_back_c_success = ChatMessage(
        type="tool",
        content="Successfully transferred back via transfer_back_to_supervisor",
        tool_call_id="back-c-super",
    )

    # 最终响应
    supervisor_final = ChatMessage(
        type="ai", content="All agents have completed their tasks successfully."
    )

    return {
        "tool_1": tool_1,
        "tool_1_result": tool_1_result,
        "tool_2": tool_2,
        "tool_2_result": tool_2_result,
        "transfer_a": transfer_a,
        "transfer_a_success": transfer_a_success,
        "transfer_b_from_a": transfer_b_from_a,
        "transfer_b_success": transfer_b_success,
        "transfer_back_b": transfer_back_b,
        "transfer_back_b_success": transfer_back_b_success,
        "transfer_back_a": transfer_back_a,
        "transfer_back_a_success": transfer_back_a_success,
        "supervisor_continues": supervisor_continues,
        "transfer_c_success": transfer_c_success,
        "transfer_back_c": transfer_back_c,
        "transfer_back_c_success": transfer_back_c_success,
        "supervisor_final": supervisor_final,
    }


@pytest.mark.asyncio
async def test_app_streaming_single_sub_agent(mock_agent_client, multi_agent_messages):
    """测试单个子 Agent 的多次工具调用，验证弹出面板功能。"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test single sub-agent with multiple tools"

    # 使用测试夹具并包含多个工作工具，以测试多个弹出面板
    # 主管 Agent → Agent A（调用 tool_1 和 tool_2）→ 主管 Agent
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message"
    )

    status_agent = ai_message.status[0]
    assert status_agent == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_agent.label

    assert status_agent.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )

    popover_1 = status_agent.children[1]
    assert hasattr(popover_1, "type") and popover_1.type == "popover", (
        "Second child of status should be a popover for the first tool call"
    )
    assert popover_1.proto.popover.label == "do_work_1"
    assert popover_1.proto.popover.icon == "🛠️"
    assert popover_1.markdown[0].value == "**Tool:** do_work_1"
    assert popover_1.markdown[1].value == "**Input:**"
    assert '"my-arg": "value"' in popover_1.json[0].value
    assert popover_1.markdown[2].value == "**Output:**"
    assert popover_1.markdown[3].value == "Tool 1 complete"

    assert status_agent.children[2].value == "Starting tool 2...", (
        "Third child of status should be tool 2 message"
    )

    popover_2 = status_agent.children[3]
    assert hasattr(popover_2, "type") and popover_2.type == "popover", (
        "Fourth child of the status should be a popover for the second tool call"
    )
    assert popover_2.proto.popover.label == "do_work_2"

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_sequential_sub_agents(mock_agent_client, multi_agent_messages):
    """测试主管将控制权交给子 Agent A、收回后再交给子 Agent C，并再次收回的流程。"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test multiple transfer back patterns"

    # 创建顺序执行的消息流：主管 → Agent A（工具 1）→ 主管 → Agent C（工具 2）→ 主管
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_continues"],
            messages["transfer_c_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_c"],
            messages["transfer_back_c_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # 状态容器的第二个子元素应为第一次工具调用的弹出面板
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**Tool:** do_work_1"
    assert popover_a.markdown[1].value == "**Input:**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**Output:**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert ai_message.children[2].value == "Now transferring to agent C...", (
        "Third child should be transfer message to agent C"
    )

    status_c = ai_message.status[1]
    assert status_c == ai_message.children[3], "Fourth child should be the second status"
    assert "transfer_to_agent_c" in status_c.label

    assert status_c.children[0].value == "Starting tool 2...", (
        "First child of next status should be tool 2 message"
    )
    popover_c = status_c.children[1]
    assert popover_c.type == "popover"
    assert popover_c.proto.popover.label == "do_work_2"
    assert popover_c.proto.popover.icon == "🛠️"
    assert popover_c.markdown[0].value == "**Tool:** do_work_2"
    assert popover_c.markdown[1].value == "**Input:**"
    assert popover_c.json[0].value == '{"my-arg-2": "value"}'
    assert popover_c.markdown[2].value == "**Output:**"
    assert popover_c.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[4].value == "All agents have completed their tasks successfully.", (
        "Fifth child should be final supervisor message"
    )

    assert len(ai_message.children) == 6, (
        f"Should have 6 children: transfer to a, status for a, transfer to c, status for c, final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception


@pytest.mark.asyncio
async def test_app_streaming_nested_sub_agents(mock_agent_client, multi_agent_messages):
    """测试嵌套子 Agent，其中 B 是 A 的子 Agent。"""

    at = AppTest.from_file("../../src/streamlit_app.py").run()

    PROMPT = "Test nested sub-agents"

    # 创建嵌套子 Agent 消息流：主管 → Agent A（工具 1）→ Agent B（工具 2）→ Agent A → 主管
    messages = multi_agent_messages

    async def amessage_iter():
        for msg in [
            messages["transfer_a"],
            messages["transfer_a_success"],
            messages["tool_1"],
            messages["tool_1_result"],
            messages["transfer_b_from_a"],
            messages["transfer_b_success"],
            messages["tool_2"],
            messages["tool_2_result"],
            messages["transfer_back_b"],
            messages["transfer_back_b_success"],
            messages["transfer_back_a"],
            messages["transfer_back_a_success"],
            messages["supervisor_final"],
        ]:
            yield msg

    mock_agent_client.astream = Mock(return_value=amessage_iter())

    at.toggle[0].set_value(True)
    at.chat_input[0].set_value(PROMPT).run()

    ai_message = at.chat_message[1]

    assert ai_message.children[0].value == "Transferring to agent A...", (
        "First child should be transfer message to agent A"
    )

    status_a = ai_message.status[0]
    assert status_a == ai_message.children[1], "Second child should be the first status"
    assert "transfer_to_agent_a" in status_a.label

    assert status_a.children[0].value == "Starting tool 1...", (
        "First child of status should be tool 1 message"
    )
    # 状态容器的第二个子元素应为第一次工具调用的弹出面板
    popover_a = status_a.children[1]
    assert popover_a.type == "popover"
    assert popover_a.proto.popover.label == "do_work_1"
    assert popover_a.proto.popover.icon == "🛠️"
    assert popover_a.markdown[0].value == "**Tool:** do_work_1"
    assert popover_a.markdown[1].value == "**Input:**"
    assert popover_a.json[0].value == '{"my-arg": "value"}'
    assert popover_a.markdown[2].value == "**Output:**"
    assert popover_a.markdown[3].value == "Tool 1 complete"

    assert status_a.children[2].value == "Agent A delegating to agent B...", (
        "Third child of status should be transfer message to agent B"
    )

    # 状态容器的第四个子元素应为 Agent B 的嵌套状态容器
    nested_status_b = status_a.children[3]
    assert "transfer_to_agent_b" in nested_status_b.label

    assert nested_status_b.children[0].value == "Starting tool 2...", (
        "First child of nested status should be tool 2 message"
    )
    # 嵌套状态容器的第二个子元素应为任务 2 工具调用的弹出面板
    popover_b = nested_status_b.children[1]
    assert popover_b.type == "popover"
    assert popover_b.proto.popover.label == "do_work_2"
    assert popover_b.proto.popover.icon == "🛠️"
    assert popover_b.markdown[0].value == "**Tool:** do_work_2"
    assert popover_b.markdown[1].value == "**Input:**"
    assert popover_b.json[0].value == '{"my-arg-2": "value"}'
    assert popover_b.markdown[2].value == "**Output:**"
    assert popover_b.markdown[3].value == "Tool 2 complete"

    assert ai_message.children[2].value == "All agents have completed their tasks successfully.", (
        "Third child should be final supervisor message"
    )

    assert len(ai_message.children) == 4, (
        f"Should have 4 children: transfer to a, status for a (with nested b), final message, feedback stars - got {len(ai_message.children)}"
    )

    assert not at.exception


@pytest.fixture
def mock_threads_data():
    """为缓存测试提供模拟会话数据的测试夹具。"""
    return UserThreads(
        threads=[
            ThreadSummary(
                thread_id="thread-1111-2222",
                agent_id="test-agent",
                updated_at="2026-07-31T20:14:19.804150+00:00",
                title="What is Python?",
            )
        ]
    )


def test_app_thread_caching_sidebar(mock_agent_client, mock_threads_data):
    """验证通过 get_user_threads 获取会话列表，并在侧边栏历史中渲染。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)

    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "user-123"
    at.run()

    mock_agent_client.get_user_threads.assert_called_with(
        user_id="user-123", agent="test-agent", limit=20
    )

    sidebar_buttons = [b.label for b in at.sidebar.button]
    assert "What is Python?" in sidebar_buttons
    assert not at.exception


def test_app_thread_click_loads_history(mock_agent_client, mock_threads_data):
    """验证点击侧边栏会话后，将对应对话加载到聊天区。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)
    mock_agent_client.get_history = Mock(
        return_value=ChatHistory(
            messages=[
                ChatMessage(type="human", content="What is Python?"),
                ChatMessage(type="ai", content="A programming language."),
            ]
        )
    )

    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "user-123"
    at.run()

    at.button(key="thread_thread-1111-2222").click().run()

    mock_agent_client.get_history.assert_called_with(
        thread_id="thread-1111-2222", agent="test-agent"
    )
    assert at.session_state.thread_id == "thread-1111-2222"
    assert [m.content for m in at.session_state.messages] == [
        "What is Python?",
        "A programming language.",
    ]
    assert not at.exception


def test_app_thread_click_history_error(mock_agent_client, mock_threads_data):
    """验证历史查询失败时显示错误，并保留当前聊天内容。"""
    mock_agent_client.get_user_threads = Mock(return_value=mock_threads_data)
    mock_agent_client.get_history = Mock(side_effect=AgentClientError("service down"))

    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "user-123"
    at.run()
    original_thread_id = at.session_state.thread_id

    at.button(key="thread_thread-1111-2222").click().run()

    assert any("Could not load that conversation." in error.value for error in at.error)
    assert at.session_state.thread_id == original_thread_id
    assert not at.exception


def test_app_thread_fetch_error_shows_caption(mock_agent_client):
    """验证会话列表接口失败时，侧边栏能够平稳降级。"""
    mock_agent_client.get_user_threads = Mock(side_effect=AgentClientError("service down"))

    at = AppTest.from_file("../../src/streamlit_app.py")
    at.query_params["user_id"] = "user-123"
    at.run()

    assert any("Couldn't load conversation history" in caption.value for caption in at.caption)
    assert not at.exception
