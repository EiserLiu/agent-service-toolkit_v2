import asyncio
import os
import urllib.parse
import uuid
from collections.abc import AsyncGenerator

import streamlit as st
from dotenv import load_dotenv
from pydantic import ValidationError

from client import AgentClient, AgentClientError
from schema import ChatHistory, ChatMessage, UserThreads
from schema.task_data import TaskData, TaskDataStatus
from voice import VoiceManager

# 通过简单聊天界面与 LangGraph Agent 交互的 Streamlit 应用。
# 应用主要包含三个异步运行的函数：

# - main()：初始化 Streamlit 应用及整体布局
# - draw_messages()：绘制聊天消息，既支持重放历史消息，
#   也支持流式显示新消息
# - handle_feedback()：绘制反馈控件并记录用户反馈

# 应用主要通过 AgentClient 与 Agent 的 FastAPI 接口交互。


APP_TITLE = "Agent Service Toolkit"
APP_ICON = "🧰"
USER_ID_COOKIE = "user_id"


def get_or_create_user_id() -> str:
    """从会话状态或 URL 参数获取用户 ID；不存在时创建新的 ID。"""
    # 检查会话状态中是否存在 user_id
    if USER_ID_COOKIE in st.session_state:
        return st.session_state[USER_ID_COOKIE]

    # 尝试通过新的 st.query_params 从 URL 参数中获取
    if USER_ID_COOKIE in st.query_params:
        user_id = st.query_params[USER_ID_COOKIE]
        st.session_state[USER_ID_COOKIE] = user_id
        return user_id

    # 未找到时生成新的 user_id
    user_id = str(uuid.uuid4())

    # 保存到当前会话状态
    st.session_state[USER_ID_COOKIE] = user_id

    # 同时写入 URL 参数，以便收藏或分享
    st.query_params[USER_ID_COOKIE] = user_id

    return user_id


@st.cache_data(ttl=600, show_spinner=False)
def fetch_user_threads_cached(
    base_url: str, user_id: str, agent_id: str | None = None, limit: int = 20
) -> UserThreads:
    """使用新的同步 get_user_threads 方法获取并缓存用户会话。"""
    client = AgentClient(base_url=base_url, get_info=False)
    return client.get_user_threads(user_id=user_id, agent=agent_id, limit=limit)


async def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon=APP_ICON,
        menu_items={},
    )

    # 隐藏 Streamlit 右上角的界面控件
    st.html(
        """
        <style>
        [data-testid="stStatusWidget"] {
                visibility: hidden;
                height: 0%;
                position: fixed;
            }
        </style>
        """,
    )
    if st.get_option("client.toolbarMode") != "minimal":
        st.set_option("client.toolbarMode", "minimal")
        await asyncio.sleep(0.1)
        st.rerun()

    # 获取或创建用户 ID
    user_id = get_or_create_user_id()

    if "agent_client" not in st.session_state:
        load_dotenv()
        agent_url = os.getenv("AGENT_URL")
        if not agent_url:
            host = os.getenv("HOST", "0.0.0.0")
            port = os.getenv("PORT", 8080)
            agent_url = f"http://{host}:{port}"
        try:
            with st.spinner("Connecting to agent service..."):
                st.session_state.agent_client = AgentClient(base_url=agent_url)
        except AgentClientError as e:
            st.error(f"Error connecting to agent service at {agent_url}: {e}")
            st.markdown("The service might be booting up. Try again in a few seconds.")
            st.stop()
    agent_client: AgentClient = st.session_state.agent_client

    # 初始化语音管理器（每个会话仅一次）
    if "voice_manager" not in st.session_state:
        st.session_state.voice_manager = VoiceManager.from_env()
    voice = st.session_state.voice_manager

    if "thread_id" not in st.session_state:
        thread_id = st.query_params.get("thread_id")
        if not thread_id:
            thread_id = str(uuid.uuid4())
            messages = []
        else:
            # 从 URL 读取 Agent，确保使用创建该会话的图
            # 来查询历史消息。
            resume_agent = st.query_params.get("agent") or agent_client.agent
            try:
                messages: ChatHistory = agent_client.get_history(
                    thread_id=thread_id, agent=resume_agent
                ).messages
            except AgentClientError:
                st.error("No message history found for this Thread ID.")
                messages = []
        st.session_state.messages = messages
        st.session_state.thread_id = thread_id

    # 将 thread_id 保留在 URL 中，使地址栏中的链接可直接分享。
    st.query_params["thread_id"] = st.session_state.thread_id

    # 配置选项
    with st.sidebar:
        st.header(f"{APP_ICON} {APP_TITLE}")

        ""
        "Full toolkit for running an AI agent service built with LangGraph, FastAPI and Streamlit"
        ""

        if st.button(":material/chat: New Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.thread_id = str(uuid.uuid4())
            # 开始新聊天时清除已保存的音频
            if "last_audio" in st.session_state:
                del st.session_state.last_audio
            st.rerun()

        with st.expander(":material/history: Previous Chats", expanded=False):
            try:
                url_agent = st.query_params.get("agent")
                if url_agent in [a.key for a in agent_client.info.agents]:
                    agent_client.agent = url_agent
                else:
                    agent_client.agent = agent_client.info.default_agent
                user_threads = fetch_user_threads_cached(
                    base_url=agent_client.base_url,
                    user_id=user_id,
                    agent_id=agent_client.agent,
                    limit=20,
                )
                thread_list = user_threads.threads
            except Exception as e:
                st.caption(f"Couldn't load conversation history: {e}")
                thread_list = []

            for t in thread_list:
                label = t.title or f"Chat {t.thread_id[:8]}"
                if st.button(label, key=f"thread_{t.thread_id}", use_container_width=True):
                    try:
                        history: ChatHistory = agent_client.get_history(
                            thread_id=t.thread_id, agent=t.agent_id
                        )
                    except AgentClientError:
                        st.error("Could not load that conversation.")
                        continue
                    st.session_state.messages = history.messages
                    st.session_state.thread_id = t.thread_id
                    st.query_params["thread_id"] = t.thread_id
                    if "last_audio" in st.session_state:
                        del st.session_state.last_audio
                    st.rerun()

        with st.popover(":material/settings: Settings", use_container_width=True):
            model_idx = agent_client.info.models.index(agent_client.info.default_model)
            model = st.selectbox("LLM to use", options=agent_client.info.models, index=model_idx)
            agent_list = [a.key for a in agent_client.info.agents]
            agent_idx = agent_list.index(agent_client.info.default_agent)
            # 将选项同步到 URL 的 ?agent= 参数；选中默认 Agent 时移除此参数。
            agent_client.agent = st.selectbox(
                "Agent to use",
                options=agent_list,
                index=agent_idx,
                key="agent",
                bind="query-params",
                on_change=fetch_user_threads_cached.clear,
            )
            use_streaming = st.toggle("Stream results", value=True)
            # 带回调的音频开关：关闭时清除缓存音频
            enable_audio = st.toggle(
                "Enable audio generation",
                value=True,
                disabled=not voice or not voice.tts,
                help="Configure VOICE_TTS_PROVIDER in .env to enable"
                if not voice or not voice.tts
                else None,
                on_change=lambda: (
                    st.session_state.pop("last_audio", None)
                    if not st.session_state.get("enable_audio", True)
                    else None
                ),
                key="enable_audio",
            )

            # 显示用户 ID（用于调试或展示用户信息）
            st.text_input("User ID (read-only)", value=user_id, disabled=True)

        @st.dialog("Architecture")
        def architecture_dialog() -> None:
            st.image(
                "https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png?raw=true"
            )
            "[View full size on Github](https://github.com/JoshuaC215/agent-service-toolkit/blob/main/media/agent_architecture.png)"
            st.caption(
                "App hosted on [Streamlit Cloud](https://share.streamlit.io/) with FastAPI service running in [Azure](https://learn.microsoft.com/en-us/azure/app-service/)"
            )

        if st.button(":material/schema: Architecture", use_container_width=True):
            architecture_dialog()

        with st.popover(":material/policy: Privacy", use_container_width=True):
            st.write(
                "Prompts, responses and feedback in this app are anonymously recorded and saved to LangSmith for product evaluation and improvement purposes only."
            )

        @st.dialog("Share/resume chat")
        def share_chat_dialog() -> None:
            # st.context.url 是已去掉查询字符串的浏览器地址。重新构建
            # 查询参数并包含 Agent，以便使用正确的图恢复会话。
            if not st.context.url:
                st.error("Could not determine the app URL to build a shareable link.")
                return
            query = urllib.parse.urlencode(
                {
                    "thread_id": st.session_state.thread_id,
                    "agent": agent_client.agent,
                    USER_ID_COOKIE: user_id,
                }
            )
            chat_url = f"{st.context.url}?{query}"
            st.markdown(f"**Chat URL:**\n```text\n{chat_url}\n```")
            st.info("Copy the above URL to share or revisit this chat")

        if st.button(":material/upload: Share/resume chat", use_container_width=True):
            share_chat_dialog()

        "[View the source code](https://github.com/JoshuaC215/agent-service-toolkit)"
        st.caption(
            "Made with :material/favorite: by [Joshua](https://www.linkedin.com/in/joshua-k-carroll/) in Oakland"
        )

    # 绘制已有消息
    messages: list[ChatMessage] = st.session_state.messages

    if len(messages) == 0:
        match agent_client.agent:
            case "chatbot":
                WELCOME = "Hello! I'm a simple chatbot. Ask me anything!"
            case "interrupt-agent":
                WELCOME = "Hello! I'm an interrupt agent. Tell me your birthday and I will predict your personality!"
            case "research-assistant":
                WELCOME = "Hello! I'm an AI-powered research assistant with web search and a calculator. Ask me anything!"
            case "rag-assistant":
                WELCOME = """Hello! I'm an AI-powered Company Policy & HR assistant with access to AcmeTech's Employee Handbook.
                I can help you find information about benefits, remote work, time-off policies, company values, and more. Ask me anything!"""
            case _:
                WELCOME = "Hello! I'm an AI agent. Ask me anything!"

        with st.chat_message("ai"):
            st.write(WELCOME)

    # draw_messages() 接收消息的异步迭代器
    async def amessage_iter() -> AsyncGenerator[ChatMessage, None]:
        for m in messages:
            yield m

    await draw_messages(amessage_iter())

    # 渲染最后一条 AI 消息已保存的音频（如果存在）
    # 确保调用 st.rerun() 后音频仍然保留
    if (
        voice
        and enable_audio
        and "last_audio" in st.session_state
        and st.session_state.last_message
        and len(messages) > 0
        and messages[-1].type == "ai"
    ):
        with st.session_state.last_message:
            audio_data = st.session_state.last_audio
            st.audio(audio_data["data"], format=audio_data["format"])

    # 用户提供新输入时生成新消息
    # 优先使用语音管理器；不可用时回退到普通输入
    # 启用语音功能必须在应用的 .env（而非服务端 .env）中设置
    # VOICE_STT_PROVIDER、VOICE_TTS_PROVIDER 和 OPENAI_API_KEY。
    if voice:
        user_input = voice.get_chat_input()
    else:
        user_input = st.chat_input()

    if user_input:
        is_first_message = len(messages) == 0
        messages.append(ChatMessage(type="human", content=user_input))
        st.chat_message("human").write(user_input)
        try:
            if use_streaming:
                stream = agent_client.astream(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                await draw_messages(stream, is_new=True)
                # 为流式响应生成 TTS 音频
                # 注意：draw_messages() 将最终消息存入 st.session_state.messages，
                # 并将容器引用存入 st.session_state.last_message
                if voice and enable_audio and st.session_state.messages:
                    last_msg = st.session_state.messages[-1]
                    # 仅为有内容的 AI 响应生成音频
                    if last_msg.type == "ai" and last_msg.content:
                        # 文本已由 draw_messages() 流式显示，因此使用 audio_only=True
                        voice.render_message(
                            last_msg.content,
                            container=st.session_state.last_message,
                            audio_only=True,
                        )
            else:
                response = await agent_client.ainvoke(
                    message=user_input,
                    model=model,
                    thread_id=st.session_state.thread_id,
                    user_id=user_id,
                )
                messages.append(response)
                # 渲染 AI 响应，并按需附带语音
                with st.chat_message("ai"):
                    if voice and enable_audio:
                        voice.render_message(response.content)
                    else:
                        st.write(response.content)
            if is_first_message:
                fetch_user_threads_cached.clear()
            st.rerun()  # 清除过期容器
        except AgentClientError as e:
            st.error(f"Error generating response: {e}")
            st.stop()

    # 已生成消息时，显示反馈控件
    if len(messages) > 0 and st.session_state.last_message:
        with st.session_state.last_message:
            await handle_feedback()


async def draw_messages(
    messages_agen: AsyncGenerator[ChatMessage | str, None],
    is_new: bool = False,
) -> None:
    """绘制聊天消息，支持重放历史消息或流式显示新消息。

    包含处理流式 token 和工具调用的附加逻辑：
    - 使用占位容器实时渲染到达的 token。
    - 使用状态容器渲染工具调用，跟踪工具输入和输出并更新容器。

    还需在会话状态中保存最后一个消息容器，因为后续消息可能继续写入同一
    容器；该引用也用于在最新聊天消息中绘制反馈控件。

    参数：
        messages_aiter：待绘制消息的异步迭代器。
        is_new：消息是否为新消息。
    """

    # 记录最后一个消息容器
    last_message_type = None
    st.session_state.last_message = None

    # 用于显示流式中间 token 的占位容器
    streaming_content = ""
    streaming_placeholder = None

    # 遍历并绘制消息
    while msg := await anext(messages_agen, None):
        # 字符串消息表示正在流式传输的中间 token
        if isinstance(msg, str):
            # 如果占位容器为空，说明这是新消息的第一个
            # 流式 token，需要先完成初始化。
            if not streaming_placeholder:
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")
                with st.session_state.last_message:
                    streaming_placeholder = st.empty()

            streaming_content += msg
            streaming_placeholder.write(streaming_content)
            continue
        if not isinstance(msg, ChatMessage):
            st.error(f"Unexpected message type: {type(msg)}")
            st.write(msg)
            st.stop()

        match msg.type:
            # 用户消息的处理最简单
            case "human":
                last_message_type = "human"
                st.chat_message("human").write(msg.content)

            # Agent 消息的处理最复杂，因为需要
            # 同时处理流式 token 和工具调用。
            case "ai":
                # 如果正在渲染新消息，将其存入会话状态
                if is_new:
                    st.session_state.messages.append(msg)

                # 如果上一条消息不是 AI 类型，则创建新的聊天消息
                if last_message_type != "ai":
                    last_message_type = "ai"
                    st.session_state.last_message = st.chat_message("ai")

                with st.session_state.last_message:
                    # 消息有内容时，将其显示出来。
                    # 重置流式变量，为下一条消息做准备。
                    if msg.content:
                        if streaming_placeholder:
                            streaming_placeholder.write(msg.content)
                            streaming_content = ""
                            streaming_placeholder = None
                        else:
                            st.write(msg.content)

                    if msg.tool_calls:
                        # 为每次工具调用创建状态容器，并按 ID
                        # 保存容器，确保结果能对应到
                        # 正确的状态容器。
                        call_results = {}
                        for tool_call in msg.tool_calls:
                            # 控制权交接和普通工具调用使用不同标签
                            if "transfer_to" in tool_call["name"]:
                                label = f"""💼 Sub Agent: {tool_call["name"]}"""
                            else:
                                label = f"""🛠️ Tool Call: {tool_call["name"]}"""

                            status = st.status(
                                label,
                                state="running" if is_new else "complete",
                            )
                            call_results[tool_call["id"]] = status

                        # 每次工具调用应对应一条 ToolMessage。
                        for tool_call in msg.tool_calls:
                            if "transfer_to" in tool_call["name"]:
                                status = call_results[tool_call["id"]]
                                status.update(expanded=True)
                                await handle_sub_agent_msgs(messages_agen, status, is_new)
                                break

                            # 只有非控制权交接的工具调用才会到达这里
                            status = call_results[tool_call["id"]]
                            status.write("Input:")
                            status.write(tool_call["args"])
                            tool_result: ChatMessage = await anext(messages_agen)

                            if tool_result.type != "tool":
                                st.error(f"Unexpected ChatMessage type: {tool_result.type}")
                                st.write(tool_result)
                                st.stop()

                            # 如果是新消息，则记录消息，并用结果更新
                            # 对应的状态容器
                            if is_new:
                                st.session_state.messages.append(tool_result)
                            if tool_result.tool_call_id:
                                status = call_results[tool_result.tool_call_id]
                            status.write("Output:")
                            status.write(tool_result.content)
                            status.update(state="complete")

            case "custom":
                # bg-task-agent 使用的 CustomData 示例
                # 参考：
                # - src/agents/utils.py CustomData
                # - src/agents/bg_task_agent/task.py
                try:
                    task_data: TaskData = TaskData.model_validate(msg.custom_data)
                except ValidationError:
                    st.error("Unexpected CustomData message received from agent")
                    st.write(msg.custom_data)
                    st.stop()

                if is_new:
                    st.session_state.messages.append(msg)

                if last_message_type != "task":
                    last_message_type = "task"
                    st.session_state.last_message = st.chat_message(
                        name="task", avatar=":material/manufacturing:"
                    )
                    with st.session_state.last_message:
                        status = TaskDataStatus()

                status.add_and_draw_task_data(task_data)

            # 遇到意外的消息类型时，记录错误并停止
            case _:
                st.error(f"Unexpected ChatMessage type: {msg.type}")
                st.write(msg)
                st.stop()


async def handle_feedback() -> None:
    """绘制反馈控件并记录用户反馈。"""

    # 记录上次发送的反馈，避免重复提交
    if "last_feedback" not in st.session_state:
        st.session_state.last_feedback = (None, None)

    latest_run_id = st.session_state.messages[-1].run_id
    feedback = st.feedback("stars", key=latest_run_id)

    # 反馈值或运行 ID 改变时，发送新的反馈记录
    if feedback is not None and (latest_run_id, feedback) != st.session_state.last_feedback:
        # 将反馈值（索引）归一化为 0 到 1 之间的分数
        normalized_score = (feedback + 1) / 5.0

        agent_client: AgentClient = st.session_state.agent_client
        try:
            await agent_client.acreate_feedback(
                run_id=latest_run_id,
                key="human-feedback-stars",
                score=normalized_score,
                kwargs={"comment": "In-line human feedback"},
            )
        except AgentClientError as e:
            st.error(f"Error recording feedback: {e}")
            st.stop()
        st.session_state.last_feedback = (latest_run_id, feedback)
        st.toast("Feedback recorded", icon=":material/reviews:")


async def handle_sub_agent_msgs(messages_agen, status, is_new):
    """将 Agent 的输出集中展示在状态容器中。
    处理最初的工具调用消息之后的所有消息，直到最终 AI 消息。

    支持带控制权交还消息的嵌套多 Agent 层级。

    参数：
        messages_agen：消息异步生成器。
        status：当前 Agent 的状态容器。
        is_new：消息是新生成的还是历史重放的。
    """
    nested_popovers = {}

    # 查找表示控制权交接成功的工具结果消息
    first_msg = await anext(messages_agen)
    if is_new:
        st.session_state.messages.append(first_msg)

    # 继续读取，直到收到明确的控制权交还消息
    while True:
        # 读取下一条消息
        sub_msg = await anext(messages_agen)

        # 只有移除 skip_stream 标记后，才应该出现这种情况
        # if isinstance(sub_msg, str):
        #     continue

        if is_new:
            st.session_state.messages.append(sub_msg)

        # 使用嵌套弹出面板处理工具结果
        if sub_msg.type == "tool" and sub_msg.tool_call_id in nested_popovers:
            popover = nested_popovers[sub_msg.tool_call_id]
            popover.write("**Output:**")
            popover.write(sub_msg.content)
            continue

        # 处理 transfer_back_to 工具调用：它们表示子 Agent 正在交还控制权
        if (
            hasattr(sub_msg, "tool_calls")
            and sub_msg.tool_calls
            and any("transfer_back_to" in tc.get("name", "") for tc in sub_msg.tool_calls)
        ):
            # 处理 transfer_back_to 工具调用
            for tc in sub_msg.tool_calls:
                if "transfer_back_to" in tc.get("name", ""):
                    # 读取对应的工具结果
                    transfer_result = await anext(messages_agen)
                    if is_new:
                        st.session_state.messages.append(transfer_result)

            # 处理完控制权交还后，结束当前 Agent 的处理
            if status:
                status.update(state="complete")
            break

        # 在同一个嵌套状态容器中显示内容和工具调用
        if status:
            if sub_msg.content:
                status.write(sub_msg.content)

            if hasattr(sub_msg, "tool_calls") and sub_msg.tool_calls:
                for tc in sub_msg.tool_calls:
                    # 检查是否为嵌套的控制权交接或委派
                    if "transfer_to" in tc["name"]:
                        # 为子 Agent 创建嵌套状态容器
                        nested_status = status.status(
                            f"""💼 Sub Agent: {tc["name"]}""",
                            state="running" if is_new else "complete",
                            expanded=True,
                        )

                        # 递归处理该子 Agent 的下级 Agent
                        await handle_sub_agent_msgs(messages_agen, nested_status, is_new)
                    else:
                        # 普通工具调用：创建弹出面板
                        popover = status.popover(f"{tc['name']}", icon="🛠️")
                        popover.write(f"**Tool:** {tc['name']}")
                        popover.write("**Input:**")
                        popover.write(tc["args"])
                        # 使用工具调用 ID 保存弹出面板引用
                        nested_popovers[tc["id"]] = popover


if __name__ == "__main__":
    asyncio.run(main())
