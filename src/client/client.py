import json
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any

import httpx

from schema import (
    ChatHistory,
    ChatHistoryInput,
    ChatMessage,
    Feedback,
    ServiceMetadata,
    StreamInput,
    UserInput,
    UserThreads,
    UserThreadsInput,
)


class AgentClientError(Exception):
    pass


class AgentClient:
    """与 Agent 服务交互的客户端。"""

    def __init__(
        self,
        base_url: str = "http://0.0.0.0",
        agent: str | None = None,
        timeout: float | None = None,
        get_info: bool = True,
    ) -> None:
        """初始化客户端。

        参数：
            base_url (str)：Agent 服务的基础地址。
            agent (str)：默认使用的 Agent 名称。
            timeout (float, 可选)：请求超时时间。
            get_info (bool, 可选)：是否在初始化时获取 Agent 信息，默认为 True。
        """
        self.base_url = base_url
        self.auth_secret = os.getenv("AUTH_SECRET")
        self.timeout = timeout
        self.info: ServiceMetadata | None = None
        self.agent: str | None = None
        if get_info:
            self.retrieve_info()
        if agent:
            self.update_agent(agent)

    @property
    def _headers(self) -> dict[str, str]:
        headers = {}
        if self.auth_secret:
            headers["Authorization"] = f"Bearer {self.auth_secret}"
        return headers

    def retrieve_info(self) -> None:
        try:
            response = httpx.get(
                f"{self.base_url}/info",
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error getting service info: {e}")

        self.info = ServiceMetadata.model_validate(response.json())
        if not self.agent or self.agent not in [a.key for a in self.info.agents]:
            self.agent = self.info.default_agent

    def update_agent(self, agent: str, verify: bool = True) -> None:
        if verify:
            if not self.info:
                self.retrieve_info()
            agent_keys = [a.key for a in self.info.agents]  # type: ignore[union-attr]
            if agent not in agent_keys:
                raise AgentClientError(
                    f"Agent {agent} not found in available agents: {', '.join(agent_keys)}"
                )
        self.agent = agent

    async def ainvoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """异步调用 Agent，仅返回最终消息。

        参数：
            message (str)：发送给 Agent 的消息。
            model (str, 可选)：Agent 使用的 LLM 模型。
            thread_id (str, 可选)：用于继续对话的会话 ID。
            user_id (str, 可选)：用于跨会话继续对话的用户 ID。
            agent_config (dict[str, Any], 可选)：透传给 Agent 的附加配置。

        返回值：
            AnyMessage：Agent 的响应。
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/{self.agent}/invoke",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def invoke(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
    ) -> ChatMessage:
        """同步调用 Agent，仅返回最终消息。

        参数：
            message (str)：发送给 Agent 的消息。
            model (str, 可选)：Agent 使用的 LLM 模型。
            thread_id (str, 可选)：用于继续对话的会话 ID。
            user_id (str, 可选)：用于跨会话继续对话的用户 ID。
            agent_config (dict[str, Any], 可选)：透传给 Agent 的附加配置。

        返回值：
            ChatMessage：Agent 的响应。
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = UserInput(message=message)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        try:
            response = httpx.post(
                f"{self.base_url}/{self.agent}/invoke",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatMessage.model_validate(response.json())

    def _parse_stream_line(self, line: str) -> ChatMessage | str | None:
        line = line.strip()
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return None
            try:
                parsed = json.loads(data)
            except Exception as e:
                raise Exception(f"Error JSON parsing message from server: {e}")
            match parsed["type"]:
                case "message":
                    # 将 JSON 格式的消息转换为 AnyMessage
                    try:
                        return ChatMessage.model_validate(parsed["content"])
                    except Exception as e:
                        raise Exception(f"Server returned invalid message: {e}")
                case "token":
                    # 直接产出字符串 token
                    return parsed["content"]
                case "error":
                    error_msg = "Error: " + parsed["content"]
                    return ChatMessage(type="ai", content=error_msg)
        return None

    def stream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> Generator[ChatMessage | str, None, None]:
        """同步流式获取 Agent 的响应。

        Agent 执行过程中的每条中间消息以 ChatMessage 形式产出。
        stream_tokens 为 True（默认值）时，还会实时产出流式模型生成的内容 token。

        参数：
            message (str)：发送给 Agent 的消息。
            model (str, 可选)：Agent 使用的 LLM 模型。
            thread_id (str, 可选)：用于继续对话的会话 ID。
            user_id (str, 可选)：用于跨会话继续对话的用户 ID。
            agent_config (dict[str, Any], 可选)：透传给 Agent 的附加配置。
            stream_tokens (bool, 可选)：是否实时输出生成的 token，默认为 True。

        返回值：
            Generator[ChatMessage | str, None, None]：Agent 响应的生成器。
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if user_id:
            request.user_id = user_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        try:
            with httpx.stream(
                "POST",
                f"{self.base_url}/{self.agent}/stream",
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.strip():
                        parsed = self._parse_stream_line(line)
                        if parsed is None:
                            break
                        yield parsed
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

    async def astream(
        self,
        message: str,
        model: str | None = None,
        thread_id: str | None = None,
        user_id: str | None = None,
        agent_config: dict[str, Any] | None = None,
        stream_tokens: bool = True,
    ) -> AsyncGenerator[ChatMessage | str, None]:
        """异步流式获取 Agent 的响应。

        Agent 执行过程中的每条中间消息以 AnyMessage 形式产出。
        stream_tokens 为 True（默认值）时，还会实时产出流式模型生成的内容 token。

        参数：
            message (str)：发送给 Agent 的消息。
            model (str, 可选)：Agent 使用的 LLM 模型。
            thread_id (str, 可选)：用于继续对话的会话 ID。
            user_id (str, 可选)：用于跨会话继续对话的用户 ID。
            agent_config (dict[str, Any], 可选)：透传给 Agent 的附加配置。
            stream_tokens (bool, 可选)：是否实时输出生成的 token，默认为 True。

        返回值：
            AsyncGenerator[ChatMessage | str, None]：Agent 响应的异步生成器。
        """
        if not self.agent:
            raise AgentClientError("No agent selected. Use update_agent() to select an agent.")
        request = StreamInput(message=message, stream_tokens=stream_tokens)
        if thread_id:
            request.thread_id = thread_id
        if model:
            request.model = model  # type: ignore[assignment]
        if agent_config:
            request.agent_config = agent_config
        if user_id:
            request.user_id = user_id
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/{self.agent}/stream",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line.strip():
                            parsed = self._parse_stream_line(line)
                            if parsed is None:
                                break
                            # 不产出空字符串 token，避免导致生成器问题
                            if parsed != "":
                                yield parsed
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    async def acreate_feedback(
        self, run_id: str, key: str, score: float, kwargs: dict[str, Any] = {}
    ) -> None:
        """为一次运行创建反馈记录。

        对 LangSmith create_feedback API 的简单封装，使凭证统一在服务端
        保存和管理，无需存放于客户端。
        参考：https://api.smith.langchain.com/redoc#tag/feedback/operation/create_feedback_api_v1_feedback_post
        """
        request = Feedback(run_id=run_id, key=key, score=score, kwargs=kwargs)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.base_url}/feedback",
                    json=request.model_dump(),
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response.json()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

    def get_history(self, thread_id: str, agent: str | None = None) -> ChatHistory:
        """获取聊天历史。

        参数：
            thread_id (str, 可选)：标识对话的会话 ID。
            agent (str, 可选)：使用哪个 Agent 的图读取该会话。
        """
        agent = agent or self.agent
        request = ChatHistoryInput(thread_id=thread_id)
        url = f"{self.base_url}/{agent}/history" if agent else f"{self.base_url}/history"
        try:
            response = httpx.post(
                url,
                json=request.model_dump(),
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return ChatHistory.model_validate(response.json())

    def _user_threads_request(
        self, user_id: str, agent: str | None, limit: int
    ) -> tuple[str, dict[str, Any]]:
        agent_id = agent or self.agent
        url = f"{self.base_url}/{agent_id}/threads" if agent_id else f"{self.base_url}/threads"
        return url, UserThreadsInput(user_id=user_id, limit=limit).model_dump()

    def get_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """列出用户的会话。

        参数：
            user_id (str)：要查询的用户 ID。
            agent (str, 可选)：要查询会话的 Agent。
            limit (int, 可选)：最多返回的会话数。
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        try:
            response = httpx.get(
                url,
                params=params,
                headers=self._headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as e:
            raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())

    async def aget_user_threads(
        self, user_id: str, agent: str | None = None, limit: int = 20
    ) -> UserThreads:
        """异步列出用户的会话。

        参数：
            user_id (str)：要查询的用户 ID。
            agent (str, 可选)：要查询会话的 Agent。
            limit (int, 可选)：最多返回的会话数。
        """
        url, params = self._user_threads_request(user_id, agent, limit)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    params=params,
                    headers=self._headers,
                    timeout=self.timeout,
                )
                response.raise_for_status()
            except httpx.HTTPError as e:
                raise AgentClientError(f"Error: {e}")

        return UserThreads.model_validate(response.json())
