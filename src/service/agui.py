"""Agent 服务的 AG-UI 协议接口。

通过 AG-UI 协议（https://docs.ag-ui.com）暴露服务中的任意 Agent，
以对接 CopilotKit 等兼容前端。LangGraph 到 AG-UI 的事件转换由官方
ag-ui-langgraph 包处理；本模块仅将它接入服务的 Agent 注册表、认证和追踪。

用法及客户端连接方法见 docs/AGUI.md。
"""

import logging
from collections.abc import AsyncGenerator
from typing import Any
from uuid import uuid4

from ag_ui.core import EventType, RunAgentInput
from ag_ui.encoder import EventEncoder
from ag_ui_langgraph import LangGraphAgent
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from langfuse.langchain import CallbackHandler  # type: ignore[import-untyped]

from agents import DEFAULT_AGENT, AgentGraph, get_agent
from core import settings
from service.utils import ensure_model_available

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agui")

# 这些字段由协议（thread_id 来自 RunAgentInput）或检查点保存器管理，
# 因此客户端不能通过 forwardedProps.configurable 覆盖它们。
RESERVED_CONFIGURABLE_KEYS = {"thread_id", "checkpoint_id", "checkpoint_ns"}


def _base_config(input_data: RunAgentInput, agent_id: str) -> RunnableConfig:
    """构建 AG-UI 运行所需的基础 RunnableConfig。

    客户端可通过 forwardedProps.configurable 传入 model、user_id 或自定义
    Agent 配置，对应原生 API 的 model、user_id 和 agent_config 字段。
    thread_id 由 ag-ui-langgraph 包直接从 AG-UI 输入中获取。
    """
    forwarded: dict[str, Any] = input_data.forwarded_props or {}
    configurable = forwarded.get("configurable") or {}
    if not isinstance(configurable, dict):
        raise HTTPException(status_code=422, detail="forwardedProps.configurable must be an object")
    if overlap := RESERVED_CONFIGURABLE_KEYS & configurable.keys():
        raise HTTPException(
            status_code=422,
            detail=f"forwardedProps.configurable contains reserved keys: {overlap}",
        )

    if (model := configurable.get("model")) is not None:
        ensure_model_available(model)

    callbacks: list[Any] = []
    if settings.LANGFUSE_TRACING:
        callbacks.append(CallbackHandler())

    configurable = dict(configurable)
    user_id = configurable.setdefault("user_id", str(uuid4()))

    return RunnableConfig(
        configurable=configurable,
        # 记录到检查点元数据中，使 AG-UI 会话也能出现在 /threads 中。
        metadata={"user_id": user_id, "agent_id": agent_id},
        callbacks=callbacks,
    )


async def _event_stream(
    agent_id: str,
    graph: AgentGraph,
    input_data: RunAgentInput,
    config: RunnableConfig,
    encoder: EventEncoder,
) -> AsyncGenerator[str, None]:
    # 每个请求创建一个 LangGraphAgent：它保存单次运行状态，且构建开销较低。
    agent = LangGraphAgent(name=agent_id, graph=graph, config=config)  # type: ignore[arg-type]
    async for event in agent.run(input_data):
        # 不转发 RAW 透传事件。标准 AG-UI 客户端会忽略这些事件，
        # 而且它们会向调用者暴露服务端内部信息，包括
        # on_chat_model_start 中已完整渲染的提示词。只有在接口由
        # 可信中间层调用，且确实需要完整事件流时才移除此过滤，
        # 例如用于 AG-UI Event Inspector。
        if event.type == EventType.RAW:
            continue
        yield encoder.encode(event)


@router.post("/run", operation_id="agui_run_default")
@router.post("/{agent_id}/run", operation_id="agui_run")
async def agui_run(
    input_data: RunAgentInput, request: Request, agent_id: str = DEFAULT_AGENT
) -> StreamingResponse:
    """通过 AG-UI 协议运行 Agent，以 SSE 流式发送 AG-UI 事件。

    将 AG-UI 客户端（如 CopilotKit 运行时或 HttpAgent）指向此接口。
    多次运行使用相同 threadId 即可继续对话；会话通过服务的检查点保存器
    持久化，并与原生 API 共享。
    """
    try:
        graph: AgentGraph = get_agent(agent_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

    config = _base_config(input_data, agent_id)
    encoder = EventEncoder(accept=request.headers.get("accept", ""))
    return StreamingResponse(
        _event_stream(agent_id, graph, input_data, config, encoder),
        media_type=encoder.get_content_type(),
    )
