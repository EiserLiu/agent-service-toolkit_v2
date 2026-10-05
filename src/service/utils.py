from collections.abc import Mapping
from typing import Any, cast

from fastapi import HTTPException
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.messages import (
    ChatMessage as LangchainChatMessage,
)

from core import settings
from schema import ChatMessage


def ensure_model_available(model: Any) -> None:
    """model 不在服务配置的 AVAILABLE_MODELS 允许列表中时，返回 400 错误。"""
    if model not in settings.AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' is not available. "
            f"Allowed: {[m.value for m in settings.AVAILABLE_MODELS]}",
        )


def convert_message_content_to_string(content: str | list[str | dict]) -> str:
    if isinstance(content, str):
        return content
    text: list[str] = []
    for content_item in content:
        if isinstance(content_item, str):
            text.append(content_item)
            continue
        if content_item["type"] == "text":
            text.append(content_item["text"])
    return "".join(text)


def langchain_to_chat_message(message: BaseMessage) -> ChatMessage:
    """将 LangChain 消息转换为 ChatMessage。"""
    match message:
        case HumanMessage():
            human_message = ChatMessage(
                type="human",
                content=convert_message_content_to_string(message.content),
            )
            return human_message
        case AIMessage():
            ai_message = ChatMessage(
                type="ai",
                content=convert_message_content_to_string(message.content),
            )
            if message.tool_calls:
                ai_message.tool_calls = message.tool_calls
            if message.response_metadata:
                ai_message.response_metadata = message.response_metadata
            return ai_message
        case ToolMessage():
            tool_message = ChatMessage(
                type="tool",
                content=convert_message_content_to_string(message.content),
                tool_call_id=message.tool_call_id,
            )
            return tool_message
        case LangchainChatMessage():
            if message.role == "custom":
                custom_message = ChatMessage(
                    type="custom",
                    content="",
                    custom_data=cast(dict[str, Any], message.content[0]),
                )
                return custom_message
            else:
                raise ValueError(f"Unsupported chat message role: {message.role}")
        case _:
            raise ValueError(f"Unsupported message type: {message.__class__.__name__}")


def messages_from_checkpoint(checkpoint: Mapping[str, Any]) -> list[BaseMessage]:
    """从原始检查点提取会话历史。

    图状态式 Agent 将对话保存在 messages 通道中；函数式 API
    （@entrypoint）Agent 则保存在 __previous__ 中。aget_state 不会暴露
    后者，只会返回入口函数的最终值。
    """
    channel_values = checkpoint.get("channel_values") or {}
    messages = channel_values.get("messages")
    if not messages:
        previous = channel_values.get("__previous__")
        if isinstance(previous, Mapping):
            messages = previous.get("messages")
        elif isinstance(previous, list):
            messages = previous
    return [m for m in (messages or []) if isinstance(m, BaseMessage)]


def remove_tool_calls(content: str | list[str | dict]) -> str | list[str | dict]:
    """从内容中移除工具调用。"""
    if isinstance(content, str):
        return content
    # 目前仅 Anthropic 模型通过 tool_use 类型的内容项流式输出工具调用。
    return [
        content_item
        for content_item in content
        if isinstance(content_item, str) or content_item["type"] != "tool_use"
    ]
