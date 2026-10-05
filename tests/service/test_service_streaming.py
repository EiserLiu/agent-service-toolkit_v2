import pytest
from langchain_core.messages import AIMessage
from pydantic_core import ValidationError

from service.service import _create_ai_message


@pytest.mark.parametrize(
    "parts, expected",
    [
        # 1）基本内容和 tool_calls
        (
            {"content": "Hello", "tool_calls": []},
            {"content": "Hello", "tool_calls": []},
        ),
        # 2）忽略未知键
        (
            {"content": "Test", "foobar": 123, "tool_calls": []},
            {"content": "Test", "tool_calls": []},
        ),
        # 3）透传 AIMessage 的其他有效参数（id、type）
        (
            {
                "content": "Hey",
                "id": "abc-123",
                "type": "ai",
                "tool_calls": [],
            },
            {"content": "Hey", "id": "abc-123", "type": "ai", "tool_calls": []},
        ),
    ],
)
def test_create_ai_message_filters_and_passes_through(parts, expected):
    """_create_ai_message 应满足：
    - 丢弃未知键（"foobar"）。
    - 保留符合 AIMessage 参数签名的键。
    - 对 parts 字典中的重复键使用最终值。
    """
    msg: AIMessage = _create_ai_message(parts)
    for key, val in expected.items():
        assert getattr(msg, key) == val


def test_create_ai_message_missing_required_content_raises():
    """AIMessage 必须包含 content；缺少时，_create_ai_message 应从构造函数
    抛出 Pydantic ValidationError。

    LangChain v1 迁移说明：
    - 以前缺少必需参数时抛出 TypeError。
    - LangChain v1 使用 Pydantic v2，校验错误改为 pydantic_core.ValidationError。
    - 相比通用 TypeError，这能更明确地表明失败发生在校验阶段。
    """
    with pytest.raises(ValidationError):
        _create_ai_message({"tool_calls": []})


def test_create_ai_message_empty_dict_raises():
    """parts 完全为空时也不应成功构造 AIMessage，
    应抛出 Pydantic ValidationError。

    LangChain v1 迁移说明：
    - 异常类型从 TypeError 改为 pydantic_core.ValidationError。
    - 这反映了 langchain_core 改用 Pydantic v2 进行模型校验。
    """
    with pytest.raises(ValidationError):
        _create_ai_message({})
