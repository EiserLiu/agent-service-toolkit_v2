from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from service import app


@pytest.fixture
def test_client():
    """创建 FastAPI 测试客户端的测试夹具。"""
    return TestClient(app)


@pytest.fixture
def mock_agent():
    """创建可针对不同场景配置的模拟 Agent 的测试夹具。"""
    agent_mock = AsyncMock()
    agent_mock.ainvoke = AsyncMock(
        return_value=[("values", {"messages": [AIMessage(content="Test response")]})]
    )
    agent_mock.aget_state = AsyncMock(
        return_value=StateSnapshot(
            values={},
            next=(),
            config={},
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
    )
    # 需要读取检查点的测试会显式设置该属性；否则 AsyncMock 会自动
    # 创建一个看起来像可用检查点保存器的属性。
    agent_mock.checkpointer = None
    with patch("service.service.get_agent", Mock(return_value=agent_mock)):
        yield agent_mock


@pytest.fixture
def mock_settings(mock_env):
    """确保每个测试的配置互不污染的测试夹具。"""
    with patch("service.service.settings") as mock_settings:
        yield mock_settings


@pytest.fixture
def mock_httpx():
    """替换 httpx.stream 和 httpx.get，使其使用测试客户端。"""

    with TestClient(app) as client:

        def mock_stream(method: str, url: str, **kwargs):
            # 去掉基础地址，因为 TestClient 只接收路径
            path = url.replace("http://0.0.0.0", "")
            return client.stream(method, path, **kwargs)

        def mock_get(url: str, **kwargs):
            # 去掉基础地址，因为 TestClient 只接收路径
            path = url.replace("http://0.0.0.0", "")
            return client.get(path, **kwargs)

        with patch("httpx.stream", mock_stream):
            with patch("httpx.get", mock_get):
                yield
