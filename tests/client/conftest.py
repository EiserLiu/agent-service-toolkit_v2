import pytest

from client import AgentClient


@pytest.fixture
def agent_client(mock_env):
    """在干净环境中创建测试客户端的测试夹具。"""
    ac = AgentClient(base_url="http://test", get_info=False)
    ac.update_agent("test-agent", verify=False)
    return ac
