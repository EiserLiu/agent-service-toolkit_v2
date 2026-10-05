"""LazyLoadingAgent 基类测试。"""

from unittest.mock import Mock

import pytest

from agents.lazy_agent import LazyLoadingAgent


class LazyLoadingAgentFixture(LazyLoadingAgent):
    """用于测试的 LazyLoadingAgent 实现。"""

    def __init__(self):
        super().__init__()

    async def load(self) -> None:
        """测试加载实现。"""
        self._loaded = True

    def _create_graph(self):
        """测试图的创建。"""
        mock_graph = Mock()
        mock_graph.name = "test-graph"
        return mock_graph


class TestLazyLoadingAgentBase:
    """测试 LazyLoadingAgent 基类的功能。"""

    def test_initialization(self):
        """测试 Agent 是否正确初始化。"""
        agent = LazyLoadingAgentFixture()
        assert not agent._loaded
        assert agent._graph is None

    @pytest.mark.asyncio
    async def test_load(self):
        """测试 load 是否正常工作。"""
        agent = LazyLoadingAgentFixture()
        await agent.load()
        assert agent._loaded

    def test_get_graph_before_load(self):
        """测试加载前调用 get_graph 是否抛出错误。"""
        agent = LazyLoadingAgentFixture()
        with pytest.raises(RuntimeError, match="Agent not loaded"):
            agent.get_graph()

    def test_get_graph_after_load(self):
        """测试加载后 get_graph 是否正常工作。"""
        agent = LazyLoadingAgentFixture()
        agent._loaded = True
        agent._graph = Mock()

        graph = agent.get_graph()
        assert graph is not None

    def test_get_graph_no_graph_created(self):
        """测试未创建图时 get_graph 是否抛出错误。"""
        agent = LazyLoadingAgentFixture()
        agent._loaded = True
        agent._graph = None

        with pytest.raises(RuntimeError, match="Agent graph not created"):
            agent.get_graph()
