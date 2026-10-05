"""支持异步初始化和动态图创建的 Agent 类型。"""

from abc import ABC, abstractmethod

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel import Pregel


class LazyLoadingAgent(ABC):
    """需要异步加载的 Agent 的基类。"""

    def __init__(self) -> None:
        """初始化 Agent。"""
        self._loaded = False
        self._graph: CompiledStateGraph | Pregel | None = None

    @abstractmethod
    async def load(self) -> None:
        """异步加载此 Agent。

        服务启动时调用此方法，负责：
        - 建立外部连接（MCP 客户端、数据库等）
        - 加载工具或资源
        - 执行其他必需的异步初始化
        - 创建 Agent 图
        """
        raise NotImplementedError  # pragma: no cover

    def get_graph(self) -> CompiledStateGraph | Pregel:
        """获取 Agent 图。

        返回 load() 期间创建的图实例。

        返回值：
            Agent 图（CompiledStateGraph 或 Pregel）。
        """
        if not self._loaded:
            raise RuntimeError("Agent not loaded. Call load() first.")
        if self._graph is None:
            raise RuntimeError("Agent graph not created during load().")
        return self._graph
