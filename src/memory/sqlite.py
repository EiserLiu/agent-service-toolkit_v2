from contextlib import AbstractAsyncContextManager, asynccontextmanager

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.store.memory import InMemoryStore

from core.settings import settings


def get_sqlite_saver() -> AbstractAsyncContextManager[AsyncSqliteSaver]:
    """初始化并返回 SQLite 保存器实例。"""
    return AsyncSqliteSaver.from_conn_string(settings.SQLITE_DB_PATH)


class AsyncInMemoryStore:
    """为 InMemoryStore 提供异步上下文管理器接口的封装。"""

    def __init__(self):
        self.store = InMemoryStore()

    async def __aenter__(self):
        return self.store

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # InMemoryStore 无需清理
        pass

    async def setup(self):
        # 为兼容 PostgresStore 提供的空操作方法
        pass


@asynccontextmanager
async def get_sqlite_store():
    """初始化并返回长期记忆存储实例。

    注意：LangGraph 未提供 SQLite 专用的存储组件，
    因此使用异步上下文管理器封装 InMemoryStore，以保持接口兼容。
    """
    store_manager = AsyncInMemoryStore()
    yield await store_manager.__aenter__()
