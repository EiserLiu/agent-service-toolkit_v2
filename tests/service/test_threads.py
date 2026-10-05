"""使用模拟检查点保存器测试 /threads。

仅覆盖难以让真实检查点保存器按需产生的情况：行数和页数上限（否则需要
创建过多检查点）、返回本应被过滤的记录、缺少时间戳和查询失败。
真实数据库能够验证的过滤、排序、标题、数量限制和子图会话行为，
则在 test_threads_sqlite.py 中使用 SQLite 测试。
"""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage


class FakeCheckpointTuple:
    def __init__(self, thread_id: str, checkpoint_id: str, checkpoint: dict, metadata: dict):
        self.config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        self.checkpoint = checkpoint
        self.metadata = metadata


class FakeCheckpointer:
    """提供 /threads 所依赖语义的检查点保存器。

    检查点按 checkpoint_id 全局排序；alist 对元数据执行精确匹配过滤；
    每个会话的首个检查点写在 step -1，与 LangGraph 写入输入检查点的方式一致。
    """

    def __init__(self):
        self.rows: list[FakeCheckpointTuple] = []
        self.alist_filters: list[dict | None] = []
        self._next_id = 0

    def _checkpoint_id(self) -> str:
        self._next_id += 1
        return f"cp-{self._next_id:06d}"

    def add_thread(
        self,
        thread_id: str,
        user_id: str = "user-123",
        agent_id: str = "research-assistant",
        turns: int = 1,
        title: str = "Hello",
        subgraph_heads: int = 0,
        tip_ts: str | None = "2024-07-31T20:14:19.804150+00:00",
    ) -> None:
        def add(step: int, channel_values: dict, ts: str | None) -> None:
            self.rows.append(
                FakeCheckpointTuple(
                    thread_id,
                    self._checkpoint_id(),
                    {"ts": ts, "channel_values": channel_values},
                    {"step": step, "user_id": user_id, "agent_id": agent_id},
                )
            )

        add(-1, {"__start__": {"messages": [HumanMessage(content=title)]}}, "2024-01-01T00:00:00Z")
        # 子图运行会写入自己的初始检查点，并继承父级运行的元数据。
        for _ in range(subgraph_heads):
            add(-1, {"__start__": {}}, "2024-01-01T00:00:00Z")

        messages: list = []
        for turn in range(turns):
            messages = messages + [
                HumanMessage(content=title if turn == 0 else f"{title} {turn}"),
                AIMessage(content="reply"),
            ]
            add(
                turn * 2,
                {"messages": messages},
                tip_ts if turn == turns - 1 else "2024-01-01T00:00:01Z",
            )

    async def alist(self, config, *, filter=None, before=None, limit=None):
        self.alist_filters.append(filter)
        rows = sorted(self.rows, key=lambda r: r.config["configurable"]["checkpoint_id"])
        rows.reverse()
        yielded = 0
        for row in rows:
            if filter and any(row.metadata.get(key) != value for key, value in filter.items()):
                continue
            if (
                before
                and row.config["configurable"]["checkpoint_id"]
                >= (before["configurable"]["checkpoint_id"])
            ):
                continue
            yield row
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    async def aget_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        rows = [r for r in self.rows if r.config["configurable"]["thread_id"] == thread_id]
        if not rows:
            return None
        return max(rows, key=lambda r: r.config["configurable"]["checkpoint_id"])


def test_threads_without_checkpointer_returns_empty(test_client, mock_agent) -> None:
    """测试 Agent 未配置检查点保存器时，/threads 返回空列表。"""
    mock_agent.checkpointer = None

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert response.json() == {"threads": []}


def test_threads_filters_by_agent_id(test_client) -> None:
    """测试 /{agent_id}/threads 是否将检查点查询限定在请求指定的 Agent。

    过滤条件中的 agent_id 是跨 Agent 隔离的保证，因此要断言查询本身，
    而不仅仅检查返回的会话。
    """
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-mine", agent_id="custom-agent", title="Mine")
    checkpointer.add_thread("thread-theirs", agent_id="other-agent", title="Theirs")

    custom_agent = AsyncMock()
    custom_agent.checkpointer = checkpointer
    default_agent = AsyncMock()
    default_agent.checkpointer = None

    agent_calls = {"default": 0, "custom": 0}

    def agent_lookup(agent_id):
        if agent_id == "custom-agent":
            agent_calls["custom"] += 1
            return custom_agent
        agent_calls["default"] += 1
        return default_agent

    with patch("service.service.get_agent", side_effect=agent_lookup):
        response = test_client.get("/custom-agent/threads", params={"user_id": "user-123"})

    assert response.status_code == 200
    payload = response.json()
    assert [thread["thread_id"] for thread in payload["threads"]] == ["thread-mine"]
    assert checkpointer.alist_filters == [
        {"user_id": "user-123", "agent_id": "custom-agent", "step": -1}
    ]
    assert agent_calls == {"custom": 1, "default": 0}


@pytest.mark.parametrize("limit", [0, -1, 101, 999999])
def test_threads_rejects_out_of_range_limit(test_client, mock_agent, limit: int) -> None:
    """测试 /threads 是否限制 limit，防止客户端请求无边界扫描。"""
    mock_agent.checkpointer = FakeCheckpointer()

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": limit})

    assert response.status_code == 422


def test_threads_skips_checkpoints_with_mismatched_metadata(test_client, mock_agent) -> None:
    """测试检查点保存器忽略过滤条件时，仍不会泄露其他用户的会话。"""

    class LeakyCheckpointer(FakeCheckpointer):
        async def alist(self, config, *, filter=None, before=None, limit=None):
            async for row in super().alist(config, before=before, limit=limit):
                yield row

    checkpointer = LeakyCheckpointer()
    checkpointer.add_thread("thread-mine", title="Mine")
    checkpointer.add_thread("thread-other-user", user_id="other-user", title="Other user")
    checkpointer.add_thread("thread-other-agent", agent_id="other-agent", title="Other agent")
    checkpointer.add_thread("thread-no-metadata", user_id=None, agent_id=None, title="No metadata")
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    assert [t["thread_id"] for t in response.json()["threads"]] == ["thread-mine"]


def test_threads_pages_past_subgraph_heads(test_client, mock_agent) -> None:
    """测试子图的初始检查点记录不会挤占真正会话的结果位置。

    包含子图的 Agent 会为每个会话写入多条初始检查点，
    因此一页记录所覆盖的会话可能远少于调用者要求的数量。
    """
    checkpointer = FakeCheckpointer()
    for index in range(30):
        checkpointer.add_thread(f"thread-{index:02d}", title=f"Thread {index}", subgraph_heads=9)
    mock_agent.checkpointer = checkpointer

    with patch("service.threads.HEAD_PAGE_SIZE", 20):
        response = test_client.get("/threads", params={"user_id": "user-123", "limit": 30})

    assert response.status_code == 200
    threads = response.json()["threads"]
    assert len(threads) == 30
    assert threads[0]["thread_id"] == "thread-29"


def test_threads_bounds_total_rows_scanned(test_client, mock_agent) -> None:
    """测试初始检查点分页在达到行数上限时停止，而非扫描整张表。"""
    checkpointer = FakeCheckpointer()
    for index in range(60):
        checkpointer.add_thread(f"thread-{index:02d}", title=f"Thread {index}", subgraph_heads=9)
    mock_agent.checkpointer = checkpointer

    with (
        patch("service.threads.HEAD_PAGE_SIZE", 20),
        patch("service.threads.MAX_HEAD_ROWS", 100),
    ):
        response = test_client.get("/threads", params={"user_id": "user-123", "limit": 30})

    assert response.status_code == 200
    assert len(checkpointer.alist_filters) == 5
    assert len(response.json()["threads"]) == 10


def test_threads_tolerates_missing_timestamp(test_client, mock_agent) -> None:
    """测试最新检查点没有时间戳的会话仍能由 /threads 返回。"""
    checkpointer = FakeCheckpointer()
    checkpointer.add_thread("thread-no-ts", title="No timestamp", tip_ts=None)
    mock_agent.checkpointer = checkpointer

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})

    assert response.status_code == 200
    payload = response.json()
    assert payload["threads"][0]["thread_id"] == "thread-no-ts"
    assert payload["threads"][0]["updated_at"] is None


def test_threads_checkpointer_error_returns_500(test_client, mock_agent) -> None:
    async def broken_alist(*args, **kwargs):
        raise RuntimeError("db unavailable")
        yield  # pragma: no cover

    mock_agent.checkpointer = type("Checkpointer", (), {})()
    mock_agent.checkpointer.alist = broken_alist

    response = test_client.get("/threads", params={"user_id": "user-123", "limit": 10})
    assert response.status_code == 500
