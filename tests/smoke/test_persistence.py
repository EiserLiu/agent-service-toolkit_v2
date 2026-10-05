import os

import pytest

from client import AgentClient

# 通过环境变量与 scripts/smoke_test.sh 共享，以便脚本验证
# 此会话的检查点确实写入了预期后端。单独运行此测试时，
# 回退到固定 ID。
THREAD_ID = os.environ.get("SMOKE_THREAD_ID", "smoke-test-persistence-thread")


@pytest.mark.docker
def test_checkpointer_persists_history():
    """确认配置的检查点保存器能跨轮次持久化对话状态。

    与后端无关，测试服务启动时配置的 DATABASE_TYPE。scripts/smoke_test.sh
    会分别使用 postgres 和 mongo 运行，再独立验证数据确实写入该后端。
    本测试本身无法区分后端，因为任何正常工作的检查点保存器都可能通过。
    要求服务已设置 USE_FAKE_MODEL=true 并连接真实数据库。

    invoke 和 get_history 均使用默认 Agent。/history 会区分 Agent，
    AgentClient 也会限定为客户端选中的 Agent，因此读取会话的图与创建会话
    的图相同。这保证了每个 Agent 拥有独立图和状态时，数据仍能正确保存和读取。
    """
    client = AgentClient("http://localhost:8080")

    client.invoke("Tell me a joke?", thread_id=THREAD_ID, model="fake")
    client.invoke("Tell me another?", thread_id=THREAD_ID, model="fake")

    history = client.get_history(thread_id=THREAD_ID)
    human_messages = [m for m in history.messages if m.type == "human"]
    assert len(human_messages) == 2
    assert human_messages[0].content == "Tell me a joke?"
    assert human_messages[1].content == "Tell me another?"


@pytest.mark.docker
def test_threads_lists_user_threads():
    """确认 /threads 通过配置的检查点保存器枚举会话。

    单元测试使用模拟保存器，SQLite 集成测试仅验证 SQLite 驱动；
    本测试确认元数据过滤（包括 step -1 初始检查点查询）在 PostgreSQL
    和 MongoDB 上具有相同行为。
    要求服务已设置 USE_FAKE_MODEL=true 并连接真实数据库。
    """
    client = AgentClient("http://localhost:8080")
    user_id = f"{THREAD_ID}-user"
    other_user_id = f"{THREAD_ID}-other"

    single_turn = f"{THREAD_ID}-single"
    multi_turn = f"{THREAD_ID}-multi"
    client.invoke("Only turn", thread_id=single_turn, user_id=user_id, model="fake")
    client.invoke("First turn", thread_id=multi_turn, user_id=user_id, model="fake")
    client.invoke("Second turn", thread_id=multi_turn, user_id=user_id, model="fake")
    client.invoke("Not mine", thread_id=f"{THREAD_ID}-other", user_id=other_user_id, model="fake")

    threads = client.get_user_threads(user_id=user_id).threads
    # 按最近更新优先排序；单轮会话即使未推进到初始检查点之后，
    # 也必须出现在列表中。
    assert [t.thread_id for t in threads] == [multi_turn, single_turn]
    assert [t.title for t in threads] == ["First turn", "Only turn"]
    assert all(t.updated_at for t in threads)

    other_threads = client.get_user_threads(user_id=other_user_id).threads
    assert [t.thread_id for t in other_threads] == [f"{THREAD_ID}-other"]
