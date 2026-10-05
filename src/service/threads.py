"""为 /threads 接口枚举会话。

会话来源于检查点保存器，而非独立的数据表，因此需要按 LangGraph 的写入
方式读取检查点元数据。实现依赖的固定约定记录在下方常量的注释中。
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from schema import ThreadSummary
from service.utils import convert_message_content_to_string, messages_from_checkpoint

logger = logging.getLogger(__name__)

# LangGraph 在每个会话的 step -1 写入一次输入检查点，后续轮次
# 从最后一步继续。不要替换为其他步骤：单轮会话永远不会到达 step 1。
THREAD_HEAD_STEP = -1

# 初始检查点按会话创建顺序排列，因此多取一些，再按最新检查点重新排序，
# 以近似实现“最近更新优先”。早于本次最早初始检查点创建的会话不会被包含。
MAX_THREAD_HEADS = 200
HEAD_PAGE_SIZE = 200

# 每条初始检查点记录不一定对应独立会话：包含子图的 Agent 每次调用子图
# 都会写入一条继承父级元数据的记录。此值限制用于补偿重复记录的分页次数。
MAX_HEAD_ROWS = 1000

TITLE_MAX_LENGTH = 60


async def _list_thread_heads(checkpointer: Any, user_id: str, agent_id: str) -> list[Any]:
    """每个会话返回一个初始检查点，较新创建的会话排在前面。"""
    heads: list[Any] = []
    seen: set[str] = set()
    rows_scanned = 0
    before = None
    while len(seen) < MAX_THREAD_HEADS and rows_scanned < MAX_HEAD_ROWS:
        page = [
            c
            async for c in checkpointer.alist(
                None,
                filter={"user_id": user_id, "agent_id": agent_id, "step": THREAD_HEAD_STEP},
                before=before,
                limit=HEAD_PAGE_SIZE,
            )
        ]
        if not page:
            break
        rows_scanned += len(page)
        short_page = len(page) < HEAD_PAGE_SIZE
        for row in page:
            thread_id = row.config["configurable"]["thread_id"]
            if thread_id in seen:
                continue
            seen.add(thread_id)
            heads.append(row)
        if short_page:
            break
        before = RunnableConfig(
            configurable={"checkpoint_id": page[-1].config["configurable"]["checkpoint_id"]}
        )
    return heads


async def list_user_threads(
    checkpointer: Any, user_id: str, agent_id: str, limit: int
) -> list[ThreadSummary]:
    """列出用户在指定 Agent 下的会话，按最近更新时间倒序排列。"""
    summaries: list[tuple[str, ThreadSummary]] = []
    for head in await _list_thread_heads(checkpointer, user_id, agent_id):
        thread_id = head.config["configurable"]["thread_id"]
        stored_user_id = head.metadata.get("user_id")
        stored_agent_id = head.metadata.get("agent_id")
        if stored_user_id != user_id or stored_agent_id != agent_id:
            logger.warning(
                f"Checkpointer returned thread {thread_id} with user_id "
                f"{stored_user_id!r}/agent_id {stored_agent_id!r}, expected "
                f"{user_id!r}/{agent_id!r} — skipping to avoid a "
                "cross-user or cross-agent leak."
            )
            continue

        # 初始检查点还没有消息，因此标题和 updated_at 从最新检查点获取。
        tip = await checkpointer.aget_tuple(RunnableConfig(configurable={"thread_id": thread_id}))
        if tip is None:
            continue
        messages = messages_from_checkpoint(tip.checkpoint)
        first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
        summaries.append(
            (
                tip.config["configurable"]["checkpoint_id"],
                ThreadSummary(
                    thread_id=thread_id,
                    agent_id=agent_id,
                    updated_at=tip.checkpoint.get("ts"),
                    title=convert_message_content_to_string(first_human.content)[:TITLE_MAX_LENGTH]
                    if first_human
                    else None,
                ),
            )
        )

    # 检查点 ID 是按时间排序的 UUID，因此最新检查点的 ID 可用于按最后更新时间排序。
    summaries.sort(key=lambda item: item[0], reverse=True)
    return [summary for _, summary in summaries[:limit]]
