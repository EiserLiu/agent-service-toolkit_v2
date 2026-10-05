import asyncio
import logging
import sys

import uvicorn
from dotenv import load_dotenv

from core import settings

load_dotenv()

if __name__ == "__main__":
    root_logger = logging.getLogger()
    if root_logger.handlers:
        print(
            f"Warning: Root logger already has {len(root_logger.handlers)} handler(s) configured. "
            f"basicConfig() will be ignored. Current level: {logging.getLevelName(root_logger.level)}"
        )

    logging.basicConfig(level=settings.LOG_LEVEL.to_logging_level())
    # 在 Windows 系统上设置兼容的事件循环策略。
    # Windows 默认的 ProactorEventLoop 可能与某些异步
    # 数据库驱动（如 PostgreSQL 的 psycopg）存在兼容问题。
    # WindowsSelectorEventLoopPolicy 可提供更好的兼容性，避免
    # 处理数据库连接时出现 "RuntimeError: Event loop is closed" 错误。
    # 必须在启动应用服务器之前设置。
    # 更多信息请参阅文档：
    # https://www.psycopg.org/psycopg3/docs/advanced/async.html#asynchronous-operations
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    uvicorn.run(
        "service:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.is_dev(),
        timeout_graceful_shutdown=settings.GRACEFUL_SHUTDOWN_TIMEOUT,
    )
