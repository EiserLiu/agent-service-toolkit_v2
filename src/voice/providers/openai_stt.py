"""基于 OpenAI Whisper 的语音转文本实现。"""

import logging
from typing import BinaryIO

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAISTT:
    """OpenAI Whisper 语音转文本服务商。"""

    def __init__(self, api_key: str | None = None):
        """初始化 OpenAI STT。

        参数：
            api_key：OpenAI API 密钥；未提供时使用环境变量。

        异常：
            Exception：OpenAI 客户端初始化失败时抛出。
        """
        # 使用传入的密钥或环境变量创建 OpenAI 客户端
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        logger.info("OpenAI STT initialized")

    def transcribe(self, audio_file: BinaryIO) -> str:
        """使用 OpenAI Whisper 转写音频。

        参数：
            audio_file：二进制音频文件。

        返回值：
            转写文本；失败时返回空字符串。

        注意：
            错误会被记录，但不会向上抛出，而是返回空字符串，
            以便面向用户的应用平稳降级。
        """
        try:
            # 将文件指针重置到开头（文件可能已在其他位置被读取）
            audio_file.seek(0)

            # 调用 OpenAI Whisper API 转写音频
            result = self.client.audio.transcriptions.create(
                model="whisper-1", file=audio_file, response_format="text"
            )

            # 清除结果首尾的空白字符
            transcribed = result.strip()
            logger.info(f"OpenAI STT: transcribed {len(transcribed)} chars")
            return transcribed

        except Exception as e:
            # 记录包含完整堆栈的错误，便于调试
            logger.error(f"OpenAI STT failed: {e}", exc_info=True)
            # 返回空字符串，让应用能够平稳降级
            return ""
