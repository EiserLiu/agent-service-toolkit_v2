"""基于 OpenAI 的文本转语音实现。"""

import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


class OpenAITTS:
    """OpenAI 文本转语音服务商。"""

    # API 限制
    MAX_TEXT_LENGTH = 4096
    MIN_TEXT_LENGTH = 3

    # 可用配置选项
    VALID_VOICES = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    VALID_MODELS = ["tts-1", "tts-1-hd"]

    def __init__(self, api_key: str | None = None, voice: str = "alloy", model: str = "tts-1"):
        """初始化 OpenAI TTS。

        参数：
            api_key：OpenAI API 密钥；未提供时使用环境变量。
            voice：音色名称（alloy、echo、fable、onyx、nova、shimmer）。
            model：模型名称（tts-1 或 tts-1-hd）。

        异常：
            ValueError：音色或模型无效时抛出。
            Exception：OpenAI 客户端初始化失败时抛出。
        """
        # 验证 voice 参数
        if voice not in self.VALID_VOICES:
            raise ValueError(f"Invalid voice '{voice}'. Must be one of {self.VALID_VOICES}")

        # 验证 model 参数
        if model not in self.VALID_MODELS:
            raise ValueError(f"Invalid model '{model}'. Must be one of {self.VALID_MODELS}")

        # 使用传入的密钥或环境变量创建 OpenAI 客户端
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.voice = voice
        self.model = model

        logger.info(f"OpenAI TTS initialized: voice={voice}, model={model}")

    def _validate_and_prepare_text(self, text: str) -> str | None:
        """验证并准备 TTS 生成所需的文本。

        参数：
            text：原始输入文本。

        返回值：
            可用于 TTS 的文本；文本过短时返回 None。

        注意：
            - 移除首尾空白字符。
            - 低于最小长度时返回 None。
            - 超过最大长度时截断。
        """
        # 移除首尾空白字符
        text = text.strip()

        # 跳过过短的文本，避免不必要的 API 调用
        if len(text) < self.MIN_TEXT_LENGTH:
            logger.debug(f"OpenAI TTS: skipping short text ({len(text)} chars)")
            return None

        # 必要时按 API 长度上限截断
        if len(text) > self.MAX_TEXT_LENGTH:
            logger.warning(
                f"OpenAI TTS: truncating from {len(text)} to {self.MAX_TEXT_LENGTH} chars"
            )
            text = text[: self.MAX_TEXT_LENGTH]

        return text

    def generate(self, text: str) -> bytes | None:
        """将文本转换为语音。

        参数：
            text：要转换为语音的文本。

        返回值：
            MP3 音频字节；文本过短或生成失败时返回 None。

        注意：
            - 少于 3 个字符时返回 None。
            - 超过 4096 个字符时截断。
            - 错误只记录、不向上抛出，改为返回 None。
        """
        # 验证并准备文本
        prepared_text = self._validate_and_prepare_text(text)
        if not prepared_text:
            return None

        try:
            # 调用 OpenAI TTS API
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=prepared_text,
                response_format="mp3",
            )

            # 从响应中提取音频字节
            audio_bytes = response.content
            logger.info(f"OpenAI TTS: generated {len(audio_bytes)} bytes")
            return audio_bytes

        except Exception as e:
            # 记录包含完整堆栈的错误，便于调试
            logger.error(f"OpenAI TTS failed: {e}", exc_info=True)
            # 返回 None，让应用能够平稳降级
            return None

    def get_format(self) -> str:
        """获取音频格式（MIME 类型）。

        返回值：
            生成音频的 MIME 类型字符串。
        """
        return "audio/mp3"
