"""语音转文本工厂。

本模块提供工厂类，根据配置加载对应的 STT 服务商。
"""

import logging
import os
from typing import BinaryIO, Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "deepgram"]


class SpeechToText:
    """语音转文本工厂。

    加载具体 STT 服务商实现，并将调用委托给它。

    示例：
        >>> stt = SpeechToText(provider="openai")
        >>> text = stt.transcribe(audio_file)
        >>>
        >>> # 也可以从环境变量创建
        >>> stt = SpeechToText.from_env()
        >>> if stt:
        ...     text = stt.transcribe(audio_file)
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """使用指定服务商初始化 STT。

        参数：
            provider：服务商名称（如 "openai"、"deepgram"）。
            api_key：API 密钥；未提供时使用环境变量。
            **config：服务商专用配置。

        异常：
            ValueError：服务商未知时抛出。
        """
        self._provider_name = provider

        # 从参数或环境变量获取 API 密钥
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置服务商
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"SpeechToText created with provider={provider}")

    def _get_api_key(self, provider: Provider, api_key: str | None) -> str | None:
        """从参数或环境变量获取 API 密钥。

        参数：
            provider：服务商名称。
            api_key：参数中传入的 API 密钥，优先使用。

        返回值：
            解析得到的 API 密钥，或 None。
        """
        # 优先使用显式传入的 API 密钥
        if api_key:
            return api_key

        # 否则根据服务商从环境变量获取
        match provider:
            case "openai":
                return os.getenv("OPENAI_API_KEY")
            case "deepgram":
                return os.getenv("DEEPGRAM_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载对应的 STT 服务商实现。

        参数：
            provider：服务商名称。
            api_key：解析得到的 API 密钥。
            config：服务商专用配置。

        返回值：
            服务商实例。

        异常：
            ValueError：服务商未知时抛出。
            NotImplementedError：服务商尚未实现时抛出。
        """
        match provider:
            case "openai":
                from voice.providers.openai_stt import OpenAISTT

                return OpenAISTT(api_key=api_key, **config)

            case "deepgram":
                # 后续扩展示例：要支持 Deepgram，请实现 DeepgramSTT 并取消下方代码的注释：
                # from voice.providers.deepgram_stt import DeepgramSTT
                # return DeepgramSTT(api_key=api_key, **config)
                raise NotImplementedError("Deepgram STT provider not yet implemented")

            case _:
                # 统一处理未知服务商
                raise ValueError(f"Unknown STT provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取服务商名称。

        返回值：
            服务商名称字符串。
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "SpeechToText | None":
        """从环境变量创建 STT。

        读取 VOICE_STT_PROVIDER 以确定服务商；未配置时返回 None。

        返回值：
            SpeechToText 实例，或 None。

        示例：
            >>> # 在 .env 中设置 VOICE_STT_PROVIDER=openai
            >>> stt = SpeechToText.from_env()
            >>> if stt:
            ...     text = stt.transcribe(audio_file)
        """
        provider = os.getenv("VOICE_STT_PROVIDER")

        # 未设置服务商时禁用语音功能
        if not provider:
            logger.debug("VOICE_STT_PROVIDER not set, STT disabled")
            return None

        try:
            # 使用环境变量指定的服务商创建实例
            # 验证服务商；无效时抛出 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不使应用崩溃，允许应用在没有语音功能时继续运行
            logger.error(f"Failed to create STT provider: {e}", exc_info=True)
            return None

    def transcribe(self, audio_file: BinaryIO) -> str:
        """将音频转写为文本。

        将调用委托给底层服务商实现。

        参数：
            audio_file：二进制音频文件。

        返回值：
            转写文本；失败时返回空字符串。
        """
        return self._provider.transcribe(audio_file)
