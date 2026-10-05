"""文本转语音工厂。

本模块提供工厂类，根据配置加载对应的 TTS 服务商。
"""

import logging
import os
from typing import Literal, cast

logger = logging.getLogger(__name__)

Provider = Literal["openai", "elevenlabs"]


class TextToSpeech:
    """文本转语音工厂。

    加载具体 TTS 服务商实现，并将调用委托给它。

    示例：
        >>> tts = TextToSpeech(provider="openai", voice="nova")
        >>> audio = tts.generate("Hello world")
        >>>
        >>> # 也可以从环境变量创建
        >>> tts = TextToSpeech.from_env()
        >>> if tts:
        ...     audio = tts.generate("Hello world")
    """

    def __init__(self, provider: Provider = "openai", api_key: str | None = None, **config):
        """使用指定服务商初始化 TTS。

        参数：
            provider：服务商名称（如 "openai"、"elevenlabs"）。
            api_key：API 密钥；未提供时使用环境变量。
            **config：服务商专用配置。
                OpenAI：voice="alloy", model="tts-1"
                ElevenLabs：voice_id="...", model_id="..."

        异常：
            ValueError：服务商未知时抛出。
        """
        self._provider_name = provider

        # 从参数或环境变量获取 API 密钥
        resolved_api_key = self._get_api_key(provider, api_key)

        # 加载并配置服务商
        self._provider = self._load_provider(provider, resolved_api_key, config)

        logger.info(f"TextToSpeech created with provider={provider}")

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
            case "elevenlabs":
                return os.getenv("ELEVENLABS_API_KEY")
            case _:
                return None

    def _load_provider(self, provider: Provider, api_key: str | None, config: dict):
        """加载对应的 TTS 服务商实现。

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
                from voice.providers.openai_tts import OpenAITTS

                # 提取 OpenAI 专用配置，并提供默认值
                voice = config.get("voice", "alloy")
                model = config.get("model", "tts-1")

                return OpenAITTS(api_key=api_key, voice=voice, model=model)

            case "elevenlabs":
                # 后续扩展示例：要支持 ElevenLabs，请实现 ElevenLabsTTS 并取消下方代码的注释：
                # from voice.providers.elevenlabs_tts import ElevenLabsTTS
                # voice_id = config.get("voice_id")
                # model_id = config.get("model_id", "eleven_monolingual_v1")
                # return ElevenLabsTTS(api_key=api_key, voice_id=voice_id, model_id=model_id)
                raise NotImplementedError("ElevenLabs TTS provider not yet implemented")

            case _:
                # 统一处理未知服务商
                raise ValueError(f"Unknown TTS provider: {provider}. Available providers: openai")

    @property
    def provider(self) -> str:
        """获取服务商名称。

        返回值：
            服务商名称字符串。
        """
        return self._provider_name

    @classmethod
    def from_env(cls) -> "TextToSpeech | None":
        """从环境变量创建 TTS。

        读取 VOICE_TTS_PROVIDER 以确定服务商；未配置时返回 None。

        返回值：
            TextToSpeech 实例，或 None。

        示例：
            >>> # 在 .env 中设置 VOICE_TTS_PROVIDER=openai
            >>> tts = TextToSpeech.from_env()
            >>> if tts:
            ...     audio = tts.generate("Hello world")
        """
        provider = os.getenv("VOICE_TTS_PROVIDER")

        # 未设置服务商时禁用语音功能
        if not provider:
            logger.debug("VOICE_TTS_PROVIDER not set, TTS disabled")
            return None

        try:
            # 使用环境变量指定的服务商创建实例
            # 验证服务商；无效时抛出 ValueError
            return cls(provider=cast(Provider, provider))
        except Exception as e:
            # 记录错误但不使应用崩溃，允许应用在没有语音功能时继续运行
            logger.error(f"Failed to create TTS provider: {e}", exc_info=True)
            return None

    def generate(self, text: str) -> bytes | None:
        """将文本转换为语音。

        将调用委托给底层服务商实现。

        参数：
            text：要转换为语音的文本。

        返回值：
            音频字节（格式取决于服务商）；失败时返回 None。
        """
        return self._provider.generate(text)

    def get_format(self) -> str:
        """获取此服务商的音频格式（MIME 类型）。

        返回值：
            MIME 类型字符串，例如 "audio/mp3"。
        """
        return self._provider.get_format()
