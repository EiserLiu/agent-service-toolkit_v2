"""语音输入和输出模块。

提供语音转文本和文本转语音功能，支持多个服务商。

模块：
    - SpeechToText：语音转文本处理器（可独立使用）
    - TextToSpeech：文本转语音处理器（可独立使用）
    - VoiceManager：面向 Streamlit 的便捷封装

快速入门：
    >>> from voice import VoiceManager
    >>>
    >>> # 简便方式：从环境变量创建
    >>> voice = VoiceManager.from_env()
    >>>
    >>> # 在 Streamlit 中使用
    >>> if voice:
    ...     user_input = voice.get_chat_input()
    ...     # 此处处理输入
    ...     with st.chat_message("ai"):
    ...         voice.render_message(response)

进阶用法：
    >>> from voice import SpeechToText, TextToSpeech, VoiceManager
    >>>
    >>> # 可混用服务商，例如 OpenAI STT 与自定义 TTS
    >>> stt = SpeechToText(provider="openai")
    >>> tts = TextToSpeech(provider="openai", voice="nova")
    >>> voice = VoiceManager(stt=stt, tts=tts)
"""

from voice.manager import VoiceManager
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

__all__ = ["VoiceManager", "SpeechToText", "TextToSpeech"]
