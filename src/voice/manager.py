"""VoiceManager：Streamlit 集成层。

本模块为语音功能提供 Streamlit 专用界面集成，
所有 Streamlit 依赖均集中在此处。
"""

import logging
from typing import Optional

import streamlit as st

from voice.stt import SpeechToText
from voice.tts import TextToSpeech

logger = logging.getLogger(__name__)


class VoiceManager:
    """面向 Streamlit 的语音功能便捷封装。

    提供 Streamlit 专用的语音输入和输出方法，处理加载提示、错误等界面反馈，
    并将实际语音处理委托给 STT 和 TTS 模块。

    示例：
        >>> voice = VoiceManager.from_env()
        >>>
        >>> if voice:
        ...     user_input = voice.get_chat_input()
        ...     if user_input:
        ...         with st.chat_message("ai"):
        ...             voice.render_message("Hello!")
    """

    def __init__(self, stt: SpeechToText | None = None, tts: TextToSpeech | None = None):
        """初始化 VoiceManager。

        参数：
            stt：SpeechToText 实例，设为 None 则禁用 STT。
            tts：TextToSpeech 实例，设为 None 则禁用 TTS。
        """
        self.stt = stt
        self.tts = tts

        logger.info(
            f"VoiceManager: STT={'enabled' if stt else 'disabled'}, "
            f"TTS={'enabled' if tts else 'disabled'}"
        )

    @classmethod
    def from_env(cls) -> Optional["VoiceManager"]:
        """从环境变量创建 VoiceManager。

        读取 VOICE_STT_PROVIDER 和 VOICE_TTS_PROVIDER，配置语音转文本
        和文本转语音服务商。

        返回值：
            配置了任一服务时返回 VoiceManager，否则返回 None。

        示例：
            >>> # 在 .env 中配置：
            >>> # VOICE_STT_PROVIDER=openai
            >>> # VOICE_TTS_PROVIDER=openai
            >>>
            >>> voice = VoiceManager.from_env()
            >>> # 返回已配置的 VoiceManager；禁用时返回 None
        """
        # 从环境变量创建 STT 和 TTS
        stt = SpeechToText.from_env()
        tts = TextToSpeech.from_env()

        # 两者都禁用时返回 None（不启用语音功能）
        if not stt and not tts:
            logger.debug("Voice features not configured")
            return None

        return cls(stt=stt, tts=tts)

    def _transcribe_audio(self, audio) -> str | None:
        """转写音频并提供界面反馈。

        转写期间显示加载提示，失败时显示错误消息。

        参数：
            audio：来自 Streamlit 聊天输入控件的音频文件对象。

        返回值：
            转写文本；转写失败时返回 None。
        """
        # 防御性检查（正确调用时不应出现这种情况）
        if not self.stt:
            st.error("⚠️ Speech-to-text not configured.")
            return None

        # 转写期间显示加载提示
        with st.spinner("🎤 Transcribing audio..."):
            transcribed = self.stt.transcribe(audio)

        # 检查转写是否成功
        if not transcribed:
            st.error("⚠️ Transcription failed. Please try again or type your message.")
            return None

        return transcribed

    def get_chat_input(self, placeholder: str = "Your message") -> str | None:
        """获取聊天输入，并按需执行语音转写。

        处理 Streamlit 音频输入控件及转写反馈（加载提示、错误等）。

        参数：
            placeholder：输入框的占位提示文字。

        返回值：
            用户消息；音频输入返回转写文本，文本输入直接返回文本，无输入时返回 None。
        """
        # 未启用 STT，使用普通文本输入
        if not self.stt:
            return st.chat_input(placeholder)

        # 已启用 STT，使用支持音频的输入控件
        chat_value = st.chat_input(placeholder, accept_audio=True)

        if not chat_value:
            return None

        # 处理字符串返回值（纯文本输入）
        if isinstance(chat_value, str):
            return chat_value

        # 处理对象或字典返回值（支持音频的输入）
        # 提取文本，兼容属性访问和字典访问
        text_content = None
        if hasattr(chat_value, "text"):
            text_content = chat_value.text
        elif isinstance(chat_value, dict):
            text_content = chat_value.get("text", "")

        # 提取音频，兼容属性访问和字典访问
        audio_content = None
        if hasattr(chat_value, "audio"):
            audio_content = chat_value.audio
        elif isinstance(chat_value, dict):
            audio_content = chat_value.get("audio")

        # 提供音频时，执行转写
        if audio_content:
            return self._transcribe_audio(audio_content)

        # 没有音频时，返回文本内容
        if text_content:
            return text_content

        # 未提供文本或音频
        return None

    def render_message(self, content: str, container=None, audio_only: bool = False) -> None:
        """渲染消息，并按需生成 TTS 音频。

        处理 Streamlit 的文本显示和音频播放器，将生成的音频保存到会话状态，
        使其在重新运行后仍然保留。

        参数：
            content：要显示的消息内容。
            container：Streamlit 容器，默认使用当前上下文。
            audio_only：为 True 时仅渲染音频，适用于文本已显示的情况。
        """
        if container is None:
            container = st

        # 显示文本，audio_only 模式除外（适用于文本已显示的流式场景）
        if not audio_only:
            container.write(content)

        # 启用 TTS 且内容非空时添加音频
        if self.tts and content.strip():
            # 生成音频时显示占位提示
            placeholder = container.empty()
            with placeholder:
                st.caption("🎙️ Generating audio...")

            # 生成 TTS 音频
            audio = self.tts.generate(content)

            # 将最后一条 AI 消息的音频保存到会话状态
            # 使其在 st.rerun() 后仍然保留
            if audio:
                st.session_state.last_audio = {"data": audio, "format": self.tts.get_format()}

            # 用音频播放器或错误消息替换占位提示
            if audio:
                placeholder.audio(audio, format=self.tts.get_format())
            else:
                placeholder.caption("🔇 Audio generation unavailable")
