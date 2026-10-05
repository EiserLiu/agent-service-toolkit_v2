"""VoiceManager 核心逻辑测试，不包含 Streamlit 界面测试。"""

from unittest.mock import Mock, patch

from voice.manager import VoiceManager


def test_init_with_both_stt_and_tts():
    """测试创建同时包含 STT 和 TTS 的 VoiceManager。"""
    mock_stt = Mock()
    mock_tts = Mock()
    manager = VoiceManager(stt=mock_stt, tts=mock_tts)
    # STT 和 TTS 均正确赋值则通过
    assert manager.stt == mock_stt
    assert manager.tts == mock_tts


def test_init_with_only_tts():
    """测试仅使用 TTS 创建 VoiceManager（STT=None）。"""
    mock_tts = Mock()
    manager = VoiceManager(stt=None, tts=mock_tts)
    # 部分语音功能（仅 TTS）正常工作则通过
    assert manager.stt is None
    assert manager.tts == mock_tts


def test_from_env_both_configured():
    """测试同时配置 STT 和 TTS 时的 from_env 行为。"""
    mock_stt = Mock()
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=mock_stt):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # 创建的 VoiceManager 同时包含 STT 和 TTS 则通过
            assert manager is not None
            assert manager.stt == mock_stt
            assert manager.tts == mock_tts


def test_from_env_only_tts_configured():
    """测试仅配置 TTS 时的 from_env 行为。"""
    mock_tts = Mock()
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=mock_tts):
            manager = VoiceManager.from_env()
            # 创建的 VoiceManager 仅包含 TTS 则通过（允许 STT=None）
            assert manager is not None
            assert manager.stt is None
            assert manager.tts == mock_tts


def test_from_env_neither_configured():
    """测试 STT 和 TTS 均未配置时的 from_env 行为。"""
    with patch("voice.manager.SpeechToText.from_env", return_value=None):
        with patch("voice.manager.TextToSpeech.from_env", return_value=None):
            manager = VoiceManager.from_env()
            # 未配置语音功能时返回 None 则通过
            assert manager is None


def test_transcribe_audio_stt_not_configured():
    """测试未配置 STT 时，_transcribe_audio 是否返回 None。"""
    manager = VoiceManager(stt=None, tts=Mock())
    mock_audio = Mock()
    # 模拟 Streamlit 的 st.error，避免真实界面调用
    with patch("voice.manager.st.error"):
        result = manager._transcribe_audio(mock_audio)
        # 返回 None 则通过（未配置 STT 时的防御性检查）
        assert result is None
