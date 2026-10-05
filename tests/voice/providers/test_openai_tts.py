"""OpenAI TTS 服务商测试。"""

from unittest.mock import patch

import pytest

from voice.providers.openai_tts import OpenAITTS


def test_init_with_valid_params(mock_openai_client):
    """测试使用有效参数创建 OpenAITTS。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key", voice="nova", model="tts-1")
        # 客户端初始化未报错则通过
        assert tts.client == mock_openai_client


def test_init_with_invalid_voice():
    """测试无效音色是否触发 ValueError。"""
    # 无效音色触发 ValueError 则通过
    with pytest.raises(ValueError, match="Invalid voice"):
        OpenAITTS(api_key="test-key", voice="invalid", model="tts-1")


def test_init_with_invalid_model():
    """测试无效模型是否触发 ValueError。"""
    # 无效模型触发 ValueError 则通过
    with pytest.raises(ValueError, match="Invalid model"):
        OpenAITTS(api_key="test-key", voice="nova", model="invalid")


def test_validate_text_too_short(mock_openai_client):
    """测试文本短于 MIN_LENGTH 时是否返回 None。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts._validate_and_prepare_text("ab")  # 2 个字符，小于 MIN_LENGTH（3）
        # 文本过短时返回 None 则通过
        assert result is None


def test_validate_text_too_long(mock_openai_client):
    """测试文本长于 MAX_LENGTH 时是否截断。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        long_text = "a" * 5000  # 超过 MAX_LENGTH（4096）
        result = tts._validate_and_prepare_text(long_text)
        # 文本已截断到 MAX_LENGTH 则通过
        assert result is not None
        assert len(result) == 4096


def test_generate_success(mock_openai_client):
    """测试成功生成音频。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # 返回音频字节则通过
        assert result == b"fake audio data"


def test_generate_api_error(mock_openai_client):
    """测试是否能妥善处理 API 错误。"""
    # 让模拟对象抛出异常
    mock_openai_client.audio.speech.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = OpenAITTS(api_key="test-key")
        result = tts.generate("Hello world")
        # 返回 None 而未崩溃则通过
        assert result is None
