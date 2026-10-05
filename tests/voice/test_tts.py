"""TextToSpeech 工厂类测试。"""

import os
from unittest.mock import patch

import pytest

from voice.tts import TextToSpeech


def test_init_with_openai_provider(mock_openai_client):
    """测试使用 OpenAI 服务商和显式 API 密钥创建 TTS。"""
    with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
        tts = TextToSpeech(provider="openai", api_key="test-key")
        # provider 属性返回 "openai" 则通过
        assert tts.provider == "openai"


def test_init_with_invalid_provider():
    """测试未知服务商是否触发 ValueError。"""
    # 抛出包含预期消息的 ValueError 则通过
    with pytest.raises(ValueError, match="Unknown TTS provider: invalid"):
        TextToSpeech(provider="invalid", api_key="test-key")


def test_init_with_unimplemented_provider():
    """测试尚未实现的服务商是否触发 NotImplementedError。"""
    # 抛出 NotImplementedError 则通过（尚未实现 elevenlabs）
    with pytest.raises(NotImplementedError, match="ElevenLabs TTS provider not yet implemented"):
        TextToSpeech(provider="elevenlabs", api_key="test-key")


def test_from_env_provider_not_set():
    """测试未设置 VOICE_TTS_PROVIDER 时，from_env 是否返回 None。"""
    with patch.dict(os.environ, {}, clear=True):
        result = TextToSpeech.from_env()
        # 环境变量未设置时返回 None 则通过
        assert result is None


def test_from_env_valid_provider(mock_openai_client):
    """测试 from_env 是否使用有效服务商创建 TTS 实例。"""
    with patch.dict(os.environ, {"VOICE_TTS_PROVIDER": "openai", "OPENAI_API_KEY": "test-key"}):
        with patch("voice.providers.openai_tts.OpenAI", return_value=mock_openai_client):
            tts = TextToSpeech.from_env()
            # 创建的 TextToSpeech 实例使用正确服务商则通过
            assert tts is not None
            assert tts.provider == "openai"


def test_from_env_invalid_provider_returns_none():
    """测试服务商无效时，from_env 是否返回 None 并记录错误。"""
    with patch.dict(os.environ, {"VOICE_TTS_PROVIDER": "invalid"}):
        result = TextToSpeech.from_env()
        # 返回 None 而未崩溃则通过
        assert result is None


def test_get_api_key_from_param(mock_openai_client):
    """测试显式 api_key 参数是否优先于环境变量。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_tts.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            TextToSpeech(provider="openai", api_key="param-key")
            # OpenAI 客户端使用参数中的密钥而非环境变量中的密钥初始化，则通过
            mock_openai.assert_called_once_with(api_key="param-key")


def test_get_api_key_from_env(mock_openai_client):
    """测试未提供 API 密钥时是否从环境变量加载。"""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
        with patch(
            "voice.providers.openai_tts.OpenAI", return_value=mock_openai_client
        ) as mock_openai:
            TextToSpeech(provider="openai")
            # OpenAI 客户端使用环境变量中的密钥初始化，则通过
            mock_openai.assert_called_once_with(api_key="env-key")
