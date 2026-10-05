"""语音模块测试的共享夹具。"""

import io
from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_openai_client():
    """为 TTS/STT 测试提供模拟 OpenAI 客户端。"""
    client = Mock()

    # 模拟 TTS 响应（返回带 .content 属性的对象）
    mock_audio_response = Mock()
    mock_audio_response.content = b"fake audio data"
    client.audio.speech.create.return_value = mock_audio_response

    # 模拟 STT 响应（response_format="text" 时直接返回字符串）
    client.audio.transcriptions.create.return_value = "transcribed text"

    return client


@pytest.fixture
def mock_audio_file():
    """为 STT 测试提供 BytesIO 模拟音频文件。"""
    return io.BytesIO(b"fake audio bytes")
