"""OpenAI STT 服务商测试。"""

from unittest.mock import patch

from voice.providers.openai_stt import OpenAISTT


def test_init_with_api_key(mock_openai_client):
    """测试使用 API 密钥创建 OpenAISTT。"""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        # 客户端初始化未报错则通过
        assert stt.client == mock_openai_client


def test_transcribe_success(mock_openai_client, mock_audio_file):
    """测试成功转写音频。"""
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 返回已去除首尾空白的转写文本则通过
        assert result == "transcribed text"


def test_transcribe_seeks_file_to_beginning(mock_openai_client, mock_audio_file):
    """测试 transcribe 是否在读取前将文件指针移到开头。"""
    # 移动文件指针，模拟文件已被读取
    mock_audio_file.seek(100)
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        stt.transcribe(mock_audio_file)
        # 转写前文件位置已重置为 0 则通过
        assert mock_audio_file.tell() == 0


def test_transcribe_strips_whitespace(mock_openai_client, mock_audio_file):
    """测试转写结果是否去除首尾空白。"""
    # 模拟 API 返回首尾带空白的文本
    mock_openai_client.audio.transcriptions.create.return_value = "  text with spaces  "
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 结果已去除首尾空白则通过
        assert result == "text with spaces"


def test_transcribe_api_error(mock_openai_client, mock_audio_file):
    """测试是否能妥善处理 API 错误。"""
    # 让模拟对象抛出异常
    mock_openai_client.audio.transcriptions.create.side_effect = Exception("API Error")
    with patch("voice.providers.openai_stt.OpenAI", return_value=mock_openai_client):
        stt = OpenAISTT(api_key="test-key")
        result = stt.transcribe(mock_audio_file)
        # 返回空字符串而未崩溃则通过
        assert result == ""
