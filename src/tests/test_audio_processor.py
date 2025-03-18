import os
import tempfile
import pytest
import torch
import torchaudio
import numpy as np
from unittest.mock import patch, MagicMock

from src.audio.processor import AudioProcessor, AudioSegment

@pytest.fixture
def audio_processor():
    return AudioProcessor()

@pytest.fixture
def test_audio():
    """Create a synthetic audio file for testing"""
    sample_rate = 16000
    duration = 2  # seconds
    t = torch.linspace(0, duration, int(sample_rate * duration))
    # Generate a 440 Hz sine wave
    waveform = torch.sin(2 * np.pi * 440 * t).unsqueeze(0)
    
    # Save to temporary file
    temp_dir = tempfile.mkdtemp()
    audio_path = os.path.join(temp_dir, "test.wav")
    torchaudio.save(audio_path, waveform, sample_rate)
    
    yield audio_path
    
    # Cleanup
    if os.path.exists(temp_dir):
        os.remove(audio_path)
        os.rmdir(temp_dir)

def test_split_stereo_channels(audio_processor, test_audio):
    """Test splitting stereo channels"""
    with tempfile.TemporaryDirectory() as temp_dir:
        left_path, right_path = audio_processor._split_stereo_channels(test_audio)
        
        assert os.path.exists(left_path)
        assert os.path.exists(right_path)
        
        # Verify both are mono
        left_audio, _ = torchaudio.load(left_path)
        right_audio, _ = torchaudio.load(right_path)
        
        assert left_audio.shape[0] == 1
        assert right_audio.shape[0] == 1

def test_process_channel(audio_processor, test_audio):
    """Test processing a single channel"""
    with tempfile.TemporaryDirectory() as temp_dir:
        segments = audio_processor._process_channel(
            test_audio,
            "customer",
            temp_dir
        )
        
        assert isinstance(segments, list)
        for segment in segments:
            assert isinstance(segment, AudioSegment)
            assert segment.channel == "customer"
            assert os.path.exists(segment.file)

@patch('speech_recognition.Recognizer.recognize_google')
def test_transcribe_audio(mock_recognize, audio_processor, test_audio):
    """Test audio transcription"""
    expected_text = "Hello, this is a test"
    mock_recognize.return_value = expected_text
    
    result = audio_processor._transcribe_audio(test_audio)
    assert result == expected_text
    
    # Test error handling
    mock_recognize.side_effect = Exception("API Error")
    with pytest.raises(Exception):
        audio_processor._transcribe_audio(test_audio)

def test_process_audio_file(audio_processor, test_audio):
    """Test complete audio processing pipeline"""
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.object(
            audio_processor,
            '_transcribe_audio',
            return_value="Test transcription"
        ):
            segments = audio_processor.process_audio_file(test_audio, temp_dir)
            
            assert isinstance(segments, list)
            assert len(segments) > 0
            
            for segment in segments:
                assert isinstance(segment, AudioSegment)
                assert segment.channel in ["customer", "customer_service"]
                assert isinstance(segment.start, int)
                assert isinstance(segment.end, int)
                assert segment.start < segment.end
                assert isinstance(segment.transcription, str)
                assert os.path.exists(segment.file) 