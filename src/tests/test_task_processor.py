import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock

from src.services.task_processor import TaskProcessor, ProcessingResult
from src.services.api import Job
from src.audio.processor import AudioSegment
from src.services.classifier import ClassificationResult

@pytest.fixture
def task_processor():
    return TaskProcessor()

@pytest.fixture
def mock_audio_segments():
    return [
        AudioSegment(
            channel="customer_service",
            start=0,
            end=1000,
            transcription="Hello, how can I help you?",
            file="test1.wav"
        ),
        AudioSegment(
            channel="customer",
            start=1000,
            end=2000,
            transcription="I need help with my device.",
            file="test2.wav"
        )
    ]

@pytest.fixture
def mock_classification_result():
    return ClassificationResult(
        classification="potential_customer",
        category="n/a",
        justification="Customer needs help with device"
    )

def test_build_dialogue(task_processor, mock_audio_segments):
    """Test dialogue building from segments"""
    dialogue = task_processor._build_dialogue(mock_audio_segments)
    
    expected = (
        "Customer Service: Hello, how can I help you?\n"
        "Customer: I need help with my device."
    )
    assert dialogue == expected

def test_process_audio_file(task_processor, mock_audio_segments, mock_classification_result):
    """Test complete audio file processing"""
    with patch.object(
        task_processor.audio_processor,
        'process_audio_file',
        return_value=mock_audio_segments
    ):
        with patch.object(
            task_processor.classifier,
            'classify_transcript',
            return_value=mock_classification_result
        ):
            result = task_processor._process_audio_file(
                "test.wav",
                "/tmp/chunks"
            )
            
            assert isinstance(result, ProcessingResult)
            assert result.classification == "potential_customer"
            assert result.category == "n/a"
            assert result.justification == "Customer needs help with device"
            assert "Hello, how can I help you?" in result.dialogue
            assert "I need help with my device." in result.dialogue

def test_process_job_success(task_processor, mock_audio_segments, mock_classification_result):
    """Test successful job processing"""
    job = Job(id="123", audio_url="http://example.com/audio.wav")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.multiple(
            task_processor.api_service,
            download_audio=MagicMock(),
            send_callback=MagicMock()
        ) as mocks:
            with patch.object(
                task_processor.audio_processor,
                'process_audio_file',
                return_value=mock_audio_segments
            ):
                with patch.object(
                    task_processor.classifier,
                    'classify_transcript',
                    return_value=mock_classification_result
                ):
                    task_processor.process_job(job)
                    
                    # Verify download was called
                    mocks['download_audio'].assert_called_once()
                    
                    # Verify callback was sent with success
                    mocks['send_callback'].assert_called_once()
                    args = mocks['send_callback'].call_args[1]
                    assert args['status'] == "finished"
                    assert args['unique_id'] == "123"
                    assert args['classification'] == "potential_customer"
                    assert args['category'] == "n/a"
                    assert "Hello, how can I help you?" in args['dialogue']

def test_process_job_download_error(task_processor):
    """Test job processing with download error"""
    job = Job(id="123", audio_url="http://example.com/audio.wav")
    
    with patch.multiple(
        task_processor.api_service,
        download_audio=MagicMock(side_effect=Exception("Download failed")),
        send_callback=MagicMock()
    ) as mocks:
        with pytest.raises(Exception):
            task_processor.process_job(job)
            
        # Verify error callback was sent
        mocks['send_callback'].assert_called_once()
        args = mocks['send_callback'].call_args[1]
        assert args['status'] == "error"
        assert args['unique_id'] == "123"
        assert "Download failed" in args['error_msg']

def test_process_job_processing_error(task_processor):
    """Test job processing with audio processing error"""
    job = Job(id="123", audio_url="http://example.com/audio.wav")
    
    with patch.multiple(
        task_processor.api_service,
        download_audio=MagicMock(),
        send_callback=MagicMock()
    ) as mocks:
        with patch.object(
            task_processor.audio_processor,
            'process_audio_file',
            side_effect=Exception("Processing failed")
        ):
            with pytest.raises(Exception):
                task_processor.process_job(job)
                
            # Verify error callback was sent
            mocks['send_callback'].assert_called_once()
            args = mocks['send_callback'].call_args[1]
            assert args['status'] == "error"
            assert args['unique_id'] == "123"
            assert "Processing failed" in args['error_msg']

def test_process_job_cleanup(task_processor):
    """Test temporary directory cleanup after job processing"""
    job = Job(id="123", audio_url="http://example.com/audio.wav")
    temp_dir = None
    
    with patch.multiple(
        task_processor.api_service,
        download_audio=MagicMock(),
        send_callback=MagicMock()
    ):
        with patch.object(
            task_processor.audio_processor,
            'process_audio_file',
            side_effect=Exception("Test error")
        ):
            try:
                task_processor.process_job(job)
            except:
                pass
            
            # Verify temp directory was cleaned up
            if temp_dir and os.path.exists(temp_dir):
                pytest.fail("Temporary directory was not cleaned up") 