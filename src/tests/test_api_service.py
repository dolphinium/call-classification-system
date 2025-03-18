import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock
import requests

from src.services.api import APIService, Job

@pytest.fixture
def api_service():
    return APIService()

@pytest.fixture
def mock_response():
    response = MagicMock()
    response.status_code = 200
    return response

def test_poll_for_jobs_wait(api_service, mock_response):
    """Test polling when no jobs are available"""
    mock_response.json.return_value = {"status": "wait"}
    
    with patch('requests.get', return_value=mock_response):
        jobs = api_service.poll_for_jobs()
        assert len(jobs) == 0

def test_poll_for_jobs_success(api_service, mock_response):
    """Test polling with available jobs"""
    mock_response.json.return_value = {
        "status": "success",
        "jobs": [
            {"id": "123", "audio_url": "http://example.com/1.wav"},
            {"id": "456", "audio_url": "http://example.com/2.wav"}
        ]
    }
    
    with patch('requests.get', return_value=mock_response):
        jobs = api_service.poll_for_jobs()
        
        assert len(jobs) == 2
        assert all(isinstance(job, Job) for job in jobs)
        assert jobs[0].id == "123"
        assert jobs[0].audio_url == "http://example.com/1.wav"

def test_poll_for_jobs_error(api_service):
    """Test polling with request error"""
    with patch('requests.get', side_effect=requests.RequestException("Network error")):
        with pytest.raises(requests.RequestException):
            api_service.poll_for_jobs()

def test_send_callback_success(api_service, mock_response):
    """Test successful callback"""
    with patch('requests.post', return_value=mock_response) as mock_post:
        api_service.send_callback(
            status="finished",
            unique_id="123",
            classification="potential_customer",
            category="n/a",
            dialogue="Test dialogue"
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        assert kwargs["json"]["status"] == "finished"
        assert kwargs["json"]["unique_id"] == "123"
        assert kwargs["json"]["classification"] == "potential_customer"
        assert kwargs["json"]["category"] == "n/a"
        assert kwargs["json"]["dialogue"] == "Test dialogue"

def test_send_callback_error(api_service, mock_response):
    """Test error callback"""
    with patch('requests.post', return_value=mock_response) as mock_post:
        api_service.send_callback(
            status="error",
            unique_id="123",
            error_msg="Test error"
        )
        
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        
        assert kwargs["json"]["status"] == "error"
        assert kwargs["json"]["unique_id"] == "123"
        assert kwargs["json"]["error"] == "Test error"

def test_send_callback_request_error(api_service):
    """Test callback with request error"""
    with patch('requests.post', side_effect=requests.RequestException("Network error")):
        with pytest.raises(requests.RequestException):
            api_service.send_callback(
                status="finished",
                unique_id="123"
            )

def test_download_audio_success(api_service):
    """Test successful audio download"""
    mock_response = MagicMock()
    mock_response.iter_content.return_value = [b"test audio data"]
    
    with tempfile.NamedTemporaryFile() as tmp_file:
        with patch('requests.get', return_value=mock_response):
            api_service.download_audio(
                "http://example.com/audio.wav",
                tmp_file.name
            )
            
            # Verify file was written
            with open(tmp_file.name, 'rb') as f:
                assert f.read() == b"test audio data"

def test_download_audio_request_error(api_service):
    """Test audio download with request error"""
    with tempfile.NamedTemporaryFile() as tmp_file:
        with patch('requests.get', side_effect=requests.RequestException("Network error")):
            with pytest.raises(requests.RequestException):
                api_service.download_audio(
                    "http://example.com/audio.wav",
                    tmp_file.name
                )

def test_download_audio_write_error(api_service, mock_response):
    """Test audio download with file write error"""
    mock_response.iter_content.return_value = [b"test audio data"]
    
    with patch('requests.get', return_value=mock_response):
        with pytest.raises(IOError):
            # Try to write to a directory instead of a file
            api_service.download_audio(
                "http://example.com/audio.wav",
                "/nonexistent/directory/audio.wav"
            ) 