import pytest
from unittest.mock import patch, MagicMock
from google.genai import types

from src.services.classifier import TranscriptClassifier, ClassificationResult

@pytest.fixture
def classifier():
    with patch('src.services.classifier.GEMINI_API_KEYS', ['test_key']):
        return TranscriptClassifier()

@pytest.fixture
def mock_response():
    response = MagicMock()
    response.text = """
<analysis>
<classification>potential_customer</classification>
<category>n/a</category>
<justification>
Customer shows interest in service and discusses scheduling.
</justification>
</analysis>
"""
    return response

def test_classifier_initialization():
    """Test classifier initialization with no API keys"""
    with pytest.raises(ValueError):
        with patch('src.services.classifier.GEMINI_API_KEYS', []):
            TranscriptClassifier()

def test_rotate_api_key(classifier):
    """Test API key rotation"""
    with patch('src.services.classifier.GEMINI_API_KEYS', ['key1', 'key2']):
        classifier = TranscriptClassifier()
        assert classifier.current_key_index == 0
        
        classifier._rotate_api_key()
        assert classifier.current_key_index == 1
        
        classifier._rotate_api_key()
        assert classifier.current_key_index == 0

def test_parse_response_valid(classifier, mock_response):
    """Test parsing a valid response"""
    result = classifier._parse_response(mock_response.text)
    
    assert isinstance(result, ClassificationResult)
    assert result.classification == "potential_customer"
    assert result.category == "n/a"
    assert "Customer shows interest" in result.justification

def test_parse_response_invalid_format(classifier):
    """Test parsing an invalid response format"""
    invalid_response = "Invalid format"
    with pytest.raises(ValueError):
        classifier._parse_response(invalid_response)

def test_parse_response_invalid_classification(classifier):
    """Test parsing response with invalid classification"""
    invalid_response = """
<analysis>
<classification>invalid_class</classification>
<category>n/a</category>
<justification>Test</justification>
</analysis>
"""
    with pytest.raises(ValueError):
        classifier._parse_response(invalid_response)

def test_parse_response_invalid_category(classifier):
    """Test parsing response with invalid category"""
    invalid_response = """
<analysis>
<classification>unnecessary_call</classification>
<category>invalid_category</category>
<justification>Test</justification>
</analysis>
"""
    with pytest.raises(ValueError):
        classifier._parse_response(invalid_response)

@patch('google.genai.GenerativeModel')
def test_classify_transcript_success(mock_model, classifier, mock_response):
    """Test successful transcript classification"""
    mock_instance = MagicMock()
    mock_instance.generate_content.return_value = mock_response
    mock_model.return_value = mock_instance
    
    result = classifier.classify_transcript("Test transcript")
    
    assert isinstance(result, ClassificationResult)
    assert result.classification == "potential_customer"
    assert result.category == "n/a"
    assert "Customer shows interest" in result.justification

@patch('google.genai.GenerativeModel')
def test_classify_transcript_rate_limit(mock_model, classifier):
    """Test handling of rate limit errors"""
    mock_instance = MagicMock()
    mock_instance.generate_content.side_effect = types.RateLimitError("Rate limit")
    mock_model.return_value = mock_instance
    
    with patch('src.services.classifier.GEMINI_API_KEYS', ['key1']):
        with pytest.raises(Exception) as exc_info:
            classifier.classify_transcript("Test transcript")
        assert "All API keys exhausted" in str(exc_info.value)

@patch('google.genai.GenerativeModel')
def test_classify_transcript_other_error(mock_model, classifier):
    """Test handling of other API errors"""
    mock_instance = MagicMock()
    mock_instance.generate_content.side_effect = Exception("API Error")
    mock_model.return_value = mock_instance
    
    with pytest.raises(Exception) as exc_info:
        classifier.classify_transcript("Test transcript")
    assert "API Error" in str(exc_info.value) 