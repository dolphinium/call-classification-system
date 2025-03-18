# Call Classification System

A robust, production-grade system for automated analysis and classification of customer service call recordings. The system processes thousands of audio files daily with over 95% accuracy, helping businesses optimize their customer service operations through intelligent call categorization.

## Overview

This system automates the analysis of customer service calls by:
1. Processing dual-channel audio recordings (customer and agent channels)
2. Performing speech-to-text transcription
3. Analyzing conversations using Google's Gemini
4. Classifying calls into actionable categories
5. Providing detailed analysis with justifications

## Key Features

- **Dual Channel Processing**: Separates and processes customer and agent audio channels independently
- **Voice Activity Detection**: Uses Silero VAD for precise speech detection
- **Automated Transcription**: Converts speech to text with support for Turkish language
- **Intelligent Classification**: Categorizes calls using Google's Gemini 
- **Scalable Architecture**: Celery-based distributed task processing
- **High Throughput**: Processes thousands of audio files daily
- **Production Reliability**: Maintains >95% classification accuracy
- **Robust Error Handling**: Comprehensive retry mechanisms and error reporting

## Tools and Technologies

### Core Technologies
- **Python 3.10**: Primary development language
- **FastAPI**: High-performance API framework
- **Celery**: Distributed task queue
- **MongoDB**: Data persistence
- **Redis**: Task broker and result backend
- **Docker**: Containerization and deployment

### AI/ML Components
- **Google Gemini Pro**: Advanced language model for call classification
- **Silero VAD**: State-of-the-art voice activity detection
- **PyTorch & TorchAudio**: Audio processing and deep learning operations
- **SpeechRecognition**: Google Speech-to-Text integration for transcription

### Audio Processing
- **FFmpeg**: Professional-grade audio manipulation
- **TorchAudio**: Audio processing pipeline
- **NumPy**: Numerical operations for audio signal processing

### Development & Testing
- **pytest**: Comprehensive test suite
- **Black & isort**: Code formatting
- **Flake8**: Code linting
- **mypy**: Static type checking

## System Architecture

The system follows a microservices architecture with the following components:

1. **API Service**: FastAPI-based endpoint for job management
2. **Task Processor**: Core processing pipeline for audio analysis
3. **Celery Workers**: Distributed task execution
4. **MongoDB**: Persistent storage
5. **Redis**: Message broker and result backend

## Classification Categories

The system classifies calls into the following categories:

- Potential Customer
- Unnecessary Call (with sub-categories):
  - Guaranteed Product
  - Irrelevant Sector
  - Installation
  - Service Fee Rejected
  - Price Research
  - Complaint
  - Call Later
  - Basic Job
  - Platform Membership
  - Others
- Empty Call
- Uncertain

## Production Performance

- Processes thousands of audio files daily
- Maintains >95% classification accuracy
- Robust error handling and retry mechanisms
- Scalable architecture for high throughput
- Production-tested with real customer data

## Environment Requirements

- Python 3.10+
- FFmpeg
- Redis
- MongoDB
- Docker & Docker Compose
