import os
import tempfile
import shutil
from typing import List, Optional
from dataclasses import dataclass

from src.audio.processor import AudioProcessor, AudioSegment
from src.services.classifier import TranscriptClassifier, ClassificationResult
from src.services.api import APIService, Job
from src.utils.logging import logger

@dataclass
class ProcessingResult:
    classification: str
    category: str
    justification: str
    dialogue: str

class TaskProcessor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.classifier = TranscriptClassifier()
        self.api_service = APIService()
        
    def process_job(self, job: Job) -> None:
        """
        Process a single job from start to finish
        
        Args:
            job: Job object containing ID and audio URL
        """
        temp_dir = None
        try:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            audio_path = os.path.join(temp_dir, "input.wav")
            chunks_dir = os.path.join(temp_dir, "chunks")
            os.makedirs(chunks_dir, exist_ok=True)
            
            # Download audio
            logger.info(f"Downloading audio for job {job.id}")
            self.api_service.download_audio(job.audio_url, audio_path)
            
            # Process audio and get result
            result = self._process_audio_file(audio_path, chunks_dir)
            
            # Send success callback
            logger.info(f"Sending success callback for job {job.id}")
            self.api_service.send_callback(
                status="finished",
                unique_id=job.id,
                classification=result.classification,
                category=result.category,
                dialogue=result.dialogue
            )
            
        except Exception as e:
            # Log error and send error callback
            error_msg = f"Error processing job {job.id}: {str(e)}"
            logger.error(error_msg)
            self.api_service.send_callback(
                status="error",
                unique_id=job.id,
                error_msg=error_msg
            )
            raise
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                
    def _process_audio_file(self, audio_path: str, chunks_dir: str) -> ProcessingResult:
        """
        Process an audio file and return classification results
        
        Args:
            audio_path: Path to the audio file
            chunks_dir: Directory to store audio chunks
            
        Returns:
            ProcessingResult containing classification and dialogue
        """
        # Process audio file
        logger.info("Processing audio file")
        segments = self.audio_processor.process_audio_file(audio_path, chunks_dir)
        
        # Build dialogue text
        dialogue = self._build_dialogue(segments)
        
        # Classify transcript
        logger.info("Classifying transcript")
        classification = self.classifier.classify_transcript(dialogue)
        
        return ProcessingResult(
            classification=classification.classification,
            category=classification.category,
            justification=classification.justification,
            dialogue=dialogue
        )
        
    def _build_dialogue(self, segments: List[AudioSegment]) -> str:
        """
        Build a formatted dialogue string from audio segments
        
        Args:
            segments: List of AudioSegment objects
            
        Returns:
            Formatted dialogue string
        """
        dialogue_parts = []
        
        for segment in segments:
            speaker = "Customer Service" if segment.channel == "customer_service" else "Customer"
            dialogue_parts.append(f"{speaker}: {segment.transcription}")
            
        return "\n".join(dialogue_parts) 