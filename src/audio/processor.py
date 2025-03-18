import os
import subprocess
from typing import Tuple, List, Dict, Any
import torch
import torchaudio
import speech_recognition as sr
from dataclasses import dataclass

from src.config.settings import (
    TARGET_SAMPLE_RATE,
    VAD_THRESHOLD,
    MIN_SPEECH_DURATION_MS,
    SPEECH_LANGUAGE
)
from src.utils.logging import logger

@dataclass
class AudioSegment:
    channel: str
    start: int
    end: int
    transcription: str
    file: str

class AudioProcessor:
    def __init__(self):
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False
        )
        (
            self.get_speech_ts,
            self.save_audio,
            self.read_audio,
            _,
            _
        ) = utils
        
        self.recognizer = sr.Recognizer()
    
    def process_audio_file(self, input_path: str, chunks_dir: str) -> List[AudioSegment]:
        """
        Process an audio file by splitting channels and transcribing speech segments
        
        Args:
            input_path: Path to the input audio file
            chunks_dir: Directory to store audio chunks
            
        Returns:
            List of AudioSegment objects containing transcribed segments
        """
        try:
            # Split stereo channels
            left_path, right_path = self._split_stereo_channels(input_path)
            
            # Process each channel
            left_segments = self._process_channel(left_path, "customer_service", chunks_dir)
            right_segments = self._process_channel(right_path, "customer", chunks_dir)
            
            # Combine and sort segments
            all_segments = left_segments + right_segments
            all_segments.sort(key=lambda x: x.start)
            
            return all_segments
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            raise
    
    def _split_stereo_channels(self, input_path: str) -> Tuple[str, str]:
        """Split stereo audio file into two mono channels"""
        base, ext = os.path.splitext(input_path)
        left_path = f"{base}_customer_service{ext}"
        right_path = f"{base}_customer{ext}"
        
        try:
            # Extract left channel
            subprocess.run([
                "ffmpeg", "-y",
                "-i", input_path,
                "-af", "pan=mono|c0=FL",
                left_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Extract right channel
            subprocess.run([
                "ffmpeg", "-y",
                "-i", input_path,
                "-af", "pan=mono|c0=FR",
                right_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            return left_path, right_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error splitting audio channels: {str(e)}")
            raise
    
    def _process_channel(self, audio_path: str, channel_label: str, chunks_dir: str) -> List[AudioSegment]:
        """Process a single audio channel"""
        # Load and preprocess audio
        audio_tensor, sr = torchaudio.load(audio_path)
        if audio_tensor.shape[0] > 1:
            audio_tensor = audio_tensor[0].unsqueeze(0)
            
        # Resample if necessary
        if sr != TARGET_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SAMPLE_RATE)
            audio_tensor = resampler(audio_tensor)
            sr = TARGET_SAMPLE_RATE
        
        # Get speech segments
        speech_timestamps = self.get_speech_ts(
            audio_tensor,
            self.model,
            sampling_rate=sr,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=MIN_SPEECH_DURATION_MS
        )
        
        return self._transcribe_segments(
            audio_tensor,
            sr,
            speech_timestamps,
            channel_label,
            chunks_dir
        )
    
    def _transcribe_segments(
        self,
        audio_tensor: torch.Tensor,
        sr: int,
        segments: List[Dict[str, Any]],
        channel_label: str,
        chunks_dir: str
    ) -> List[AudioSegment]:
        """Transcribe individual speech segments"""
        extra_samples = int(0.4 * sr)  # Add 0.4s padding
        transcribed_segments = []
        
        for i, seg in enumerate(segments):
            start = int(seg["start"])
            end = int(seg["end"])
            
            # Add padding but stay within bounds
            new_start = max(0, start - extra_samples)
            new_end = min(audio_tensor.shape[1], end + extra_samples)
            
            # Extract segment
            segment_audio = audio_tensor[:, new_start:new_end]
            
            # Save raw segment
            raw_file_path = os.path.join(chunks_dir, f"{channel_label}_segment_{i}_raw.wav")
            torchaudio.save(raw_file_path, segment_audio, sample_rate=sr)
            
            # Convert to format suitable for speech recognition
            converted_file_path = os.path.join(chunks_dir, f"{channel_label}_segment_{i}.wav")
            subprocess.run([
                "ffmpeg", "-y",
                "-i", raw_file_path,
                "-ar", str(TARGET_SAMPLE_RATE),
                "-ac", "1",
                converted_file_path
            ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Transcribe
            transcription = self._transcribe_audio(converted_file_path)
            
            transcribed_segments.append(AudioSegment(
                channel=channel_label,
                start=new_start,
                end=new_end,
                transcription=transcription,
                file=converted_file_path
            ))
        
        return transcribed_segments
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe a single audio file"""
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
            
            transcription = self.recognizer.recognize_google(
                audio,
                language=SPEECH_LANGUAGE
            )
            return transcription
            
        except sr.UnknownValueError:
            return "[Anlaşılmayan ses]"
        except sr.RequestError as e:
            error_msg = f"Speech Recognition error: {str(e)}"
            logger.error(error_msg)
            return f"[Request error: {str(e)}]"
        except Exception as e:
            logger.error(f"Unexpected error in transcription: {str(e)}")
            raise 