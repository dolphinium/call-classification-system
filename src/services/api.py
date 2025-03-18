from typing import Dict, Optional, List, Any
import requests
from dataclasses import dataclass

from src.config.settings import (
    CALLBACK_URL,
    POLL_URL,
    API_HEADERS
)
from src.utils.logging import logger

@dataclass
class Job:
    id: str
    audio_url: str

class APIService:
    def __init__(self):
        self.headers = API_HEADERS
        
    def poll_for_jobs(self) -> List[Job]:
        """
        Poll the server for new jobs
        
        Returns:
            List of Job objects if available, empty list if should wait
        """
        try:
            response = requests.get(POLL_URL, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            if data["status"] == "wait":
                return []
                
            return [
                Job(id=job["id"], audio_url=job["audio_url"])
                for job in data.get("jobs", [])
            ]
            
        except requests.RequestException as e:
            logger.error(f"Error polling for jobs: {str(e)}")
            raise
            
    def send_callback(
        self,
        status: str,
        unique_id: str,
        classification: Optional[str] = None,
        category: Optional[str] = None,
        error_msg: Optional[str] = None,
        dialogue: Optional[str] = None
    ) -> None:
        """
        Send callback to server with job results
        
        Args:
            status: Job status ('finished' or 'error')
            unique_id: Job ID
            classification: Call classification (if finished)
            category: Call category (if finished)
            error_msg: Error message (if error)
            dialogue: Full dialogue transcript (if finished)
        """
        payload = {
            "status": status,
            "unique_id": unique_id
        }
        
        if status == "finished":
            payload.update({
                "classification": classification,
                "category": category,
                "dialogue": dialogue
            })
        elif status == "error":
            payload["error"] = error_msg
            
        try:
            response = requests.post(
                CALLBACK_URL,
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            
        except requests.RequestException as e:
            logger.error(f"Error sending callback for job {unique_id}: {str(e)}")
            raise
            
    def download_audio(self, url: str, local_path: str) -> None:
        """
        Download audio file from URL
        
        Args:
            url: Audio file URL
            local_path: Path to save the file locally
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        except requests.RequestException as e:
            logger.error(f"Error downloading audio from {url}: {str(e)}")
            raise
        except IOError as e:
            logger.error(f"Error saving audio to {local_path}: {str(e)}")
            raise 