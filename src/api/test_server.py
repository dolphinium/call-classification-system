from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Call Classification Test Server")

# In-memory job queue for testing
# In a real service, this might come from a database, etc.
jobs_queue = []

# Job model for pulling
class Job(BaseModel):
    """Job model for audio processing requests"""
    id: str
    audio_url: str
    
    class Config:
        schema_extra = {
            "example": {
                "id": "12345",
                "audio_url": "https://example.com/audio.wav"
            }
        }

# Callback model to demonstrate receiving your Celery results
class CallbackData(BaseModel):
    """Model for receiving processing results"""
    status: str
    unique_id: str
    classification: Optional[str] = None
    category: Optional[str] = None
    error: Optional[str] = None
    dialogue: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "status": "finished",
                "unique_id": "12345",
                "classification": "potential_customer",
                "category": "n/a",
                "dialogue": "Customer: Hello\nAgent: Hi, how can I help?"
            }
        }

@app.get("/jobs/pull", response_model=dict)
async def pull_jobs():
    """
    Pull pending jobs from the queue.
    
    Returns:
        - If jobs available: {"status": "success", "jobs": [Job objects]}
        - If no jobs: {"status": "wait"}
    """
    if not jobs_queue:
        return {"status": "wait"}

    # In a real app, you'd only pop a subset, or handle concurrency, etc.
    batch_of_jobs = jobs_queue.copy()
    jobs_queue.clear()

    logger.info(f"Returning {len(batch_of_jobs)} jobs")
    return {
        "status": "success",
        "jobs": [job.dict() for job in batch_of_jobs]
    }

@app.post("/jobs/callback")
async def job_callback(data: CallbackData):
    """
    Receive callback with job processing results
    
    Args:
        data: CallbackData object containing processing results
    """
    if data.status == "finished":
        logger.info(
            f"Job {data.unique_id} finished successfully.\n"
            f"Classification: {data.classification}\n"
            f"Category: {data.category}"
        )
    elif data.status == "error":
        logger.error(f"Job {data.unique_id} failed: {data.error}")
    else:
        logger.warning(f"Unknown status {data.status} for job {data.unique_id}")

    return {"detail": "Callback received"}

@app.post("/jobs/add")
async def add_job(job: Job):
    """
    Add a new job to the processing queue
    
    Args:
        job: Job object containing ID and audio URL
    """
    # Validate job ID uniqueness
    if any(existing_job.id == job.id for existing_job in jobs_queue):
        raise HTTPException(
            status_code=400,
            detail=f"Job with ID {job.id} already exists"
        )
    
    jobs_queue.append(job)
    logger.info(f"Added job {job.id} to queue")
    return {"detail": f"Job {job.id} added successfully"}

@app.get("/jobs/count")
async def get_queue_count():
    """Get current number of jobs in queue"""
    return {"count": len(jobs_queue)}

@app.delete("/jobs/clear")
async def clear_queue():
    """Clear all jobs from queue (for testing)"""
    jobs_queue.clear()
    return {"detail": "Queue cleared"} 