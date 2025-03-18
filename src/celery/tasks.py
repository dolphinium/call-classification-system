from src.celery.app import app
from src.services.task_processor import TaskProcessor
from src.services.api import Job
from src.utils.logging import logger
from celery.exceptions import Ignore

processor = TaskProcessor()

@app.task
def poll_server_for_jobs():
    """
    Poll server for new jobs and create tasks for each job
    """
    try:
        jobs = processor.api_service.poll_for_jobs()
        for job in jobs:
            process_audio_job.delay(job.id, job.audio_url)
            
    except Exception as e:
        logger.error(f"Error polling for jobs: {str(e)}")
        raise

@app.task(bind=True, max_retries=3)
def process_audio_job(self, unique_id: str, audio_url: str):
    """
    Process a single audio job
    
    Args:
        unique_id: Job ID
        audio_url: URL to download the audio file
    """
    try:
        job = Job(id=unique_id, audio_url=audio_url)
        processor.process_job(job)
        
    except Exception as e:
        # Log error and retry
        logger.error(f"Error processing job {unique_id}: {str(e)}")
        
        try:
            self.retry(exc=e)
        except self.MaxRetriesExceededError:
            # If max retries exceeded, send error callback and give up
            processor.api_service.send_callback(
                status="error",
                unique_id=unique_id,
                error_msg=f"Max retries exceeded: {str(e)}"
            )
            raise Ignore() 