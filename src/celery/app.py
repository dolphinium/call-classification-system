from celery import Celery
from src.config.settings import MONGO_URI

app = Celery(
    'call_classification',
    broker=MONGO_URI,
    backend=MONGO_URI,
    include=['src.celery.tasks']
)

# Optional configuration
app.conf.update(
    result_expires=3600,  # Results expire in 1 hour
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)

if __name__ == '__main__':
    app.start() 