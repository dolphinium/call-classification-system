import logging
from datetime import datetime
from typing import Optional, Any, Dict

from pymongo import MongoClient
from src.config.settings import (
    MONGO_URI,
    DB_NAME,
    LOGS_COLLECTION,
    LOG_FORMAT,
    LOG_LEVEL
)

class MongoLogger:
    def __init__(self):
        self.client = MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[LOGS_COLLECTION]
        
        # Setup standard Python logging
        logging.basicConfig(
            level=getattr(logging, LOG_LEVEL),
            format=LOG_FORMAT
        )
        self.logger = logging.getLogger(__name__)
    
    def log(self, event_type: str, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an event to both MongoDB and standard Python logger
        
        Args:
            event_type: Type of the event (e.g., 'error', 'info', 'warning')
            message: The log message
            extra: Additional data to store with the log
        """
        doc = {
            "event_type": event_type,
            "message": message,
            "timestamp": datetime.utcnow(),
        }
        if extra:
            doc.update(extra)
            
        try:
            self.collection.insert_one(doc)
            self.logger.info(f"[{event_type}] {message}")
        except Exception as e:
            self.logger.error(f"Failed to log to MongoDB: {str(e)}")
            
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("error", message, extra)
        
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("info", message, extra)
        
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        self.log("warning", message, extra)

# Global logger instance
logger = MongoLogger() 