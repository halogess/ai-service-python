
"""
Merging Worker (Structure Analysis)
Processes 'struktur' tasks from the antrian queue.
Orchestrates PDF Extraction, Alignment, and Docling Classification using MergingExtractionService.
"""
import time
import logging
import os
import sys

# Ensure src is in pythonpath if running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import SessionLocal, engine
from models import Base
from services.antrian_service import AntrianService
from services.merging_extraction_service import MergingExtractionService

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/merging_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create tables if not exist (ensures DB is ready)
Base.metadata.create_all(bind=engine)

def process_struktur_task():
    """Process one structure task from the queue."""
    db = SessionLocal()
    try:
        antrian_service = AntrianService(db)
        
        # Get next task
        task = antrian_service.get_next_struktur_task()
        if not task:
            return False
        
        try:
            # Update status to processing
            antrian_service.update_struktur_status(task, 'processing')
            
            logger.info(f"Processing structure task ID: {task.antrian_id}, DocID: {task.dokumen_id}")
            
            if not task.dokumen_id:
                 raise ValueError(f"Task {task.antrian_id} has no dokumen_id")

            # Run Merging Process
            service = MergingExtractionService()
            success = service.process_document(task.dokumen_id)
            
            if success:
                logger.info(f"Structure task {task.antrian_id} completed successfully")
                antrian_service.update_struktur_status(task, 'completed')
                return True
            else:
                logger.error(f"Structure task {task.antrian_id} failed in processing")
                antrian_service.update_struktur_status(task, 'failed', "Processing service returned failure")
                return False
            
        except Exception as e:
            logger.error(f"Structure task {task.antrian_id} failed: {str(e)}", exc_info=True)
            try:
                antrian_service.update_struktur_status(task, 'failed', str(e))
            except Exception as commit_error:
                logger.error(f"Failed to update error status: {commit_error}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking structure queue: {str(e)}")
        return False
    finally:
        db.close()


def run_worker(check_interval: int = 5):
    """
    Run merging worker with specified check interval.
    """
    logger.info(f"Starting merging worker (STRUKTUR) (check every {check_interval} seconds)")
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        while True:
            processed = process_struktur_task()
            if not processed:
                time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Merging worker stopped by user")
    except Exception as e:
        logger.error(f"Merging worker error: {str(e)}")

if __name__ == "__main__":
    run_worker(check_interval=5)
