import time
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import init_engine, SessionLocal
from config import get_settings
from jobs.antrian_repository import ambil_satu_job_visual
from jobs.worker_visual import process_visual_job
from utils.logging import setup_logger

def main():
    logger = setup_logger()
    settings = get_settings()
    
    logger.info("Starting AI Service Python Worker")
    logger.info(f"Database: {settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")
    logger.info(f"Base path: {settings.BASE_PATH}")
    
    # Initialize database
    init_engine()
    logger.info("Database engine initialized")
    
    # Main worker loop
    while True:
        try:
            with SessionLocal() as session:
                # Try to claim a visual job
                job = ambil_satu_job_visual(session)
                
                if job is None:
                    # No job available, sleep and continue
                    time.sleep(5)
                    continue
                
                logger.info(f"Claimed job {job.antrian_id}")
                
                # Process the job
                try:
                    process_visual_job(session, job, settings)
                except Exception as e:
                    logger.error(f"Failed to process job {job.antrian_id}: {str(e)}")
                    # Error handling is done in process_visual_job
                    continue
                
        except KeyboardInterrupt:
            logger.info("Received interrupt signal, shutting down...")
            break
        except Exception as e:
            logger.error(f"Unexpected error in main loop: {str(e)}")
            time.sleep(10)  # Wait before retrying
            continue

if __name__ == "__main__":
    main()