import time
import logging
import os
import json
from database import SessionLocal, engine
from models import Base, Antrian, Bab, Dokumen, DokumenElemen, DokumenPart, DokumenSection
from services.pdf_extraction_service import PDFExtractor
from workers.visual_worker import run_visual_worker

STORAGE_BASE = "/app/storage"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visual_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create tables
Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    run_visual_worker(check_interval=5)