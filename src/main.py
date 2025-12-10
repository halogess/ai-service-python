import time
import logging
import os
from database import SessionLocal, engine
from models import Base, Antrian, Bab, Dokumen
from pdf_processor import convert_pdf_to_images

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

def check_visual_queue():
    """Cek antrian visual yang in_queue"""
    db = SessionLocal()
    try:
        # Ambil 1 task dengan status in_queue, FIFO
        task = db.query(Antrian).filter(
            Antrian.antrian_visual_status == 'in_queue'
        ).order_by(Antrian.antrian_created_at).first()
        
        if task:
            logger.info(f"Found visual task in queue: ID {task.antrian_id}")
            
            try:
                # Update status ke processing
                task.antrian_visual_status = 'processing'
                db.commit()
                logger.info(f"Processing visual task ID: {task.antrian_id}, Type: {task.antrian_tipe}")
                
                # Get PDF path based on type
                if task.antrian_tipe == 'buku':
                    # Get bab data
                    bab = db.query(Bab).filter(Bab.bab_id == task.bab_id).first()
                    if not bab or not bab.bab_pdf_path:
                        raise Exception(f"PDF path not found for bab_id: {task.bab_id}")
                    pdf_path = bab.bab_pdf_path
                    docx_path = bab.bab_docx_path or pdf_path
                    
                elif task.antrian_tipe == 'dokumen':
                    # Get dokumen data
                    dokumen = db.query(Dokumen).filter(Dokumen.dokumen_id == task.dokumen_id).first()
                    if not dokumen or not dokumen.dokumen_pdf_path:
                        raise Exception(f"PDF path not found for dokumen_id: {task.dokumen_id}")
                    pdf_path = dokumen.dokumen_pdf_path
                    docx_path = dokumen.dokumen_docx_path or pdf_path
                else:
                    raise Exception(f"Unknown antrian_tipe: {task.antrian_tipe}")
                
                # Get base directory
                pdf_dir = os.path.dirname(pdf_path)
                base_dir = os.path.dirname(pdf_dir)
                
                # Get filename without extension
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
                
                # Create images directory
                images_dir = os.path.join(base_dir, 'images', pdf_filename)
                full_pdf_path = os.path.join(STORAGE_BASE, pdf_path)
                full_images_dir = os.path.join(STORAGE_BASE, images_dir)
                
                logger.info(f"Converting PDF: {full_pdf_path}")
                
                # Convert PDF to images only
                image_paths = convert_pdf_to_images(full_pdf_path, full_images_dir)
                
                logger.info(f"Created {len(image_paths)} images")
                
                # Update status ke completed
                task.antrian_visual_status = 'completed'
                task.antrian_error_message = None
                db.commit()
                db.refresh(task)
                
                logger.info(f"Visual task {task.antrian_id} completed successfully")
                
            except Exception as e:
                try:
                    db.rollback()
                    task = db.query(Antrian).filter(Antrian.antrian_id == task.antrian_id).first()
                    if task:
                        task.antrian_visual_status = 'failed'
                        task.antrian_error_message = str(e)[:255]
                        db.commit()
                except Exception as commit_error:
                    logger.error(f"Failed to update error status: {commit_error}")
                
                logger.error(f"Visual task {task.antrian_id} failed: {str(e)}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error checking visual queue: {str(e)}")
    finally:
        db.close()

def run_visual_worker(check_interval=5):
    """Jalankan visual worker dengan interval tertentu (detik)"""
    logger.info(f"Starting visual worker (check every {check_interval} seconds)")
    
    try:
        while True:
            check_visual_queue()
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Visual worker stopped by user")
    except Exception as e:
        logger.error(f"Visual worker error: {str(e)}")

if __name__ == "__main__":
    # Jalankan worker, cek setiap 5 detik
    run_visual_worker(check_interval=5)