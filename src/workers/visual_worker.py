"""
Visual Worker - Main Entry Point
Processes visual tasks from the antrian queue

New Flow (Backend-Only PDF Processing):
1. Extract merging data from PDF (text groups, tables, images, shapes)
2. Align extraction items with DokumenElemen from database
3. Run classification (LayoutLM + Docling)
4. Save results to files
"""

import time
import logging
import os
from database import SessionLocal, engine
from models import Base, Bab, Dokumen
from services.antrian_service import AntrianService, STORAGE_BASE
from services.merging_extraction_service import MergingExtractionService
from services.pdf_image_service import convert_pdf_to_images

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/visual_worker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create tables if not exist
Base.metadata.create_all(bind=engine)


def _resolve_doc_root_from_pdf_path(pdf_path: str) -> str:
    pdf_dir = os.path.dirname(pdf_path)
    if os.path.basename(pdf_dir).lower() == "pdf":
        return os.path.dirname(pdf_dir)
    return pdf_dir


def _resolve_image_output_dir(db, ref_tipe: str, ref_id: int, pdf_path: str) -> str:
    base_images_dir = os.path.join(_resolve_doc_root_from_pdf_path(pdf_path), "images")
    if ref_tipe not in ('bab', 'buku'):
        return base_images_dir

    bab = db.query(Bab).get(ref_id)
    if not bab:
        logger.warning(f"Bab {ref_id} not found, fallback to default images directory")
        return base_images_dir

    # Chapter images are grouped by chapter order: .../images/<bab_order>
    folder_name = bab.bab_order if bab.bab_order is not None else bab.bab_id
    return os.path.join(base_images_dir, str(folder_name))


def _to_storage_relative_path(path: str):
    if not path:
        return None

    normalized_storage = os.path.normpath(STORAGE_BASE)
    normalized_path = os.path.normpath(path)
    try:
        rel_path = os.path.relpath(normalized_path, normalized_storage)
    except ValueError:
        return None

    if rel_path.startswith('..'):
        return None

    return '/' + rel_path.replace('\\', '/').lstrip('/')


def process_visual_task():
    """Process one visual task from the queue."""
    db = SessionLocal()
    try:
        antrian_service = AntrianService(db)
        
        # Get next task
        task = antrian_service.get_next_labeling_task()
        if not task:
            return False
        
        try:
            # Update status to processing
            antrian_service.update_labeling_status(task, 'processing')
            
            logger.info(f"Processing labeling task ID: {task.antrian_id}, Type: {task.antrian_tipe}")
            ref_tipe, ref_id = antrian_service.get_task_reference(task)
            logger.info(f"Reference target: {ref_tipe}:{ref_id}")
            
            # Get PDF path
            pdf_path = antrian_service.get_full_pdf_path(task)
            output_dir = antrian_service.get_output_directory(task)
            
            logger.info(f"PDF: {pdf_path}")
            logger.info(f"Output: {output_dir}")

            # Convert PDF pages to images under the document directory
            images_output_dir = _resolve_image_output_dir(db, ref_tipe, ref_id, pdf_path)
            logger.info(f"Images output directory: {images_output_dir}")
            convert_pdf_to_images(pdf_path, output_dir=images_output_dir)

            # Update images_path after conversion.
            images_relative = _to_storage_relative_path(images_output_dir)
            if ref_tipe == 'dokumen':
                doc = db.query(Dokumen).get(ref_id)
                if doc:
                    images_relative = f"/dokumen/{doc.mhs_nrp}/{doc.dokumen_id}/images"
                    if doc.dokumen_images_path != images_relative:
                        doc.dokumen_images_path = images_relative
                        db.commit()
                        logger.info(f"Updated dokumen_images_path: {images_relative}")
                else:
                    logger.warning(f"Dokumen {ref_id} not found for images path update")
            elif ref_tipe in ('bab', 'buku'):
                bab = db.query(Bab).get(ref_id)
                if bab:
                    if not images_relative:
                        logger.warning(
                            f"Cannot derive storage-relative path for bab_images_path from: {images_output_dir}"
                        )
                    elif bab.bab_images_path != images_relative:
                        bab.bab_images_path = images_relative
                        db.commit()
                        logger.info(f"Updated bab_images_path: {images_relative}")
                else:
                    logger.warning(f"Bab {ref_id} not found for images path update")
            
            # ==============================
            # Delegates to MergingExtractionService
            # This handles Extraction, Alignment, Docling Fusion, JSON saving, and Visualization
            # ==============================
            logger.info("Delegating to MergingExtractionService...")
            
            # Determine output directory for visualizations
            # Goal: .../pdf/classification
            # pdf_path is usually .../pdf/filename.pdf
            pdf_dir = os.path.dirname(pdf_path)
            classification_dir = os.path.join(pdf_dir, "classification")
            
            # Ensure directory exists
            os.makedirs(classification_dir, exist_ok=True)
            
            logger.info(f"Visual output directory: {classification_dir}")

            # Initialize service
            merging_service = MergingExtractionService()

            # Run processing with visualizations enabled
            # This will:
            # 1. Extract PDF
            # 2. Align with DB
            # 3. Run Docling & Fuse Results
            # 4. Save to DB (Skipped if save_to_db=False)
            # 5. Generate Visualizations & JSON in 'classification_dir'
            success = merging_service.process_document(
                doc_id=ref_id,
                generate_visualizations=True,
                save_to_db=True,  # Save DokumenElemenVisual to database
                output_dir=classification_dir,
                ref_tipe=ref_tipe
            )
            
            if success:
                logger.info(f"Labeling flow completed for {ref_tipe}:{ref_id}")
                antrian_service.update_labeling_status(task, 'completed')
                # Set validation status to in_queue so it can be picked up by validation worker
                antrian_service.update_validation_status(task, 'in_queue')
                return True
            else:
                logger.error(f"Labeling flow failed for {ref_tipe}:{ref_id}")
                antrian_service.update_labeling_status(task, 'failed', "MergingExtractionService returned failure")
                return False
            
            # ==============================
            # STEP 3: Legacy char groups
            # ==============================
            # Skipped - already extracted in Step 1
            
            # Update status to completed
            antrian_service.update_labeling_status(task, 'completed')
            
            logger.info(f"Visual task {task.antrian_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Visual task {task.antrian_id} failed: {str(e)}", exc_info=True)
            try:
                db.rollback()
                antrian_service.update_labeling_status(task, 'failed', str(e))
            except Exception as commit_error:
                logger.error(f"Failed to update error status: {commit_error}")
            return False
            
    except Exception as e:
        logger.error(f"Error checking visual queue: {str(e)}")
        return False
    finally:
        db.close()


def run_visual_worker(check_interval: int = 5):
    """
    Run visual worker with specified check interval.
    
    Args:
        check_interval: Seconds between queue checks
    """
    logger.info(f"Starting labeling worker (check every {check_interval} seconds)")
    logger.info("Flow: Labeling (Extraction+Alignment+Docling) -> Validation")
    
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    try:
        while True:
            process_visual_task()
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        logger.info("Visual worker stopped by user")
    except Exception as e:
        logger.error(f"Visual worker error: {str(e)}")


if __name__ == "__main__":
    run_visual_worker(check_interval=5)
