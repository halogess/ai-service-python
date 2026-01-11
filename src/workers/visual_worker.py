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
import json
import logging
import os
from database import SessionLocal, engine
from models import Base, Dokumen
from services.antrian_service import AntrianService
from services.merging_extraction_service import MergingExtractionService
from services.alignment_service import AlignmentService

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


def process_visual_task():
    """Process one visual task from the queue."""
    db = SessionLocal()
    try:
        antrian_service = AntrianService(db)
        
        # Get next task
        task = antrian_service.get_next_visual_task()
        if not task:
            return False
        
        try:
            # Update status to processing
            antrian_service.update_status(task, 'processing')
            
            logger.info(f"Processing visual task ID: {task.antrian_id}, Type: {task.antrian_tipe}")
            
            # Get PDF path
            pdf_path = antrian_service.get_full_pdf_path(task)
            output_dir = antrian_service.get_output_directory(task)
            
            logger.info(f"PDF: {pdf_path}")
            logger.info(f"Output: {output_dir}")
            
            # ==============================
            # STEP 1: Extract merging data
            # ==============================
            logger.info("Step 1: Extracting PDF content...")
            
            with MergingExtractionService(pdf_path) as extractor:
                extraction_results = extractor.extract_document()
            
            # Save extraction results
            extraction_file = os.path.join(output_dir, "extraction_results.json")
            with open(extraction_file, 'w', encoding='utf-8') as f:
                json.dump(extraction_results, f, indent=2, ensure_ascii=False)
            
            total_items = sum(len(r.get('items', [])) for r in extraction_results)
            logger.info(f"Extracted {len(extraction_results)} pages, {total_items} items")
            
            # ==============================
            # STEP 2: Align with DokumenElemen
            # ==============================
            alignment_results = []
            
            # Only run alignment if we have a dokumen_id
            if task.dokumen_id:
                logger.info("Step 2: Aligning with DokumenElemen...")
                
                alignment_service = AlignmentService(db)
                alignment_results = alignment_service.align_document(
                    extraction_results, 
                    task.dokumen_id
                )
                
                # Save alignment results
                alignment_file = os.path.join(output_dir, "alignment_results.json")
                with open(alignment_file, 'w', encoding='utf-8') as f:
                    # Convert to serializable format
                    serializable = []
                    for r in alignment_results:
                        serializable.append({
                            'success': r.get('success', False),
                            'page': r.get('page', 0),
                            'alignments': r.get('alignments', []),
                            'unaligned_pdf_units': r.get('unaligned_pdf_units', []),
                            'max_openxml_idx': r.get('max_openxml_idx', 0),
                            'stats': r.get('stats', {})
                        })
                    json.dump(serializable, f, indent=2, ensure_ascii=False)
                
                total_alignments = sum(len(r.get('alignments', [])) for r in alignment_results)
                logger.info(f"Created {total_alignments} alignments across {len(alignment_results)} pages")
            else:
                logger.info("Step 2: Skipped alignment (no dokumen_id)")
            
            # ==============================
            # STEP 3: Classification (TODO)
            # ==============================
            # Classification will be added in a future iteration
            # For now, we have extraction and alignment working
            
            # ==============================
            # Legacy: Also process char groups for backward compatibility
            # ==============================
            result = antrian_service.process_char_groups(task)
            logger.info(f"Legacy char groups: {result['page_count']} pages, {result['total_groups']} groups")
            
            # Update status to completed
            antrian_service.update_status(task, 'completed')
            
            logger.info(f"Visual task {task.antrian_id} completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Visual task {task.antrian_id} failed: {str(e)}", exc_info=True)
            try:
                db.rollback()
                antrian_service.update_status(task, 'failed', str(e))
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
    logger.info(f"Starting visual worker (check every {check_interval} seconds)")
    logger.info("Flow: Extraction -> Alignment -> Classification")
    
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
