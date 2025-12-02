import time
import logging
import os
from sqlalchemy import and_
from database import SessionLocal, engine
from models import Base, Antrian, Bab, Dokumen
from pdf_processor import convert_pdf_to_images, get_images_dir_from_docx, extract_layout_data_from_pdf
from ocr_processor import extract_text_with_tesseract
from layoutlm_processor import process_document
from PIL import Image
import json

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
                
                # Get base directory (buku/id_orang/id_dokumen or dokumen/id_orang/id_dokumen)
                pdf_dir = os.path.dirname(pdf_path)  # e.g., buku/222117032/5639/pdf
                base_dir = os.path.dirname(pdf_dir)  # e.g., buku/222117032/5639
                
                # Get filename without extension for subdirectory
                pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]  # e.g., Bab 1 - Pendahuluan
                
                # Create separate directories per bab
                images_dir = os.path.join(base_dir, 'images', pdf_filename)
                image_result_dir = os.path.join(base_dir, 'image-result', pdf_filename)
                image_result_pdf_dir = os.path.join(base_dir, 'image-result-pdf', pdf_filename)
                image_result_ocr_dir = os.path.join(base_dir, 'image-result-ocr', pdf_filename)
                image_result_layoutlm_ocr_dir = os.path.join(base_dir, 'image-result-layoutlm-ocr', pdf_filename)
                
                # Full paths
                full_pdf_path = os.path.join(STORAGE_BASE, pdf_path)
                full_images_dir = os.path.join(STORAGE_BASE, images_dir)
                full_image_result_dir = os.path.join(STORAGE_BASE, image_result_dir)
                full_image_result_pdf_dir = os.path.join(STORAGE_BASE, image_result_pdf_dir)
                full_image_result_ocr_dir = os.path.join(STORAGE_BASE, image_result_ocr_dir)
                full_image_result_layoutlm_ocr_dir = os.path.join(STORAGE_BASE, image_result_layoutlm_ocr_dir)
                
                # Check ENV for enabled versions
                enable_full = os.getenv('ENABLE_VERSION_FULL', 'true').lower() == 'true'
                enable_pdf = os.getenv('ENABLE_VERSION_PDF', 'true').lower() == 'true'
                enable_ocr = os.getenv('ENABLE_VERSION_OCR', 'true').lower() == 'true'
                enable_layoutlm_ocr = os.getenv('ENABLE_VERSION_LAYOUTLM_OCR', 'false').lower() == 'true'
                
                logger.info(f"Converting PDF: {full_pdf_path}")
                
                # Extract layout data from PDF
                logger.info(f"Extracting layout data from PDF...")
                pdf_text_data = extract_layout_data_from_pdf(full_pdf_path)
                
                # Convert PDF to images
                image_paths = convert_pdf_to_images(full_pdf_path, full_images_dir)
                
                logger.info(f"Created {len(image_paths)} images")
                
                # VERSION 1: /image-result (full post-processing: 1,2,3,4)
                if enable_full:
                    logger.info(f"Processing VERSION 1: Full post-processing (PyMuPDF)...")
                    results_full = process_document(image_paths, output_dir=full_image_result_dir, pdf_text_data=pdf_text_data, post_processing_level="full")
                    
                    results_full_path = os.path.join(full_images_dir, "layoutlm_results_full.json")
                    with open(results_full_path, 'w', encoding='utf-8') as f:
                        json.dump(results_full, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"VERSION 1 (full post-processing) saved to {results_full_path}")
                else:
                    logger.info(f"VERSION 1 (full) DISABLED by ENV - skipping...")
                    results_full = None
                
                # VERSION 2: /image-result-pdf (sliding_window post-processing: hanya #3)
                if enable_pdf:
                    logger.info(f"Processing VERSION 2: Sliding window aggregation only (PyMuPDF)...")
                    results_pdf = process_document(image_paths, output_dir=full_image_result_pdf_dir, pdf_text_data=pdf_text_data, post_processing_level="sliding_window")
                    
                    results_pdf_path = os.path.join(full_images_dir, "layoutlm_results_pdf.json")
                    with open(results_pdf_path, 'w', encoding='utf-8') as f:
                        json.dump(results_pdf, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"VERSION 2 (sliding window aggregation only) saved to {results_pdf_path}")
                else:
                    logger.info(f"VERSION 2 (pdf) DISABLED by ENV")
                
                # VERSION 3: /image-result-ocr (no post-processing - Tesseract OCR)
                if enable_ocr:
                    logger.info(f"Processing VERSION 3: No post-processing (Tesseract OCR)...")
                    ocr_text_data = []
                    for img_path in image_paths:
                        image = Image.open(img_path).convert("RGB")
                        ocr_data = extract_text_with_tesseract(image)
                        ocr_text_data.append(ocr_data)
                    
                    results_ocr = process_document(image_paths, output_dir=full_image_result_ocr_dir, pdf_text_data=ocr_text_data, post_processing_level="none")
                    
                    results_ocr_path = os.path.join(full_images_dir, "layoutlm_results_ocr.json")
                    with open(results_ocr_path, 'w', encoding='utf-8') as f:
                        json.dump(results_ocr, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"VERSION 3 (Tesseract OCR) saved to {results_ocr_path}")
                else:
                    logger.info(f"VERSION 3 (ocr) DISABLED by ENV")
                
                # VERSION 4: /image-result-layoutlm-ocr (LayoutLMv3 built-in OCR)
                if enable_layoutlm_ocr:
                    logger.info(f"Processing VERSION 4: LayoutLMv3 built-in OCR (no text_data)...")
                    results_layoutlm_ocr = process_document(image_paths, output_dir=full_image_result_layoutlm_ocr_dir, pdf_text_data=None, post_processing_level="none", use_builtin_ocr=True)
                    
                    results_layoutlm_ocr_path = os.path.join(full_images_dir, "layoutlm_results_layoutlm_ocr.json")
                    with open(results_layoutlm_ocr_path, 'w', encoding='utf-8') as f:
                        json.dump(results_layoutlm_ocr, f, indent=2, ensure_ascii=False)
                    
                    logger.info(f"VERSION 4 (LayoutLMv3 OCR) saved to {results_layoutlm_ocr_path}")
                else:
                    logger.info(f"VERSION 4 (layoutlm-ocr) DISABLED by ENV")
                
                # Update status ke completed
                task.antrian_visual_status = 'completed'
                task.antrian_error_message = None
                db.commit()
                db.refresh(task)
                
                enabled_versions = []
                if enable_full: enabled_versions.append('Full')
                if enable_pdf: enabled_versions.append('PDF')
                if enable_ocr: enabled_versions.append('OCR')
                if enable_layoutlm_ocr: enabled_versions.append('LayoutLM-OCR')
                logger.info(f"Visual task {task.antrian_id} completed successfully ({len(enabled_versions)} versions: {' + '.join(enabled_versions)})")
                
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