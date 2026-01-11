import time
import logging
import os
import json
from database import SessionLocal, engine
from models import Base, Antrian, Bab, Dokumen, DokumenElemen, DokumenPart, DokumenSection
from processors.docling_processor import process_pdf_with_docling, draw_bboxes_on_images
from services.pdf_extraction_service import PDFExtractor
from services.matching_service import match_db_with_docling

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
                
                # Create directories
                images_dir = os.path.join(base_dir, 'images', pdf_filename)
                result_images_dir = os.path.join(base_dir, 'image-result', pdf_filename)
                full_pdf_path = os.path.join(STORAGE_BASE, pdf_path)
                full_images_dir = os.path.join(STORAGE_BASE, images_dir)
                full_result_images_dir = os.path.join(STORAGE_BASE, result_images_dir)
                
                logger.info(f"Processing PDF: {full_pdf_path}")
                
                # Convert PDF to images
                with PDFExtractor(full_pdf_path) as extractor:
                    image_paths = []
                    for page_num in range(extractor.page_count):
                        output_path = os.path.join(full_images_dir, f"page_{page_num + 1}.png")
                        extractor.render_page_to_image(page_num, output_path, dpi=300)
                        image_paths.append(output_path)
                logger.info(f"Created {len(image_paths)} images")
                
                # Get dokumen_id for matching
                dokumen_id = task.dokumen_id if task.antrian_tipe == 'dokumen' else None
                
                # Process with Docling
                docling_result = process_pdf_with_docling(full_pdf_path, full_images_dir)
                
                # Match with DB if dokumen
                if dokumen_id:
                    # Query through new hierarchy: DokumenElemen → DokumenPart → DokumenSection → Dokumen
                    db_elements = db.query(DokumenElemen).join(
                        DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id
                    ).join(
                        DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id
                    ).filter(
                        DokumenSection.dokumen_id == dokumen_id,
                        DokumenPart.dpart_type == 'body'  # Only body parts, not header/footer
                    ).order_by(DokumenElemen.delemen_sequence).all()
                    
                    logger.info(f"Found {len(db_elements)} elements in DB for matching")
                    
                    # Match DB elements with Docling results
                    matched_results = match_db_with_docling(db_elements, docling_result['text_blocks'])
                    docling_result['matched_elements'] = matched_results
                else:
                    docling_result['matched_elements'] = []
                
                # Save Docling results
                docling_json_path = os.path.join(full_images_dir, "docling_result.json")
                with open(docling_json_path, 'w', encoding='utf-8') as f:
                    json.dump(docling_result['document'], f, indent=2, ensure_ascii=False)
                
                docling_md_path = os.path.join(full_images_dir, "docling_result.md")
                with open(docling_md_path, 'w', encoding='utf-8') as f:
                    f.write(docling_result['markdown'])
                
                # Save segmentation result: page, bbox, text, label
                segmentation_path = os.path.join(full_images_dir, "segmentation.json")
                with open(segmentation_path, 'w', encoding='utf-8') as f:
                    json.dump(docling_result['pages_with_bbox'], f, indent=2, ensure_ascii=False)
                
                # Save text blocks for matching with OpenXML SDK
                text_blocks_path = os.path.join(full_images_dir, "text_blocks.json")
                with open(text_blocks_path, 'w', encoding='utf-8') as f:
                    json.dump(docling_result['text_blocks'], f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved {len(docling_result['text_blocks'])} text blocks for matching")
                
                # Save matched results
                if docling_result['matched_elements']:
                    matched_path = os.path.join(full_images_dir, "matched_elements.json")
                    with open(matched_path, 'w', encoding='utf-8') as f:
                        json.dump(docling_result['matched_elements'], f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved {len(docling_result['matched_elements'])} matched elements")
                
                # Draw bboxes on images
                draw_bboxes_on_images(image_paths, docling_result['pages_with_bbox'], full_result_images_dir)
                
                logger.info(f"Docling results saved: {docling_json_path}")
                
                # Update status ke completed
                task.antrian_visual_status = 'completed'
                task.antrian_error_message = None
                db.commit()
                db.refresh(task)
                
                logger.info(f"Visual task {task.antrian_id} completed: {len(image_paths)} images with bbox visualization")
                
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