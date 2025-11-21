import os
import json
from sqlalchemy.orm import Session
from models.antrian import Antrian
from models.dokumen import Dokumen
from models.hasil_validasi import HasilValidasiVisual
from models.detail_error import DetailErrorVisual
from processing.pdf_loader import load_pages
from processing.ocr import extract_text
from processing.layout_model import LayoutModel
from processing.rules_visual import VisualRulesEvaluator
from processing.scoring import calculate_score
from config import Settings
from utils.logging import setup_logger

logger = setup_logger()

def process_visual_job(session: Session, job: Antrian, settings: Settings) -> None:
    try:
        logger.info(f"Processing visual job {job.antrian_id}")
        
        # 1. Tentukan target
        if job.dokumen_id:
            dokumen = session.query(Dokumen).filter(Dokumen.dokumen_id == job.dokumen_id).first()
            if not dokumen:
                raise Exception(f"Dokumen {job.dokumen_id} tidak ditemukan")
            
            pdf_rel_path = dokumen.dokumen_pdf_path
            target_type = 'dokumen'
            target_id = job.dokumen_id
        else:
            raise Exception("Target job tidak didukung (hanya dokumen)")
        
        # 2. Gabungkan dengan BASE_PATH
        pdf_abs_path = os.path.join(settings.BASE_PATH, pdf_rel_path.lstrip('/'))
        
        if not os.path.exists(pdf_abs_path):
            raise Exception(f"File PDF tidak ditemukan: {pdf_abs_path}")
        
        logger.info(f"Processing PDF: {pdf_abs_path}")
        
        # 3. Jalankan pipeline processing
        # Load PDF pages
        page_images = load_pages(pdf_abs_path)
        logger.info(f"Loaded {len(page_images)} pages")
        
        # OCR
        ocr_result = extract_text(page_images)
        logger.info(f"OCR extracted {len(ocr_result.words)} words")
        
        # Layout model
        layout_model = LayoutModel()
        layout_blocks = layout_model.predict_layout(ocr_result)
        logger.info(f"Layout model found {len(layout_blocks)} blocks")
        
        # Visual rules evaluation
        page_sizes = [(img.page_width, img.page_height) for img in page_images]
        rules_evaluator = VisualRulesEvaluator()
        errors = rules_evaluator.evaluate(layout_blocks, page_sizes)
        logger.info(f"Found {len(errors)} visual errors")
        
        # Calculate score
        skor_visual = calculate_score(errors, len(page_images))
        logger.info(f"Visual score: {skor_visual}")
        
        # 4. Simpan hasil
        # Create hasil_validasi_visual record
        hasil = HasilValidasiVisual(
            antrian_id=job.antrian_id,
            tipe_target=target_type,
            target_id=target_id,
            total_error=len(errors),
            skor_visual=skor_visual
        )
        session.add(hasil)
        session.flush()  # Get the ID
        
        # Create detail_error_visual records
        for error in errors:
            detail = DetailErrorVisual(
                hasil_id=hasil.id,
                kode=error.code,
                pesan_singkat=error.pesan_singkat,
                penjelasan=error.penjelasan,
                severity=error.severity,
                page=error.page,
                bbox_left=error.bbox.left,
                bbox_top=error.bbox.top,
                bbox_right=error.bbox.right,
                bbox_bottom=error.bbox.bottom,
                meta=json.dumps(error.meta) if error.meta else None
            )
            session.add(detail)
        
        # Update dokumen score if target is dokumen
        if target_type == 'dokumen':
            dokumen.dokumen_skor = int(skor_visual)
            dokumen.dokumen_jumlah_kesalahan = len(errors)
        
        # 5. Update status job
        job.antrian_visual_status = 'completed'
        job.antrian_error_message = None
        
        session.commit()
        logger.info(f"Successfully completed visual job {job.antrian_id}")
        
    except Exception as e:
        logger.error(f"Error processing visual job {job.antrian_id}: {str(e)}")
        
        # Update job status to failed
        job.antrian_visual_status = 'failed'
        job.antrian_error_message = str(e)[:255]  # Truncate to 255 chars
        
        session.commit()
        raise e