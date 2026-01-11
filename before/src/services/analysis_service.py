import os
import time
from flask import current_app
from models import db, TestingDokumen, TestingHistory, TestingPrediction
from pdf_processor import convert_pdf_to_images, extract_text_from_pdf
from layoutlm_processor import process_image_with_layoutlm

def analyze_document(doc_id, progress_tracker):
    document = TestingDokumen.query.get_or_404(doc_id)
    
    try:
        progress_tracker[doc_id] = 0
        
        doc_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id))
        images_folder = os.path.join(doc_folder, 'images')
        os.makedirs(images_folder, exist_ok=True)
        
        text_data = extract_text_from_pdf(document.testing_dokumen_path)
        image_paths = convert_pdf_to_images(document.testing_dokumen_path, images_folder)
        
        history = TestingHistory(
            testing_dokumen_id=doc_id,
            testing_history_description=None
        )
        db.session.add(history)
        db.session.flush()
        
        result_folder = os.path.join(doc_folder, f'result_{history.testing_history_id}')
        os.makedirs(result_folder, exist_ok=True)
        
        results = []
        total_pages = len(image_paths)
        start_time = time.time()
        
        for i, image_path in enumerate(image_paths):
            progress_tracker[doc_id] = i + 1
            page_start = time.time()
            print(f"Processing page {i + 1}/{total_pages}...")
            page_result = process_image_with_layoutlm(image_path, text_data[i])
            page_time = time.time() - page_start
            page_result['page_number'] = i + 1
            page_result['processing_time'] = round(page_time, 2)
            results.append(page_result)
            print(f"Page {i + 1} completed in {page_time:.2f}s")
        
        total_time = time.time() - start_time
        print(f"Completed processing {total_pages} pages in {total_time:.2f}s (avg: {total_time/total_pages:.2f}s/page)")
        
        try:
            history.testing_history_processing_time = round(total_time, 2)
        except:
            pass
        
        for page_result in results:
            page_num = page_result['page_number']
            for pred in page_result['predictions']:
                bbox = pred['bbox']
                prediction = TestingPrediction(
                    testing_history_id=history.testing_history_id,
                    testing_prediction_page=page_num,
                    testing_prediction_bbox_x0=bbox[0],
                    testing_prediction_bbox_y0=bbox[1],
                    testing_prediction_bbox_x1=bbox[2],
                    testing_prediction_bbox_y1=bbox[3],
                    testing_prediction_label=pred['label'],
                    testing_prediction_word=pred.get('word', ''),
                    testing_prediction_confidence=pred.get('confidence', 1.0)
                )
                db.session.add(prediction)
        
        db.session.commit()
        progress_tracker.pop(doc_id, None)
        
        return {'success': True, 'results': results, 'history_id': history.testing_history_id}
    
    except Exception as e:
        import traceback
        progress_tracker.pop(doc_id, None)
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        return {'error': str(e)}, 500
