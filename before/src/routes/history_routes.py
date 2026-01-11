from flask import Blueprint, render_template, jsonify, request, send_file
import os
import shutil
from models import db, TestingDokumen, TestingHistory, TestingPrediction
from services.evaluation_service import calculate_confusion_matrix

history_bp = Blueprint('history', __name__)

@history_bp.route('/history/<int:history_id>')
def history_detail(history_id):
    history = TestingHistory.query.get_or_404(history_id)
    document = TestingDokumen.query.get_or_404(history.testing_dokumen_id)
    
    # Use different template for Docling
    if 'Docling' in (history.testing_history_description or ''):
        return render_template('history_detail_docling.html', history=history, document=document)
    
    return render_template('history_detail.html', history=history, document=document, is_alignment_view=False)

@history_bp.route('/history/<int:history_id>/confusion_matrix_page')
def confusion_matrix_page(history_id):
    history = TestingHistory.query.get_or_404(history_id)
    document = TestingDokumen.query.get_or_404(history.testing_dokumen_id)
    return render_template('confusion_matrix.html', history=history, document=document)

@history_bp.route('/history/<int:history_id>/confusion_matrix')
def get_confusion_matrix(history_id):
    return jsonify(calculate_confusion_matrix(history_id))

@history_bp.route('/history/<int:history_id>/image/<int:page_num>')
def serve_result_image(history_id, page_num):
    from flask import current_app
    from pdf_processor import convert_pdf_to_images
    
    history = TestingHistory.query.get_or_404(history_id)
    doc_id = history.testing_dokumen_id
    document = TestingDokumen.query.get_or_404(doc_id)
    
    images_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), 'images')
    filename = os.path.splitext(document.testing_dokumen_nama)[0]
    image_path = os.path.join(images_folder, f'{filename}-page-{page_num}.jpg')
    
    # Auto-generate if not exists
    if not os.path.exists(image_path):
        os.makedirs(images_folder, exist_ok=True)
        convert_pdf_to_images(document.testing_dokumen_path, images_folder)
    
    return send_file(image_path, mimetype='image/jpeg')

@history_bp.route('/history/<int:history_id>/alignment')
def history_alignment(history_id):
    from flask import current_app
    import json
    
    history = TestingHistory.query.get_or_404(history_id)
    document = TestingDokumen.query.get_or_404(history.testing_dokumen_id)
    
    alignment_file = os.path.join(current_app.config['ASSETS_FOLDER'], str(document.testing_dokumen_id), 'alignment_result.json')
    
    if not os.path.exists(alignment_file):
        from flask import flash, redirect, url_for
        flash(f'No alignment result found. Please run alignment first.')
        return redirect(url_for('document.document_detail', doc_id=document.testing_dokumen_id))
    
    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)
    
    return render_template('history_detail.html', 
                         document=document,
                         history=history,
                         is_alignment_view=True)

@history_bp.route('/history/<int:history_id>/annotations', methods=['GET'])
def get_annotations(history_id):
    from flask import current_app, request
    import json
    
    print(f"[ANNOTATIONS] ===== ROUTE CALLED =====")
    print(f"[ANNOTATIONS] History ID: {history_id}")
    print(f"[ANNOTATIONS] Query args: {request.args}")
    
    # Check if alignment mode
    is_alignment = request.args.get('alignment') == 'true'
    print(f"[ANNOTATIONS] is_alignment: {is_alignment}")
    
    if is_alignment:
        print(f"[ANNOTATIONS] Entering alignment mode")
        history = TestingHistory.query.get_or_404(history_id)
        alignment_file = os.path.join(current_app.config['ASSETS_FOLDER'], str(history.testing_dokumen_id), 'alignment_result.json')
        
        print(f"[ANNOTATIONS] Doc ID: {history.testing_dokumen_id}")
        print(f"[ANNOTATIONS] Looking for file: {alignment_file}")
        print(f"[ANNOTATIONS] File exists: {os.path.exists(alignment_file)}")
        
        if not os.path.exists(alignment_file):
            print(f"[ANNOTATIONS] File not found!")
            return jsonify({})
        
        with open(alignment_file, 'r', encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        blocks = alignment_data.get('blocks', [])
        print(f"[ANNOTATIONS] Found {len(blocks)} blocks in alignment file")
        
        if blocks:
            print(f"[ANNOTATIONS] Sample block: {blocks[0]}")
        
        annotations = {}
        for idx, block in enumerate(blocks):
            segments = block.get('segments', [])
            if not segments:
                print(f"[ANNOTATIONS] Block {idx} has no segments!")
                continue
            
            for seg in segments:
                page_num = seg.get('page')
                bbox = seg.get('bbox')
                if not page_num or not bbox:
                    print(f"[ANNOTATIONS] Block {idx} segment missing page or bbox: {seg}")
                    continue
                
                if page_num not in annotations:
                    annotations[page_num] = []
                
                ann_data = {
                    'bbox': bbox,
                    'label': block.get('label', 'text'),
                    'word': block.get('text', '')
                }
                
                annotations[page_num].append(ann_data)
        
        print(f"[ANNOTATIONS] Generated annotations for {len(annotations)} pages")
        for page, anns in list(annotations.items())[:3]:
            print(f"[ANNOTATIONS] Page {page}: {len(anns)} annotations")
        print(f"[ANNOTATIONS] ===== RETURNING ALIGNMENT ANNOTATIONS =====")
        
        return jsonify(annotations)
    
    # Normal mode
    print(f"[ANNOTATIONS] Using normal mode (not alignment)")
    predictions = TestingPrediction.query.filter_by(testing_history_id=history_id).all()
    print(f"[ANNOTATIONS] Found {len(predictions)} predictions")
    
    annotations = {}
    for pred in predictions:
        page_num = pred.testing_prediction_page
        if page_num not in annotations:
            annotations[page_num] = []
        
        annotations[page_num].append({
            'id': pred.testing_prediction_id,
            'bbox': pred.testing_prediction_bbox,
            'label': pred.testing_prediction_label,
            'word': pred.testing_prediction_word,
            'confidence': pred.testing_prediction_confidence
        })
    
    print(f"[ANNOTATIONS] Normal mode: {len(annotations)} pages")
    print(f"[ANNOTATIONS] ===== RETURNING NORMAL ANNOTATIONS =====")
    return jsonify(annotations)

@history_bp.route('/history/<int:history_id>/pdf_words', methods=['GET'])
def get_pdf_words(history_id):
    # PDF words feature removed
    return jsonify({})



@history_bp.route('/history/<int:history_id>/openxml_text', methods=['GET'])
def get_openxml_text(history_id):
    return jsonify({'error': 'Fusion algorithm removed', 'elements': [], 'pred_to_openxml': {}, 'orphan_elements': []}), 404

@history_bp.route('/update_history_desc/<int:history_id>', methods=['POST'])
def update_history_desc(history_id):
    history = TestingHistory.query.get_or_404(history_id)
    desc = request.json.get('description', '')
    history.testing_history_description = desc
    db.session.commit()
    return {'success': True}

@history_bp.route('/history/<int:history_id>/export_json', methods=['GET'])
def export_history_json(history_id):
    from flask import current_app
    from models import DokumenElemen
    import json
    
    try:
        history = TestingHistory.query.get_or_404(history_id)
        doc_id = history.testing_dokumen_id
        
        # Get ALL predictions
        all_predictions = TestingPrediction.query.filter(
            TestingPrediction.testing_history_id == history_id
        ).order_by(
            TestingPrediction.testing_prediction_page,
            TestingPrediction.testing_prediction_bbox_y0,
            TestingPrediction.testing_prediction_bbox_x0
        ).all()
        
        # Separate: for alignment vs visualization only
        excluded_labels = ['page_header', 'page_footer', 'footnote']
        
        alignment_elements = []
        visualization_only = []
        
        for p in all_predictions:
            elem = {
                'testing_prediction_id': p.testing_prediction_id,
                'testing_prediction_page': p.testing_prediction_page,
                'testing_prediction_bbox': p.testing_prediction_bbox,
                'testing_prediction_label': p.testing_prediction_label,
                'testing_prediction_word': p.testing_prediction_word,
                'testing_prediction_confidence': p.testing_prediction_confidence
            }
            
            if p.testing_prediction_label in excluded_labels:
                visualization_only.append(elem)
            else:
                alignment_elements.append(elem)
        
        pred_data = {
            'dokumen_id': doc_id,
            'history_id': history.testing_history_id,
            'elements': alignment_elements,
            'visualization_only': visualization_only
        }
        
        # Get OpenXML elements from body parts only, with section margin data
        from models import DokumenPart, DokumenSection
        
        elements = db.session.query(DokumenElemen, DokumenSection)\
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)\
            .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)\
            .filter(DokumenSection.dokumen_id == doc_id)\
            .filter(DokumenPart.dpart_type == 'body')\
            .order_by(DokumenElemen.delemen_sequence)\
            .all()
        
        elem_list = [{
            'dokumen_elemen_id': e.DokumenElemen.delemen_id,
            'dokumen_elemen_sequence': e.DokumenElemen.delemen_sequence,
            'dokumen_elemen_type': e.DokumenElemen.delemen_type,
            'dokumen_elemen_json_tree': e.DokumenElemen.delemen_json_tree,
            'section_margins': {
                'top_twips': e.DokumenSection.dsec_margin_top_twips,
                'bottom_twips': e.DokumenSection.dsec_margin_bottom_twips,
                'left_twips': e.DokumenSection.dsec_margin_left_twips,
                'right_twips': e.DokumenSection.dsec_margin_right_twips,
                'header_twips': e.DokumenSection.dsec_header_margin_twips,
                'footer_twips': e.DokumenSection.dsec_footer_margin_twips
            }
        } for e in elements]
        
        elem_data = {
            'dokumen_id': doc_id,
            'elements': elem_list
        }
        
        # Save to assets folder
        assets_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id))
        os.makedirs(assets_folder, exist_ok=True)
        
        pred_file = os.path.join(assets_folder, f'testing_prediction_{history_id}.json')
        elem_file = os.path.join(assets_folder, f'dokumen_elemen_{doc_id}.json')
        
        with open(pred_file, 'w', encoding='utf-8') as f:
            json.dump(pred_data, f, indent=2, ensure_ascii=False)
        
        with open(elem_file, 'w', encoding='utf-8') as f:
            json.dump(elem_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'message': 'Export completed',
            'files': {
                'predictions': f'testing_prediction_{history_id}.json',
                'elements': f'dokumen_elemen_{doc_id}.json'
            },
            'location': f'assets/{doc_id}/',
            'counts': {
                'predictions': len(alignment_elements),
                'visualization_only': len(visualization_only),
                'elements': len(elem_list)
            }
        })
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'trace': traceback.format_exc()}), 500

@history_bp.route('/delete_history/<int:history_id>', methods=['DELETE'])
def delete_history(history_id):
    from flask import current_app
    
    try:
        history = TestingHistory.query.get_or_404(history_id)
        doc_id = history.testing_dokumen_id
        
        result_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), f'result_{history_id}')
        if os.path.exists(result_folder):
            shutil.rmtree(result_folder)
        
        TestingPrediction.query.filter_by(testing_history_id=history_id).delete()
        db.session.delete(history)
        db.session.commit()
        
        return {'success': True}
    except Exception as e:
        return {'success': False, 'error': str(e)}, 500
