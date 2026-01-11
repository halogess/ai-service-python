from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify
import os
from models import db, TestingDokumen, TestingHistory, TestingGroundTruth, TestingPrediction

ground_truth_bp = Blueprint('ground_truth', __name__)

@ground_truth_bp.route('/document/<int:doc_id>/ground_truth')
def view_ground_truth(doc_id):
    from flask import current_app
    
    document = TestingDokumen.query.get_or_404(doc_id)
    images_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), 'images')
    
    if not os.path.exists(images_folder):
        flash('No analysis found. Please run analysis first.')
        return redirect(url_for('document.document_detail', doc_id=doc_id))
    
    return render_template('ground_truth.html', document=document)

@ground_truth_bp.route('/document/<int:doc_id>/ground_truth_data')
def get_ground_truth_data(doc_id):
    gt_data = TestingGroundTruth.query.filter_by(testing_dokumen_id=doc_id).all()
    
    if gt_data:
        result = []
        for gt in gt_data:
            result.append({
                'page': gt.testing_ground_truth_page,
                'bbox': gt.testing_ground_truth_bbox,
                'label': gt.testing_ground_truth_label,
                'word': gt.testing_ground_truth_word,
                'confidence': gt.testing_ground_truth_confidence
            })
        return jsonify(result)
    
    latest_history = TestingHistory.query.filter_by(testing_dokumen_id=doc_id).order_by(TestingHistory.testing_history_created_at.desc()).first()
    if latest_history:
        predictions = TestingPrediction.query.filter_by(testing_history_id=latest_history.testing_history_id).all()
        result = []
        for pred in predictions:
            result.append({
                'page': pred.testing_prediction_page,
                'bbox': pred.testing_prediction_bbox,
                'label': pred.testing_prediction_label,
                'word': pred.testing_prediction_word,
                'confidence': pred.testing_prediction_confidence
            })
        return jsonify(result)
    
    return jsonify([])

@ground_truth_bp.route('/save_ground_truth/<int:doc_id>', methods=['POST'])
def save_ground_truth(doc_id):
    annotations = request.json
    
    TestingGroundTruth.query.filter_by(testing_dokumen_id=doc_id).delete()
    
    for page_num_str, page_annotations in annotations.items():
        page_num = int(page_num_str)
        for ann in page_annotations:
            gt = TestingGroundTruth(
                testing_dokumen_id=doc_id,
                testing_ground_truth_page=page_num,
                testing_ground_truth_bbox=ann['bbox'],
                testing_ground_truth_label=ann['label'],
                testing_ground_truth_word=ann.get('word', ''),
                testing_ground_truth_confidence=ann.get('confidence', 1.0)
            )
            db.session.add(gt)
    
    db.session.commit()
    return {'success': True}
