from flask import Blueprint, jsonify
from models import db, TestingDokumen, TestingHistory, TestingPrediction
from services.analysis_service import analyze_document

analysis_bp = Blueprint('analysis', __name__)

analysis_progress = {}

@analysis_bp.route('/analyze_progress/<int:doc_id>')
def get_analysis_progress(doc_id):
    return jsonify({'current': analysis_progress.get(doc_id, 0)})

@analysis_bp.route('/analyze/<int:doc_id>', methods=['POST'])
def analyze(doc_id):
    try:
        return jsonify(analyze_document(doc_id, analysis_progress))
    except Exception as e:
        import traceback
        print('ERROR in analyze:', traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500



@analysis_bp.route('/analyze_docling/<int:doc_id>', methods=['POST'])
def analyze_docling(doc_id):
    try:
        from docling_processor import process_pdf_with_docling
        import os
        
        print(f"[DOCLING] Starting Docling analysis for doc_id={doc_id}")
        
        document = TestingDokumen.query.get_or_404(doc_id)
        pdf_path = document.testing_dokumen_path
        output_folder = os.path.join(os.path.dirname(pdf_path), 'docling_output')
        os.makedirs(output_folder, exist_ok=True)
        
        print(f"[DOCLING] PDF path: {pdf_path}")
        print(f"[DOCLING] Output folder: {output_folder}")
        print(f"[DOCLING] Calling process_pdf_with_docling...")
        
        process_pdf_with_docling(pdf_path, output_folder, doc_id=doc_id, save_to_db=True)
        
        print(f"[DOCLING] Docling analysis completed successfully")
        
        return jsonify({
            'success': True, 
            'message': 'Docling analysis saved to database'
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print('[DOCLING] ERROR in analyze_docling:', error_trace)
        return jsonify({
            'success': False,
            'error': str(e),
            'trace': error_trace
        }), 500
