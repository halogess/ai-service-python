from flask import Blueprint, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename
import os
import fitz
import re
import unicodedata
import sys
from models import db, TestingDokumen, TestingHistory, TestingGroundTruth, DokumenElemen

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

document_bp = Blueprint('document', __name__)

@document_bp.route('/')
def home():
    return render_template('home.html')

@document_bp.route('/documents')
def documents():
    documents = TestingDokumen.query.all()
    return render_template('documents.html', documents=documents)

@document_bp.route('/upload', methods=['POST'])
def upload_pdf():
    from flask import current_app
    
    if 'pdf_file' not in request.files:
        flash('No file selected')
        return redirect(url_for('document.documents'))
    
    file = request.files['pdf_file']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        flash('Please select a PDF file')
        return redirect(url_for('document.documents'))
    
    filename = secure_filename(file.filename)
    
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            max_id = db.session.query(db.func.max(TestingDokumen.testing_dokumen_id)).scalar() or 0
            next_id = max_id + 1
            
            doc_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(next_id))
            os.makedirs(doc_folder, exist_ok=True)
            
            pdf_path = os.path.join(doc_folder, filename)
            file.save(pdf_path)
            
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            from pdf_processor import convert_pdf_to_images
            images_folder = os.path.join(doc_folder, 'images')
            os.makedirs(images_folder, exist_ok=True)
            convert_pdf_to_images(pdf_path, images_folder)
            
            new_doc = TestingDokumen(
                testing_dokumen_id=next_id,
                testing_dokumen_nama=filename,
                testing_dokumen_path=pdf_path,
                testing_dokumen_total_pages=total_pages
            )
            db.session.add(new_doc)
            db.session.commit()
            
            flash(f'Document uploaded successfully! {total_pages} pages')
            return redirect(url_for('document.documents'))
            
        except Exception as e:
            db.session.rollback()
            if attempt == max_attempts - 1:
                flash(f'Upload failed: {str(e)}')
                return redirect(url_for('document.documents'))
            continue

@document_bp.route('/document/<int:doc_id>')
def document_detail(doc_id):
    document = TestingDokumen.query.get_or_404(doc_id)
    histories = TestingHistory.query.filter_by(testing_dokumen_id=doc_id).order_by(TestingHistory.testing_history_created_at.desc()).all()
    has_ground_truth = TestingGroundTruth.query.filter_by(testing_dokumen_id=doc_id).first() is not None
    return render_template('document_detail.html', document=document, histories=histories, has_ground_truth=has_ground_truth)

@document_bp.route('/document/<int:doc_id>/alignment')
def view_alignment(doc_id):
    from flask import current_app
    import json
    
    document = TestingDokumen.query.get_or_404(doc_id)
    
    # Try multiple file names
    alignment_file = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), 'alignment_new.json')
    if not os.path.exists(alignment_file):
        alignment_file = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), 'alignment_result.json')
    if not os.path.exists(alignment_file):
        alignment_file = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), f'alignment_difflib_{doc_id}.json')
    
    if not os.path.exists(alignment_file):
        flash('No alignment result found. Please run alignment first.')
        return redirect(url_for('document.document_detail', doc_id=doc_id))
    
    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)
    
    # Handle difflib format (nested in 'data' key)
    if 'data' in alignment_data:
        alignment_data = alignment_data['data']
    
    # Convert difflib unaligned format to frontend format
    unaligned_elements = alignment_data.get('unaligned_elements', [])
    if unaligned_elements and 'sequence' not in unaligned_elements[0]:
        # Difflib format - convert to frontend format
        converted_unaligned = []
        for elem in unaligned_elements:
            converted_unaligned.append({
                'sequence': elem.get('element_id', 0),
                'type': 'unknown',
                'status': 'unmatched',
                'raw_text': elem.get('text', ''),
                'segments': []
            })
        alignment_data['unaligned_elements'] = converted_unaligned
    
    # Extract unaligned from mapping if not present
    if not alignment_data.get('unaligned_elements') and 'mapping' in alignment_data:
        alignment_data['unaligned_elements'] = [m for m in alignment_data['mapping'] if m.get('status') == 'unmatched']
    
    return render_template('alignment_viewer.html', 
                         document=document,
                         stats=alignment_data.get('stats', {}),
                         mapping=alignment_data.get('mapping', []),
                         unaligned=alignment_data.get('unaligned_elements', []),
                         reconstructed=alignment_data.get('reconstructed_blocks', []))

@document_bp.route('/run_alignment/<int:history_id>', methods=['POST'])
def run_alignment(history_id):
    from flask import current_app, jsonify
    import subprocess
    import json
    
    print(f"[ALIGNMENT] Starting alignment for history_id={history_id}")
    source_history = TestingHistory.query.get_or_404(history_id)
    document = TestingDokumen.query.get_or_404(source_history.testing_dokumen_id)
    doc_id = document.testing_dokumen_id
    print(f"[ALIGNMENT] Document ID: {doc_id}")
    
    try:
        from routes.history_routes import export_history_json
        assets_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id))
        openxml_file = os.path.join(assets_folder, f'dokumen_elemen_{doc_id}.json')
        docling_file = os.path.join(assets_folder, f'testing_prediction_{history_id}.json')
        
        # Only export if OpenXML file doesn't exist (export once per document, not per history)
        if not os.path.exists(openxml_file):
            print(f"[ALIGNMENT] OpenXML file not found, running export...")
            with current_app.test_request_context():
                export_result = export_history_json(history_id)
                if isinstance(export_result, tuple):
                    export_data = export_result[0].get_json()
                else:
                    export_data = export_result.get_json()
                
                if not export_data.get('success'):
                    return jsonify({'success': False, 'error': 'Auto-export failed: ' + export_data.get('error', 'Unknown')}), 500
            print(f"[ALIGNMENT] Export completed")
        else:
            print(f"[ALIGNMENT] OpenXML file already exists, skipping export")
        
        if not os.path.exists(openxml_file):
            print(f"[ALIGNMENT] OpenXML file not found: {openxml_file}")
            return jsonify({'success': False, 'error': 'OpenXML file not found. Please export first.'}), 404
        
        if not docling_file or not os.path.exists(docling_file):
            print(f"[ALIGNMENT] Docling file not found: {docling_file}")
            return jsonify({'success': False, 'error': 'Docling prediction file not found. Please run Docling analysis first.'}), 404
        
        print(f"[ALIGNMENT] Files found - OpenXML: {openxml_file}, Docling: {docling_file}")
        
        output_file = os.path.join(assets_folder, 'alignment_result.json')
        
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script_path = os.path.join(root_dir, 'align_openxml_docling.py')
        
        if not os.path.exists(script_path):
            return jsonify({'success': False, 'error': f'Script not found: {script_path}'}), 404
        
        pdf_path = document.testing_dokumen_path
        
        cmd = ['python', script_path, openxml_file, docling_file, pdf_path, output_file]
        print(f"[ALIGNMENT] Running subprocess: {' '.join(cmd)}")
        print(f"[ALIGNMENT] CWD: {root_dir}")
        print(f"[ALIGNMENT] PDF path: {pdf_path}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=root_dir
        )
        print(f"[ALIGNMENT] Subprocess completed with return code: {result.returncode}")
        if result.stdout:
            print(f"[ALIGNMENT] STDOUT:\n{result.stdout}")
        if result.stderr:
            print(f"[ALIGNMENT] STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            error_msg = result.stderr if result.stderr else result.stdout
            return jsonify({'success': False, 'error': f'Alignment failed: {error_msg}'}), 500
        
        print(f"[ALIGNMENT] Loading result from: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            alignment_data = json.load(f)
        
        blocks = alignment_data.get('blocks', [])
        print(f"[ALIGNMENT] Alignment successful")
        print(f"[ALIGNMENT] Stats: {alignment_data['stats']}")
        print(f"[ALIGNMENT] Total blocks in result: {len(blocks)}")
        
        if blocks:
            print(f"[ALIGNMENT] Sample block: {blocks[0]}")
        else:
            print(f"[ALIGNMENT] WARNING: No blocks generated!")
        
        return jsonify({
            'success': True,
            'stats': alignment_data['stats']
        })
        
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Alignment timeout (>5 minutes)'}), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@document_bp.route('/document/<int:doc_id>/image/<int:page_num>')
def serve_document_image(doc_id, page_num):
    from flask import current_app
    
    document = TestingDokumen.query.get_or_404(doc_id)
    images_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id), 'images')
    filename = os.path.splitext(document.testing_dokumen_nama)[0]
    image_path = os.path.join(images_folder, f'{filename}-page-{page_num}.jpg')
    
    return send_file(image_path, mimetype='image/jpeg')

@document_bp.route('/update_testing_dokumen_id/<int:old_id>', methods=['POST'])
def update_testing_dokumen_id(old_id):
    from flask import current_app
    import shutil
    
    new_id = request.json.get('new_id')
    
    existing = TestingDokumen.query.get(new_id)
    if existing:
        return {'success': False, 'error': f'ID {new_id} already exists'}, 400
    
    document = TestingDokumen.query.get_or_404(old_id)
    
    try:
        db.session.execute(db.text('UPDATE testing_ground_truth SET testing_dokumen_id = :new_id WHERE testing_dokumen_id = :old_id'), {'new_id': new_id, 'old_id': old_id})
        db.session.execute(db.text('UPDATE testing_history SET testing_dokumen_id = :new_id WHERE testing_dokumen_id = :old_id'), {'new_id': new_id, 'old_id': old_id})
        db.session.execute(db.text('UPDATE testing_dokumen SET testing_dokumen_id = :new_id WHERE testing_dokumen_id = :old_id'), {'new_id': new_id, 'old_id': old_id})
        db.session.commit()
        
        old_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(old_id))
        new_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(new_id))
        if os.path.exists(old_folder):
            shutil.move(old_folder, new_folder)
        
        document = TestingDokumen.query.get(new_id)
        if document:
            old_path = document.testing_dokumen_path
            new_path = old_path.replace(f'\\{old_id}\\', f'\\{new_id}\\')
            document.testing_dokumen_path = new_path
            db.session.commit()
        
        return {'success': True}
    except Exception as e:
        import traceback
        db.session.rollback()
        return {'success': False, 'error': str(e), 'trace': traceback.format_exc()}, 500


@document_bp.route('/export_dokumen_elemen/<int:doc_id>', methods=['POST'])
def export_dokumen_elemen(doc_id):
    """Export dokumen_elemen for a document (independent of history)"""
    from flask import current_app
    import json
    
    try:
        elements = (
            DokumenElemen.query
            .filter_by(dokumen_id=doc_id)
            .order_by(DokumenElemen.delemen_sequence)
            .all()
        )
        
        if not elements:
            return jsonify({'success': False, 'error': 'No elements found for this document'}), 404
        
        elements_data = []
        for elem in elements:
            elements_data.append({
                'dokumen_elemen_id': elem.delemen_id,
                'dokumen_id': elem.dokumen_id,
                'dokumen_elemen_sequence': elem.delemen_sequence,
                'dokumen_elemen_type': elem.delemen_type,
                'dokumen_elemen_json_tree': elem.delemen_json_tree
            })
        
        assets_folder = os.path.join(current_app.config['ASSETS_FOLDER'], str(doc_id))
        os.makedirs(assets_folder, exist_ok=True)
        
        output_file = os.path.join(assets_folder, f'dokumen_elemen_{doc_id}.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(elements_data, f, indent=2, ensure_ascii=False)
        
        return jsonify({
            'success': True,
            'file': output_file,
            'count': len(elements_data)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@document_bp.route('/run_merge_alignment/<int:history_id>', methods=['POST'])
def run_merge_alignment(history_id):
    """Run merge alignment between difflib and docling"""
    import json
    import traceback
    
    try:
        from src.merge_alignment_docling import merge_alignment_with_docling
        
        history = TestingHistory.query.get_or_404(history_id)
        doc_id = history.testing_dokumen_id
        document = TestingDokumen.query.get_or_404(doc_id)
        
        elements = (
            DokumenElemen.query
            .filter_by(dokumen_id=doc_id)
            .order_by(DokumenElemen.delemen_sequence)
            .all()
        )
        
        result = merge_alignment_with_docling(document.testing_dokumen_path, elements, db.session)
        
        from flask import current_app
        assets_folder = current_app.config['ASSETS_FOLDER']
        output_file = os.path.join(assets_folder, str(doc_id), f'merged_alignment_{doc_id}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return jsonify({'success': True, 'total': len(result)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500
