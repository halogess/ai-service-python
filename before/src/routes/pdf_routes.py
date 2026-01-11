from flask import Blueprint, render_template, jsonify
import fitz
import re
import unicodedata
import difflib

from models import db, TestingDokumen, DokumenElemen

pdf_bp = Blueprint('pdf', __name__)

def normalize_text(s: str) -> str:
    """Normalisasi teks supaya ekstraksi DOCX dan PDF lebih mudah di-align."""
    if not s:
        return ''
    s = unicodedata.normalize('NFKC', s)
    s = s.replace('\u00ad', '')  # soft hyphen
    s = s.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
    # normalisasi berbagai jenis dash ke '-'
    s = s.replace('–', '-').replace('—', '-').replace('−', '-')
    # karakter aneh lain bisa ditambahkan di sini kalau ketemu
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def tokenize(s: str):
    """
    Tokenizer yang cocok untuk alignment:
    - angka berpola 5.1 tetap satu token
    - kata biasa
    - punctuation berdiri sendiri
    """
    s = normalize_text(s)
    if not s:
        return []
    return re.findall(r'\d+(?:\.\d+)*|[A-Za-zÀ-ÿ_]+|[^\w\s]', s, flags=re.UNICODE)

def extract_text_from_json_tree(json_tree):
    """
    Ekstrak teks dari dokumen_elemen_json_tree (OpenXML) secara rekursif.
    Menangani:
    - node type 'text'
    - node type 'math'
    - tabel: rows/cells
    """
    if not json_tree:
        return ""

    texts = []

    def rec(node):
        if isinstance(node, dict):
            # Struktur tabel
            if 'rows' in node:
                for row in node.get('rows', []):
                    if isinstance(row, dict):
                        for cell in row.get('cells', []):
                            if isinstance(cell, str):
                                texts.append(cell)
                            elif isinstance(cell, dict):
                                rec(cell)
                return

            # Node teks umum
            if node.get('type') == 'text' and 'value' in node:
                texts.append(node['value'])
            elif node.get('type') == 'math' and 'text' in node:
                texts.append(node['text'])

            # Telusuri child lain
            for k, v in node.items():
                if k not in ('type', 'value', 'text', 'rows', 'cells'):
                    rec(v)

        elif isinstance(node, list):
            for x in node:
                rec(x)

    rec(json_tree)
    return ' '.join(texts)

def union_bbox(bboxes):
    """Union beberapa bbox [x0,y0,x1,y1] menjadi satu bbox besar."""
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return [x0, y0, x1, y1]

@pdf_bp.route('/pdf-documents')
def pdf_documents():
    documents = TestingDokumen.query.all()
    return render_template('pdf_documents.html', documents=documents)

@pdf_bp.route('/pdf-view/<int:doc_id>')
def pdf_view(doc_id):
    document = TestingDokumen.query.get_or_404(doc_id)
    return render_template('pdf_viewer.html', doc=document)

@pdf_bp.route('/pdf-words/<int:doc_id>/<int:page_num>')
def pdf_words(doc_id, page_num):
    """
    Ambil semua kata di 1 halaman PDF dan mapping ke DokumenElemen (OpenXML)
    via difflib.SequenceMatcher token-level dengan sliding window.
    """
    document = TestingDokumen.query.get_or_404(doc_id)

    with fitz.open(document.testing_dokumen_path) as pdf:
        if page_num < 1 or page_num > pdf.page_count:
            return jsonify({'error': 'page out of range'}), 400
        page = pdf[page_num - 1]
        pdf_words_page = page.get_text('words')

    # Build PDF tokens untuk halaman ini
    pdf_tokens = []
    pdf_bboxes = []
    for (x0, y0, x1, y1, text, *_) in pdf_words_page:
        toks = tokenize(text)
        for tok in toks:
            pdf_tokens.append(tok)
            pdf_bboxes.append([x0, y0, x1, y1])

    if not pdf_tokens:
        return jsonify([])

    # Ambil semua elemen OpenXML
    elements = (
        DokumenElemen.query
        .filter_by(dokumen_id=doc_id)
        .order_by(DokumenElemen.delemen_sequence)
        .all()
    )

    # Coba align setiap elemen secara individual dengan sliding window di PDF
    result = []
    used_pdf_indices = set()

    for elem in elements:
        elem_text = extract_text_from_json_tree(elem.delemen_json_tree)
        elem_tokens = tokenize(elem_text)
        if not elem_tokens:
            continue

        # Cari best match window di PDF tokens
        best_ratio = 0
        best_start = -1
        best_end = -1

        window_size = len(elem_tokens)
        for start in range(len(pdf_tokens) - window_size + 1):
            # Skip jika sudah dipakai
            if any(i in used_pdf_indices for i in range(start, start + window_size)):
                continue

            window = pdf_tokens[start:start + window_size]
            sm = difflib.SequenceMatcher(None, elem_tokens, window, autojunk=False)
            ratio = sm.ratio()

            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = start + window_size

        # Jika match cukup bagus (threshold 0.6)
        if best_ratio >= 0.6 and best_start >= 0:
            bboxes = []
            for i in range(best_start, best_end):
                if i not in used_pdf_indices:
                    bboxes.append(pdf_bboxes[i])
                    used_pdf_indices.add(i)

            if bboxes:
                result.append({
                    'element_id': elem.delemen_id,
                    'element_type': elem.delemen_type,
                    'bbox': union_bbox(bboxes),
                    'words': [{'bbox': bb} for bb in bboxes]
                })

    return jsonify(result)

@pdf_bp.route('/pdf-words-raw/<int:doc_id>/<int:page_num>')
def pdf_words_raw(doc_id, page_num):
    """
    Endpoint debug: mengembalikan semua bbox kata mentah di 1 halaman PDF.
    """
    document = TestingDokumen.query.get_or_404(doc_id)

    with fitz.open(document.testing_dokumen_path) as pdf:
        if page_num < 1 or page_num > pdf.page_count:
            return jsonify({'error': 'page out of range'}), 400
        page = pdf[page_num - 1]
        pdf_words = page.get_text('words')

    result = []
    for (x0, y0, x1, y1, text, *_) in pdf_words:
        result.append({'bbox': [x0, y0, x1, y1], 'text': text})

    return jsonify(result)


@pdf_bp.route("/openxml-align/<int:doc_id>")
def openxml_align(doc_id):
    """OpenXML alignment untuk dokumen"""
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    try:
        from alignment_api import AlignmentAPI
        from extract_pdf_words import extract_pdf_words, merge_close_words
        from alignment_utils import extract_page_heights_from_pdf
        import json
        
        document = TestingDokumen.query.get_or_404(doc_id)
        
        # Check if alignment already exists
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_{doc_id}.json')
        
        if os.path.exists(alignment_file):
            with open(alignment_file, 'r', encoding='utf-8') as f:
                cached_result = json.load(f)
            return jsonify(cached_result)
        
        # Load OpenXML elements dari database
        elements = (
            DokumenElemen.query
            .filter_by(dokumen_id=doc_id)
            .order_by(DokumenElemen.delemen_sequence)
            .all()
        )
        
        openxml_elements = []
        for elem in elements:
            openxml_elements.append({
                'dokumen_elemen_id': elem.delemen_id,
                'dokumen_elemen_sequence': elem.delemen_sequence,
                'dokumen_elemen_type': elem.delemen_type,
                'dokumen_elemen_json_tree': elem.delemen_json_tree or {}
            })
        
        # Extract PDF words
        pdf_words_raw = extract_pdf_words(document.testing_dokumen_path)
        pdf_words = merge_close_words(pdf_words_raw, threshold=2.0)
        
        # Extract page heights
        page_heights = extract_page_heights_from_pdf(document.testing_dokumen_path)
        
        # Perform alignment
        api = AlignmentAPI()
        result = api.align_document(openxml_elements, pdf_words, page_heights)
        
        # Save to file
        if result['success']:
            os.makedirs(os.path.dirname(alignment_file), exist_ok=True)
            with open(alignment_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pdf_bp.route("/openxml-align-status/<int:doc_id>")
def openxml_align_status(doc_id):
    """Check if alignment exists"""
    import os
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_{doc_id}.json')
    
    exists = os.path.exists(alignment_file)
    return jsonify({'exists': exists, 'path': alignment_file if exists else None})


@pdf_bp.route("/openxml-align-difflib/<int:doc_id>")
def openxml_align_difflib(doc_id):
    """OpenXML alignment menggunakan difflib (decomposed version)"""
    import os
    import json
    import sys
    
    log_file = None
    
    try:
        # Import dari folder difflib_alignment (decomposed)
        from difflib_alignment import align_document
        
        log_file = open('alignment_trace.log', 'w', encoding='utf-8')
        log_file.write(f"\n\n========== DIFFLIB ALIGN CALLED FOR DOC {doc_id} ==========\n")
        log_file.flush()
        
        sys.stderr.write(f"\n\n========== DIFFLIB ALIGN CALLED FOR DOC {doc_id} ==========\n")
        sys.stderr.flush()
        
        document = TestingDokumen.query.get_or_404(doc_id)
        log_file.write(f"Document found: {document.testing_dokumen_nama}\n")
        log_file.flush()
        sys.stderr.write(f"Document found: {document.testing_dokumen_nama}\n")
        sys.stderr.flush()
        
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_{doc_id}.json')
        
        if os.path.exists(alignment_file):
            os.remove(alignment_file)
        
        elements = (
            DokumenElemen.query
            .filter_by(dokumen_id=doc_id)
            .order_by(DokumenElemen.delemen_sequence)
            .all()
        )
        
        log_file.write(f"Total elements: {len(elements)}\n")
        log_file.flush()
        sys.stderr.write(f"Total elements: {len(elements)}\n")
        sys.stderr.flush()
        
        data = align_document(document.testing_dokumen_path, elements, log_file)
        
        log_file.write("\n=== ALIGNMENT COMPLETE ===\n")
        log_file.flush()
        sys.stderr.write("\n=== ALIGNMENT COMPLETE ===\n")
        sys.stderr.flush()
        
        result = {
            'success': True,
            'data': data
        }
        
        os.makedirs(os.path.dirname(alignment_file), exist_ok=True)
        with open(alignment_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_msg = f"Failed to perform difflib alignment: {str(e)}"
        error_trace = traceback.format_exc()
        
        if log_file:
            log_file.write(f"\n=== ERROR ===\n{error_msg}\n{error_trace}\n")
            log_file.close()
        
        sys.stderr.write(f"\n=== ERROR ===\n{error_msg}\n{error_trace}\n")
        sys.stderr.flush()
        
        # ALWAYS return JSON, never let Flask return HTML error page
        return jsonify({
            'success': False,
            'error': error_msg,
            'traceback': error_trace
        }), 500
    finally:
        if log_file and not log_file.closed:
            log_file.close()


@pdf_bp.route('/difflib-align/<int:doc_id>')
def difflib_align(doc_id):
    return openxml_align_difflib(doc_id)


@pdf_bp.route('/difflib-align-backup/<int:doc_id>')
def difflib_align_backup(doc_id):
    """Alignment menggunakan difflib_alignment_monolith.py (versi monolitik)"""
    import os
    import json
    import difflib_alignment_monolith
    
    log_file = open('alignment_trace_monolith.log', 'w', encoding='utf-8')
    log_file.write(f"\n\n========== DIFFLIB MONOLITH ALIGN CALLED FOR DOC {doc_id} ==========\n")
    log_file.flush()
    
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        log_file.write(f"Document found: {document.testing_dokumen_nama}\n")
        log_file.flush()
        
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_monolith_{doc_id}.json')
        
        if os.path.exists(alignment_file):
            os.remove(alignment_file)
        
        elements = (
            DokumenElemen.query
            .filter_by(dokumen_id=doc_id)
            .order_by(DokumenElemen.delemen_sequence)
            .all()
        )
        
        log_file.write(f"Total elements: {len(elements)}\n")
        log_file.flush()
        
        data = difflib_alignment_monolith.align_document(document.testing_dokumen_path, elements, log_file)
        
        log_file.write("\n=== ALIGNMENT COMPLETE ===\n")
        log_file.flush()
        log_file.close()
        
        result = {
            'success': True,
            'data': data
        }
        
        os.makedirs(os.path.dirname(alignment_file), exist_ok=True)
        with open(alignment_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return jsonify(result)
        
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pdf_bp.route('/difflib-align-backup-status/<int:doc_id>')
def difflib_align_backup_status(doc_id):
    import os
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_monolith_{doc_id}.json')
    exists = os.path.exists(alignment_file)
    return jsonify({'exists': exists, 'path': alignment_file if exists else None})


@pdf_bp.route('/difflib-align-backup-load/<int:doc_id>')
def difflib_align_backup_load(doc_id):
    """Load existing difflib monolith alignment result"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_monolith_{doc_id}.json')
    
    if not os.path.exists(alignment_file):
        return jsonify({'success': False, 'error': 'Alignment file not found'}), 404
    
    try:
        with open(alignment_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_bp.route('/reassign-tokens/<int:doc_id>', methods=['POST'])
def reassign_tokens(doc_id):
    """Reassign unaligned tokens to nearest elements"""
    import os
    from reassign_tokens import reassign_unaligned_tokens
    
    try:
        result = reassign_unaligned_tokens(doc_id)
        
        if 'error' in result:
            return jsonify({'success': False, 'error': result['error']}), 404
        
        return jsonify({
            'success': True,
            'initial_unaligned': result['initial_unaligned'],
            'reassigned': result['reassigned'],
            'remaining_unaligned': result['remaining_unaligned']
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pdf_bp.route('/show-alignment/<int:doc_id>/<alignment_type>')
def show_alignment(doc_id, alignment_type):
    """Load alignment JSON (difflib or final)"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    
    if alignment_type == 'difflib':
        alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_{doc_id}.json')
    elif alignment_type == 'final':
        alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_final_{doc_id}.json')
    else:
        return jsonify({'success': False, 'error': 'Invalid alignment type'}), 400
    
    if not os.path.exists(alignment_file):
        return jsonify({'success': False, 'error': f'{alignment_type} alignment not found'}), 404
    
    try:
        with open(alignment_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_bp.route('/difflib-align-status/<int:doc_id>')
def difflib_align_status(doc_id):
    import os
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_{doc_id}.json')
    exists = os.path.exists(alignment_file)
    return jsonify({'exists': exists, 'path': alignment_file if exists else None})


@pdf_bp.route('/difflib-align-load/<int:doc_id>')
def difflib_align_load(doc_id):
    """Load existing difflib alignment result"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_{doc_id}.json')
    
    if not os.path.exists(alignment_file):
        return jsonify({'success': False, 'error': 'Alignment file not found'}), 404
    
    try:
        with open(alignment_file, 'r', encoding='utf-8') as f:
            result = json.load(f)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@pdf_bp.route('/docling-bbox/<int:doc_id>/<int:page_num>')
def docling_bbox(doc_id, page_num):
    """Get Docling bounding boxes for a specific page"""
    import os
    import json
    
    document = TestingDokumen.query.get_or_404(doc_id)
    
    # Look for Docling output
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    docling_folder = os.path.join(assets_folder, str(doc_id), 'docling_output')
    
    if not os.path.exists(docling_folder):
        return jsonify([])
    
    # Find JSON file in docling_output
    json_files = [f for f in os.listdir(docling_folder) if f.endswith('.json')]
    if not json_files:
        return jsonify([])
    
    docling_file = os.path.join(docling_folder, json_files[0])
    
    try:
        with open(docling_file, 'r', encoding='utf-8') as f:
            docling_data = json.load(f)
        
        # Extract bboxes for the requested page
        bboxes = []
        
        # Parse texts array
        for item in docling_data.get('texts', []):
            if item.get('prov'):
                for prov in item['prov']:
                    if prov.get('page_no') == page_num:
                        bbox = prov.get('bbox')
                        if bbox and 'l' in bbox and 't' in bbox and 'r' in bbox and 'b' in bbox:
                            # Convert BOTTOMLEFT to TOPLEFT (PDF coordinate system)
                            import fitz
                            with fitz.open(document.testing_dokumen_path) as pdf:
                                page_height = pdf[page_num - 1].rect.height
                            
                            bboxes.append({
                                'bbox': [bbox['l'], page_height - bbox['b'], bbox['r'], page_height - bbox['t']],
                                'type': item.get('label', 'text')
                            })
        
        # Parse tables array
        for item in docling_data.get('tables', []):
            if item.get('prov'):
                for prov in item['prov']:
                    if prov.get('page_no') == page_num:
                        bbox = prov.get('bbox')
                        if bbox and 'l' in bbox and 't' in bbox and 'r' in bbox and 'b' in bbox:
                            import fitz
                            with fitz.open(document.testing_dokumen_path) as pdf:
                                page_height = pdf[page_num - 1].rect.height
                            
                            bboxes.append({
                                'bbox': [bbox['l'], page_height - bbox['b'], bbox['r'], page_height - bbox['t']],
                                'type': 'table'
                            })
        
        # Parse pictures array
        for item in docling_data.get('pictures', []):
            if item.get('prov'):
                for prov in item['prov']:
                    if prov.get('page_no') == page_num:
                        bbox = prov.get('bbox')
                        if bbox and 'l' in bbox and 't' in bbox and 'r' in bbox and 'b' in bbox:
                            import fitz
                            with fitz.open(document.testing_dokumen_path) as pdf:
                                page_height = pdf[page_num - 1].rect.height
                            
                            bboxes.append({
                                'bbox': [bbox['l'], page_height - bbox['b'], bbox['r'], page_height - bbox['t']],
                                'type': 'picture'
                            })
        
        return jsonify(bboxes)
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
@pdf_bp.route('/analyze-docling/<int:doc_id>')
def analyze_docling(doc_id):
    """Analyze document with Docling and save result"""
    import os
    import json
    import subprocess
    
    document = TestingDokumen.query.get_or_404(doc_id)
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    output_folder = os.path.join(assets_folder, str(doc_id), 'docling_output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    try:
        # Get docling executable from venv
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        docling_exe = os.path.join(base_dir, 'venv310', 'Scripts', 'docling.exe')
        
        if not os.path.exists(docling_exe):
            return jsonify({'success': False, 'error': f'Docling not found at {docling_exe}'}), 500
        
        # Run docling with JSON output
        result = subprocess.run(
            [docling_exe, document.testing_dokumen_path, '--output', output_folder, '--to', 'json'],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode != 0:
            return jsonify({'success': False, 'error': result.stderr or result.stdout}), 500
        
        return jsonify({'success': True, 'output': result.stdout})
        
    except subprocess.TimeoutExpired:
        return jsonify({'success': False, 'error': 'Docling analysis timeout'}), 500
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@pdf_bp.route('/docling-status/<int:doc_id>')
def docling_status(doc_id):
    """Check if Docling analysis exists"""
    import os
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    docling_folder = os.path.join(assets_folder, str(doc_id), 'docling_output')
    
    if not os.path.exists(docling_folder):
        return jsonify({'exists': False})
    
    json_files = [f for f in os.listdir(docling_folder) if f.endswith('.json')]
    return jsonify({'exists': len(json_files) > 0})


@pdf_bp.route('/run_merge_alignment/<int:history_id>', methods=['POST'])
def run_merge_alignment(history_id):
    """Run merge alignment between difflib and docling"""
    import os
    import json
    from merge_alignment_docling import merge_alignment_with_docling
    from models import TestingHistory, DokumenElemen
    
    try:
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
        
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        output_file = os.path.join(assets_folder, str(doc_id), f'merged_alignment_{doc_id}.json')
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        return jsonify({'success': True, 'total': len(result)})
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@pdf_bp.route('/merged-alignment/<int:doc_id>/<int:page_num>')
def merged_alignment(doc_id, page_num):
    """Get merged alignment data for a specific page"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    merged_file = os.path.join(assets_folder, str(doc_id), f'merged_alignment_{doc_id}.json')
    
    if not os.path.exists(merged_file):
        return jsonify({'error': 'Merged alignment not found'}), 404
    
    try:
        with open(merged_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        merged_results = data.get('merged_results', data if isinstance(data, list) else [])
        unaligned_tokens = data.get('unaligned_tokens', [])
        
        page_data = [item for item in merged_results if item['page'] == page_num - 1]
        page_unaligned = [t for t in unaligned_tokens if t['page'] == page_num - 1]
        
        return jsonify({
            'merged': page_data,
            'unaligned_tokens': page_unaligned
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_bp.route('/reassign1-alignment/<int:doc_id>/<int:page_num>')
def reassign1_alignment(doc_id, page_num):
    """Get reassign1 alignment data"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    file_path = os.path.join(assets_folder, str(doc_id), f'reassign1_merged_alignment_{doc_id}.json')
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Reassign1 not found'}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use merged_results which has correctly rebuilt text with reassigned tokens
        merged_results = data.get('merged_results', [])
        unaligned_tokens = data.get('unaligned_tokens', [])
        
        # Fallback to aligned_words if merged_results is empty
        if not merged_results:
            merged_results = data.get('aligned_words', [])
        
        page_data = [item for item in merged_results if item['page'] == page_num - 1]
        page_unaligned = [t for t in unaligned_tokens if t['page'] == page_num - 1]
        
        return jsonify({
            'aligned_words': page_data,
            'unaligned_tokens': page_unaligned
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500



@pdf_bp.route('/reassign2-alignment/<int:doc_id>/<int:page_num>')
def reassign2_alignment(doc_id, page_num):
    """Get reassign2 alignment data"""
    import os
    import json
    
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    file_path = os.path.join(assets_folder, str(doc_id), f'reassign2_merged_alignment_{doc_id}.json')
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'Reassign2 not found'}), 404
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        merged_results = data.get('merged_results', [])
        unaligned_tokens = data.get('unaligned_tokens', [])
        
        page_data = [item for item in merged_results if item['page'] == page_num - 1]
        page_unaligned = [t for t in unaligned_tokens if t['page'] == page_num - 1]
        
        return jsonify({
            'merged': page_data,
            'unaligned_tokens': page_unaligned
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@pdf_bp.route('/run_reassign1/<int:doc_id>', methods=['POST'])
def run_reassign1(doc_id):
    """Run reassign step 1: merge same Y elements"""
    import json
    import traceback
    import os
    from merge_json_files import merge_step1
    
    try:
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        doc_folder = os.path.join(assets_folder, str(doc_id))
        
        alignment_file = os.path.join(doc_folder, f'alignment_difflib_{doc_id}.json')
        docling_folder = os.path.join(doc_folder, 'docling_output')
        docling_file = None
        
        if os.path.exists(docling_folder):
            for file in os.listdir(docling_folder):
                if file.endswith('.json'):
                    docling_file = os.path.join(docling_folder, file)
                    break
        
        if not os.path.exists(alignment_file):
            return jsonify({'success': False, 'error': 'Alignment file not found'}), 404
        
        if not docling_file or not os.path.exists(docling_file):
            return jsonify({'success': False, 'error': 'Docling file not found'}), 404
        
        output_file = os.path.join(doc_folder, f'reassign1_merged_alignment_{doc_id}.json')
        
        result = merge_step1(alignment_file, docling_file, output_file)
        
        return jsonify({'success': True, 'stats': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@pdf_bp.route('/run_reassign2/<int:doc_id>', methods=['POST'])
def run_reassign2(doc_id):
    """Run reassign step 2: reassign via docling overlap
    
    Input: reassign1_merged_alignment_xxx.json + docling
    Output: reassign2_merged_alignment_xxx.json
    """
    import json
    import traceback
    import os
    from merge_json_files import merge_step2
    
    try:
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        doc_folder = os.path.join(assets_folder, str(doc_id))
        
        reassign1_file = os.path.join(doc_folder, f'reassign1_merged_alignment_{doc_id}.json')
        docling_folder = os.path.join(doc_folder, 'docling_output')
        docling_file = None
        
        if os.path.exists(docling_folder):
            for file in os.listdir(docling_folder):
                if file.endswith('.json'):
                    docling_file = os.path.join(docling_folder, file)
                    break
        
        if not os.path.exists(reassign1_file):
            return jsonify({'success': False, 'error': 'Reassign1 file not found. Run Reassign1 first.'}), 404
        
        if not docling_file or not os.path.exists(docling_file):
            return jsonify({'success': False, 'error': 'Docling file not found'}), 404
        
        output_file = os.path.join(doc_folder, f'reassign2_merged_alignment_{doc_id}.json')
        
        result = merge_step2(reassign1_file, docling_file, output_file)
        
        return jsonify({'success': True, 'stats': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500
@pdf_bp.route('/run_merge_alignment_pdf/<int:doc_id>', methods=['POST'])
def run_merge_alignment_pdf(doc_id):
    """Run final merge: rebuild text from reassign2 result
    
    Input: reassign2_merged_alignment_xxx.json + docling
    Output: merged_alignment_xxx.json (final bbox + label)
    """
    import json
    import traceback
    import os
    from merge_json_files import merge_from_json_files
    from models import DokumenElemenVisual
    
    try:
        # Delete old merged data from DB
        DokumenElemenVisual.query.filter_by(dokumen_id=doc_id).delete()
        db.session.commit()
        
        assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        doc_folder = os.path.join(assets_folder, str(doc_id))
        
        # Use reassign2 result as input
        reassign2_file = os.path.join(doc_folder, f'reassign2_merged_alignment_{doc_id}.json')
        
        docling_file = None
        
        # Find docling file in docling_output folder
        docling_folder = os.path.join(doc_folder, 'docling_output')
        if os.path.exists(docling_folder):
            for file in os.listdir(docling_folder):
                if file.endswith('.json'):
                    docling_file = os.path.join(docling_folder, file)
                    break
        
        if not os.path.exists(reassign2_file):
            return jsonify({'success': False, 'error': 'Reassign2 file not found. Run Reassign2 first.'}), 404
        
        if not docling_file or not os.path.exists(docling_file):
            return jsonify({'success': False, 'error': 'Docling file not found. Run docling analysis first.'}), 404
        
        output_file = os.path.join(doc_folder, f'merged_alignment_{doc_id}.json')
        
        # Delete old merged file
        if os.path.exists(output_file):
            os.remove(output_file)
        
        result = merge_from_json_files(reassign2_file, docling_file, output_file)
        
        # Load the saved output to get unaligned tokens
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        
        merged_results = output_data.get('merged_results', result)
        
        # Insert to DB
        for item in merged_results:
            dev = DokumenElemenVisual(
                dokumen_id=doc_id,
                dev_bbox_x0=item['bbox']['x0'],
                dev_bbox_y0=item['bbox']['y0'],
                dev_bbox_x1=item['bbox']['x1'],
                dev_bbox_y1=item['bbox']['y1'],
                dev_page=item['page'],
                dev_label=item['docling_label'],
                dev_text=item['text'],
                dokumen_elemen_id=item['element_id'] if isinstance(item['element_id'], int) else None
            )
            db.session.add(dev)
        db.session.commit()
        
        return jsonify({
            'success': True, 
            'total': len(merged_results),
            'stats': output_data.get('stats', {})
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 500


@pdf_bp.route('/unaligned-tokens/<int:doc_id>/<int:page_num>')
def unaligned_tokens(doc_id, page_num):
    """Get unaligned PDF tokens for a specific page"""
    import os
    import json
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from difflib_alignment import iter_pdf_tokens_with_bboxes
    
    document = TestingDokumen.query.get_or_404(doc_id)
    
    # Load alignment data
    assets_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
    alignment_file = os.path.join(assets_folder, str(doc_id), f'alignment_difflib_{doc_id}.json')
    
    if not os.path.exists(alignment_file):
        return jsonify({'error': 'Alignment not found'}), 404
    
    with open(alignment_file, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)
    
    if not alignment_data.get('success'):
        return jsonify({'error': 'Invalid alignment data'}), 400
    
    # Get aligned PDF token keys (before_align_bboxes are dicts)
    aligned_keys = set()
    for word in alignment_data['data']['aligned_words']:
        if word['page'] == page_num - 1 and word.get('before_align_bboxes'):
            for bbox in word['before_align_bboxes']:
                key = f"{bbox['x0']:.1f},{bbox['y0']:.1f},{bbox['x1']:.1f},{bbox['y1']:.1f}"
                aligned_keys.add(key)
    
    # Get all PDF tokens using same method as alignment
    with fitz.open(document.testing_dokumen_path) as pdf:
        if page_num < 1 or page_num > pdf.page_count:
            return jsonify({'error': 'Invalid page number'}), 400
        page = pdf[page_num - 1]
        
        unaligned = []
        for token, bbox, _ in iter_pdf_tokens_with_bboxes(page, page_num - 1):
            # bbox is list [x0, y0, x1, y1]
            key = f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
            if key not in aligned_keys:
                unaligned.append({
                    'text': token,
                    'bbox': bbox
                })
    
    return jsonify(unaligned)
