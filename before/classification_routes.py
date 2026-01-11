from flask import Blueprint, render_template, jsonify, request
from models import db, TestingDokumen
import os

classification_bp = Blueprint('classification', __name__)

def clamp_bbox(bbox, min_val=0, max_val=1000):
    """Clamp bounding box coordinates to valid range for LayoutLM.
    
    LayoutLM expects normalized bounding boxes in range 0-1000.
    Negative values or values > 1000 cause CUDA device-side assert errors.
    """
    return [
        max(min_val, min(max_val, int(bbox[0]))),
        max(min_val, min(max_val, int(bbox[1]))),
        max(min_val, min(max_val, int(bbox[2]))),
        max(min_val, min(max_val, int(bbox[3])))
    ]

@classification_bp.route('/classification')
def classification_home():
    """List all documents for classification"""
    documents = TestingDokumen.query.order_by(TestingDokumen.testing_dokumen_id.desc()).all()
    return render_template('classification_documents.html', documents=documents)

@classification_bp.route('/classification/<int:doc_id>')
def classification_viewer(doc_id):
    """Classification viewer for a specific document"""
    doc = TestingDokumen.query.get_or_404(doc_id)
    return render_template('classification.html', doc=doc)


@classification_bp.route('/classification-api/classify/<int:doc_id>/<int:page>', methods=['POST'])
def api_classify_page(doc_id, page):
    """
    Run LayoutLM classification on aligned bboxes + header/footer units.
    
    Request body:
    {
        "aligned_units": [
            {"text": "...", "bbox": [x0, y0, x1, y1]},
            ...
        ],
        "header_footer_units": [
            {"text": "...", "bbox": [x0, y0, x1, y1]},
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": [
            {"text": "...", "bbox": [...], "label": "...", "confidence": 0.95},
            ...
        ]
    }
    """
    from PIL import Image
    import fitz
    
    data = request.get_json() or {}
    aligned_units = data.get('aligned_units', [])
    header_footer_units = data.get('header_footer_units', [])
    
    # Get document
    doc = TestingDokumen.query.get_or_404(doc_id)
    pdf_path = doc.testing_dokumen_path
    
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'}), 404
    
    try:
        # Open PDF and get page
        pdf_doc = fitz.open(pdf_path)
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            return jsonify({'success': False, 'error': f'Invalid page number: {page}'}), 400
        
        pdf_page = pdf_doc[page - 1]
        page_width = pdf_page.rect.width
        page_height = pdf_page.rect.height
        
        # Render page to image (300 DPI)
        scale = 300 / 72
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Convert to PIL Image
        import io
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pdf_doc.close()
        
        # Combine all units (aligned + header/footer)
        # NOTE: bbox is now received in 300 DPI pixels from frontend
        all_units = []
        dpi_scale = 300 / 72  # Conversion factor
        
        # Process aligned units (bbox already in 300 DPI)
        for unit in aligned_units:
            text = unit.get('text', '') or ''
            bbox = unit.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox:
                all_units.append({
                    'text': text,
                    'bbox': bbox,  # Already in 300 DPI
                    'source': 'aligned'
                })
        
        # Process header/footer units (bbox already in 300 DPI from frontend)
        for unit in header_footer_units:
            text = unit.get('text', '') or ''
            bbox = unit.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox:
                all_units.append({
                    'text': text,
                    'bbox': bbox,  # Already in 300 DPI
                    'source': 'header_footer'
                })
        
        if not all_units:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No units to classify'
            })
        
        # Sort by reading order (top to bottom, left to right)
        all_units.sort(key=lambda u: (u['bbox'][1], u['bbox'][0]))
        
        # Prepare words and boxes for LayoutLM
        # IMPORTANT: Extract words with accurate bboxes directly from PDF
        # Instead of estimating word positions, we use PyMuPDF's text extraction
        
        # Reopen PDF to extract words with accurate bboxes
        pdf_doc = fitz.open(pdf_path)
        pdf_page = pdf_doc[page - 1]
        
        # Extract words with bboxes from PDF (returns list of (x0, y0, x1, y1, word, block_no, line_no, word_no))
        word_data = pdf_page.get_text("words")
        pdf_doc.close()
        
        # Build unit bboxes for matching (convert to PDF points if needed)
        dpi_scale = 300 / 72
        
        # Image size in 300 DPI pixels
        img_width = page_width * dpi_scale
        img_height = page_height * dpi_scale
        
        # Scale factors for normalization (from 300 DPI pixels to 0-1000)
        scale_x = 1000 / img_width
        scale_y = 1000 / img_height
        
        words = []
        boxes = []
        original_bboxes = []  # In 300 DPI for display
        unit_indices = []  # Track which unit each word belongs to
        
        # NEW APPROACH: Extract ALL words from PDF page and run LayoutLM on entire page
        # Then map predictions back to units by bbox overlap
        
        # Debug: Log units received
        print(f"[Classification] Page {page}: Received {len(all_units)} units")
        for i, unit in enumerate(all_units[:3]):
            print(f"  Unit {i}: bbox={unit['bbox']}, src={unit['source']}")
        
        # Build word list from PDF (all words, no filtering)
        for wd in word_data:
            x0, y0, x1, y1 = wd[0], wd[1], wd[2], wd[3]
            word_text = wd[4]
            
            if not word_text.strip():
                continue
            
            # Word bbox in 300 DPI for display
            word_bbox_300dpi = [x0 * dpi_scale, y0 * dpi_scale, x1 * dpi_scale, y1 * dpi_scale]
            
            words.append(word_text)
            
            # Normalized bbox for model (0-1000), clamped to valid range
            boxes.append(clamp_bbox([
                word_bbox_300dpi[0] * scale_x,
                word_bbox_300dpi[1] * scale_y,
                word_bbox_300dpi[2] * scale_x,
                word_bbox_300dpi[3] * scale_y
            ]))
            
            original_bboxes.append(word_bbox_300dpi)
        
        print(f"[Classification] Extracted {len(words)} words from PDF")
        
        if not words:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No words to classify'
            })
        
        # Run LayoutLM
        from layoutlm_processor import load_model
        model, processor = load_model()
        device = model.device
        
        import torch
        
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        
        # Move to device
        for key in ("input_ids", "attention_mask", "bbox"):
            if key in encoding and isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(device)
        
        pv = encoding.get("pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_tensor = pv.to(device)
        elif isinstance(pv, list) and len(pv) > 0:
            pixel_values_tensor = pv[0].unsqueeze(0).to(device)
        else:
            return jsonify({'success': False, 'error': 'Failed to process pixel values'}), 500
        
        num_windows = len(encoding["input_ids"])
        token_predictions = {}
        
        for window_idx in range(num_windows):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"][window_idx:window_idx+1],
                    attention_mask=encoding["attention_mask"][window_idx:window_idx+1],
                    bbox=encoding["bbox"][window_idx:window_idx+1],
                    pixel_values=pixel_values_tensor,
                )
            
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)
            word_ids = encoding.word_ids(batch_index=window_idx)
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    label_id = pred_ids[token_idx].item()
                    label_prob = probs[token_idx, label_id].item()
                    
                    if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                        token_predictions[word_id] = (label_id, label_prob)
        
        # Build predictions grouped by original unit
        # Map each word's prediction to units by bbox overlap
        unit_predictions = {}  # unit_idx -> {labels: Counter, words: [], bbox: merged}
        
        def word_in_unit(word_bbox, unit_bbox, margin=20):
            """Check if word center is inside unit bbox"""
            cx = (word_bbox[0] + word_bbox[2]) / 2
            cy = (word_bbox[1] + word_bbox[3]) / 2
            return (unit_bbox[0] - margin <= cx <= unit_bbox[2] + margin and
                    unit_bbox[1] - margin <= cy <= unit_bbox[3] + margin)
        
        for word_id, pred in sorted(token_predictions.items()):
            if word_id >= len(words):
                continue
                
            raw_label = model.config.id2label.get(pred[0], "UNKNOWN")
            label = raw_label.lower().replace("-", "_")
            confidence = pred[1]
            
            # Get word bbox
            word_bbox = original_bboxes[word_id]  # In 300 DPI
            
            # Find which unit this word belongs to
            for unit_idx, unit in enumerate(all_units):
                unit_bbox = unit['bbox']  # In 300 DPI
                
                if word_in_unit(word_bbox, unit_bbox):
                    if unit_idx not in unit_predictions:
                        unit_predictions[unit_idx] = {
                            'text': unit['text'],
                            'bbox': unit['bbox'],
                            'source': unit['source'],
                            'labels': {},
                            'words': [],
                            'total_confidence': 0,
                            'word_count': 0
                        }
                    
                    up = unit_predictions[unit_idx]
                    up['labels'][label] = up['labels'].get(label, 0) + 1
                    up['words'].append({'word': words[word_id], 'label': label, 'confidence': confidence})
                    up['total_confidence'] += confidence
                    up['word_count'] += 1
                    break  # Word belongs to first matching unit
        
        print(f"[Classification] Mapped words to {len(unit_predictions)} units")
        
        # Determine final label for each unit (majority vote)
        predictions = []
        for unit_idx in sorted(unit_predictions.keys()):
            up = unit_predictions[unit_idx]
            
            # Get majority label
            if up['labels']:
                majority_label = max(up['labels'], key=up['labels'].get)
            else:
                majority_label = 'unknown'
            
            avg_confidence = up['total_confidence'] / up['word_count'] if up['word_count'] > 0 else 0
            
            predictions.append({
                'text': up['text'],
                'bbox': up['bbox'],
                'label': majority_label,
                'confidence': avg_confidence,
                'source': up['source'],
                'word_labels': up['labels']
            })
        
        return jsonify({
            'success': True,
            'page': page,
            'total_words': len(words),
            'total_units': len(all_units),
            'predictions': predictions
        })
        
    except Exception as e:
        import traceback
        print(f"[Classification] Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@classification_bp.route('/classification-api/docling-classify/<int:doc_id>', methods=['POST'])
def api_docling_classify_document(doc_id):
    """
    Run Docling classification on ENTIRE document (all pages at once).
    Frontend should call this once and cache the results.
    
    Returns:
    {
        "success": true,
        "total_pages": N,
        "predictions_by_page": {
            "1": [{"text": "...", "bbox": [...], "label": "...", "confidence": 1.0}, ...],
            "2": [...],
            ...
        }
    }
    """
    from docling.document_converter import DocumentConverter
    import fitz
    
    # Get document
    doc = TestingDokumen.query.get_or_404(doc_id)
    pdf_path = doc.testing_dokumen_path
    
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'}), 404
    
    try:
        # Convert PDF with Docling (processes entire document once)
        print(f"[Docling] Processing entire document: {doc_id}")
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        docling_doc = result.document
        
        # Open PDF to get page dimensions for each page
        pdf_doc = fitz.open(pdf_path)
        total_pages = len(pdf_doc)
        
        # Get page heights for Y coordinate conversion
        page_heights = {}
        for page_num in range(total_pages):
            page_heights[page_num + 1] = pdf_doc[page_num].rect.height
        
        pdf_doc.close()
        
        # Collect predictions for ALL pages
        predictions_by_page = {str(p): [] for p in range(1, total_pages + 1)}
        
        # Extract from texts (paragraphs, titles, list items, etc)
        for text in docling_doc.texts:
            if hasattr(text, 'prov') and text.prov:
                prov = text.prov[0] if isinstance(text.prov, list) else text.prov
                if hasattr(prov, 'page_no') and hasattr(prov, 'bbox') and prov.bbox:
                    page = prov.page_no
                    if page < 1 or page > total_pages:
                        continue
                    
                    bbox = prov.bbox
                    page_height = page_heights.get(page, 842)
                    y_top = page_height - bbox.t
                    y_bottom = page_height - bbox.b
                    text_bbox = [bbox.l, y_top, bbox.r, y_bottom]
                    
                    label = str(text.label).split('.')[-1].lower() if hasattr(text, 'label') else 'text'
                    text_content = text.text if hasattr(text, 'text') else ''
                    
                    predictions_by_page[str(page)].append({
                        'text': text_content,
                        'bbox': text_bbox,
                        'label': label,
                        'confidence': 1.0,
                        'source': 'docling'
                    })
        
        # Extract from tables
        for table in docling_doc.tables:
            if hasattr(table, 'prov') and table.prov:
                prov = table.prov[0] if isinstance(table.prov, list) else table.prov
                if hasattr(prov, 'page_no') and hasattr(prov, 'bbox') and prov.bbox:
                    page = prov.page_no
                    if page < 1 or page > total_pages:
                        continue
                    
                    bbox = prov.bbox
                    page_height = page_heights.get(page, 842)
                    y_top = page_height - bbox.t
                    y_bottom = page_height - bbox.b
                    
                    predictions_by_page[str(page)].append({
                        'text': '[Table]',
                        'bbox': [bbox.l, y_top, bbox.r, y_bottom],
                        'label': 'table',
                        'confidence': 1.0,
                        'source': 'docling'
                    })
        
        # Extract from pictures
        if hasattr(docling_doc, 'pictures'):
            for picture in docling_doc.pictures:
                if hasattr(picture, 'prov') and picture.prov:
                    prov = picture.prov[0] if isinstance(picture.prov, list) else picture.prov
                    if hasattr(prov, 'page_no') and hasattr(prov, 'bbox') and prov.bbox:
                        page = prov.page_no
                        if page < 1 or page > total_pages:
                            continue
                        
                        bbox = prov.bbox
                        page_height = page_heights.get(page, 842)
                        y_top = page_height - bbox.t
                        y_bottom = page_height - bbox.b
                        
                        predictions_by_page[str(page)].append({
                            'text': '[Picture]',
                            'bbox': [bbox.l, y_top, bbox.r, y_bottom],
                            'label': 'picture',
                            'confidence': 1.0,
                            'source': 'docling'
                        })
        
        # Extract from formulas
        if hasattr(docling_doc, 'formulas'):
            for formula in docling_doc.formulas:
                if hasattr(formula, 'prov') and formula.prov:
                    prov = formula.prov[0] if isinstance(formula.prov, list) else formula.prov
                    if hasattr(prov, 'page_no') and hasattr(prov, 'bbox') and prov.bbox:
                        page = prov.page_no
                        if page < 1 or page > total_pages:
                            continue
                        
                        bbox = prov.bbox
                        page_height = page_heights.get(page, 842)
                        y_top = page_height - bbox.t
                        y_bottom = page_height - bbox.b
                        formula_text = formula.text if hasattr(formula, 'text') else '[Formula]'
                        
                        predictions_by_page[str(page)].append({
                            'text': formula_text,
                            'bbox': [bbox.l, y_top, bbox.r, y_bottom],
                            'label': 'formula',
                            'confidence': 1.0,
                            'source': 'docling'
                        })
        
        # Sort predictions by reading order for each page
        for page_key in predictions_by_page:
            predictions_by_page[page_key].sort(key=lambda p: (p['bbox'][1], p['bbox'][0]))
        
        total_predictions = sum(len(preds) for preds in predictions_by_page.values())
        print(f"[Docling] Document {doc_id}: Found {total_predictions} predictions across {total_pages} pages")
        
        return jsonify({
            'success': True,
            'total_pages': total_pages,
            'bbox_unit': 'pdf_points',
            'predictions_by_page': predictions_by_page
        })
        
    except Exception as e:
        import traceback
        print(f"[Docling] Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# LINE-BASED LAYOUTLM CLASSIFICATION
# =============================================================================

def merge_line_units(units):
    """Merge units in same line: sort by X, concatenate text, merge bbox."""
    if not units:
        return None
    
    # Sort by x0 (left to right)
    sorted_units = sorted(units, key=lambda u: u['bbox'][0])
    
    # Merge text
    texts = [u['text'] for u in sorted_units if u.get('text')]
    merged_text = ' '.join(texts)
    
    # Merge bbox
    x0 = min(u['bbox'][0] for u in sorted_units)
    y0 = min(u['bbox'][1] for u in sorted_units)
    x1 = max(u['bbox'][2] for u in sorted_units)
    y1 = max(u['bbox'][3] for u in sorted_units)
    
    return {
        'text': merged_text,
        'bbox': [x0, y0, x1, y1],
        'unit_count': len(units),
        'source': units[0].get('source', 'aligned')
    }


def merge_groups_into_lines(units, y_tolerance_ratio=0.5):
    """
    Merge units that are on the same line (similar Y position).
    
    Args:
        units: list of {text, bbox} where bbox is [x0, y0, x1, y1]
        y_tolerance_ratio: overlap ratio threshold for same line
    
    Returns:
        lines: list of {text, bbox} merged lines
    """
    if not units:
        return []
    
    # Filter out units without valid bbox or text
    valid_units = [u for u in units if u.get('bbox') and len(u['bbox']) >= 4 and u.get('text')]
    
    if not valid_units:
        return []
    
    # Sort by y0 (top position)
    sorted_units = sorted(valid_units, key=lambda u: u['bbox'][1])
    
    lines = []
    current_line = [sorted_units[0]]
    
    for i in range(1, len(sorted_units)):
        curr = sorted_units[i]
        prev = current_line[-1]
        
        # Check Y overlap
        prev_y0, prev_y1 = prev['bbox'][1], prev['bbox'][3]
        curr_y0, curr_y1 = curr['bbox'][1], curr['bbox'][3]
        prev_height = prev_y1 - prev_y0
        
        if prev_height > 0:
            overlap = max(0, min(prev_y1, curr_y1) - max(prev_y0, curr_y0))
            overlap_ratio = overlap / prev_height
        else:
            overlap_ratio = 0
        
        if overlap_ratio >= y_tolerance_ratio:
            # Same line - add to current line
            current_line.append(curr)
        else:
            # New line - merge current and start new
            merged = merge_line_units(current_line)
            if merged:
                lines.append(merged)
            current_line = [curr]
    
    # Merge last line
    if current_line:
        merged = merge_line_units(current_line)
        if merged:
            lines.append(merged)
    
    return lines


@classification_bp.route('/classification-api/classify-lines/<int:doc_id>/<int:page>', methods=['POST'])
def api_classify_lines(doc_id, page):
    """
    Run LayoutLM classification on LINE-LEVEL merged bboxes.
    
    Input aligned_units are merged into lines based on Y position overlap,
    then each line is classified as a single unit.
    
    Request body:
    {
        "aligned_units": [
            {"text": "...", "bbox": [x0, y0, x1, y1]},
            ...
        ],
        "header_footer_units": [
            {"text": "...", "bbox": [x0, y0, x1, y1]},
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": [
            {"text": "...", "bbox": [...], "label": "...", "confidence": 0.95},
            ...
        ],
        "line_count": N,
        "original_unit_count": M
    }
    """
    from PIL import Image
    import fitz
    import torch
    
    data = request.get_json() or {}
    aligned_units = data.get('aligned_units', [])
    header_footer_units = data.get('header_footer_units', [])
    
    # Get document
    doc = TestingDokumen.query.get_or_404(doc_id)
    pdf_path = doc.testing_dokumen_path
    
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'}), 404
    
    try:
        # Open PDF and get page
        pdf_doc = fitz.open(pdf_path)
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            return jsonify({'success': False, 'error': f'Invalid page number: {page}'}), 400
        
        pdf_page = pdf_doc[page - 1]
        page_width = pdf_page.rect.width
        page_height = pdf_page.rect.height
        
        # Render page to image (300 DPI)
        scale = 300 / 72
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Convert to PIL Image
        import io
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pdf_doc.close()
        
        # Combine aligned + header/footer units
        all_units = []
        dpi_scale = 300 / 72
        
        # Process aligned units (bbox already in 300 DPI from frontend)
        for unit in aligned_units:
            text = unit.get('text', '') or ''
            bbox = unit.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox:
                all_units.append({
                    'text': text,
                    'bbox': bbox,
                    'source': 'aligned'
                })
        
        # Process header/footer units (bbox in PDF points, convert to 300 DPI)
        for unit in header_footer_units:
            text = unit.get('text', '') or ''
            bbox = unit.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox:
                # Convert from PDF points to 300 DPI
                bbox_300dpi = [b * dpi_scale for b in bbox]
                all_units.append({
                    'text': text,
                    'bbox': bbox_300dpi,
                    'source': 'header_footer'
                })
        
        original_unit_count = len(all_units)
        
        if not all_units:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No units to classify',
                'line_count': 0,
                'original_unit_count': 0
            })
        
        # MERGE UNITS INTO LINES
        merged_lines = merge_groups_into_lines(all_units, y_tolerance_ratio=0.5)
        
        print(f"[ClassifyLines] Page {page}: {original_unit_count} units → {len(merged_lines)} lines")
        
        # Debug: show first few lines
        for i, line in enumerate(merged_lines[:3]):
            print(f"  Line {i}: text='{line['text'][:50]}...', units={line['unit_count']}")
        
        if not merged_lines:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No lines after merging',
                'line_count': 0,
                'original_unit_count': original_unit_count
            })
        
        # Sort lines by reading order (top to bottom)
        merged_lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
        
        # Prepare words and boxes for LayoutLM
        # IMPORTANT: LayoutLM expects individual words, not sentences
        # So we tokenize each line into words, but keep the SAME bbox for all words in a line
        # Then aggregate predictions per-line using majority voting
        img_width = page_width * dpi_scale
        img_height = page_height * dpi_scale
        
        # Scale factors for normalization (from 300 DPI pixels to 0-1000)
        scale_x = 1000 / img_width
        scale_y = 1000 / img_height
        
        words = []
        boxes = []
        word_to_line = []  # Track which line each word belongs to
        
        for line_idx, line in enumerate(merged_lines):
            line_text = line['text']
            bbox = line['bbox']  # In 300 DPI
            
            # Normalized bbox for model (0-1000), clamped to valid range
            normalized_bbox = clamp_bbox([
                bbox[0] * scale_x,
                bbox[1] * scale_y,
                bbox[2] * scale_x,
                bbox[3] * scale_y
            ])
            
            # Tokenize line into words
            line_words = line_text.split()
            
            for word in line_words:
                if word.strip():
                    words.append(word.strip())
                    boxes.append(normalized_bbox)  # Same bbox for all words in line
                    word_to_line.append(line_idx)
        
        print(f"[ClassifyLines] Sending {len(words)} words from {len(merged_lines)} lines to LayoutLM")
        
        if not words:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No words after tokenization',
                'line_count': len(merged_lines),
                'original_unit_count': original_unit_count
            })
        
        # Validate all boxes are in valid range
        for i, box in enumerate(boxes):
            if any(v < 0 or v > 1000 for v in box):
                print(f"[ClassifyLines] Warning: Invalid bbox at index {i}: {box}")
                boxes[i] = clamp_bbox(box)
        
        # Clear CUDA cache before running to avoid memory issues
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Run LayoutLM
        from layoutlm_processor import load_model
        model, processor = load_model()

        device = model.device
        
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        
        # Move to device
        for key in ("input_ids", "attention_mask", "bbox"):
            if key in encoding and isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(device)
        
        pv = encoding.get("pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_tensor = pv.to(device)
        elif isinstance(pv, list) and len(pv) > 0:
            pixel_values_tensor = pv[0].unsqueeze(0).to(device)
        else:
            return jsonify({'success': False, 'error': 'Failed to process pixel values'}), 500
        
        num_windows = len(encoding["input_ids"])
        token_predictions = {}
        
        for window_idx in range(num_windows):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"][window_idx:window_idx+1],
                    attention_mask=encoding["attention_mask"][window_idx:window_idx+1],
                    bbox=encoding["bbox"][window_idx:window_idx+1],
                    pixel_values=pixel_values_tensor,
                )
            
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)
            word_ids = encoding.word_ids(batch_index=window_idx)
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    label_id = pred_ids[token_idx].item()
                    label_prob = probs[token_idx, label_id].item()
                    
                    if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                        token_predictions[word_id] = (label_id, label_prob)
        
        # Aggregate predictions per-line using majority voting
        from collections import Counter
        line_predictions = {}  # line_idx -> {labels: Counter, total_conf: float, count: int}
        
        for word_id, pred in token_predictions.items():
            if word_id >= len(word_to_line):
                continue
            
            line_idx = word_to_line[word_id]
            raw_label = model.config.id2label.get(pred[0], "UNKNOWN")
            label = raw_label.lower().replace("-", "_")
            confidence = pred[1]
            
            if line_idx not in line_predictions:
                line_predictions[line_idx] = {
                    'labels': Counter(),
                    'total_conf': 0.0,
                    'count': 0
                }
            
            line_predictions[line_idx]['labels'][label] += 1
            line_predictions[line_idx]['total_conf'] += confidence
            line_predictions[line_idx]['count'] += 1
        
        # Build final predictions per line
        predictions = []
        for line_idx in range(len(merged_lines)):
            line = merged_lines[line_idx]
            
            if line_idx in line_predictions:
                lp = line_predictions[line_idx]
                # Majority vote for label
                majority_label = lp['labels'].most_common(1)[0][0]
                avg_confidence = lp['total_conf'] / lp['count'] if lp['count'] > 0 else 0.0
            else:
                majority_label = 'unknown'
                avg_confidence = 0.0
            
            predictions.append({
                'text': line['text'],
                'bbox': line['bbox'],  # In 300 DPI
                'label': majority_label,
                'confidence': avg_confidence,
                'source': line.get('source', 'aligned'),
                'unit_count': line['unit_count']
            })
        
        print(f"[ClassifyLines] Got {len(predictions)} line predictions")
        
        return jsonify({
            'success': True,
            'page': page,
            'predictions': predictions,
            'line_count': len(merged_lines),
            'original_unit_count': original_unit_count,
            'word_count': len(words)
        })
        
    except Exception as e:
        import traceback
        print(f"[ClassifyLines] Error: {traceback.format_exc()}")
        
        # Try to recover CUDA context on error
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if "CUDA" in str(e) or "device-side assert" in str(e):
                print("[ClassifyLines] CUDA error detected. Clearing cache and resetting device...")
                torch.cuda.synchronize()
        
        return jsonify({'success': False, 'error': str(e)}), 500



# =============================================================================
# EXTRACTED LINE-BASED LAYOUTLM CLASSIFICATION
# Uses TRUE line-level input: 1 line = 1 text + 1 bbox
# Source: char_groups from PyMuPDF extraction (before alignment)
# =============================================================================

@classification_bp.route('/classification-api/classify-extracted-lines/<int:doc_id>/<int:page>', methods=['POST'])
def api_classify_extracted_lines(doc_id, page):
    """
    Run LayoutLM classification using TRUE LINE-LEVEL input.
    
    Each line from char_groups is treated as a single "word" by LayoutLM,
    with its own bounding box. No word splitting or aggregation needed.
    
    Request body:
    {
        "lines": [
            {"text": "full line text...", "bbox": [x0, y0, x1, y1]},  // bbox in PDF points
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions": [
            {"text": "...", "bbox": [...], "label": "...", "confidence": 0.95},
            ...
        ]
    }
    """
    from PIL import Image
    import fitz
    import torch
    
    data = request.get_json() or {}
    lines = data.get('lines', [])
    
    # Get document
    doc = TestingDokumen.query.get_or_404(doc_id)
    pdf_path = doc.testing_dokumen_path
    
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'}), 404
    
    try:
        # Open PDF and get page
        pdf_doc = fitz.open(pdf_path)
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            return jsonify({'success': False, 'error': f'Invalid page number: {page}'}), 400
        
        pdf_page = pdf_doc[page - 1]
        page_width = pdf_page.rect.width
        page_height = pdf_page.rect.height
        
        # Render page to image (300 DPI)
        scale = 300 / 72
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Convert to PIL Image
        import io
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pdf_doc.close()
        
        # Filter valid lines
        valid_lines = []
        for line in lines:
            text = line.get('text', '') or ''
            bbox = line.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox and len(bbox) >= 4:
                valid_lines.append({
                    'text': text.strip(),
                    'bbox': bbox  # In PDF points
                })
        
        print(f"[ExtractedLines] Page {page}: Received {len(lines)} lines, {len(valid_lines)} valid")
        
        if not valid_lines:
            return jsonify({
                'success': True,
                'predictions': [],
                'message': 'No valid lines to classify',
                'line_count': 0
            })
        
        # Sort lines by reading order (top to bottom, left to right)
        valid_lines.sort(key=lambda l: (l['bbox'][1], l['bbox'][0]))
        
        # DPI conversion factor
        dpi_scale = 300 / 72
        
        # Image size in 300 DPI pixels
        img_width = page_width * dpi_scale
        img_height = page_height * dpi_scale
        
        # Scale factors for normalization (from 300 DPI pixels to 0-1000)
        scale_x = 1000 / img_width
        scale_y = 1000 / img_height
        
        # Prepare words and boxes for LayoutLM
        # KEY DIFFERENCE: Each LINE is treated as a single "word" 
        # This gives true line-level classification without aggregation
        words = []
        boxes = []
        original_bboxes_300dpi = []  # Store 300 DPI bboxes for response
        
        for i, line in enumerate(valid_lines):
            # Line text becomes a single "word"
            words.append(line['text'])
            
            # Convert bbox from PDF points to 300 DPI
            bbox_pdf = line['bbox']
            bbox_300dpi = [
                bbox_pdf[0] * dpi_scale,
                bbox_pdf[1] * dpi_scale,
                bbox_pdf[2] * dpi_scale,
                bbox_pdf[3] * dpi_scale
            ]
            original_bboxes_300dpi.append(bbox_300dpi)
            
            # Normalized bbox for model (0-1000), clamped to valid range
            normalized_bbox = clamp_bbox([
                bbox_300dpi[0] * scale_x,
                bbox_300dpi[1] * scale_y,
                bbox_300dpi[2] * scale_x,
                bbox_300dpi[3] * scale_y
            ])
            boxes.append(normalized_bbox)
            
            # Debug: show first 3 lines
            if i < 3:
                print(f"  Line {i}: \"{line['text'][:50]}...\" bbox_norm={normalized_bbox}")
        
        print(f"[ExtractedLines] Sending {len(words)} lines to LayoutLM")
        
        # Run LayoutLM
        from layoutlm_processor import load_model
        model, processor = load_model()
        device = model.device
        
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        
        # Move to device
        for key in ("input_ids", "attention_mask", "bbox"):
            if key in encoding and isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(device)
        
        pv = encoding.get("pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_tensor = pv.to(device)
        elif isinstance(pv, list) and len(pv) > 0:
            pixel_values_tensor = pv[0].unsqueeze(0).to(device)
        else:
            return jsonify({'success': False, 'error': 'Failed to process pixel values'}), 500
        
        num_windows = len(encoding["input_ids"])
        token_predictions = {}
        
        for window_idx in range(num_windows):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"][window_idx:window_idx+1],
                    attention_mask=encoding["attention_mask"][window_idx:window_idx+1],
                    bbox=encoding["bbox"][window_idx:window_idx+1],
                    pixel_values=pixel_values_tensor,
                )
            
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)
            word_ids = encoding.word_ids(batch_index=window_idx)
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    label_id = pred_ids[token_idx].item()
                    label_prob = probs[token_idx, label_id].item()
                    
                    # For sliding windows, keep the highest confidence prediction
                    if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                        token_predictions[word_id] = (label_id, label_prob)
        
        # Build predictions - each word_id corresponds to one line
        predictions = []
        for word_id in range(len(valid_lines)):
            if word_id in token_predictions:
                pred = token_predictions[word_id]
                raw_label = model.config.id2label.get(pred[0], "UNKNOWN")
                label = raw_label.lower().replace("-", "_")
                confidence = pred[1]
            else:
                label = 'unknown'
                confidence = 0.0
            
            predictions.append({
                'text': valid_lines[word_id]['text'],
                'bbox': original_bboxes_300dpi[word_id],  # Return in 300 DPI for canvas display
                'bbox_pdf': valid_lines[word_id]['bbox'],  # Also include original PDF points
                'label': label,
                'confidence': confidence
            })
        
        print(f"[ExtractedLines] Got {len(predictions)} line predictions")
        
        return jsonify({
            'success': True,
            'page': page,
            'predictions': predictions,
            'line_count': len(valid_lines)
        })
        
    except Exception as e:
        import traceback
        print(f"[ExtractedLines] Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


# =============================================================================
# GROUP-BASED LAYOUTLM CLASSIFICATION WITH ALIGNMENT GROUPING
# Uses individual extraction groups (not line-merged)
# Output grouped by aligned OpenXML elements
# =============================================================================

@classification_bp.route('/classification-api/classify-groups/<int:doc_id>/<int:page>', methods=['POST'])
def api_classify_groups(doc_id, page):
    """
    Run LayoutLM classification on individual extraction groups.
    Results are grouped by aligned OpenXML elements.
    
    Request body:
    {
        "extraction_groups": [
            {"text": "...", "bbox": [x0, y0, x1, y1]},  // bbox in PDF points
            ...
        ],
        "alignments": [
            {
                "element_id": 123,
                "element_sequence": 45,
                "element_type": "paragraph",
                "matched_pdf_units": [
                    {"bbox": [x0, y0, x1, y1], "text": "..."},
                    ...
                ]
            },
            ...
        ]
    }
    
    Returns:
    {
        "success": true,
        "predictions_by_alignment": [
            {
                "element_id": 123,
                "element_sequence": 45,
                "element_type": "paragraph",
                "predictions": [
                    {"text": "...", "bbox": [...], "label": "...", "confidence": 0.95},
                    ...
                ],
                "majority_label": "paragraph",
                "avg_confidence": 0.85
            },
            ...
        ],
        "unaligned_predictions": [...]  // Groups that didn't match any alignment
    }
    """
    from PIL import Image
    import fitz
    import torch
    from collections import Counter
    
    data = request.get_json() or {}
    extraction_groups = data.get('extraction_groups', [])
    alignments = data.get('alignments', [])
    
    print(f"[ClassifyGroups] Received {len(extraction_groups)} extraction_groups, {len(alignments)} alignments")
    
    # Get document
    doc = TestingDokumen.query.get_or_404(doc_id)
    pdf_path = doc.testing_dokumen_path
    
    if not os.path.exists(pdf_path):
        return jsonify({'success': False, 'error': 'PDF file not found'}), 404
    
    try:
        # Open PDF and get page
        pdf_doc = fitz.open(pdf_path)
        if page < 1 or page > len(pdf_doc):
            pdf_doc.close()
            return jsonify({'success': False, 'error': f'Invalid page number: {page}'}), 400
        
        pdf_page = pdf_doc[page - 1]
        page_width = pdf_page.rect.width
        page_height = pdf_page.rect.height
        
        # Render page to image (300 DPI)
        scale = 300 / 72
        pix = pdf_page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        
        # Convert to PIL Image
        import io
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        pdf_doc.close()
        
        # DPI conversion factor
        dpi_scale = 300 / 72
        
        # Filter valid groups
        valid_groups = []
        for i, group in enumerate(extraction_groups):
            text = group.get('text', '') or ''
            bbox = group.get('bbox', [0, 0, 0, 0])
            if text.strip() and bbox and len(bbox) >= 4:
                valid_groups.append({
                    'idx': i,
                    'text': text.strip(),
                    'bbox_pdf': bbox,  # Original PDF points
                    'bbox_300dpi': [b * dpi_scale for b in bbox]  # 300 DPI
                })
        
        print(f"[ClassifyGroups] Page {page}: Received {len(extraction_groups)} groups, {len(valid_groups)} valid")
        
        if not valid_groups:
            return jsonify({
                'success': True,
                'predictions_by_alignment': [],
                'unaligned_predictions': [],
                'message': 'No valid groups to classify',
                'group_count': 0
            })
        
        # Sort groups by reading order
        valid_groups.sort(key=lambda g: (g['bbox_pdf'][1], g['bbox_pdf'][0]))
        
        # Image size in 300 DPI pixels
        img_width = page_width * dpi_scale
        img_height = page_height * dpi_scale
        
        # Scale factors for normalization
        scale_x = 1000 / img_width
        scale_y = 1000 / img_height
        
        # Prepare words and boxes for LayoutLM
        # Each group is treated as a single "word"
        words = []
        boxes = []
        
        for group in valid_groups:
            words.append(group['text'])
            
            bbox_300dpi = group['bbox_300dpi']
            normalized_bbox = clamp_bbox([
                bbox_300dpi[0] * scale_x,
                bbox_300dpi[1] * scale_y,
                bbox_300dpi[2] * scale_x,
                bbox_300dpi[3] * scale_y
            ])
            boxes.append(normalized_bbox)
        
        print(f"[ClassifyGroups] Sending {len(words)} groups to LayoutLM")
        
        # Clear CUDA cache before inference to prevent OOM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f"[ClassifyGroups] CUDA cache cleared. Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB allocated")
        
        # Run LayoutLM
        from layoutlm_processor import load_model
        model, processor = load_model()
        device = model.device
        
        encoding = processor(
            image,
            words,
            boxes=boxes,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True
        )
        
        # Move to device
        for key in ("input_ids", "attention_mask", "bbox"):
            if key in encoding and isinstance(encoding[key], torch.Tensor):
                encoding[key] = encoding[key].to(device)
        
        pv = encoding.get("pixel_values")
        if isinstance(pv, torch.Tensor):
            pixel_values_tensor = pv.to(device)
        elif isinstance(pv, list) and len(pv) > 0:
            pixel_values_tensor = pv[0].unsqueeze(0).to(device)
        else:
            return jsonify({'success': False, 'error': 'Failed to process pixel values'}), 500
        
        num_windows = len(encoding["input_ids"])
        token_predictions = {}
        
        for window_idx in range(num_windows):
            with torch.no_grad():
                outputs = model(
                    input_ids=encoding["input_ids"][window_idx:window_idx+1],
                    attention_mask=encoding["attention_mask"][window_idx:window_idx+1],
                    bbox=encoding["bbox"][window_idx:window_idx+1],
                    pixel_values=pixel_values_tensor,
                )
            
            logits = outputs.logits[0]
            probs = torch.softmax(logits, dim=-1)
            pred_ids = logits.argmax(-1)
            word_ids = encoding.word_ids(batch_index=window_idx)
            
            for token_idx, word_id in enumerate(word_ids):
                if word_id is not None:
                    label_id = pred_ids[token_idx].item()
                    label_prob = probs[token_idx, label_id].item()
                    
                    if word_id not in token_predictions or label_prob > token_predictions[word_id][1]:
                        token_predictions[word_id] = (label_id, label_prob)
        
        # Build predictions for each group
        group_predictions = []
        for word_id, group in enumerate(valid_groups):
            if word_id in token_predictions:
                pred = token_predictions[word_id]
                raw_label = model.config.id2label.get(pred[0], "UNKNOWN")
                label = raw_label.lower().replace("-", "_")
                confidence = pred[1]
            else:
                label = 'unknown'
                confidence = 0.0
            
            group_predictions.append({
                'idx': group['idx'],
                'text': group['text'],
                'bbox': group['bbox_300dpi'],
                'bbox_pdf': group['bbox_pdf'],
                'label': label,
                'confidence': confidence
            })
        
        # Now group predictions by alignment
        # Build a lookup from bbox to group prediction
        def bbox_overlap(bbox1, bbox2, threshold=0.3):
            """Check if two bboxes overlap significantly."""
            x0 = max(bbox1[0], bbox2[0])
            y0 = max(bbox1[1], bbox2[1])
            x1 = min(bbox1[2], bbox2[2])
            y1 = min(bbox1[3], bbox2[3])
            
            if x0 >= x1 or y0 >= y1:
                return False
            
            intersection = (x1 - x0) * (y1 - y0)
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            if area1 <= 0 or area2 <= 0:
                return False
            
            overlap_ratio = intersection / min(area1, area2)
            return overlap_ratio >= threshold
        
        matched_group_indices = set()
        predictions_by_alignment = []
        
        # Debug: log first alignment and first few groups
        if alignments and len(alignments) > 0:
            first_align = alignments[0]
            print(f"[ClassifyGroups] First alignment: element_id={first_align.get('element_id')}, "
                  f"matched_units={len(first_align.get('matched_pdf_units', []))}")
            if first_align.get('matched_pdf_units'):
                first_unit = first_align['matched_pdf_units'][0]
                print(f"  First unit bbox: {first_unit.get('bbox')}")
        
        if group_predictions and len(group_predictions) > 0:
            print(f"[ClassifyGroups] First group: bbox_300dpi={group_predictions[0]['bbox']}, bbox_pdf={group_predictions[0]['bbox_pdf']}")
        
        debug_first_comparison = True  # Only log first comparison
        
        for alignment in alignments:
            element_id = alignment.get('element_id')
            element_sequence = alignment.get('element_sequence')
            element_type = alignment.get('element_type', 'paragraph')
            matched_pdf_units = alignment.get('matched_pdf_units', [])
            
            # Find group predictions that match this alignment's bboxes
            alignment_predictions = []
            
            for pdf_unit in matched_pdf_units:
                unit_bbox = pdf_unit.get('bbox', [0, 0, 0, 0])
                
                # Debug: log first unit bbox
                if debug_first_comparison and unit_bbox:
                    print(f"[ClassifyGroups] Alignment unit_bbox (raw from frontend): {unit_bbox}")
                
                for gp in group_predictions:
                    if gp['idx'] in matched_group_indices:
                        continue
                    
                    # Try comparing directly first (both might be in same unit)
                    # Don't convert - compare raw unit_bbox with bbox_pdf
                    if debug_first_comparison:
                        print(f"[ClassifyGroups] Comparing: unit_bbox={unit_bbox} vs gp.bbox_pdf={gp['bbox_pdf']}")
                        debug_first_comparison = False
                    
                    # Try multiple comparison strategies
                    if bbox_overlap(unit_bbox, gp['bbox_pdf'], threshold=0.1):  # Compare directly
                        alignment_predictions.append(gp)
                        matched_group_indices.add(gp['idx'])
            
            # Calculate majority label and average confidence
            if alignment_predictions:
                labels = Counter(p['label'] for p in alignment_predictions)
                majority_label = labels.most_common(1)[0][0]
                avg_confidence = sum(p['confidence'] for p in alignment_predictions) / len(alignment_predictions)
            else:
                majority_label = 'unknown'
                avg_confidence = 0.0
            
            predictions_by_alignment.append({
                'element_id': element_id,
                'element_sequence': element_sequence,
                'element_type': element_type,
                'predictions': alignment_predictions,
                'majority_label': majority_label,
                'avg_confidence': avg_confidence,
                'prediction_count': len(alignment_predictions)
            })
        
        # Collect unaligned predictions
        unaligned_predictions = [gp for gp in group_predictions if gp['idx'] not in matched_group_indices]
        
        print(f"[ClassifyGroups] Got {len(group_predictions)} predictions, "
              f"{len(predictions_by_alignment)} alignments, {len(unaligned_predictions)} unaligned")
        
        return jsonify({
            'success': True,
            'page': page,
            'predictions_by_alignment': predictions_by_alignment,
            'unaligned_predictions': unaligned_predictions,
            'group_count': len(valid_groups),
            'alignment_count': len(alignments)
        })
        
    except Exception as e:
        import traceback
        print(f"[ClassifyGroups] Error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500

