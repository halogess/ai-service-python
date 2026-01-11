"""Table cell processing utilities"""

import fitz
from .bbox_utils import merge_bboxes_token_level
from .pdf_extractor import extract_pdf_table_cells
from .table_matcher import match_docx_cell_to_pdf_cell


def process_table_cell(cell_idx, words, cell_content, page_num, elem_id, 
                       pdf_path, pymupdf_table_cache, used_pdf_cells, is_image_only=False, table_cell_images=None):
    """Process single table cell"""
    # Ensure cell_content is list
    if isinstance(cell_content, str):
        cell_text = cell_content
        cell_content = [{'type': 'text', 'value': cell_text}] if cell_text else []
    else:
        # Reconstruct full text from list
        texts = [item['value'] for item in cell_content if item.get('type') == 'text']
        cell_text = " ".join(texts)

    matched_text = " ".join(w["text"] for w in words)
    if not isinstance(cell_text, str) or not cell_text or not cell_text.strip():
        # Fallback to matched text if no docx content
        cell_text = matched_text

    x0, y0, x1, y1 = 0, 0, 0, 0
    
    if words:
        x0 = min(w["bbox"]["x0"] for w in words)
        y0 = min(w["bbox"]["y0"] for w in words)
        x1 = max(w["bbox"]["x1"] for w in words)
        y1 = max(w["bbox"]["y1"] for w in words)
        page_num = words[0]["page"]

    if page_num not in pymupdf_table_cache:
        pymupdf_pdf_doc = fitz.open(pdf_path)
        pymupdf_table_cache[page_num] = extract_pdf_table_cells(pymupdf_pdf_doc, page_num)
        pymupdf_pdf_doc.close()
    
    pdf_tables = pymupdf_table_cache[page_num]
    pdf_match_found = False
    
    for t_idx, pdf_table in enumerate(pdf_tables):
        table_used_indices = {c for (p, t, c) in used_pdf_cells if p == page_num and t == t_idx}
        
        pdf_cell_idx = match_docx_cell_to_pdf_cell(
            cell_text, 
            pdf_table['cells'], 
            pdf_table['cell_texts'],
            exclude_indices=table_used_indices
        )
        
        if pdf_cell_idx is not None:
            used_pdf_cells.add((page_num, t_idx, pdf_cell_idx))
            
            pdf_cell_bbox = pdf_table['cells'][pdf_cell_idx]
            if pdf_cell_bbox:
                x0, y0, x1, y1 = pdf_cell_bbox
                pdf_match_found = True
                break
    
    # For image-only cells, create placeholder even without words
    if not words and not pdf_match_found and not is_image_only:
        return None
    
    merged_segments = merge_bboxes_token_level(words, is_formula=False)

    result = {
        "text": cell_text,
        "matched_text": matched_text,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "bboxes": merged_segments,
        "page": page_num,
        "element_id": f"{elem_id}_cell_{cell_idx}",
        "parent_element_id": elem_id,
        "confidence": 1.0 if words or pdf_match_found else 0.0,
        "before_align_bboxes": [w["bbox"] for w in words],
        "words": words,
        "is_table_cell": True
    }
    
    if is_image_only:
        result['is_image_only_cell'] = True
    
    # Build content_items for frontend display with granular alignment
    content_items = []
    content_bboxes = []
    
    # Sort words by Y then X for sequential matching
    sorted_words = sorted(words, key=lambda w: (w['bbox']['y0'], w['bbox']['x0']))
    current_word_idx = 0
    
    if cell_content:
        for item in cell_content:
            item_type = item.get('type')
            
            if item_type == 'text':
                item_text = item.get('value', '').strip()
                if item_text:
                    content_items.append({'type': 'text', 'value': item_text})
                    
                    # Try to find corresponding tokens for this item
                    item_tokens = []
                    item_text_len = len(item_text.replace(" ", ""))
                    collected_len = 0
                    
                    # Consume tokens until length matches (heuristic)
                    while current_word_idx < len(sorted_words):
                        w = sorted_words[current_word_idx]
                        w_text = w['text']
                        item_tokens.append(w)
                        collected_len += len(w_text)
                        current_word_idx += 1
                        
                        # Use a fuzzy length check or simple heuristic
                        # Ideally use SequenceMatcher but sequential consuming usually works for cells
                        if collected_len >= item_text_len * 0.8: # Threshold
                             break
                    
                    # Calculate item bbox from tokens
                    if item_tokens:
                        ix0 = min(w["bbox"]["x0"] for w in item_tokens)
                        iy0 = min(w["bbox"]["y0"] for w in item_tokens)
                        ix1 = max(w["bbox"]["x1"] for w in item_tokens)
                        iy1 = max(w["bbox"]["y1"] for w in item_tokens)
                        content_bboxes.append({'x0': ix0, 'y0': iy0, 'x1': ix1, 'y1': iy1})
                    else:
                        content_bboxes.append(None)
                        
            elif item_type == 'image':
                 # Placeholder, will be filled by image aligner logic below or external
                 # But we keep it in the list to maintain order
                 # If this image corresponds to an already aligned image from table_cell_images
                 # we might need to link them.
                 # For now, just add structure.
                 content_items.append({'type': 'image'})
                 content_bboxes.append(None)
            
            elif item_type == 'shape':
                 if 'content' in item:
                     # Flatten shape content for now or handle nested?
                     # Simplifying: Add shape as one item with text
                     shape_content = item.get('content', [])
                     shape_text = " ".join(s['value'] for s in shape_content if s['type'] == 'text')
                     content_items.append({'type': 'shape', 'value': shape_text, 'content': shape_content})
                     content_bboxes.append(None) # TODO: Shape alignment

    else:
         # Fallback for empty/legacy
         if cell_text and cell_text.strip():
            content_items.append({'type': 'text', 'value': cell_text})
            if words:
                 content_bboxes.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
            else:
                 content_bboxes.append(None)

    # Add images if available (merging logic needs to generally match content list image placeholders)
    # If content_list has images, we try to fill their bboxes using table_cell_images
    if table_cell_images:
        cell_images = [(e_id, c_idx, rId) for (e_id, c_idx, rId) in table_cell_images if e_id == elem_id and c_idx == cell_idx]
        
        # If we have explicit image placeholders in content_items
        image_placeholders = [i for i, item in enumerate(content_items) if item['type'] == 'image']
        
        for i, (_, _, rId) in enumerate(cell_images):
            if i < len(image_placeholders):
                idx = image_placeholders[i]
                content_items[idx]['rId'] = rId
                # Bbox will be filled by image_aligner later
            else:
                # Extra images defined in table_cell_images but not in extracted content?
                content_items.append({'type': 'image', 'rId': rId})
                content_bboxes.append(None)
    
    if content_items:
        result['content_items'] = content_items
        result['content_bboxes'] = content_bboxes
    
    return result

