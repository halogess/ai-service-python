"""Fallback alignment for completely unaligned elements"""

from .text_utils import tokenize


def add_unaligned_elements_fallback(final_aligned, elements, element_texts, aligned_parent_ids, log_file=None):
    """Add elements that were not aligned at all as unaligned entries"""
    
    if log_file:
        log_file.write(f"\nChecking for unaligned elements...\n")
    
    for elem in elements:
        elem_id = elem.delemen_id
        if elem_id in aligned_parent_ids:
            continue
        
        # Element tidak ter-align sama sekali
        elem_text = element_texts.get(elem_id, "")
        if not elem_text or not elem_text.strip():
            continue
        
        if log_file:
            log_file.write(f"Element {elem_id} not aligned, text: {elem_text[:50]}...\n")
        
        # Tambahkan sebagai unaligned element dengan bbox dummy
        final_aligned.append({
            "text": elem_text,
            "matched_text": "",
            "bbox": {"x0": 0, "y0": 0, "x1": 0, "y1": 0},
            "bboxes": [],
            "page": 0,
            "element_id": elem_id,
            "confidence": 0.0,
            "before_align_bboxes": [],
            "unaligned": True,
        })


def try_flexible_cell_matching(unmapped_table_cells, pdf_tokens, pdf_bboxes, pdf_pages, used_pdf_tokens, log_file=None):
    """Try flexible sequence matching for unmapped table cells"""
    
    matched_cells = []
    
    for elem_id, cell_idx, cell_text in unmapped_table_cells:
        cell_tokens = tokenize(cell_text)
        if not cell_tokens:
            continue
        
        # Flexible sequence matching: allow skipping up to 2 tokens
        best_match = None
        best_score = 0
        target_page = 0  # Simplified - should get from context
        
        for start_j in range(len(pdf_tokens)):
            if pdf_pages[start_j] != target_page:
                continue
            if start_j in used_pdf_tokens:
                continue
            
            matched_indices = []
            cell_tok_idx = 0
            pdf_tok_idx = start_j
            skip_count = 0
            
            while cell_tok_idx < len(cell_tokens) and pdf_tok_idx < len(pdf_tokens):
                if pdf_pages[pdf_tok_idx] != target_page:
                    break
                
                if pdf_tok_idx in used_pdf_tokens:
                    pdf_tok_idx += 1
                    continue
                
                if pdf_tokens[pdf_tok_idx] == cell_tokens[cell_tok_idx]:
                    # Spatial check
                    if matched_indices:
                        prev_idx = matched_indices[-1]
                        prev_bbox = pdf_bboxes[prev_idx]
                        curr_bbox = pdf_bboxes[pdf_tok_idx]
                        
                        gap_y = max(0, curr_bbox[1] - prev_bbox[3], prev_bbox[1] - curr_bbox[3])
                        
                        if gap_y > 150:
                            break
                    
                    matched_indices.append(pdf_tok_idx)
                    cell_tok_idx += 1
                    pdf_tok_idx += 1
                    skip_count = 0
                else:
                    if skip_count < 2:
                        pdf_tok_idx += 1
                        skip_count += 1
                    else:
                        break
            
            if matched_indices:
                score = len(matched_indices) / len(cell_tokens)
                if score > best_score:
                    best_score = score
                    best_match = matched_indices
        
        if best_match and best_score >= 0.5:
            matched_words = []
            for pdf_idx in best_match:
                bbox = pdf_bboxes[pdf_idx]
                matched_words.append({
                    "text": pdf_tokens[pdf_idx],
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "page": pdf_pages[pdf_idx],
                })
                used_pdf_tokens.add(pdf_idx)
            
            if matched_words:
                from .bbox_utils import merge_bboxes_token_level
                
                x0 = min(w["bbox"]["x0"] for w in matched_words)
                y0 = min(w["bbox"]["y0"] for w in matched_words)
                x1 = max(w["bbox"]["x1"] for w in matched_words)
                y1 = max(w["bbox"]["y1"] for w in matched_words)
                
                merged_segments = merge_bboxes_token_level(matched_words, is_formula=False)
                
                matched_cells.append({
                    "text": cell_text,
                    "matched_text": " ".join(w["text"] for w in matched_words),
                    "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
                    "bboxes": merged_segments,
                    "page": matched_words[0]["page"],
                    "element_id": f"{elem_id}_cell_{cell_idx}",
                    "parent_element_id": elem_id,
                    "confidence": best_score,
                    "before_align_bboxes": [w["bbox"] for w in matched_words],
                })
    
    return matched_cells
