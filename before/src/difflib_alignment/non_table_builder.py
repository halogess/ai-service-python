"""Build non-table elements"""

from collections import defaultdict
from .bbox_utils import merge_bboxes_token_level
from .formula_utils import has_formula_in_tree
from .formula_expander import expand_formula_bbox


def build_non_table_elements(elem_id, cell_groups, element_texts, elements=None, pdf_on_page=None):
    """Build non-table elements (paragraphs, etc)"""
    final_aligned = []
    
    words = cell_groups.get(-1, [])
    if not words:
        return final_aligned
    
    # Check if this is formula
    is_formula = False
    if elements:
        elem = next((e for e in elements if e.delemen_id == elem_id), None)
        if elem:
            is_formula = elem.delemen_type == 'math' or has_formula_in_tree(elem.delemen_json_tree)
    
    # Group by page
    page_groups = defaultdict(list)
    for w in words:
        page_groups[w['page']].append(w)
    
    # Build element per page
    for page_num in sorted(page_groups.keys()):
        page_words = page_groups[page_num]
        
        if not page_words:  # Skip if no words on this page
            continue
        
        x0 = min(w["bbox"]["x0"] for w in page_words)
        y0 = min(w["bbox"]["y0"] for w in page_words)
        x1 = max(w["bbox"]["x1"] for w in page_words)
        y1 = max(w["bbox"]["y1"] for w in page_words)
        
        # Apply formula expansion if needed
        if is_formula and len(page_words) > 1 and pdf_on_page:
            aligned_indices = set(w.get('pdf_index') for w in page_words if 'pdf_index' in w)
            x0, y0, x1, y1 = expand_formula_bbox(x0, y0, x1, y1, page_words, pdf_on_page, page_num, aligned_indices)

        merged_segments = merge_bboxes_token_level(page_words, is_formula=is_formula)
        
        final_elem_id = elem_id if len(page_groups) == 1 else f"{elem_id}_page_{page_num}"
        
        elem_text = element_texts.get(elem_id, "")
        matched_text = " ".join(w["text"] for w in page_words)
        
        if not elem_text or not elem_text.strip():
            elem_text = matched_text
        
        final_aligned.append({
            "text": elem_text,
            "matched_text": matched_text,
            "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
            "bboxes": merged_segments,
            "page": page_num,
            "element_id": final_elem_id,
            "confidence": 1.0,
            "before_align_bboxes": [w["bbox"] for w in page_words],
            "words": page_words,
            "is_formula": is_formula,
        })
    
    return final_aligned
