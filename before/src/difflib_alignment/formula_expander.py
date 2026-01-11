"""Formula bbox expansion utilities"""

from .config import (
    FORMULA_EXPAND_VERT_TOP_RATIO,
    FORMULA_EXPAND_VERT_BOTTOM_RATIO,
    FORMULA_EXPAND_HORZ_RIGHT_RATIO,
    FORMULA_EXPAND_HORZ_RIGHT_MAX_PX,
    FORMULA_EXPAND_HORZ_LEFT_RATIO,
    FORMULA_EXPAND_HORZ_LEFT_MAX_PX,
    FORMULA_PAGE_MARGIN_LEFT,
    FORMULA_PAGE_MARGIN_RIGHT
)


def expand_formula_bbox(x0, y0, x1, y1, page_words, pdf_on_page, page_num, aligned_indices):
    """Expand formula bbox to capture superscript/subscript/fraction"""
    
    if len(page_words) <= 1:
        return x0, y0, x1, y1
    
    # Calculate average line height
    heights = [w["bbox"]["y1"] - w["bbox"]["y0"] for w in page_words]
    avg_height = sum(heights) / len(heights)
    
    # Expand vertically
    y0 = y0 - avg_height * FORMULA_EXPAND_VERT_TOP_RATIO
    y1 = y1 + avg_height * FORMULA_EXPAND_VERT_BOTTOM_RATIO
    
    # Conditional horizontal expansion
    current_width = x1 - x0
    
    # Right expansion
    max_expand_right = min(current_width * FORMULA_EXPAND_HORZ_RIGHT_RATIO, FORMULA_EXPAND_HORZ_RIGHT_MAX_PX)
    proposed_x1 = min(x1 + max_expand_right, FORMULA_PAGE_MARGIN_RIGHT)
    
    found_extra_right_x = x1
    
    if proposed_x1 > x1:
        for p_idx, p_bbox in pdf_on_page[page_num]:
            if p_idx in aligned_indices:
                continue
            
            p_y_mid = (p_bbox[1] + p_bbox[3]) / 2
            if y0 <= p_y_mid <= y1:
                if x1 < p_bbox[0] < proposed_x1:
                    found_extra_right_x = max(found_extra_right_x, p_bbox[2])
    
    if found_extra_right_x > x1:
        x1 = found_extra_right_x + 2
    
    # Left expansion
    max_expand_left = min(current_width * FORMULA_EXPAND_HORZ_LEFT_RATIO, FORMULA_EXPAND_HORZ_LEFT_MAX_PX)
    proposed_x0 = max(x0 - max_expand_left, FORMULA_PAGE_MARGIN_LEFT)
    
    found_extra_left_x = x0
    
    if proposed_x0 < x0:
        for p_idx, p_bbox in pdf_on_page[page_num]:
            if p_idx in aligned_indices:
                continue
            
            p_y_mid = (p_bbox[1] + p_bbox[3]) / 2
            if y0 <= p_y_mid <= y1:
                if proposed_x0 < p_bbox[2] < x0:
                    found_extra_left_x = min(found_extra_left_x, p_bbox[0])
    
    if found_extra_left_x < x0:
        x0 = found_extra_left_x - 2
    
    return x0, y0, x1, y1
