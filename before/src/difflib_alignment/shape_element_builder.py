"""Build shape elements"""

from .bbox_utils import merge_bboxes_token_level


def build_shape_container(elem_id, sorted_shape_indices, cell_groups, element_texts):
    """Build shape container element"""
    if not sorted_shape_indices:
        return None
    
    first_shape_words = cell_groups[sorted_shape_indices[0]]
    all_shape_words = []
    for shape_idx in sorted_shape_indices:
        all_shape_words.extend(cell_groups[shape_idx])
    
    if not all_shape_words:
        return None
    
    container_x0 = min(w["bbox"]["x0"] for w in all_shape_words)
    container_y0 = min(w["bbox"]["y0"] for w in all_shape_words)
    container_x1 = max(w["bbox"]["x1"] for w in all_shape_words)
    container_y1 = max(w["bbox"]["y1"] for w in all_shape_words)
    
    return {
        "text": element_texts.get(elem_id, ""),
        "matched_text": "",
        "bbox": {"x0": container_x0, "y0": container_y0, "x1": container_x1, "y1": container_y1},
        "bboxes": [],
        "page": first_shape_words[0]["page"],
        "element_id": elem_id,
        "confidence": 1.0,
        "before_align_bboxes": [],
        "is_shape_container": True,
        "children": [],
    }


def build_single_shape_element(elem_id, shape_idx, words, element_shape_data):
    """Build single shape element"""
    if not words:
        return None
    
    shape_text = ""
    shape_item = None
    if elem_id in element_shape_data and shape_idx in element_shape_data[elem_id]:
        shape_text, shape_item = element_shape_data[elem_id][shape_idx]
    
    matched_text = " ".join(w["text"] for w in words)
    if not shape_text or not shape_text.strip():
        shape_text = matched_text
    
    x0 = min(w["bbox"]["x0"] for w in words)
    y0 = min(w["bbox"]["y0"] for w in words)
    x1 = max(w["bbox"]["x1"] for w in words)
    y1 = max(w["bbox"]["y1"] for w in words)
    
    merged_segments = merge_bboxes_token_level(words, is_formula=False)
    
    return {
        "text": shape_text,
        "matched_text": matched_text,
        "bbox": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "bboxes": merged_segments,
        "page": words[0]["page"],
        "element_id": f"{elem_id}_shape_{shape_idx}",
        "parent_element_id": elem_id,
        "confidence": 1.0,
        "before_align_bboxes": [w["bbox"] for w in words],
        "words": words,
    }
