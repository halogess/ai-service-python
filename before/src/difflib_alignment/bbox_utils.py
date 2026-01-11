"""Bounding box utilities"""

from .config import DEFAULT_MERGE_X_GAP, DEFAULT_MERGE_Y_OVERLAP, FORMULA_MERGE_Y_THRESHOLD


def merge_bboxes_token_level(items, x_gap=DEFAULT_MERGE_X_GAP, y_overlap_min=DEFAULT_MERGE_Y_OVERLAP, is_formula=False):
    """Merge bbox token-level jadi beberapa segmen"""
    
    def overlap_ratio(a0, a1, b0, b1):
        inter = max(0.0, min(a1, b1) - max(a0, b0))
        denom = min(a1 - a0, b1 - b0) if min(a1 - a0, b1 - b0) > 0 else 1.0
        return inter / denom

    if not items:
        return []

    # Filter items dengan Y terlalu jauh dari median (untuk formula)
    if is_formula and len(items) > 1:
        y_mids = [(item['bbox']['y0'] + item['bbox']['y1']) / 2 for item in items]
        y_median = sorted(y_mids)[len(y_mids) // 2]
        items = [item for item in items 
                if abs((item['bbox']['y0'] + item['bbox']['y1']) / 2 - y_median) <= FORMULA_MERGE_Y_THRESHOLD]
    
    if not items:
        return []

    # Sort stabil
    items = sorted(items, key=lambda w: (w.get("page", 0), w["bbox"]["y0"], w["bbox"]["x0"]))

    merged = []
    cur = None

    for w in items:
        p = w.get("page", 0)
        b = w["bbox"]
        if cur is None:
            cur = {"page": p, "bbox": dict(b)}
            continue

        cb = cur["bbox"]
        same_page = p == cur["page"]
        y_ok = overlap_ratio(cb["y0"], cb["y1"], b["y0"], b["y1"]) >= y_overlap_min
        x_ok = b["x0"] <= cb["x1"] + x_gap

        if same_page and y_ok and x_ok:
            cb["x0"] = min(cb["x0"], b["x0"])
            cb["y0"] = min(cb["y0"], b["y0"])
            cb["x1"] = max(cb["x1"], b["x1"])
            cb["y1"] = max(cb["y1"], b["y1"])
        else:
            merged.append(cur)
            cur = {"page": p, "bbox": dict(b)}

    if cur is not None:
        merged.append(cur)
    return merged
