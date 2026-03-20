from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional


DEFAULT_PAGE_HEIGHT = 842.0
DEFAULT_EDGE_MARGIN_RATIO = 0.22
DEFAULT_EDGE_MARGIN_PT = 110.0


def merge_bboxes(bboxes: Iterable[Optional[List[float]]]) -> Optional[List[float]]:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) >= 4]
    if not valid:
        return None
    return [
        min(float(bbox[0]) for bbox in valid),
        min(float(bbox[1]) for bbox in valid),
        max(float(bbox[2]) for bbox in valid),
        max(float(bbox[3]) for bbox in valid),
    ]


def _coerce_int(value) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _edge_margin(page_height: Optional[float], edge_margin_ratio: float, edge_margin_pt: float) -> float:
    height = float(page_height or DEFAULT_PAGE_HEIGHT)
    return max(float(edge_margin_pt), height * float(edge_margin_ratio))


def is_bbox_near_page_top(
    bbox: Optional[List[float]],
    page_height: Optional[float],
    *,
    edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
    edge_margin_pt: float = DEFAULT_EDGE_MARGIN_PT,
) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    return float(bbox[1]) <= _edge_margin(page_height, edge_margin_ratio, edge_margin_pt)


def is_bbox_near_page_bottom(
    bbox: Optional[List[float]],
    page_height: Optional[float],
    *,
    edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
    edge_margin_pt: float = DEFAULT_EDGE_MARGIN_PT,
) -> bool:
    if not bbox or len(bbox) < 4:
        return False
    height = float(page_height or DEFAULT_PAGE_HEIGHT)
    return float(bbox[3]) >= height - _edge_margin(height, edge_margin_ratio, edge_margin_pt)


def analyze_cross_page_entries(
    entries: Iterable[dict],
    page_heights: Optional[Dict[int, float]] = None,
    *,
    default_page_height: float = DEFAULT_PAGE_HEIGHT,
    edge_margin_ratio: float = DEFAULT_EDGE_MARGIN_RATIO,
    edge_margin_pt: float = DEFAULT_EDGE_MARGIN_PT,
) -> Dict:
    rows_by_page: Dict[int, List[dict]] = defaultdict(list)
    for entry in entries or []:
        page = _coerce_int((entry or {}).get("page"))
        bbox = (entry or {}).get("bbox")
        if page is None or not bbox or len(bbox) < 4:
            continue
        rows_by_page[page].append(entry)

    pages = sorted(rows_by_page)
    page_infos: Dict[int, Dict] = {}
    for page in pages:
        bbox = merge_bboxes(row.get("bbox") for row in rows_by_page[page])
        page_height = float((page_heights or {}).get(page) or default_page_height)
        page_infos[page] = {
            "bbox": bbox,
            "page_height": page_height,
            "near_top": is_bbox_near_page_top(
                bbox,
                page_height,
                edge_margin_ratio=edge_margin_ratio,
                edge_margin_pt=edge_margin_pt,
            ),
            "near_bottom": is_bbox_near_page_bottom(
                bbox,
                page_height,
                edge_margin_ratio=edge_margin_ratio,
                edge_margin_pt=edge_margin_pt,
            ),
        }

    if len(pages) <= 1:
        return {
            "pages": pages,
            "page_infos": page_infos,
            "valid_pairs": [],
            "invalid_pairs": [],
            "invalid_pages": [],
            "is_multi_page": False,
            "is_valid_continuation": False,
            "is_invalid_duplicate": False,
        }

    valid_pairs = []
    invalid_pairs = []
    invalid_pages = set()

    for prev_page, current_page in zip(pages, pages[1:]):
        if current_page != prev_page + 1:
            invalid_pairs.append(
                {
                    "pages": [prev_page, current_page],
                    "reason": "page_gap",
                }
            )
            invalid_pages.update((prev_page, current_page))
            continue

        prev_info = page_infos.get(prev_page) or {}
        current_info = page_infos.get(current_page) or {}
        if prev_info.get("near_bottom") and current_info.get("near_top"):
            valid_pairs.append([prev_page, current_page])
            continue

        invalid_pairs.append(
            {
                "pages": [prev_page, current_page],
                "reason": "not_boundary_continuation",
            }
        )
        invalid_pages.update((prev_page, current_page))

    is_valid_continuation = bool(valid_pairs) and not invalid_pairs
    return {
        "pages": pages,
        "page_infos": page_infos,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "invalid_pages": sorted(invalid_pages),
        "is_multi_page": True,
        "is_valid_continuation": is_valid_continuation,
        "is_invalid_duplicate": bool(invalid_pairs),
    }
