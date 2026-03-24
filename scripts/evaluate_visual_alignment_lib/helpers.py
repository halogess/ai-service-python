from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz
from rapidfuzz import fuzz

from services.alignment_service import AlignmentService
from utils.cross_page_claims import analyze_cross_page_entries

ALIGNER = AlignmentService()
CAPTION_TEXT_RE = re.compile(
    r"^\s*(?:gambar|figure|fig\.?|grafik|graph|chart|tabel|table)\s*\d",
    re.IGNORECASE,
)
ACTIONABLE_ORPHAN_EXCLUDED_LABELS = {"page_header", "page_footer"}

def normalize_text(value: Optional[str]) -> str:
    return ALIGNER._normalize_text(value or "")

def get_visual_label(row: dict) -> str:
    return str(row.get("label") or "").strip().lower()

def is_actionable_orphan_label(label: Optional[str]) -> bool:
    return str(label or "").strip().lower() not in ACTIONABLE_ORPHAN_EXCLUDED_LABELS

def is_caption_like_text(text: Optional[str]) -> bool:
    if not text:
        return False
    return bool(CAPTION_TEXT_RE.match(str(text).strip()))

def bbox_x_overlap_ratio(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    left = max(float(a[0]), float(b[0]))
    right = min(float(a[2]), float(b[2]))
    if right <= left:
        return 0.0
    width_a = max(0.0, float(a[2]) - float(a[0]))
    width_b = max(0.0, float(b[2]) - float(b[0]))
    min_width = min(width_a, width_b)
    if min_width <= 0:
        return 0.0
    return (right - left) / min_width

def is_valid_same_page_chart_caption_pair(rows: List[dict]) -> bool:
    if len(rows) != 2:
        return False
    picture_rows = [row for row in rows if get_visual_label(row) == "picture"]
    caption_rows = [
        row for row in rows
        if get_visual_label(row) == "caption" or is_caption_like_text(row.get("text"))
    ]
    if len(picture_rows) != 1 or len(caption_rows) != 1:
        return False

    picture_row = picture_rows[0]
    caption_row = caption_rows[0]
    picture_bbox = picture_row.get("bbox")
    caption_bbox = caption_row.get("bbox")
    if not picture_bbox or not caption_bbox:
        return False

    gap = float(caption_bbox[1]) - float(picture_bbox[3])
    if gap < -4 or gap > 80:
        return False
    if bbox_x_overlap_ratio(picture_bbox, caption_bbox) < 0.15:
        return False
    return True

def is_table_like_row(row: dict) -> bool:
    if not row:
        return False
    if get_visual_label(row) == "table":
        return True
    if row.get("has_table_units"):
        return True
    element_type = str(row.get("element_type") or "").strip().lower()
    return "table" in element_type

def is_valid_same_page_table_claim_set(rows: List[dict]) -> bool:
    if len(rows) <= 1:
        return False
    return all(is_table_like_row(row) for row in rows)

def parse_json_tree(raw_value):
    return ALIGNER._parse_json_tree(raw_value)

def extract_openxml_text(raw_value) -> str:
    return ALIGNER._extract_text_from_json_tree(parse_json_tree(raw_value))

def json_tree_has_visual_bearing_content(raw_value) -> bool:
    tree = parse_json_tree(raw_value)
    found = False

    def walk(node):
        nonlocal found
        if found:
            return
        if isinstance(node, dict):
            node_type = str(node.get("type") or "").strip().lower()
            if node_type in {"image", "chart", "table", "drawing"}:
                found = True
                return
            if node_type == "text":
                raw_value = node.get("value")
                if raw_value is None:
                    raw_value = node.get("text")
                if raw_value is None:
                    raw_value = node.get("t")
                value = normalize_text(str(raw_value or "").strip())
                if value:
                    found = True
                    return
            for key, value in node.items():
                if key in {"type", "value", "text", "t"}:
                    continue
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, str):
            if normalize_text(node):
                found = True

    walk(tree)
    return found

def bbox_area(bbox: Optional[List[float]]) -> float:
    if not bbox or len(bbox) < 4:
        return 0.0
    width = max(0.0, float(bbox[2]) - float(bbox[0]))
    height = max(0.0, float(bbox[3]) - float(bbox[1]))
    return width * height

def merge_bboxes(bboxes: Iterable[Optional[List[float]]]) -> Optional[List[float]]:
    valid = [bbox for bbox in bboxes if bbox and len(bbox) >= 4]
    if not valid:
        return None
    return [
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    ]

def intersection_area(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return (x1 - x0) * (y1 - y0)

def center_in_bbox(inner_bbox: List[float], outer_bbox: List[float]) -> bool:
    cx = (inner_bbox[0] + inner_bbox[2]) / 2
    cy = (inner_bbox[1] + inner_bbox[3]) / 2
    return outer_bbox[0] <= cx <= outer_bbox[2] and outer_bbox[1] <= cy <= outer_bbox[3]

def word_overlap_ratio(word_bbox: List[float], outer_bbox: List[float]) -> float:
    area = bbox_area(word_bbox)
    if area <= 0:
        return 0.0
    return intersection_area(word_bbox, outer_bbox) / area

def word_is_covered_by_rows(word_bbox: List[float], page_rows: List[dict]) -> bool:
    for row in page_rows or []:
        bbox = row.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        if center_in_bbox(word_bbox, bbox) or word_overlap_ratio(word_bbox, bbox) >= 0.5:
            return True
    return False

def text_similarity(a: str, b: str) -> Optional[float]:
    norm_a = normalize_text(a)
    norm_b = normalize_text(b)
    if not norm_a or not norm_b:
        return None
    ratio = fuzz.ratio(norm_a, norm_b) / 100.0
    partial = fuzz.partial_ratio(norm_a, norm_b) / 100.0
    if norm_a in norm_b or norm_b in norm_a:
        contained = min(len(norm_a), len(norm_b)) / max(len(norm_a), len(norm_b))
        return max(ratio, partial, contained)
    return max(ratio, partial)

def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * q))))
    return ordered[idx]

class PdfWordCache:
    def __init__(self, pdf_path: Path):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.cache: Dict[int, List[dict]] = {}
        self.page_height_cache: Dict[int, float] = {}

    def close(self):
        self.doc.close()

    def get_page_words(self, page_num: int) -> List[dict]:
        if page_num in self.cache:
            return self.cache[page_num]
        page = self.doc[page_num - 1]
        words = []
        for entry in page.get_text("words"):
            x0, y0, x1, y1, text_value = entry[:5]
            if not text_value or not str(text_value).strip():
                continue
            words.append(
                {
                    "bbox": [float(x0), float(y0), float(x1), float(y1)],
                    "text": str(text_value),
                }
            )
        self.cache[page_num] = words
        return words

    def get_page_height(self, page_num: int) -> float:
        if page_num in self.page_height_cache:
            return self.page_height_cache[page_num]
        page = self.doc[page_num - 1]
        height = float(page.rect.height)
        self.page_height_cache[page_num] = height
        return height

def choose_representative_row(rows: List[dict]) -> dict:
    def sort_key(row: dict):
        bbox = row["bbox"]
        center_y = ((bbox[1] + bbox[3]) / 2) if bbox else 0.0
        return (
            row["page"],
            center_y,
            bbox[0] if bbox else 0.0,
            bbox_area(bbox),
            row["dev_id"],
        )

    return min(rows, key=sort_key)

def choose_order_anchor_row(
    rows: List[dict],
    page_heights: Optional[Dict[int, float]] = None,
) -> dict:
    if not rows:
        raise ValueError("rows must not be empty")

    def sort_key(row: dict):
        bbox = row["bbox"]
        center_y = ((bbox[1] + bbox[3]) / 2) if bbox else 0.0
        return (
            row["page"],
            center_y,
            bbox[0] if bbox else 0.0,
            bbox_area(bbox),
            row["dev_id"],
        )

    pages = {row["page"] for row in rows if row.get("page") is not None}
    if len(pages) > 1:
        duplicate_analysis = analyze_element_rows_for_duplicates(rows, page_heights=page_heights)
        if duplicate_analysis.get("is_valid_continuation"):
            return max(rows, key=sort_key)

    return min(rows, key=sort_key)

def group_rows_by_element(rows: List[dict], body_element_ids: set[int]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = defaultdict(list)
    for row in rows:
        element_id = row["element_id"]
        if element_id in body_element_ids:
            grouped[element_id].append(row)
    return grouped

def analyze_element_rows_for_duplicates(
    rows: List[dict],
    page_heights: Optional[Dict[int, float]] = None,
) -> Dict:
    return analyze_cross_page_entries(rows, page_heights=page_heights)

def should_ignore_body_element(
    element_type: Optional[str],
    openxml_text_norm: str,
    raw_value=None,
) -> bool:
    normalized_type = str(element_type or "").strip().lower()
    if normalized_type == "bookmarkend":
        return False
    if openxml_text_norm:
        return False
    if raw_value is not None and json_tree_has_visual_bearing_content(raw_value):
        return False
    return True

def evaluate_element_on_pdf(
    page_rows: List[dict],
    openxml_text: str,
    pdf_words: List[dict],
) -> Tuple[Optional[float], Optional[float], str]:
    merged_bbox = merge_bboxes(row["bbox"] for row in page_rows)
    if not merged_bbox:
        return None, None, ""

    overlapping_words = []
    for word in pdf_words:
        word_bbox = word["bbox"]
        if center_in_bbox(word_bbox, merged_bbox) or word_overlap_ratio(word_bbox, merged_bbox) >= 0.5:
            overlapping_words.append(word)

    pdf_text = " ".join(word["text"] for word in overlapping_words).strip()
    similarity = text_similarity(openxml_text, pdf_text)

    if not overlapping_words:
        return similarity, 0.0, pdf_text

    merged_word_bbox = merge_bboxes(word["bbox"] for word in overlapping_words)
    tightness = 0.0
    merged_area = bbox_area(merged_bbox)
    if merged_area > 0 and merged_word_bbox:
        tightness = intersection_area(merged_bbox, merged_word_bbox) / merged_area

    return similarity, max(0.0, min(1.0, tightness)), pdf_text
