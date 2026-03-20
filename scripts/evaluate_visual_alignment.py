from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import socket
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import fitz
from dotenv import load_dotenv
from rapidfuzz import fuzz
from sqlalchemy import create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from utils.cross_page_claims import analyze_cross_page_entries


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def configure_env() -> Tuple[str, int, str, str, str]:
    load_dotenv(dotenv_path=REPO_ROOT / ".env")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = int(os.getenv("DB_PORT", "3306"))
    db_name = os.getenv("DB_NAME", "")
    db_user = os.getenv("DB_USER", "")
    db_password = os.getenv("DB_PASSWORD", "")

    if db_host == "host.docker.internal" and port_open("localhost", db_port):
        db_host = "localhost"
        os.environ["DB_HOST"] = db_host

    return db_host, db_port, db_name, db_user, db_password


DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD = configure_env()

from services.alignment_service import AlignmentService  # noqa: E402


ALIGNER = AlignmentService()
VOLUME_BASE = Path(os.getenv("VOLUME_BASE_PATH", ""))
CAPTION_TEXT_RE = re.compile(
    r"^\s*(?:gambar|figure|fig\.?|grafik|graph|chart|tabel|table)\s*\d",
    re.IGNORECASE,
)
ACTIONABLE_ORPHAN_EXCLUDED_LABELS = {"page_header", "page_footer"}


REF_QUERIES = {
    "dokumen": text(
        """
        select
            'dokumen' as ref_type,
            dokumen_id as ref_id,
            mhs_nrp as owner_key,
            dokumen_filename as filename,
            dokumen_pdf_path as pdf_path
        from dokumen
        order by dokumen_id
        """
    ),
    "bab": text(
        """
        select
            'bab' as ref_type,
            bab_id as ref_id,
            cast(buku_id as char) as owner_key,
            bab_filename as filename,
            bab_pdf_path as pdf_path
        from bab
        order by bab_id
        """
    ),
}

BODY_ELEMENTS_QUERY = text(
    """
    select
        de.delemen_id,
        de.delemen_sequence,
        de.delemen_type,
        de.delemen_json_tree
    from dokumen_elemen de
    join dokumen_part dp on de.dpart_id = dp.dpart_id
    join dokumen_section ds on dp.dsec_id = ds.dsec_id
    where ds.dsec_ref_tipe = :ref_type
      and ds.dsec_ref_id = :ref_id
      and dp.dpart_type = 'body'
    order by de.delemen_sequence, de.delemen_id
    """
)

ALL_ELEMENTS_QUERY = text(
    """
    select de.delemen_id
    from dokumen_elemen de
    join dokumen_part dp on de.dpart_id = dp.dpart_id
    join dokumen_section ds on dp.dsec_id = ds.dsec_id
    where ds.dsec_ref_tipe = :ref_type
      and ds.dsec_ref_id = :ref_id
    """
)

NOTE_IDS_QUERY = text(
    """
    select dnote_id
    from dokumen_note
    where dokumen_id = :ref_id
    """
)

VISUAL_ROWS_QUERY = text(
    """
    select
        dev_id,
        dev_page,
        dokumen_elemen_id,
        dev_bbox_x0,
        dev_bbox_y0,
        dev_bbox_x1,
        dev_bbox_y1,
        dev_label,
        dev_text
    from dokumen_elemen_visual
    where dev_ref_tipe = :ref_type
      and dev_ref_id = :ref_id
    order by dev_page, dev_bbox_y0, dev_bbox_x0, dev_id
    """
)

ORPHAN_VISUAL_QUERY = text(
    """
    select
        v.dev_ref_tipe as ref_type,
        v.dev_ref_id as ref_id,
        count(*) as visual_rows
    from dokumen_elemen_visual v
    left join dokumen d
        on v.dev_ref_tipe = 'dokumen'
       and v.dev_ref_id = d.dokumen_id
    left join bab b
        on v.dev_ref_tipe = 'bab'
       and v.dev_ref_id = b.bab_id
    where v.dev_ref_tipe in ('dokumen', 'bab')
      and (
        (v.dev_ref_tipe = 'dokumen' and d.dokumen_id is null)
        or
        (v.dev_ref_tipe = 'bab' and b.bab_id is null)
      )
    group by v.dev_ref_tipe, v.dev_ref_id
    order by v.dev_ref_tipe, v.dev_ref_id
    """
)

UNSCOPED_VISUAL_QUERY = text(
    """
    select count(*) as rows_count
    from dokumen_elemen_visual
    where dev_ref_tipe is null or dev_ref_id is null
    """
)


def build_engine():
    url = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(url)


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


def parse_json_tree(raw_value):
    return ALIGNER._parse_json_tree(raw_value)


def extract_openxml_text(raw_value) -> str:
    return ALIGNER._extract_text_from_json_tree(parse_json_tree(raw_value))


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


def should_ignore_body_element(element_type: Optional[str], openxml_text_norm: str) -> bool:
    return (element_type == "paragraph" and not openxml_text_norm) or (
        element_type in {"h1", "h2"} and not openxml_text_norm
    )


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


def evaluate_ref(conn, ref_row: dict) -> dict:
    ref_type = str(ref_row["ref_type"])
    ref_id = int(ref_row["ref_id"])
    pdf_rel_path = str(ref_row["pdf_path"] or "")
    pdf_path = (VOLUME_BASE / pdf_rel_path).resolve() if pdf_rel_path else VOLUME_BASE.resolve()
    pdf_cache = None
    page_heights: Dict[int, float] = {}

    body_elements = {}
    body_element_ids = set()
    ignored_empty_paragraph_ids = set()
    ignored_empty_heading_ids = set()
    for row in conn.execute(BODY_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id}):
        mapping = dict(row._mapping)
        element_id = int(mapping["delemen_id"])
        openxml_text = extract_openxml_text(mapping["delemen_json_tree"])
        openxml_text_norm = normalize_text(openxml_text)
        if should_ignore_body_element(mapping["delemen_type"], openxml_text_norm):
            if mapping["delemen_type"] == "paragraph":
                ignored_empty_paragraph_ids.add(element_id)
            elif mapping["delemen_type"] in {"h1", "h2"}:
                ignored_empty_heading_ids.add(element_id)
            continue
        body_elements[element_id] = {
            "sequence": int(mapping["delemen_sequence"]) if mapping["delemen_sequence"] is not None else None,
            "type": mapping["delemen_type"],
            "openxml_text": openxml_text,
            "openxml_text_norm": openxml_text_norm,
        }
        body_element_ids.add(element_id)

    all_element_ids = {
        int(row[0]) for row in conn.execute(ALL_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id})
    }
    note_ids = set()
    if ref_type == "dokumen":
        note_ids = {int(row[0]) for row in conn.execute(NOTE_IDS_QUERY, {"ref_id": ref_id})}

    visual_rows = []
    for row in conn.execute(VISUAL_ROWS_QUERY, {"ref_type": ref_type, "ref_id": ref_id}):
        mapping = dict(row._mapping)
        element_id = mapping["dokumen_elemen_id"]
        if element_id is not None:
            element_id = int(element_id)
        visual_rows.append(
            {
                "dev_id": int(mapping["dev_id"]),
                "page": int(mapping["dev_page"]) if mapping["dev_page"] is not None else None,
                "element_id": element_id,
                "bbox": [
                    float(mapping["dev_bbox_x0"]),
                    float(mapping["dev_bbox_y0"]),
                    float(mapping["dev_bbox_x1"]),
                    float(mapping["dev_bbox_y1"]),
                ],
                "label": mapping["dev_label"],
                "text": mapping["dev_text"] or "",
            }
        )

    if pdf_path.exists() and pdf_path.is_file():
        pdf_cache = PdfWordCache(pdf_path)
        page_heights = {
            row["page"]: pdf_cache.get_page_height(row["page"])
            for row in visual_rows
            if row["page"] is not None
        }

    body_groups = group_rows_by_element(visual_rows, body_element_ids)
    claimed_body_ids = set(body_groups)
    total_body_elements = len(body_elements)
    coverage = (len(claimed_body_ids) / total_body_elements) if total_body_elements else None
    missing_body_ids = body_element_ids - claimed_body_ids
    missing_bookmark_end = sum(
        1 for element_id in missing_body_ids
        if body_elements[element_id]["type"] == "bookmarkEnd"
    )
    missing_non_bookmark = sum(
        1 for element_id in missing_body_ids
        if body_elements[element_id]["type"] != "bookmarkEnd"
    )

    cross_page_duplicates = 0
    valid_cross_page_continuations = 0
    same_page_fragments = 0
    for rows in body_groups.values():
        pages = {row["page"] for row in rows if row["page"] is not None}
        if len(pages) > 1:
            duplicate_analysis = analyze_element_rows_for_duplicates(rows, page_heights=page_heights)
            if duplicate_analysis["is_valid_continuation"]:
                valid_cross_page_continuations += 1
            else:
                cross_page_duplicates += 1
        elif len(rows) > 1:
            if is_valid_same_page_chart_caption_pair(rows):
                continue
            same_page_fragments += 1

    duplicate_rate = (cross_page_duplicates / len(claimed_body_ids)) if claimed_body_ids else 0.0
    fragment_rate = (same_page_fragments / len(claimed_body_ids)) if claimed_body_ids else 0.0

    ordered_body = []
    for element_id, rows in body_groups.items():
        sequence = body_elements[element_id]["sequence"]
        if sequence is None:
            continue
        representative = choose_representative_row(rows)
        bbox = representative["bbox"]
        ordered_body.append(
            (
                sequence,
                representative["page"],
                (bbox[1] + bbox[3]) / 2 if bbox else 0.0,
                bbox[0] if bbox else 0.0,
            )
        )
    ordered_body.sort(key=lambda item: item[0])
    order_violations = 0
    for previous, current in zip(ordered_body, ordered_body[1:]):
        prev_loc = previous[1:]
        curr_loc = current[1:]
        if prev_loc > curr_loc:
            order_violations += 1
    order_consistency = 1.0
    if len(ordered_body) > 1:
        order_consistency = 1.0 - (order_violations / (len(ordered_body) - 1))

    null_claim_rows = sum(1 for row in visual_rows if row["element_id"] is None)
    note_claim_rows = sum(1 for row in visual_rows if row["element_id"] in note_ids)
    foreign_claim_rows = sum(
        1
        for row in visual_rows
        if row["element_id"] is not None
        and row["element_id"] not in all_element_ids
        and row["element_id"] not in note_ids
    )
    total_visual_rows = len(visual_rows)
    null_claim_rate = (null_claim_rows / total_visual_rows) if total_visual_rows else 0.0
    foreign_claim_rate = (foreign_claim_rows / total_visual_rows) if total_visual_rows else 0.0
    orphan_visual_rows = null_claim_rows + foreign_claim_rows
    orphan_visual_rate = (orphan_visual_rows / total_visual_rows) if total_visual_rows else 0.0
    actionable_null_claim_rows = sum(
        1
        for row in visual_rows
        if row["element_id"] is None and is_actionable_orphan_label(row.get("label"))
    )
    actionable_foreign_claim_rows = sum(
        1
        for row in visual_rows
        if row["element_id"] is not None
        and row["element_id"] not in all_element_ids
        and row["element_id"] not in note_ids
        and is_actionable_orphan_label(row.get("label"))
    )
    actionable_orphan_visual_rows = actionable_null_claim_rows + actionable_foreign_claim_rows
    actionable_orphan_visual_rate = (
        actionable_orphan_visual_rows / total_visual_rows
    ) if total_visual_rows else 0.0

    support_scores: List[float] = []
    bbox_tightness_scores: List[float] = []
    support_details = []

    if pdf_cache is not None:
        try:
            for element_id, rows in body_groups.items():
                openxml_text = body_elements[element_id]["openxml_text"]
                if len(body_elements[element_id]["openxml_text_norm"]) < 8:
                    continue

                rows_by_page: Dict[int, List[dict]] = defaultdict(list)
                for row in rows:
                    if row["page"] is not None:
                        rows_by_page[row["page"]].append(row)

                best_support = None
                best_tightness = None
                best_page = None
                for page_num, page_rows in rows_by_page.items():
                    pdf_words = pdf_cache.get_page_words(page_num)
                    similarity, tightness, pdf_text = evaluate_element_on_pdf(
                        page_rows,
                        openxml_text,
                        pdf_words,
                    )
                    if similarity is None:
                        continue
                    if best_support is None or similarity > best_support:
                        best_support = similarity
                        best_tightness = tightness
                        best_page = page_num
                        best_pdf_text = pdf_text

                if best_support is None:
                    continue

                support_scores.append(best_support)
                if best_tightness is not None:
                    bbox_tightness_scores.append(best_tightness)
                support_details.append(
                    {
                        "element_id": element_id,
                        "sequence": body_elements[element_id]["sequence"],
                        "page": best_page,
                        "support": round(best_support, 4),
                        "bbox_tightness": round(best_tightness or 0.0, 4),
                        "pdf_text_preview": (best_pdf_text or "")[:160],
                        "openxml_text_preview": openxml_text[:160],
                    }
                )
        finally:
            pdf_cache.close()

    support_median = median(support_scores) if support_scores else None
    support_p10 = percentile(support_scores, 0.10)
    bbox_tightness_median = median(bbox_tightness_scores) if bbox_tightness_scores else None
    bbox_tightness_p10 = percentile(bbox_tightness_scores, 0.10)

    thresholds = {
        "coverage": 0.90,
        "duplicate_rate": 0.05,
        "order_consistency": 0.95,
        "support_median": 0.70,
        "bbox_tightness_median": 0.40,
        "null_claim_rate": 0.05,
    }
    check_results = {
        "coverage": coverage is not None and coverage >= thresholds["coverage"],
        "duplicate_rate": duplicate_rate <= thresholds["duplicate_rate"],
        "order_consistency": order_consistency >= thresholds["order_consistency"],
        "support_median": support_median is not None and support_median >= thresholds["support_median"],
        "bbox_tightness_median": bbox_tightness_median is not None and bbox_tightness_median >= thresholds["bbox_tightness_median"],
        "null_claim_rate": null_claim_rate <= thresholds["null_claim_rate"],
    }

    score_parts = [
        ("coverage", coverage),
        ("dup", 1.0 - duplicate_rate),
        ("order", order_consistency),
        ("support", support_median),
        ("tightness", bbox_tightness_median),
        ("null", 1.0 - null_claim_rate),
    ]
    weights = {
        "coverage": 0.30,
        "dup": 0.15,
        "order": 0.15,
        "support": 0.20,
        "tightness": 0.10,
        "null": 0.10,
    }
    weighted_total = 0.0
    weight_sum = 0.0
    for key, value in score_parts:
        if value is None:
            continue
        weighted_total += weights[key] * max(0.0, min(1.0, value))
        weight_sum += weights[key]
    overall_score = (weighted_total / weight_sum) if weight_sum > 0 else None

    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "owner_key": ref_row["owner_key"],
        "filename": ref_row["filename"],
        "pdf_path": pdf_rel_path,
        "ignored_empty_paragraphs": len(ignored_empty_paragraph_ids),
        "ignored_empty_headings": len(ignored_empty_heading_ids),
        "total_body_elements": total_body_elements,
        "claimed_body_elements": len(claimed_body_ids),
        "missing_body_elements": max(0, total_body_elements - len(claimed_body_ids)),
        "missing_bookmark_end": missing_bookmark_end,
        "missing_non_bookmark": missing_non_bookmark,
        "coverage": coverage,
        "cross_page_duplicates": cross_page_duplicates,
        "valid_cross_page_continuations": valid_cross_page_continuations,
        "same_page_fragments": same_page_fragments,
        "duplicate_rate": duplicate_rate,
        "fragment_rate": fragment_rate,
        "order_consistency": order_consistency,
        "order_violations": order_violations,
        "support_sample_count": len(support_scores),
        "support_median": support_median,
        "support_p10": support_p10,
        "bbox_tightness_sample_count": len(bbox_tightness_scores),
        "bbox_tightness_median": bbox_tightness_median,
        "bbox_tightness_p10": bbox_tightness_p10,
        "total_visual_rows": total_visual_rows,
        "null_claim_rows": null_claim_rows,
        "null_claim_rate": null_claim_rate,
        "note_claim_rows": note_claim_rows,
        "foreign_claim_rows": foreign_claim_rows,
        "foreign_claim_rate": foreign_claim_rate,
        "orphan_visual_rows": orphan_visual_rows,
        "orphan_visual_rate": orphan_visual_rate,
        "actionable_null_claim_rows": actionable_null_claim_rows,
        "actionable_foreign_claim_rows": actionable_foreign_claim_rows,
        "actionable_orphan_visual_rows": actionable_orphan_visual_rows,
        "actionable_orphan_visual_rate": actionable_orphan_visual_rate,
        "checks_passed": sum(1 for passed in check_results.values() if passed),
        "check_results": check_results,
        "overall_score": overall_score,
        "support_details_top5": sorted(
            support_details,
            key=lambda item: item["support"],
            reverse=True,
        )[:5],
        "support_details_bottom5": sorted(
            support_details,
            key=lambda item: item["support"],
        )[:5],
    }


def round_metric(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def round_nested(value):
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    return round_metric(value)


def build_payload(results, summary, orphan_visual_refs, unscoped_visual_rows):
    return {
        "db_host_used": DB_HOST,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "refs": round_nested(results),
        "orphan_visual_refs": round_nested(orphan_visual_refs),
        "unscoped_visual_rows": round_nested(unscoped_visual_rows),
        "summary": round_nested(summary),
    }


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        return
    fields = [
        "ref_type",
        "ref_id",
        "owner_key",
        "filename",
        "ignored_empty_paragraphs",
        "ignored_empty_headings",
        "total_body_elements",
        "claimed_body_elements",
        "missing_body_elements",
        "missing_bookmark_end",
        "missing_non_bookmark",
        "coverage",
        "cross_page_duplicates",
        "valid_cross_page_continuations",
        "duplicate_rate",
        "same_page_fragments",
        "fragment_rate",
        "order_consistency",
        "order_violations",
        "support_sample_count",
        "support_median",
        "support_p10",
        "bbox_tightness_sample_count",
        "bbox_tightness_median",
        "bbox_tightness_p10",
        "total_visual_rows",
        "null_claim_rows",
        "null_claim_rate",
        "note_claim_rows",
        "foreign_claim_rows",
        "foreign_claim_rate",
        "orphan_visual_rows",
        "orphan_visual_rate",
        "actionable_null_claim_rows",
        "actionable_foreign_claim_rows",
        "actionable_orphan_visual_rows",
        "actionable_orphan_visual_rate",
        "checks_passed",
        "overall_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: round_metric(row.get(field)) for field in fields})


def fmt_number(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4f}"
    return html.escape(str(value))


def fmt_percent(value) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.2f}%"


def metric_class(
    value,
    *,
    good_threshold: Optional[float] = None,
    bad_threshold: Optional[float] = None,
    lower_is_better: bool = False,
) -> str:
    if value is None:
        return "metric-na"
    if lower_is_better:
        if good_threshold is not None and value <= good_threshold:
            return "metric-good"
        if bad_threshold is not None and value > bad_threshold:
            return "metric-bad"
        return "metric-warn"
    if good_threshold is not None and value >= good_threshold:
        return "metric-good"
    if bad_threshold is not None and value < bad_threshold:
        return "metric-bad"
    return "metric-warn"


def render_summary_card(title: str, value: str, note: str, tone: str = "neutral") -> str:
    return (
        f'<section class="summary-card summary-{tone}">'
        f'<div class="summary-title">{html.escape(title)}</div>'
        f'<div class="summary-value">{value}</div>'
        f'<div class="summary-note">{html.escape(note)}</div>'
        "</section>"
    )


def render_top_list(items: List[dict], metric_key: str, title: str, percent: bool = False) -> str:
    rows = []
    for item in items:
        metric_value = item.get(metric_key)
        metric_text = fmt_percent(metric_value) if percent else fmt_number(metric_value)
        rows.append(
            "<li>"
            f'<span class="top-file">{html.escape(str(item.get("filename", "")))}</span>'
            f'<span class="top-metric">{metric_text}</span>'
            f'<span class="top-meta">ref {html.escape(str(item.get("ref_id", "")))}</span>'
            "</li>"
        )
    return (
        '<section class="panel">'
        f"<h2>{html.escape(title)}</h2>"
        f"<ol class=\"top-list\">{''.join(rows) if rows else '<li>Tidak ada data</li>'}</ol>"
        "</section>"
    )


def render_table_rows(rows: List[dict]) -> str:
    rendered_rows = []
    for row in rows:
        coverage_class = metric_class(row.get("coverage"), good_threshold=0.98, bad_threshold=0.90)
        duplicate_class = metric_class(
            row.get("duplicate_rate"), good_threshold=0.03, bad_threshold=0.05, lower_is_better=True
        )
        order_class = metric_class(row.get("order_consistency"), good_threshold=0.99, bad_threshold=0.95)
        null_class = metric_class(
            row.get("null_claim_rate"), good_threshold=0.01, bad_threshold=0.05, lower_is_better=True
        )
        foreign_class = metric_class(
            row.get("foreign_claim_rate"), good_threshold=0.0, bad_threshold=0.0, lower_is_better=True
        )
        orphan_class = metric_class(
            row.get("orphan_visual_rate"), good_threshold=0.01, bad_threshold=0.05, lower_is_better=True
        )
        actionable_orphan_class = metric_class(
            row.get("actionable_orphan_visual_rate"),
            good_threshold=0.01,
            bad_threshold=0.05,
            lower_is_better=True,
        )
        row_class = (
            "row-issue"
            if (
                row.get("checks_passed", 0) < 6 or
                row.get("actionable_orphan_visual_rows", 0) > 0 or
                row.get("total_visual_rows", 0) == 0
            )
            else ""
        )
        rendered_rows.append(
            f'<tr class="{row_class}">'
            f'<td>{html.escape(str(row.get("ref_id", "")))}</td>'
            f'<td class="cell-file">{html.escape(str(row.get("filename", "")))}</td>'
            f'<td>{html.escape(str(row.get("owner_key", "")))}</td>'
            f'<td class="{coverage_class}">{fmt_percent(row.get("coverage"))}</td>'
            f'<td>{fmt_number(row.get("missing_body_elements"))}</td>'
            f'<td>{fmt_number(row.get("missing_non_bookmark"))}</td>'
            f'<td>{fmt_number(row.get("missing_bookmark_end"))}</td>'
            f'<td class="{duplicate_class}">{fmt_percent(row.get("duplicate_rate"))}</td>'
            f'<td>{fmt_number(row.get("cross_page_duplicates"))}</td>'
            f'<td>{fmt_number(row.get("valid_cross_page_continuations"))}</td>'
            f'<td>{fmt_number(row.get("same_page_fragments"))}</td>'
            f'<td class="{order_class}">{fmt_percent(row.get("order_consistency"))}</td>'
            f'<td class="{null_class}">{fmt_percent(row.get("null_claim_rate"))}</td>'
            f'<td class="{foreign_class}">{fmt_percent(row.get("foreign_claim_rate"))}</td>'
            f'<td class="{actionable_orphan_class}">{fmt_percent(row.get("actionable_orphan_visual_rate"))}</td>'
            f'<td>{fmt_number(row.get("actionable_orphan_visual_rows"))}</td>'
            f'<td class="{orphan_class}">{fmt_percent(row.get("orphan_visual_rate"))}</td>'
            f'<td>{fmt_number(row.get("orphan_visual_rows"))}</td>'
            f'<td>{fmt_number(row.get("total_visual_rows"))}</td>'
            f'<td>{fmt_number(row.get("ignored_empty_paragraphs"))}</td>'
            f'<td>{fmt_number(row.get("ignored_empty_headings"))}</td>'
            f'<td>{fmt_number(row.get("checks_passed"))}/6</td>'
            "</tr>"
        )
    return "".join(rendered_rows)


def write_html(
    path: Path,
    payload: dict,
    *,
    report_title: str,
    json_filename: str,
    csv_filename: str,
):
    summary = payload["summary"]
    refs = payload["refs"]

    summary_cards = "".join(
        [
            render_summary_card(
                "Dokumen",
                fmt_number(summary.get("documents")),
                "Jumlah ref dokumen yang dievaluasi",
            ),
            render_summary_card(
                "Processed Docs",
                fmt_number((summary.get("processed_docs_only") or {}).get("documents")),
                "Ref yang sudah punya visual rows",
                "good",
            ),
            render_summary_card(
                "Zero Visual Docs",
                fmt_number(summary.get("documents_with_zero_visual_rows")),
                "Ref yang masih 0 visual row",
                "warn" if (summary.get("documents_with_zero_visual_rows") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Coverage Global",
                fmt_percent(summary.get("global_coverage")),
                "Coverage setelah paragraf kosong diabaikan",
                "good" if (summary.get("global_coverage") or 0) >= 0.98 else "warn",
            ),
            render_summary_card(
                "Missing Non-Bookmark",
                fmt_number(summary.get("missing_non_bookmark")),
                "Missing actionable di luar bookmarkEnd",
                "warn" if (summary.get("missing_non_bookmark") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Missing BookmarkEnd",
                fmt_number(summary.get("missing_bookmark_end")),
                "Tetap dihitung, tapi bukan fokus perbaikan",
            ),
            render_summary_card(
                "Duplicate Rate",
                fmt_percent(summary.get("average_duplicate_rate")),
                "Rata-rata invalid cross-page duplicate per dokumen",
                "warn" if (summary.get("average_duplicate_rate") or 0) > 0.05 else "good",
            ),
            render_summary_card(
                "Valid Continuations",
                fmt_number(summary.get("total_valid_cross_page_continuations")),
                "Cross-page claim valid yang diabaikan dari duplicate",
                "good",
            ),
            render_summary_card(
                "Order Consistency",
                fmt_percent(summary.get("median_order_consistency")),
                "Median konsistensi urutan elemen",
                "good" if (summary.get("median_order_consistency") or 0) >= 0.99 else "warn",
            ),
            render_summary_card(
                "Null Claim Rate",
                fmt_percent(summary.get("average_null_claim_rate")),
                "Rata-rata row visual tanpa element_id",
                "good" if (summary.get("average_null_claim_rate") or 0) <= 0.01 else "warn",
            ),
            render_summary_card(
                "Orphan Visual Rows",
                fmt_number(summary.get("total_orphan_visual_rows")),
                "Total row visual in-scope yang null/foreign",
                "warn" if (summary.get("total_orphan_visual_rows") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Actionable Orphan",
                fmt_number(summary.get("total_actionable_orphan_visual_rows")),
                "Orphan tanpa page_header/page_footer",
                "warn" if (summary.get("total_actionable_orphan_visual_rows") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Docs With Orphan",
                fmt_number(summary.get("documents_with_orphan_visual_rows")),
                "Dokumen yang masih punya orphan visual row",
                "warn" if (summary.get("documents_with_orphan_visual_rows") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Unscoped Visual Rows",
                fmt_number(payload.get("unscoped_visual_rows")),
                "Row visual global yang tidak punya scope ref",
                "warn" if (payload.get("unscoped_visual_rows") or 0) > 0 else "good",
            ),
        ]
    )

    panels = "".join(
        [
            render_top_list(summary.get("top_5_lowest_coverage", []), "coverage", "Coverage Terendah", percent=True),
            render_top_list(
                summary.get("top_5_highest_missing_non_bookmark", []),
                "missing_non_bookmark",
                "Missing Non-Bookmark Tertinggi",
                percent=False,
            ),
            render_top_list(
                summary.get("top_5_highest_duplicate_rate", []),
                "duplicate_rate",
                "Duplicate Tertinggi",
                percent=True,
            ),
            render_top_list(
                summary.get("top_5_docs_with_zero_visual_rows", []),
                "total_body_elements",
                "Zero Visual Rows",
                percent=False,
            ),
            render_top_list(
                summary.get("top_5_highest_orphan_visual_rate", []),
                "orphan_visual_rate",
                "Orphan Visual Tertinggi",
                percent=True,
            ),
            render_top_list(
                summary.get("top_5_highest_actionable_orphan_rate", []),
                "actionable_orphan_visual_rate",
                "Actionable Orphan Tertinggi",
                percent=True,
            ),
        ]
    )

    table_rows = render_table_rows(refs)
    generated_at = html.escape(str(payload.get("generated_at", "")))
    db_host = html.escape(str(payload.get("db_host_used", "")))

    document = f"""<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report_title)}</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --paper: #fffaf2;
      --ink: #1f2937;
      --muted: #6b7280;
      --line: #d6cbb7;
      --accent: #0f4c5c;
      --accent-soft: #d8ecef;
      --good: #1d6f42;
      --good-soft: #dff3e6;
      --warn: #a15c00;
      --warn-soft: #fff0d6;
      --bad: #9b2226;
      --bad-soft: #fde7e6;
      --shadow: 0 18px 40px rgba(31, 41, 55, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI Variable Text", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 76, 92, 0.12), transparent 28rem),
        linear-gradient(180deg, #f7f1e7 0%, #f4efe6 100%);
    }}
    .wrap {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 32px 24px 48px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(15, 76, 92, 0.95), rgba(34, 55, 90, 0.92));
      color: white;
      padding: 28px 32px;
      border-radius: 24px;
      box-shadow: var(--shadow);
      margin-bottom: 24px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 32px;
      line-height: 1.1;
    }}
    .hero p {{
      margin: 4px 0;
      color: rgba(255, 255, 255, 0.84);
      max-width: 72rem;
    }}
    .hero-links {{
      margin-top: 14px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .hero-links a {{
      color: white;
      text-decoration: none;
      border: 1px solid rgba(255, 255, 255, 0.28);
      padding: 8px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }}
    .summary-card {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
      min-height: 132px;
    }}
    .summary-good {{ border-color: rgba(29, 111, 66, 0.35); }}
    .summary-warn {{ border-color: rgba(161, 92, 0, 0.35); }}
    .summary-title {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 10px;
    }}
    .summary-value {{
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 8px;
    }}
    .summary-note {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }}
    .panel-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .panel {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .panel h2 {{
      margin: 0 0 14px;
      font-size: 18px;
    }}
    .top-list {{
      margin: 0;
      padding-left: 20px;
      display: grid;
      gap: 10px;
    }}
    .top-list li {{
      display: grid;
      gap: 2px;
    }}
    .top-file {{
      font-weight: 600;
    }}
    .top-metric {{
      font-family: "Cascadia Code", "Consolas", monospace;
      color: var(--accent);
    }}
    .top-meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .table-wrap {{
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 22px;
      overflow: hidden;
      box-shadow: var(--shadow);
    }}
    .table-head {{
      padding: 18px 20px 8px;
      border-bottom: 1px solid var(--line);
    }}
    .table-head h2 {{
      margin: 0 0 8px;
      font-size: 20px;
    }}
    .table-head p {{
      margin: 0;
      color: var(--muted);
    }}
    .scroll {{
      overflow: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 1380px;
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid rgba(214, 203, 183, 0.72);
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f0e8dc;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-size: 12px;
    }}
    .cell-file {{
      min-width: 280px;
      font-weight: 600;
    }}
    .metric-good {{
      background: var(--good-soft);
      color: var(--good);
      font-weight: 700;
    }}
    .metric-warn {{
      background: var(--warn-soft);
      color: var(--warn);
      font-weight: 700;
    }}
    .metric-bad {{
      background: var(--bad-soft);
      color: var(--bad);
      font-weight: 700;
    }}
    .metric-na {{
      color: var(--muted);
    }}
    .row-issue {{
      background: rgba(255, 242, 219, 0.35);
    }}
    .legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.55);
    }}
    .dot {{
      width: 10px;
      height: 10px;
      border-radius: 999px;
      display: inline-block;
    }}
    .dot-good {{ background: var(--good); }}
    .dot-warn {{ background: var(--warn); }}
    .dot-bad {{ background: var(--bad); }}
  </style>
</head>
<body>
  <main class="wrap">
    <section class="hero">
      <h1>{html.escape(report_title)}</h1>
      <p>Aspek utama yang dievaluasi: coverage setelah paragraf kosong dan heading kosong dibuang, duplicate invalid lintas halaman, order consistency, null/foreign claim, raw orphan visual rows, actionable orphan tanpa header/footer, serta breakdown missing non-bookmark vs bookmarkEnd.</p>
      <p>Dihasilkan pada {generated_at} dari DB host {db_host}.</p>
      <div class="hero-links">
        <a href="{html.escape(json_filename)}">Buka JSON</a>
        <a href="{html.escape(csv_filename)}">Buka CSV</a>
      </div>
    </section>

    <section class="summary-grid">
      {summary_cards}
    </section>

    <section class="panel-grid">
      {panels}
    </section>

    <section class="table-wrap">
      <div class="table-head">
        <h2>Semua Dokumen</h2>
        <p>Tabel ini difokuskan ke metrik yang benar-benar dipakai saat ini. Coverage sudah mengecualikan paragraf kosong dan h1/h2 kosong. Actionable orphan mengabaikan page_header/page_footer agar noise header tidak terlihat seperti failure body alignment.</p>
        <div class="legend">
          <span class="chip"><span class="dot dot-good"></span> sehat</span>
          <span class="chip"><span class="dot dot-warn"></span> perlu perhatian</span>
          <span class="chip"><span class="dot dot-bad"></span> bermasalah</span>
        </div>
      </div>
      <div class="scroll">
        <table>
          <thead>
            <tr>
              <th>Ref</th>
              <th>Filename</th>
              <th>Owner</th>
              <th>Coverage</th>
              <th>Missing</th>
              <th>Missing Non-BM</th>
              <th>Missing BM</th>
              <th>Duplicate</th>
              <th>Cross-page Dup</th>
              <th>Valid Cont</th>
              <th>Fragments</th>
              <th>Order</th>
              <th>Null</th>
              <th>Foreign</th>
              <th>Act Orphan</th>
              <th>Act Orphan Rows</th>
              <th>Orphan</th>
              <th>Orphan Rows</th>
              <th>Visual Rows</th>
              <th>Ignored Empty P</th>
              <th>Ignored Empty H</th>
              <th>Checks</th>
            </tr>
          </thead>
          <tbody>
            {table_rows}
          </tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""

    path.write_text(document, encoding="utf-8")


def summarize(results: List[dict]) -> dict:
    def summarize_rows(rows: List[dict]) -> dict:
        valid_coverages = [row["coverage"] for row in rows if row["coverage"] is not None]
        valid_support = [row["support_median"] for row in rows if row["support_median"] is not None]
        valid_tightness = [row["bbox_tightness_median"] for row in rows if row["bbox_tightness_median"] is not None]

        return {
            "documents": len(rows),
            "ignored_empty_paragraphs": sum(row["ignored_empty_paragraphs"] for row in rows),
            "ignored_empty_headings": sum(row["ignored_empty_headings"] for row in rows),
            "total_body_elements": sum(row["total_body_elements"] for row in rows),
            "claimed_body_elements": sum(row["claimed_body_elements"] for row in rows),
            "missing_bookmark_end": sum(row["missing_bookmark_end"] for row in rows),
            "missing_non_bookmark": sum(row["missing_non_bookmark"] for row in rows),
            "total_valid_cross_page_continuations": sum(row["valid_cross_page_continuations"] for row in rows),
            "documents_with_valid_cross_page_continuations": sum(
                1 for row in rows if row["valid_cross_page_continuations"] > 0
            ),
            "global_coverage": (
                sum(row["claimed_body_elements"] for row in rows)
                / max(1, sum(row["total_body_elements"] for row in rows))
            ),
            "median_document_coverage": median(valid_coverages) if valid_coverages else None,
            "documents_with_cross_page_duplicates": sum(1 for row in rows if row["cross_page_duplicates"] > 0),
            "documents_with_null_claims": sum(1 for row in rows if row["null_claim_rows"] > 0),
            "median_order_consistency": median([row["order_consistency"] for row in rows]) if rows else None,
            "median_support_median": median(valid_support) if valid_support else None,
            "median_bbox_tightness_median": median(valid_tightness) if valid_tightness else None,
            "average_null_claim_rate": (
                sum(row["null_claim_rate"] for row in rows) / len(rows)
            ) if rows else None,
            "total_orphan_visual_rows": sum(row["orphan_visual_rows"] for row in rows),
            "documents_with_orphan_visual_rows": sum(1 for row in rows if row["orphan_visual_rows"] > 0),
            "average_orphan_visual_rate": (
                sum(row["orphan_visual_rate"] for row in rows) / len(rows)
            ) if rows else None,
            "total_actionable_orphan_visual_rows": sum(row["actionable_orphan_visual_rows"] for row in rows),
            "documents_with_actionable_orphan_visual_rows": sum(
                1 for row in rows if row["actionable_orphan_visual_rows"] > 0
            ),
            "average_actionable_orphan_visual_rate": (
                sum(row["actionable_orphan_visual_rate"] for row in rows) / len(rows)
            ) if rows else None,
            "average_duplicate_rate": (
                sum(row["duplicate_rate"] for row in rows) / len(rows)
            ) if rows else None,
        }

    overall_summary = summarize_rows(results)
    processed_rows = [row for row in results if (row.get("total_visual_rows") or 0) > 0]
    zero_visual_rows = [row for row in results if (row.get("total_visual_rows") or 0) == 0]

    overall_summary.update({
        "processed_docs_only": summarize_rows(processed_rows),
        "documents_with_zero_visual_rows": len(zero_visual_rows),
        "top_5_docs_with_zero_visual_rows": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "total_body_elements": row["total_body_elements"],
                "missing_non_bookmark": row["missing_non_bookmark"],
                "coverage": round_metric(row["coverage"]),
            }
            for row in sorted(
                zero_visual_rows,
                key=lambda item: (item["total_body_elements"], item["missing_non_bookmark"]),
                reverse=True,
            )[:5]
        ],
        "by_type": {
            ref_type: {
                "refs": len(type_rows),
                "ignored_empty_paragraphs": sum(row["ignored_empty_paragraphs"] for row in type_rows),
                "ignored_empty_headings": sum(row["ignored_empty_headings"] for row in type_rows),
                "total_body_elements": sum(row["total_body_elements"] for row in type_rows),
                "claimed_body_elements": sum(row["claimed_body_elements"] for row in type_rows),
                "missing_bookmark_end": sum(row["missing_bookmark_end"] for row in type_rows),
                "missing_non_bookmark": sum(row["missing_non_bookmark"] for row in type_rows),
                "total_valid_cross_page_continuations": sum(
                    row["valid_cross_page_continuations"] for row in type_rows
                ),
                "documents_with_valid_cross_page_continuations": sum(
                    1 for row in type_rows if row["valid_cross_page_continuations"] > 0
                ),
                "global_coverage": (
                    sum(row["claimed_body_elements"] for row in type_rows)
                    / max(1, sum(row["total_body_elements"] for row in type_rows))
                ),
                "median_document_coverage": median(
                    [row["coverage"] for row in type_rows if row["coverage"] is not None]
                ) if type_rows else None,
                "median_order_consistency": median([row["order_consistency"] for row in type_rows]) if type_rows else None,
                "median_support_median": median(
                    [row["support_median"] for row in type_rows if row["support_median"] is not None]
                ) if any(row["support_median"] is not None for row in type_rows) else None,
                "median_bbox_tightness_median": median(
                    [row["bbox_tightness_median"] for row in type_rows if row["bbox_tightness_median"] is not None]
                ) if any(row["bbox_tightness_median"] is not None for row in type_rows) else None,
                "average_duplicate_rate": (
                    sum(row["duplicate_rate"] for row in type_rows) / len(type_rows)
                ) if type_rows else None,
                "average_null_claim_rate": (
                    sum(row["null_claim_rate"] for row in type_rows) / len(type_rows)
                ) if type_rows else None,
                "total_orphan_visual_rows": sum(row["orphan_visual_rows"] for row in type_rows),
                "documents_with_orphan_visual_rows": sum(
                    1 for row in type_rows if row["orphan_visual_rows"] > 0
                ),
                "average_orphan_visual_rate": (
                    sum(row["orphan_visual_rate"] for row in type_rows) / len(type_rows)
                ) if type_rows else None,
                "total_actionable_orphan_visual_rows": sum(
                    row["actionable_orphan_visual_rows"] for row in type_rows
                ),
                "documents_with_actionable_orphan_visual_rows": sum(
                    1 for row in type_rows if row["actionable_orphan_visual_rows"] > 0
                ),
                "average_actionable_orphan_visual_rate": (
                    sum(row["actionable_orphan_visual_rate"] for row in type_rows) / len(type_rows)
                ) if type_rows else None,
            }
            for ref_type in sorted({row["ref_type"] for row in results})
            for type_rows in [[row for row in results if row["ref_type"] == ref_type]]
        },
        "top_5_lowest_coverage": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "coverage": round_metric(row["coverage"]),
                "checks_passed": row["checks_passed"],
                "overall_score": round_metric(row["overall_score"]),
            }
            for row in sorted(results, key=lambda item: (item["coverage"] if item["coverage"] is not None else -1.0))[:5]
        ],
        "top_5_highest_missing_non_bookmark": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "missing_non_bookmark": row["missing_non_bookmark"],
                "missing_bookmark_end": row["missing_bookmark_end"],
                "coverage": round_metric(row["coverage"]),
            }
            for row in sorted(
                results,
                key=lambda item: (
                    item["missing_non_bookmark"],
                    item["missing_bookmark_end"],
                    -1 * (item["coverage"] if item["coverage"] is not None else 0.0),
                ),
                reverse=True,
            )[:5]
        ],
        "top_5_highest_duplicate_rate": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "duplicate_rate": round_metric(row["duplicate_rate"]),
                "cross_page_duplicates": row["cross_page_duplicates"],
            }
            for row in sorted(results, key=lambda item: item["duplicate_rate"], reverse=True)[:5]
        ],
        "top_5_lowest_order_consistency": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "order_consistency": round_metric(row["order_consistency"]),
                "order_violations": row["order_violations"],
            }
            for row in sorted(results, key=lambda item: item["order_consistency"])[:5]
        ],
        "top_5_highest_orphan_visual_rate": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "orphan_visual_rows": row["orphan_visual_rows"],
                "orphan_visual_rate": round_metric(row["orphan_visual_rate"]),
                "total_visual_rows": row["total_visual_rows"],
            }
            for row in sorted(results, key=lambda item: item["orphan_visual_rate"], reverse=True)[:5]
        ],
        "top_5_highest_actionable_orphan_rate": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "actionable_orphan_visual_rows": row["actionable_orphan_visual_rows"],
                "actionable_orphan_visual_rate": round_metric(row["actionable_orphan_visual_rate"]),
                "total_visual_rows": row["total_visual_rows"],
            }
            for row in sorted(results, key=lambda item: item["actionable_orphan_visual_rate"], reverse=True)[:5]
        ],
        "top_5_lowest_support": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "support_median": round_metric(row["support_median"]),
                "support_sample_count": row["support_sample_count"],
            }
            for row in sorted(
                [row for row in results if row["support_median"] is not None],
                key=lambda item: item["support_median"],
            )[:5]
        ],
        "top_5_lowest_bbox_tightness": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "bbox_tightness_median": round_metric(row["bbox_tightness_median"]),
                "bbox_tightness_sample_count": row["bbox_tightness_sample_count"],
            }
            for row in sorted(
                [row for row in results if row["bbox_tightness_median"] is not None],
                key=lambda item: item["bbox_tightness_median"],
            )[:5]
        ],
    })
    return overall_summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate visual alignment quality for all dokumen and bab refs."
    )
    parser.add_argument(
        "--ref-type",
        choices=["all", "dokumen", "bab"],
        default="all",
        help="Limit evaluation to one ref_type, or use all.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "reports" / "alignment_eval"),
        help="Directory to store JSON and CSV reports.",
    )
    parser.add_argument(
        "--min-ref-id",
        type=int,
        default=None,
        help="Optional minimum ref_id to include.",
    )
    parser.add_argument(
        "--max-ref-id",
        type=int,
        default=None,
        help="Optional maximum ref_id to include.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = build_engine()
    with engine.connect() as conn:
        ref_rows = []
        selected_ref_types = (
            [args.ref_type] if args.ref_type != "all" else list(REF_QUERIES.keys())
        )
        for ref_type in selected_ref_types:
            query = REF_QUERIES[ref_type]
            ref_rows.extend(dict(row._mapping) for row in conn.execute(query))
        if args.min_ref_id is not None:
            ref_rows = [
                row for row in ref_rows
                if row.get("ref_id") is not None and int(row["ref_id"]) >= args.min_ref_id
            ]
        if args.max_ref_id is not None:
            ref_rows = [
                row for row in ref_rows
                if row.get("ref_id") is not None and int(row["ref_id"]) <= args.max_ref_id
            ]
        results = [evaluate_ref(conn, ref_row) for ref_row in ref_rows]
        orphan_visual_refs = [
            row for row in (dict(item._mapping) for item in conn.execute(ORPHAN_VISUAL_QUERY))
            if row["ref_type"] in selected_ref_types
        ]
        unscoped_visual_rows = conn.execute(UNSCOPED_VISUAL_QUERY).scalar() or 0

    summary = summarize(results)

    file_suffix = args.ref_type
    report_range = None
    if args.min_ref_id is not None or args.max_ref_id is not None:
        range_start = args.min_ref_id if args.min_ref_id is not None else "min"
        range_end = args.max_ref_id if args.max_ref_id is not None else "max"
        report_range = f"{range_start}_{range_end}"
        file_suffix = f"{file_suffix}_{report_range}"
    json_path = output_dir / f"visual_alignment_eval_{file_suffix}.json"
    csv_path = output_dir / f"visual_alignment_eval_{file_suffix}.csv"
    html_path = output_dir / f"visual_alignment_eval_{file_suffix}.html"
    payload = build_payload(results, summary, orphan_visual_refs, unscoped_visual_rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    write_csv(csv_path, results)
    write_html(
        html_path,
        payload,
        report_title=(
            f"Visual Alignment Report ({args.ref_type}, ref_id {report_range})"
            if report_range
            else f"Visual Alignment Report ({args.ref_type})"
        ),
        json_filename=json_path.name,
        csv_filename=csv_path.name,
    )

    print(f"DB host used: {DB_HOST}:{DB_PORT}")
    print(f"Ref type filter: {args.ref_type}")
    if args.min_ref_id is not None or args.max_ref_id is not None:
        print(
            "Ref id range: "
            f"{args.min_ref_id if args.min_ref_id is not None else '*'}"
            f"..{args.max_ref_id if args.max_ref_id is not None else '*'}"
        )
    print(f"Refs evaluated: {summary['documents']}")
    print(f"Ignored empty paragraphs: {summary['ignored_empty_paragraphs']}")
    print(f"Ignored empty headings: {summary['ignored_empty_headings']}")
    print(f"Global coverage: {round_metric(summary['global_coverage'])}")
    print(f"Missing bookmarkEnd: {summary['missing_bookmark_end']}")
    print(f"Missing non-bookmark: {summary['missing_non_bookmark']}")
    print(f"Valid cross-page continuations: {summary['total_valid_cross_page_continuations']}")
    print(f"Median document coverage: {round_metric(summary['median_document_coverage'])}")
    print(f"Median order consistency: {round_metric(summary['median_order_consistency'])}")
    print(f"Median support median: {round_metric(summary['median_support_median'])}")
    print(f"Median bbox tightness median: {round_metric(summary['median_bbox_tightness_median'])}")
    print(f"Average duplicate rate: {round_metric(summary['average_duplicate_rate'])}")
    print(f"Average null claim rate: {round_metric(summary['average_null_claim_rate'])}")
    print(f"Total orphan visual rows: {summary['total_orphan_visual_rows']}")
    print(f"Total actionable orphan visual rows: {summary['total_actionable_orphan_visual_rows']}")
    print(f"Documents with orphan visual rows: {summary['documents_with_orphan_visual_rows']}")
    print(
        "Documents with actionable orphan visual rows: "
        f"{summary['documents_with_actionable_orphan_visual_rows']}"
    )
    print(f"Average orphan visual rate: {round_metric(summary['average_orphan_visual_rate'])}")
    print(
        "Average actionable orphan visual rate: "
        f"{round_metric(summary['average_actionable_orphan_visual_rate'])}"
    )
    print(
        "Processed docs only: "
        f"{(summary.get('processed_docs_only') or {}).get('documents', 0)}"
    )
    print(f"Docs with zero visual rows: {summary['documents_with_zero_visual_rows']}")
    print(f"Orphan visual refs: {len(orphan_visual_refs)}")
    print(f"Unscoped visual rows: {unscoped_visual_rows}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print(f"HTML report: {html_path}")


if __name__ == "__main__":
    main()
