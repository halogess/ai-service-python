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

from evaluate_visual_alignment_lib.helpers import (
    PdfWordCache,
    analyze_element_rows_for_duplicates,
    bbox_area,
    bbox_x_overlap_ratio,
    center_in_bbox,
    choose_order_anchor_row,
    choose_representative_row,
    evaluate_element_on_pdf,
    extract_openxml_text,
    get_visual_label,
    group_rows_by_element,
    intersection_area,
    is_actionable_orphan_label,
    is_caption_like_text,
    is_table_like_row,
    is_valid_same_page_chart_caption_pair,
    is_valid_same_page_table_claim_set,
    json_tree_has_visual_bearing_content,
    merge_bboxes,
    normalize_text,
    parse_json_tree,
    percentile,
    should_ignore_body_element,
    text_similarity,
    word_is_covered_by_rows,
    word_overlap_ratio,
)
from evaluate_visual_alignment_lib.reporting import (
    fmt_number,
    fmt_percent,
    metric_class,
    render_summary_card,
    render_table_rows,
    render_top_list,
    round_metric,
    round_nested,
    write_csv,
    write_html,
)


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
    body_rows = [dict(row._mapping) for row in conn.execute(BODY_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id})]
    ignored_toc_stub_sequences = ALIGNER._collect_toc_stub_sequences(body_rows)
    for mapping in body_rows:
        element_id = int(mapping["delemen_id"])
        openxml_text = extract_openxml_text(mapping["delemen_json_tree"])
        openxml_text_norm = normalize_text(openxml_text)
        if (
            mapping["delemen_sequence"] is not None and
            int(mapping["delemen_sequence"]) in ignored_toc_stub_sequences
        ):
            continue
        if should_ignore_body_element(
            mapping["delemen_type"],
            openxml_text_norm,
            mapping["delemen_json_tree"],
        ):
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
            if is_valid_same_page_table_claim_set(rows):
                continue
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
        representative = choose_order_anchor_row(rows, page_heights=page_heights)
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

    total_pdf_words = 0
    uncovered_pdf_words = 0
    pages_with_uncovered_pdf_words = 0
    uncovered_pdf_word_rate = 0.0
    support_scores: List[float] = []
    bbox_tightness_scores: List[float] = []
    support_details = []

    if pdf_cache is not None:
        try:
            visual_rows_by_page: Dict[int, List[dict]] = defaultdict(list)
            for row in visual_rows:
                if row["page"] is not None and row.get("bbox"):
                    visual_rows_by_page[row["page"]].append(row)

            for page_num in range(1, pdf_cache.doc.page_count + 1):
                page_words = pdf_cache.get_page_words(page_num)
                total_pdf_words += len(page_words)
                page_uncovered = sum(
                    1
                    for word in page_words
                    if not word_is_covered_by_rows(word["bbox"], visual_rows_by_page.get(page_num, []))
                )
                uncovered_pdf_words += page_uncovered
                if page_uncovered > 0:
                    pages_with_uncovered_pdf_words += 1

            uncovered_pdf_word_rate = (
                uncovered_pdf_words / total_pdf_words
            ) if total_pdf_words else 0.0

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
        "total_pdf_words": total_pdf_words,
        "uncovered_pdf_words": uncovered_pdf_words,
        "uncovered_pdf_word_rate": uncovered_pdf_word_rate,
        "pages_with_uncovered_pdf_words": pages_with_uncovered_pdf_words,
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






def build_payload(results, summary, orphan_visual_refs, unscoped_visual_rows):
    return {
        "db_host_used": DB_HOST,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "refs": round_nested(results),
        "orphan_visual_refs": round_nested(orphan_visual_refs),
        "unscoped_visual_rows": round_nested(unscoped_visual_rows),
        "summary": round_nested(summary),
    }


















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
            "total_pdf_words": sum(row["total_pdf_words"] for row in rows),
            "total_uncovered_pdf_words": sum(row["uncovered_pdf_words"] for row in rows),
            "documents_with_uncovered_pdf_words": sum(
                1 for row in rows if row["uncovered_pdf_words"] > 0
            ),
            "average_uncovered_pdf_word_rate": (
                sum(row["uncovered_pdf_word_rate"] for row in rows) / len(rows)
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
        "top_5_highest_uncovered_pdf_word_rate": [
            {
                "ref_type": row["ref_type"],
                "ref_id": row["ref_id"],
                "filename": row["filename"],
                "uncovered_pdf_words": row["uncovered_pdf_words"],
                "uncovered_pdf_word_rate": round_metric(row["uncovered_pdf_word_rate"]),
                "total_pdf_words": row["total_pdf_words"],
            }
            for row in sorted(results, key=lambda item: item["uncovered_pdf_word_rate"], reverse=True)[:5]
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
    print(f"Total uncovered PDF words: {summary['total_uncovered_pdf_words']}")
    print(
        "Average uncovered PDF word rate: "
        f"{round_metric(summary['average_uncovered_pdf_word_rate'])}"
    )
    print(
        "Documents with uncovered PDF words: "
        f"{summary['documents_with_uncovered_pdf_words']}"
    )
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
