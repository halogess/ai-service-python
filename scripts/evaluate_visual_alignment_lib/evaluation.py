from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List

from evaluate_visual_alignment_lib.helpers import (
    PdfWordCache,
    analyze_element_rows_for_duplicates,
    choose_order_anchor_row,
    evaluate_element_on_pdf,
    extract_openxml_text,
    group_rows_by_element,
    is_actionable_orphan_label,
    is_valid_same_page_chart_caption_pair,
    is_valid_same_page_table_claim_set,
    normalize_text,
    percentile,
    should_ignore_body_element,
    word_is_covered_by_rows,
)


def evaluate_ref(
    conn,
    ref_row: dict,
    *,
    aligner,
    volume_base: Path,
    body_elements_query,
    all_elements_query,
    note_ids_query,
    visual_rows_query,
) -> dict:
    ref_type = str(ref_row["ref_type"])
    ref_id = int(ref_row["ref_id"])
    pdf_rel_path = str(ref_row["pdf_path"] or "")
    pdf_path = (volume_base / pdf_rel_path).resolve() if pdf_rel_path else volume_base.resolve()
    pdf_cache = None
    page_heights: Dict[int, float] = {}

    body_elements = {}
    body_element_ids = set()
    ignored_empty_paragraph_ids = set()
    ignored_empty_heading_ids = set()
    body_rows = [dict(row._mapping) for row in conn.execute(body_elements_query, {"ref_type": ref_type, "ref_id": ref_id})]
    ignored_toc_stub_sequences = aligner._collect_toc_stub_sequences(body_rows)
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
        int(row[0]) for row in conn.execute(all_elements_query, {"ref_type": ref_type, "ref_id": ref_id})
    }
    note_ids = set()
    if ref_type == "dokumen":
        note_ids = {int(row[0]) for row in conn.execute(note_ids_query, {"ref_id": ref_id})}

    visual_rows = []
    for row in conn.execute(visual_rows_query, {"ref_type": ref_type, "ref_id": ref_id}):
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
