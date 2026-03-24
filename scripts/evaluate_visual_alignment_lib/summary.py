from __future__ import annotations

from statistics import median
from typing import List

from evaluate_visual_alignment_lib.reporting import round_metric


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
