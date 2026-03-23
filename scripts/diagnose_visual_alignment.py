from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from sqlalchemy import text

import evaluate_visual_alignment as evalmod


REPO_ROOT = Path(__file__).resolve().parents[1]


def classify_element_issue(element_type: Optional[str], text_norm: str) -> str:
    normalized_type = str(element_type or "").strip().lower()
    if normalized_type == "bookmarkend":
        return "bookmarkEnd"
    if "[img" in text_norm:
        return "image_placeholder"
    if ("gambar" in text_norm or "tabel" in text_norm) and len(text_norm) <= 24:
        return "short_caption_fragment"
    if "gambar" in text_norm or "tabel" in text_norm:
        return "caption_or_table_ref"
    if "table" in normalized_type:
        return "table"
    if "list-item" in normalized_type:
        return "list_item"
    if len(text_norm) <= 4:
        return "very_short_text"
    if len(text_norm) >= 180:
        return "long_paragraph"
    return "paragraph_other"


def collect_ref_rows(conn, ref_type: str, min_ref_id: Optional[int], max_ref_id: Optional[int]) -> List[dict]:
    selected_ref_types = [ref_type] if ref_type != "all" else list(evalmod.REF_QUERIES.keys())
    ref_rows: List[dict] = []
    for current_ref_type in selected_ref_types:
        query = evalmod.REF_QUERIES[current_ref_type]
        ref_rows.extend(dict(row._mapping) for row in conn.execute(query))
    if min_ref_id is not None:
        ref_rows = [row for row in ref_rows if int(row["ref_id"]) >= min_ref_id]
    if max_ref_id is not None:
        ref_rows = [row for row in ref_rows if int(row["ref_id"]) <= max_ref_id]
    return ref_rows


def collect_ref_diagnosis(conn, ref_row: dict) -> dict:
    ref_type = str(ref_row["ref_type"])
    ref_id = int(ref_row["ref_id"])
    pdf_rel_path = str(ref_row.get("pdf_path") or "")
    pdf_path = (evalmod.VOLUME_BASE / pdf_rel_path).resolve()
    page_heights: Dict[int, float] = {}

    body_elements: Dict[int, dict] = {}
    body_element_ids = set()
    for row in conn.execute(evalmod.BODY_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id}):
        mapping = dict(row._mapping)
        element_id = int(mapping["delemen_id"])
        openxml_text = evalmod.extract_openxml_text(mapping["delemen_json_tree"])
        openxml_text_norm = evalmod.normalize_text(openxml_text)
        if evalmod.should_ignore_body_element(
            mapping["delemen_type"],
            openxml_text_norm,
            mapping["delemen_json_tree"],
        ):
            continue
        body_elements[element_id] = {
            "sequence": int(mapping["delemen_sequence"]) if mapping["delemen_sequence"] is not None else None,
            "type": mapping["delemen_type"],
            "openxml_text": openxml_text,
            "openxml_text_norm": openxml_text_norm,
        }
        body_element_ids.add(element_id)

    visual_rows: List[dict] = []
    null_rows: List[dict] = []
    foreign_rows: List[dict] = []
    for row in conn.execute(evalmod.VISUAL_ROWS_QUERY, {"ref_type": ref_type, "ref_id": ref_id}):
        mapping = dict(row._mapping)
        element_id = mapping["dokumen_elemen_id"]
        current_row = {
            "dev_id": int(mapping["dev_id"]),
            "page": int(mapping["dev_page"]) if mapping["dev_page"] is not None else None,
            "element_id": int(element_id) if element_id is not None else None,
            "label": mapping["dev_label"],
            "text": mapping["dev_text"] or "",
            "bbox": [
                float(mapping["dev_bbox_x0"]),
                float(mapping["dev_bbox_y0"]),
                float(mapping["dev_bbox_x1"]),
                float(mapping["dev_bbox_y1"]),
            ],
        }
        visual_rows.append(current_row)
        if current_row["element_id"] is None:
            null_rows.append(current_row)
        elif current_row["element_id"] not in body_element_ids:
            foreign_rows.append(current_row)

    actionable_null_rows = [
        row for row in null_rows
        if evalmod.is_actionable_orphan_label(row.get("label"))
    ]
    actionable_foreign_rows = [
        row for row in foreign_rows
        if evalmod.is_actionable_orphan_label(row.get("label"))
    ]

    if pdf_path.exists() and pdf_path.is_file():
        pdf_cache = evalmod.PdfWordCache(pdf_path)
        try:
            page_heights = {
                row["page"]: pdf_cache.get_page_height(row["page"])
                for row in visual_rows
                if row["page"] is not None
            }
        finally:
            pdf_cache.close()

    grouped = evalmod.group_rows_by_element(visual_rows, body_element_ids)
    missing_examples = []
    duplicate_examples = []
    valid_continuation_examples = []
    order_violations = []
    duplicate_class_counts = Counter()

    missing_body_ids = body_element_ids - set(grouped)
    for element_id in sorted(missing_body_ids, key=lambda item: (body_elements[item]["sequence"] or 0, item)):
        meta = body_elements[element_id]
        missing_examples.append(
            {
                "element_id": element_id,
                "sequence": meta["sequence"],
                "element_type": meta["type"],
                "issue_class": classify_element_issue(meta["type"], meta["openxml_text_norm"]),
                "text_preview": meta["openxml_text"][:200],
            }
        )

    ordered_body = []
    for element_id, rows in grouped.items():
        representative = evalmod.choose_representative_row(rows)
        pages = sorted({row["page"] for row in rows if row["page"] is not None})
        meta = body_elements[element_id]
        if len(pages) > 1:
            duplicate_analysis = evalmod.analyze_element_rows_for_duplicates(rows, page_heights=page_heights)
            example_payload = {
                "element_id": element_id,
                "sequence": meta["sequence"],
                "element_type": meta["type"],
                "pages": pages,
                "row_count": len(rows),
                "label_set": sorted({str(row.get("label") or "") for row in rows}),
                "text_preview": meta["openxml_text"][:200],
            }
            if duplicate_analysis["is_valid_continuation"]:
                valid_continuation_examples.append(example_payload)
            else:
                issue_class = classify_element_issue(meta["type"], meta["openxml_text_norm"])
                duplicate_class_counts[issue_class] += 1
                duplicate_examples.append(
                    {
                        **example_payload,
                        "issue_class": issue_class,
                        "invalid_pairs": duplicate_analysis.get("invalid_pairs") or [],
                    }
                )

        bbox = representative["bbox"]
        ordered_body.append(
            {
                "element_id": element_id,
                "sequence": meta["sequence"],
                "page": representative["page"],
                "y_center": ((bbox[1] + bbox[3]) / 2) if bbox else None,
                "label": representative["label"],
                "text_preview": meta["openxml_text"][:160],
            }
        )

    ordered_body.sort(
        key=lambda item: (
            item["page"] if item["page"] is not None else 10**9,
            item["y_center"] if item["y_center"] is not None else 10**9,
            item["sequence"] if item["sequence"] is not None else 10**9,
        )
    )
    previous = None
    for current in ordered_body:
        if previous and previous["sequence"] is not None and current["sequence"] is not None:
            if current["sequence"] < previous["sequence"]:
                order_violations.append(
                    {
                        "previous": previous,
                        "current": current,
                    }
                )
        previous = current

    return {
        "ref_type": ref_type,
        "ref_id": ref_id,
        "filename": ref_row["filename"],
        "total_visual_rows": len(visual_rows),
        "null_row_count": len(null_rows),
        "foreign_row_count": len(foreign_rows),
        "actionable_null_row_count": len(actionable_null_rows),
        "actionable_foreign_row_count": len(actionable_foreign_rows),
        "actionable_orphan_row_count": len(actionable_null_rows) + len(actionable_foreign_rows),
        "null_label_counts": dict(Counter(str(row.get("label") or "") for row in null_rows).most_common()),
        "foreign_label_counts": dict(Counter(str(row.get("label") or "") for row in foreign_rows).most_common()),
        "missing_examples": missing_examples,
        "duplicate_examples": sorted(
            duplicate_examples,
            key=lambda item: (len(item["pages"]), item["row_count"], item["sequence"] or 0),
            reverse=True,
        ),
        "order_violations": order_violations,
        "null_rows": null_rows,
        "foreign_rows": foreign_rows,
        "duplicate_class_counts": dict(duplicate_class_counts),
        "valid_continuation_examples": sorted(
            valid_continuation_examples,
            key=lambda item: (len(item["pages"]), item["row_count"], item["sequence"] or 0),
            reverse=True,
        ),
    }


def summarize_diagnosis(results: List[dict]) -> dict:
    duplicate_class_counts = Counter()
    missing_class_counts = Counter()
    null_label_counts = Counter()
    foreign_label_counts = Counter()
    docs_with_order_issues = []
    docs_with_duplicate_examples = []
    docs_with_valid_continuations = []
    total_valid_cross_page_continuations = 0
    actionable_orphan_total = 0

    for result in results:
        duplicate_class_counts.update(result.get("duplicate_class_counts") or {})
        null_label_counts.update(result.get("null_label_counts") or {})
        foreign_label_counts.update(result.get("foreign_label_counts") or {})
        actionable_orphan_total += int(result.get("actionable_orphan_row_count") or 0)
        if result.get("duplicate_examples"):
            docs_with_duplicate_examples.append(
                {
                    "ref_id": result["ref_id"],
                    "filename": result["filename"],
                    "duplicate_examples": len(result["duplicate_examples"]),
                }
            )
        if result.get("valid_continuation_examples"):
            docs_with_valid_continuations.append(
                {
                    "ref_id": result["ref_id"],
                    "filename": result["filename"],
                    "valid_continuations": len(result["valid_continuation_examples"]),
                }
            )
            total_valid_cross_page_continuations += len(result["valid_continuation_examples"])
        if result.get("order_violations"):
            docs_with_order_issues.append(
                {
                    "ref_id": result["ref_id"],
                    "filename": result["filename"],
                    "order_violations": len(result["order_violations"]),
                }
            )
        for item in result.get("missing_examples") or []:
            missing_class_counts[item["issue_class"]] += 1

    return {
        "duplicate_class_counts": dict(duplicate_class_counts.most_common()),
        "missing_class_counts": dict(missing_class_counts.most_common()),
        "null_label_counts": dict(null_label_counts.most_common()),
        "foreign_label_counts": dict(foreign_label_counts.most_common()),
        "actionable_orphan_total": actionable_orphan_total,
        "total_valid_cross_page_continuations": total_valid_cross_page_continuations,
        "docs_with_duplicate_examples": docs_with_duplicate_examples,
        "docs_with_valid_continuations": docs_with_valid_continuations,
        "docs_with_order_issues": docs_with_order_issues,
    }


def main():
    parser = argparse.ArgumentParser(description="Diagnose duplicate, missing, and order root causes.")
    parser.add_argument("--ref-type", choices=["all", "dokumen", "bab"], default="dokumen")
    parser.add_argument("--min-ref-id", type=int, default=None)
    parser.add_argument("--max-ref-id", type=int, default=None)
    parser.add_argument(
        "--focus-ids",
        default="",
        help="Comma-separated ref_ids to include detailed focus output.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "reports" / "alignment_eval"),
        help="Directory to store diagnosis output.",
    )
    args = parser.parse_args()

    focus_ids = {
        int(part.strip())
        for part in str(args.focus_ids or "").split(",")
        if part.strip()
    }

    engine = evalmod.build_engine()
    with engine.connect() as conn:
        ref_rows = collect_ref_rows(conn, args.ref_type, args.min_ref_id, args.max_ref_id)
        results = [collect_ref_diagnosis(conn, ref_row) for ref_row in ref_rows]

    summary = summarize_diagnosis(results)
    focus_results = [result for result in results if result["ref_id"] in focus_ids] if focus_ids else []

    suffix = args.ref_type
    if args.min_ref_id is not None or args.max_ref_id is not None:
        range_start = args.min_ref_id if args.min_ref_id is not None else "min"
        range_end = args.max_ref_id if args.max_ref_id is not None else "max"
        suffix = f"{suffix}_{range_start}_{range_end}"

    payload = {
        "db_host_used": evalmod.DB_HOST,
        "summary": summary,
        "focus_results": focus_results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"visual_alignment_diagnosis_{suffix}.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    print(f"DB host used: {evalmod.DB_HOST}:{evalmod.DB_PORT}")
    print(f"Ref type filter: {args.ref_type}")
    if args.min_ref_id is not None or args.max_ref_id is not None:
        print(
            "Ref id range: "
            f"{args.min_ref_id if args.min_ref_id is not None else '*'}"
            f"..{args.max_ref_id if args.max_ref_id is not None else '*'}"
        )
    print(f"Refs diagnosed: {len(results)}")
    print(f"Valid cross-page continuations: {summary['total_valid_cross_page_continuations']}")
    print("Duplicate class counts:")
    for key, value in summary["duplicate_class_counts"].items():
        print(f"  {key}: {value}")
    print("Missing class counts:")
    for key, value in summary["missing_class_counts"].items():
        print(f"  {key}: {value}")
    print(f"Actionable orphan total: {summary['actionable_orphan_total']}")
    print(f"Diagnosis report: {output_path}")


if __name__ == "__main__":
    main()
