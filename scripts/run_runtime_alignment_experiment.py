from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import evaluate_visual_alignment as evalmod  # noqa: E402
from services.merging_extraction_service import MergingExtractionService  # noqa: E402


def parse_doc_ids(raw_value: str) -> List[int]:
    doc_ids: List[int] = []
    for part in str(raw_value or "").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value not in doc_ids:
            doc_ids.append(value)
    return doc_ids


def load_ref_rows(conn) -> Dict[int, dict]:
    rows = {}
    for row in conn.execute(evalmod.REF_QUERIES["dokumen"]):
        mapping = dict(row._mapping)
        rows[int(mapping["ref_id"])] = mapping
    return rows


def load_visual_rows(doc_dir: Path) -> List[dict]:
    rows: List[dict] = []
    next_dev_id = 1
    for path in sorted(doc_dir.glob("page_*_fusion_data.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        page = int(data.get("page"))
        for result in data.get("fused_results") or []:
            bbox = result.get("bbox") or []
            if len(bbox) != 4:
                continue
            element_id = result.get("element_id")
            if element_id is not None:
                try:
                    element_id = int(element_id)
                except (TypeError, ValueError):
                    element_id = None
            rows.append(
                {
                    "dev_id": next_dev_id,
                    "page": page,
                    "element_id": element_id,
                    "bbox": [float(value) for value in bbox],
                    "label": result.get("label") or result.get("docling_label"),
                    "text": result.get("text") or "",
                }
            )
            next_dev_id += 1
    return rows


def evaluate_runtime_doc(conn, ref_row: dict, visual_rows: List[dict]) -> dict:
    ref_type = str(ref_row["ref_type"])
    ref_id = int(ref_row["ref_id"])
    pdf_rel_path = str(ref_row["pdf_path"] or "")
    pdf_path = (evalmod.VOLUME_BASE / pdf_rel_path).resolve() if pdf_rel_path else evalmod.VOLUME_BASE.resolve()

    body_elements = {}
    body_element_ids = set()
    for row in conn.execute(evalmod.BODY_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id}):
        mapping = dict(row._mapping)
        element_id = int(mapping["delemen_id"])
        openxml_text = evalmod.extract_openxml_text(mapping["delemen_json_tree"])
        openxml_text_norm = evalmod.normalize_text(openxml_text)
        if evalmod.should_ignore_body_element(mapping["delemen_type"], openxml_text_norm):
            continue
        body_elements[element_id] = {
            "sequence": int(mapping["delemen_sequence"]) if mapping["delemen_sequence"] is not None else None,
            "type": mapping["delemen_type"],
        }
        body_element_ids.add(element_id)

    all_element_ids = {
        int(row[0])
        for row in conn.execute(evalmod.ALL_ELEMENTS_QUERY, {"ref_type": ref_type, "ref_id": ref_id})
    }
    note_ids = {
        int(row[0])
        for row in conn.execute(evalmod.NOTE_IDS_QUERY, {"ref_id": ref_id})
    } if ref_type == "dokumen" else set()

    page_heights = {}
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

    body_groups = evalmod.group_rows_by_element(visual_rows, body_element_ids)
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
            analysis = evalmod.analyze_element_rows_for_duplicates(rows, page_heights=page_heights)
            if analysis["is_valid_continuation"]:
                valid_cross_page_continuations += 1
            else:
                cross_page_duplicates += 1
        elif len(rows) > 1:
            if evalmod.is_valid_same_page_chart_caption_pair(rows):
                continue
            same_page_fragments += 1
    duplicate_rate = (cross_page_duplicates / len(claimed_body_ids)) if claimed_body_ids else 0.0

    ordered_body = []
    for element_id, rows in body_groups.items():
        sequence = body_elements[element_id]["sequence"]
        if sequence is None:
            continue
        representative = evalmod.choose_representative_row(rows)
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
    order_violations = sum(
        1 for previous, current in zip(ordered_body, ordered_body[1:])
        if previous[1:] > current[1:]
    )
    order_consistency = 1.0
    if len(ordered_body) > 1:
        order_consistency = 1.0 - (order_violations / (len(ordered_body) - 1))

    total_visual_rows = len(visual_rows)
    orphan_visual_rows = sum(
        1
        for row in visual_rows
        if (
            row["element_id"] is None or
            (row["element_id"] not in all_element_ids and row["element_id"] not in note_ids)
        )
    )
    actionable_orphan_visual_rows = sum(
        1
        for row in visual_rows
        if (
            (
                row["element_id"] is None or
                (row["element_id"] not in all_element_ids and row["element_id"] not in note_ids)
            ) and
            evalmod.is_actionable_orphan_label(row.get("label"))
        )
    )

    return {
        "ref_id": ref_id,
        "filename": ref_row["filename"],
        "coverage": round(coverage, 4) if coverage is not None else None,
        "missing_non_bookmark": missing_non_bookmark,
        "missing_bookmark_end": missing_bookmark_end,
        "duplicate_rate": round(duplicate_rate, 4),
        "cross_page_duplicates": cross_page_duplicates,
        "valid_cross_page_continuations": valid_cross_page_continuations,
        "same_page_fragments": same_page_fragments,
        "order_consistency": round(order_consistency, 4),
        "order_violations": order_violations,
        "total_visual_rows": total_visual_rows,
        "orphan_visual_rows": orphan_visual_rows,
        "actionable_orphan_visual_rows": actionable_orphan_visual_rows,
    }


def summarize_results(results: List[dict]) -> dict:
    if not results:
        return {
            "documents": 0,
            "average_coverage": None,
            "average_duplicate_rate": None,
            "average_order_consistency": None,
            "total_missing_non_bookmark": 0,
            "total_actionable_orphan_visual_rows": 0,
        }
    return {
        "documents": len(results),
        "average_coverage": round(sum((row["coverage"] or 0.0) for row in results) / len(results), 4),
        "average_duplicate_rate": round(sum(row["duplicate_rate"] for row in results) / len(results), 4),
        "average_order_consistency": round(sum(row["order_consistency"] for row in results) / len(results), 4),
        "total_missing_non_bookmark": sum(row["missing_non_bookmark"] for row in results),
        "total_actionable_orphan_visual_rows": sum(row["actionable_orphan_visual_rows"] for row in results),
    }


def main():
    parser = argparse.ArgumentParser(description="Run runtime alignment experiments without DB writes.")
    parser.add_argument("--doc-ids", required=True, help="Comma-separated dokumen_id values.")
    parser.add_argument("--exp-name", required=True, help="Experiment output folder name.")
    parser.add_argument(
        "--output-root",
        default=str(REPO_ROOT / "reports" / "alignment_eval" / "experiment_runs"),
        help="Root folder for experiment outputs.",
    )
    parser.add_argument(
        "--set-env",
        action="append",
        default=[],
        help="Environment override in KEY=VALUE form. May be repeated.",
    )
    args = parser.parse_args()

    os.environ.setdefault("DB_HOST", "localhost")
    for item in args.set_env:
        key, _, value = str(item).partition("=")
        if key.strip():
            os.environ[key.strip()] = value

    doc_ids = parse_doc_ids(args.doc_ids)
    output_root = Path(args.output_root) / args.exp_name
    output_root.mkdir(parents=True, exist_ok=True)

    engine = evalmod.build_engine()
    service = MergingExtractionService()

    with engine.connect() as conn:
        ref_rows = load_ref_rows(conn)
        results = []
        for doc_id in doc_ids:
            doc_dir = output_root / f"dokumen_{doc_id}"
            doc_dir.mkdir(parents=True, exist_ok=True)
            print(f"Processing dokumen:{doc_id}")
            service.process_document(
                doc_id,
                generate_visualizations=False,
                save_to_db=False,
                output_dir=str(doc_dir),
                ref_tipe="dokumen",
            )
            visual_rows = load_visual_rows(doc_dir)
            results.append(evaluate_runtime_doc(conn, ref_rows[doc_id], visual_rows))

    payload = {
        "experiment": args.exp_name,
        "doc_ids": doc_ids,
        "env_overrides": dict(item.split("=", 1) for item in args.set_env if "=" in item),
        "summary": summarize_results(results),
        "results": results,
    }

    output_path = output_root / "summary.json"
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Summary: {output_path}")
    for row in results:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
