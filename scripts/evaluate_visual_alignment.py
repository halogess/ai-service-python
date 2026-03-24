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
from evaluate_visual_alignment_lib.evaluation import evaluate_ref
from evaluate_visual_alignment_lib.summary import summarize


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




























































def build_payload(results, summary, orphan_visual_refs, unscoped_visual_rows):
    return {
        "db_host_used": DB_HOST,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "refs": round_nested(results),
        "orphan_visual_refs": round_nested(orphan_visual_refs),
        "unscoped_visual_rows": round_nested(unscoped_visual_rows),
        "summary": round_nested(summary),
    }




















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
        results = [
            evaluate_ref(
                conn,
                ref_row,
                aligner=ALIGNER,
                volume_base=VOLUME_BASE,
                body_elements_query=BODY_ELEMENTS_QUERY,
                all_elements_query=ALL_ELEMENTS_QUERY,
                note_ids_query=NOTE_IDS_QUERY,
                visual_rows_query=VISUAL_ROWS_QUERY,
            )
            for ref_row in ref_rows
        ]
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
