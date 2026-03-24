from __future__ import annotations

import csv
import html
from datetime import datetime
from pathlib import Path
from typing import List, Optional


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
        "total_pdf_words",
        "uncovered_pdf_words",
        "uncovered_pdf_word_rate",
        "pages_with_uncovered_pdf_words",
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
        uncovered_pdf_class = metric_class(
            row.get("uncovered_pdf_word_rate"),
            good_threshold=0.01,
            bad_threshold=0.05,
            lower_is_better=True,
        )
        row_class = (
            "row-issue"
            if (
                row.get("checks_passed", 0) < 6 or
                row.get("actionable_orphan_visual_rows", 0) > 0 or
                row.get("total_visual_rows", 0) == 0 or
                row.get("uncovered_pdf_words", 0) > 0
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
            f'<td class="{uncovered_pdf_class}">{fmt_percent(row.get("uncovered_pdf_word_rate"))}</td>'
            f'<td>{fmt_number(row.get("uncovered_pdf_words"))}</td>'
            f'<td>{fmt_number(row.get("total_visual_rows"))}</td>'
            f'<td>{fmt_number(row.get("ignored_empty_paragraphs"))}</td>'
            f'<td>{fmt_number(row.get("ignored_empty_headings"))}</td>'
            f'<td>{fmt_number(row.get("checks_passed"))}/6</td>'
            "</tr>"
        )
    return "".join(rendered_rows)
