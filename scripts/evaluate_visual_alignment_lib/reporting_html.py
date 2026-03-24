from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path
from typing import Optional

from .reporting_formatting import (
    fmt_number,
    fmt_percent,
    metric_class,
    render_summary_card,
    render_table_rows,
    render_top_list,
)


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
                "Uncovered PDF Words",
                fmt_number(summary.get("total_uncovered_pdf_words")),
                "Word bbox PyMuPDF yang belum tertutup visual rows",
                "warn" if (summary.get("total_uncovered_pdf_words") or 0) > 0 else "good",
            ),
            render_summary_card(
                "Avg Uncovered PDF",
                fmt_percent(summary.get("average_uncovered_pdf_word_rate")),
                "Rata-rata word bbox PDF yang belum tertutup",
                "warn" if (summary.get("average_uncovered_pdf_word_rate") or 0) > 0.01 else "good",
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
            render_top_list(
                summary.get("top_5_highest_uncovered_pdf_word_rate", []),
                "uncovered_pdf_word_rate",
                "Uncovered PDF Words Tertinggi",
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
      <p>Aspek utama yang dievaluasi: coverage setelah paragraf kosong dan heading kosong dibuang, duplicate invalid lintas halaman, order consistency, null/foreign claim, raw orphan visual rows, actionable orphan tanpa header/footer, breakdown missing non-bookmark vs bookmarkEnd, serta uncovered PDF words yang belum tertutup visual rows.</p>
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
        <p>Tabel ini difokuskan ke metrik yang benar-benar dipakai saat ini. Coverage sudah mengecualikan paragraf kosong dan h1/h2 kosong. Actionable orphan mengabaikan page_header/page_footer agar noise header tidak terlihat seperti failure body alignment. Uncovered PDF words menunjukkan word bbox PyMuPDF yang belum tertutup visual rows sama sekali.</p>
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
              <th>PDF Uncov</th>
              <th>Uncov Words</th>
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
