import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

class AlignmentMatchingTraceMixin:
    def _append_alignment_trace(self, trace_context, traversal_log, min_openxml_idx, pdf_units_count, openxml_units_count):
        doc_id = trace_context.get('doc_id')
        page_num = trace_context.get('page_num')
        if doc_id is None or page_num is None:
            return
        if not traversal_log:
            return

        phase = trace_context.get('phase', 'pass1')
        os.makedirs(self.TRACE_DIR, exist_ok=True)
        path = os.path.join(
            self.TRACE_DIR,
            f"{self.TRACE_PREFIX}_doc_{doc_id}_page_{page_num}.txt"
        )

        timestamp = datetime.now().isoformat(timespec='seconds')
        header = (
            f"=== {timestamp} doc_id={doc_id} page={page_num} phase={phase} "
            f"steps={len(traversal_log)} min_openxml_idx={min_openxml_idx} "
            f"pdf_units={pdf_units_count} openxml_units={openxml_units_count} ===\n"
        )

        def sanitize_char(value):
            if value is None:
                return ''
            text = str(value)
            return text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        try:
            with open(path, 'a', encoding='utf-8') as log_file:
                log_file.write(header)
                for entry in traversal_log:
                    char = sanitize_char(entry.get('char'))
                    action = entry.get('action') or ''
                    reason = entry.get('reason') or ''
                    matched_count = entry.get('matched_count')
                    matched_part = f" cnt:{matched_count}" if matched_count is not None else ''
                    log_file.write(
                        f"[{entry.get('step')}] "
                        f"Block{entry.get('block')} Char=\"{char}\" "
                        f"PDF[{entry.get('pdf_char_idx')}] -> U{entry.get('pdf_unit')}({entry.get('pdf_unit_id')}) "
                        f"OX[{entry.get('openxml_char_idx')}] -> U{entry.get('openxml_unit')}({entry.get('openxml_unit_id')}) "
                        f"| {action} {reason}{matched_part}\n"
                    )
                log_file.write("\n")
        except OSError:
            # Trace logging is best-effort and should never break document processing.
            return
