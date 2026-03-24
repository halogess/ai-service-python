import difflib
import json
import logging
import os
import re
from datetime import datetime

from sqlalchemy.orm import Session

from models import (
    Bab,
    Dokumen,
    DokumenElemen,
    DokumenElemenVisual,
    DokumenFormatParagraf,
    DokumenFormatText,
    DokumenNote,
    DokumenPart,
    DokumenSection,
)
from utils.cross_page_claims import analyze_cross_page_entries

logger = logging.getLogger(__name__)


class MergingExtractionFusionTableCollapseMixin:


    def _collapse_table_visual_results_for_page(self, fused_results):
        if not fused_results:
            return []

        def is_collapsible_table_row(row):
            if not row or not self._is_table_like_visual_result(row):
                return False
            if self._try_parse_int_id((row or {}).get('element_id')) is None:
                return False
            visual_label = self._get_visual_label(row)
            if visual_label and visual_label != 'table':
                return False
            element_type = str((row or {}).get('element_type') or '').strip().lower()
            if 'caption' in element_type:
                return False
            return True

        def merge_bbox_rows(rows):
            bboxes = [row.get('bbox') for row in rows if row.get('bbox') and len(row.get('bbox')) >= 4]
            if not bboxes:
                return None
            return [
                min(float(bbox[0]) for bbox in bboxes),
                min(float(bbox[1]) for bbox in bboxes),
                max(float(bbox[2]) for bbox in bboxes),
                max(float(bbox[3]) for bbox in bboxes),
            ]

        def first_non_empty(rows, key):
            for row in rows:
                value = row.get(key)
                if value not in (None, ''):
                    return value
            return None

        grouped_rows = {}
        for row in fused_results:
            if not is_collapsible_table_row(row):
                continue
            element_id = self._try_parse_int_id((row or {}).get('element_id'))
            grouped_rows.setdefault(element_id, []).append(row)

        collapsed = []
        seen_element_ids = set()
        for row in fused_results:
            if not is_collapsible_table_row(row):
                collapsed.append(row)
                continue

            element_id = self._try_parse_int_id((row or {}).get('element_id'))
            if element_id in seen_element_ids:
                continue
            seen_element_ids.add(element_id)

            rows = grouped_rows.get(element_id) or []
            if len(rows) <= 1:
                collapsed.append(row)
                continue

            merged_row = dict(row)
            merged_row['bbox'] = merge_bbox_rows(rows)
            merged_row['label'] = 'table'
            merged_row['docling_label'] = 'table'
            merged_row['source'] = 'table_page_merge'
            merged_row['has_table_units'] = True
            merged_row['dev_label_struktural'] = first_non_empty(rows, 'dev_label_struktural') or 'tabel'

            merged_text_parts = [
                self._coerce_text(candidate.get('text')).strip()
                for candidate in rows
                if self._coerce_text(candidate.get('text')).strip()
            ]
            merged_row['text'] = '\n'.join(merged_text_parts)

            merged_row['merged_count'] = sum(
                self._try_parse_int_id(candidate.get('merged_count')) or 1
                for candidate in rows
            )

            overlap_values = [
                float(candidate.get('overlap'))
                for candidate in rows
                if candidate.get('overlap') is not None
            ]
            if overlap_values:
                merged_row['overlap'] = max(overlap_values)

            confidence_values = [
                float(candidate.get('alignment_confidence'))
                for candidate in rows
                if candidate.get('alignment_confidence') is not None
            ]
            if confidence_values:
                merged_row['alignment_confidence'] = max(confidence_values)

            block_orders = [
                self._try_parse_int_id(candidate.get('block_order'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('block_order')) is not None
            ]
            if block_orders:
                merged_row['block_order'] = min(block_orders)

            element_sequences = [
                self._try_parse_int_id(candidate.get('element_sequence'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('element_sequence')) is not None
            ]
            if element_sequences:
                merged_row['element_sequence'] = min(element_sequences)

            openxml_indices = [
                self._try_parse_int_id(candidate.get('openxml_idx'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('openxml_idx')) is not None
            ]
            if openxml_indices:
                merged_row['openxml_idx'] = min(openxml_indices)

            collapsed.append(merged_row)

        return collapsed
