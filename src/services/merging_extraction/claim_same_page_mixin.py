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


class MergingExtractionClaimSamePageMixin:


    def _clear_same_page_covered_claims(self, page_vis_payload):
        if not page_vis_payload:
            return {'cleared_rows': 0, 'affected_pages': 0}

        cleared_rows = 0
        affected_pages = set()
        for page_num, payload in (page_vis_payload or {}).items():
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue
            rows_by_element = {}
            for row in fused_results:
                elem_id = self._try_parse_int_id((row or {}).get('element_id'))
                if elem_id is None:
                    continue
                rows_by_element.setdefault(elem_id, []).append(row)
            for rows in rows_by_element.values():
                cell_rows = [
                    row for row in rows
                    if self._is_table_like_visual_result(row) and str((row or {}).get('source') or '').strip().lower() == 'cell'
                ]
                if not cell_rows:
                    continue
                merged_cell_bbox = self.alignment_service._merge_bboxes(
                    [row.get('bbox') for row in cell_rows if row.get('bbox')]
                )
                if not merged_cell_bbox:
                    continue
                for row in rows:
                    if not self._is_table_like_visual_result(row):
                        continue
                    if str((row or {}).get('source') or '').strip().lower() == 'cell':
                        continue
                    row_bbox = row.get('bbox')
                    if not row_bbox or len(row_bbox) < 4:
                        continue
                    overlap_ratio = self._bbox_overlap_ratio(row_bbox, merged_cell_bbox)
                    if overlap_ratio < 0.98:
                        continue
                    row['_drop_from_output'] = True
                    row['duplicate_claim_reason'] = 'same_page_table_aggregate_covered_by_cells'
                    cleared_rows += 1
                    affected_pages.add(page_num)
            claimed_rows = [
                row for row in fused_results
                if self._try_parse_int_id((row or {}).get('element_id')) is not None and not (row or {}).get('_drop_from_output')
            ]
            for result in fused_results:
                if not result or self._try_parse_int_id(result.get('element_id')) is None:
                    continue
                if result.get('_drop_from_output'):
                    continue
                if not self._is_claimed_cover_clear_candidate(result):
                    continue
                covering_claim = self._find_same_page_covering_claim(result, claimed_rows)
                if not covering_claim or covering_claim is result:
                    continue
                if self._clear_visual_result_claim(
                    result,
                    'same_page_covered_claim',
                    {'page': page_num, 'result': covering_claim},
                    drop_from_output=True
                ):
                    cleared_rows += 1
                    affected_pages.add(page_num)
        return {'cleared_rows': cleared_rows, 'affected_pages': len(affected_pages)}

    def _drop_redundant_same_page_proxies(self, page_vis_payload):
        if not page_vis_payload:
            return {'dropped_rows': 0, 'affected_pages': 0}
        dropped_rows = 0
        affected_pages = set()
        for page_num, payload in (page_vis_payload or {}).items():
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue
            rows_by_element = {}
            for result in fused_results:
                elem_id = self._try_parse_int_id((result or {}).get('element_id'))
                if elem_id is None:
                    continue
                rows_by_element.setdefault(elem_id, []).append(result)
            for rows in rows_by_element.values():
                real_rows = [
                    row for row in rows
                    if str((row or {}).get('source') or '').strip().lower() not in {'bookmark_proxy', 'body_text_proxy'}
                ]
                if not real_rows:
                    continue
                for row in rows:
                    source = str((row or {}).get('source') or '').strip().lower()
                    if source not in {'bookmark_proxy', 'body_text_proxy'}:
                        continue
                    row_bbox = row.get('bbox')
                    if not row_bbox or len(row_bbox) < 4:
                        continue
                    if any(self._bbox_overlap_ratio(row_bbox, other.get('bbox')) >= 0.98 for other in real_rows):
                        row['_drop_from_output'] = True
                        dropped_rows += 1
                        affected_pages.add(page_num)
        return {'dropped_rows': dropped_rows, 'affected_pages': len(affected_pages)}

    def _assign_result_to_existing_owner(self, result, owner_result, assignment_reason):
        if not result or not owner_result:
            return False
        owner_element_id = self._try_parse_int_id(owner_result.get('element_id'))
        if owner_element_id is None:
            return False
        result['element_id'] = owner_element_id
        result['element_type'] = owner_result.get('element_type')
        result['element_sequence'] = owner_result.get('element_sequence')
        result['block_kind'] = owner_result.get('block_kind')
        result['block_key'] = owner_result.get('block_key')
        result['content_role'] = owner_result.get('content_role')
        result['block_order'] = owner_result.get('block_order')
        result['target_kind'] = owner_result.get('target_kind') or 'body'
        result['assigned_existing_owner'] = True
        result['assigned_existing_owner_reason'] = assignment_reason
        result['alignment_confidence'] = max(
            float(result.get('alignment_confidence') or 0.0),
            max(0.62, float(owner_result.get('alignment_confidence') or 0.0) - 0.08)
        )
        result['candidate_source'] = assignment_reason
        result.pop('duplicate_claim_conflict', None)
        result.pop('duplicate_claim_reason', None)
        result.pop('duplicate_claim_winner_page', None)
        result.pop('duplicate_claim_winner_element_id', None)
        result.pop('_drop_from_output', None)
        return True

    def _merge_same_page_null_fragment_into_owner(self, result, owner_result):
        if not result or not owner_result:
            return False
        if not self._is_cover_drop_candidate(result):
            return False
        overlap_ratio = self._bbox_overlap_ratio(result.get('bbox'), owner_result.get('bbox'))
        if overlap_ratio < 0.98:
            return False
        self._assign_result_to_existing_owner(result, owner_result, 'same_page_fragment_merge')
        result['_drop_from_output'] = True
        result['merged_same_page_owner_id'] = owner_result.get('element_id')
        return True

    def _reassign_null_result_to_same_page_owner(self, result, page_results):
        if not self._result_supports_target_assignment(result):
            return None
        bbox = result.get('bbox')
        if not bbox or len(bbox) < 4:
            return None

        result_text_norm = self._normalize_text_value(result.get('text'))
        result_label = self._get_visual_label(result)
        result_block_key = str(result.get('block_key') or '').strip().lower()
        result_block_order = self._try_parse_int_id(result.get('block_order'))
        result_is_table = self._result_prefers_table_target(result)
        result_is_code = self._result_is_code_like(result)
        result_is_note = self._result_is_note_like(result)
        result_is_picture_family = result_label in {'picture', 'caption'}

        best_owner = None
        best_score = None
        best_overlap = 0.0

        for owner in page_results or []:
            if owner is result or (owner or {}).get('_drop_from_output'):
                continue
            owner_element_id = self._try_parse_int_id((owner or {}).get('element_id'))
            if owner_element_id is None:
                continue
            owner_bbox = owner.get('bbox')
            if not owner_bbox or len(owner_bbox) < 4:
                continue

            overlap_ratio = self._bbox_overlap_ratio(bbox, owner_bbox)
            x_overlap = self._bbox_x_overlap_ratio(bbox, owner_bbox)
            y_overlap = self._bbox_y_overlap_ratio(bbox, owner_bbox)
            owner_text_norm = self._normalize_text_value(owner.get('text'))
            text_similarity = self._compute_text_similarity(result_text_norm, owner_text_norm)
            owner_label = self._get_visual_label(owner)
            owner_block_key = str(owner.get('block_key') or '').strip().lower()
            owner_block_order = self._try_parse_int_id(owner.get('block_order'))
            block_key_match = bool(result_block_key and owner_block_key and result_block_key == owner_block_key)
            block_order_match = result_block_order is not None and owner_block_order == result_block_order
            chart_caption_pair_match = False
            chart_caption_gap = None

            if result_is_note:
                owner_is_note = (
                    str(owner.get('target_kind') or '').strip().lower() == 'note' or
                    self._result_is_note_like(owner)
                )
                if not owner_is_note:
                    continue
            elif result_is_table:
                if not self._is_table_like_visual_result(owner):
                    continue
            elif result_is_picture_family:
                owner_picture_family = owner_label in {'picture', 'caption'}
                if owner_picture_family and {result_label, owner_label} == {'picture', 'caption'}:
                    picture_result = result if result_label == 'picture' else owner
                    caption_result = result if result_label == 'caption' else owner
                    chart_caption_pair_match = self._is_valid_same_page_chart_caption_pair(
                        picture_result,
                        caption_result,
                    )
                    if chart_caption_pair_match:
                        picture_bbox = picture_result.get('bbox') or []
                        caption_bbox = caption_result.get('bbox') or []
                        if len(picture_bbox) >= 4 and len(caption_bbox) >= 4:
                            chart_caption_gap = max(0.0, float(caption_bbox[1]) - float(picture_bbox[3]))
                if not owner_picture_family and not block_key_match:
                    continue
            elif result_is_code:
                owner_is_code = self._result_is_code_like(owner)
                strong_code_context = owner_is_code or block_key_match or block_order_match
                if not strong_code_context and overlap_ratio < 0.5 and y_overlap < 0.75:
                    continue
            else:
                labels_compatible = (
                    owner_label == result_label or
                    (result_label in {'text', 'paragraph', 'caption'} and owner_label in {'text', 'paragraph', 'caption', 'section_header'})
                )
                if not labels_compatible and not block_key_match and not block_order_match:
                    continue

            score = (overlap_ratio * 1.35) + (x_overlap * 0.25) + (y_overlap * 0.55) + (text_similarity * 0.9)
            if owner_label == result_label:
                score += 0.15
            if block_key_match:
                score += 0.35
            if block_order_match:
                score += 0.22
            if result_is_table:
                score += 0.45
                if x_overlap >= 0.65:
                    score += 0.18
                if y_overlap >= 0.18:
                    score += 0.12
            if result_is_code and self._result_is_code_like(owner):
                score += 0.30
                result_line = self._extract_leading_code_line_number(result.get('text'))
                owner_line = self._extract_leading_code_line_number(owner.get('text'))
                if result_line is not None and owner_line is not None:
                    score += max(-0.10, 0.14 - (abs(result_line - owner_line) * 0.01))
            if result_is_picture_family and owner_label == result_label:
                score += 0.18
            if chart_caption_pair_match:
                score += 1.05
                if chart_caption_gap is not None:
                    score += max(0.0, 0.16 - (min(chart_caption_gap, 80.0) / 500.0))
                if x_overlap >= 0.45:
                    score += 0.12

            if best_score is None or score > best_score:
                best_owner = owner
                best_score = score
                best_overlap = overlap_ratio

        if not best_owner:
            return None

        threshold = 1.05
        if result_is_table:
            threshold = 0.80
        elif result_is_code:
            threshold = 0.92
        elif result_is_note:
            threshold = 1.00
        elif result_is_picture_family:
            threshold = 0.88

        if best_score < threshold:
            return None

        if self._merge_same_page_null_fragment_into_owner(result, best_owner):
            return {'action': 'merged', 'owner': best_owner}
        if self._assign_result_to_existing_owner(result, best_owner, 'same_page_owner_repair'):
            return {'action': 'reassigned', 'owner': best_owner, 'overlap': best_overlap}
        return None

    def _repair_same_page_null_claims(self, page_vis_payload):
        if not page_vis_payload:
            return {
                'reassigned_claims': 0,
                'merged_fragment_rows': 0,
                'dropped_synthetic_rows': 0,
                'affected_pages': 0,
            }

        reassigned_claims = 0
        merged_fragment_rows = 0
        dropped_synthetic_rows = 0
        affected_pages = set()

        for page_num, payload in (page_vis_payload or {}).items():
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue
            claimed_rows = [
                row for row in fused_results
                if self._try_parse_int_id((row or {}).get('element_id')) is not None and not (row or {}).get('_drop_from_output')
            ]
            if not claimed_rows:
                continue

            for result in fused_results:
                if not self._result_supports_target_assignment(result):
                    continue

                covering_claim = self._find_same_page_covering_claim(result, claimed_rows)
                if covering_claim and self._merge_same_page_null_fragment_into_owner(result, covering_claim):
                    merged_fragment_rows += 1
                    dropped_synthetic_rows += 1
                    affected_pages.add(page_num)
                    continue

                repair_result = self._reassign_null_result_to_same_page_owner(result, claimed_rows)
                if not repair_result:
                    continue
                if repair_result.get('action') == 'merged':
                    merged_fragment_rows += 1
                    dropped_synthetic_rows += 1
                elif repair_result.get('action') == 'reassigned':
                    reassigned_claims += 1
                    claimed_rows.append(result)
                affected_pages.add(page_num)

        return {
            'reassigned_claims': reassigned_claims,
            'merged_fragment_rows': merged_fragment_rows,
            'dropped_synthetic_rows': dropped_synthetic_rows,
            'affected_pages': len(affected_pages),
        }
