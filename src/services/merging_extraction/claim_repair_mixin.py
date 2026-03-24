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


class MergingExtractionClaimRepairMixin:
    @staticmethod
    def _bbox_area(bbox):
        if not bbox or len(bbox) < 4:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return width * height

    @staticmethod
    def _bbox_x_overlap_ratio(bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        left = max(float(bbox_a[0]), float(bbox_b[0]))
        right = min(float(bbox_a[2]), float(bbox_b[2]))
        if right <= left:
            return 0.0
        width_a = max(0.0, float(bbox_a[2]) - float(bbox_a[0]))
        width_b = max(0.0, float(bbox_b[2]) - float(bbox_b[0]))
        min_width = min(width_a, width_b)
        if min_width <= 0.0:
            return 0.0
        return (right - left) / min_width

    @staticmethod
    def _bbox_y_overlap_ratio(bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        top = max(float(bbox_a[1]), float(bbox_b[1]))
        bottom = min(float(bbox_a[3]), float(bbox_b[3]))
        if bottom <= top:
            return 0.0
        height_a = max(0.0, float(bbox_a[3]) - float(bbox_a[1]))
        height_b = max(0.0, float(bbox_b[3]) - float(bbox_b[1]))
        min_height = min(height_a, height_b)
        if min_height <= 0.0:
            return 0.0
        return (bottom - top) / min_height

    @classmethod
    def _bbox_overlap_ratio(cls, bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        left = max(float(bbox_a[0]), float(bbox_b[0]))
        top = max(float(bbox_a[1]), float(bbox_b[1]))
        right = min(float(bbox_a[2]), float(bbox_b[2]))
        bottom = min(float(bbox_a[3]), float(bbox_b[3]))
        if right <= left or bottom <= top:
            return 0.0
        overlap_area = (right - left) * (bottom - top)
        min_area = min(cls._bbox_area(bbox_a), cls._bbox_area(bbox_b))
        if min_area <= 0.0:
            return 0.0
        return overlap_area / min_area

    def _visual_result_claim_score(self, result):
        source = str((result or {}).get('source') or '').strip().lower()
        confidence = float((result or {}).get('alignment_confidence') or 0.0)
        if source in {'note', 'bookmark_proxy', 'body_text_proxy'}:
            confidence = max(confidence, 0.98)
        elif source == 'header_footer':
            confidence = max(confidence, 0.95)
        if (result or {}).get('repair_reason'):
            confidence = max(0.0, confidence - 0.05)
        overlap = float((result or {}).get('overlap') or 0.0)
        area = self._bbox_area((result or {}).get('bbox'))
        text_len = len(self._coerce_text((result or {}).get('text')))
        return confidence, overlap, area, text_len

    def _visual_existing_claim_score(self, row):
        bbox = [
            getattr(row, 'dev_bbox_x0', None),
            getattr(row, 'dev_bbox_y0', None),
            getattr(row, 'dev_bbox_x1', None),
            getattr(row, 'dev_bbox_y1', None),
        ]
        area = self._bbox_area(bbox)
        text_len = len(self._coerce_text(getattr(row, 'dev_text', None)))
        # Historical rows do not store overlap, so default to 0.0.
        return 0.0, 0.0, area, text_len

    def _is_table_like_visual_result(self, result):
        if not result:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label == 'table':
            return True
        if result.get('has_table_units'):
            return True
        element_type = str(result.get('element_type') or '').strip().lower()
        return 'table' in element_type

    def _clear_visual_result_claim(self, result, reason, winner_claim=None, drop_from_output=False):
        if not result or result.get('element_id') is None:
            return False
        result['element_id'] = None
        result['duplicate_claim_conflict'] = True
        result['duplicate_claim_reason'] = reason
        source = str((result or {}).get('source') or '').strip().lower()
        synthetic_proxy_kind = str((result or {}).get('synthetic_proxy_kind') or '').strip().lower()
        can_drop_from_output = source in {'bookmark_proxy', 'body_text_proxy'} or synthetic_proxy_kind in {
            'bookmark_end',
            'body_text',
        }
        if drop_from_output and can_drop_from_output:
            result['_drop_from_output'] = True
        else:
            result.pop('_drop_from_output', None)
        if winner_claim:
            result['duplicate_claim_winner_page'] = winner_claim.get('page')
            winner_result = winner_claim.get('result') or {}
            result['duplicate_claim_winner_element_id'] = winner_result.get('element_id')
        return True

    @staticmethod
    def _is_synthetic_repair_reason(reason):
        return str(reason or '').strip().lower() in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
        }

    def _is_synthetic_rescue_result(self, result):
        if not result:
            return False
        source = str(result.get('source') or '').strip().lower()
        if source in {'bookmark_proxy', 'body_text_proxy'}:
            return True
        if not self._is_synthetic_repair_reason(result.get('repair_reason')):
            return False
        matched_unit_count = self._try_parse_int_id(result.get('matched_pdf_unit_count'))
        if matched_unit_count is None:
            matched_unit_count = 0
        return matched_unit_count <= 0

    def _is_short_ambiguous_result(self, result):
        label = self._get_visual_label(result)
        text = self._coerce_text((result or {}).get('text')).strip()
        if label in {'picture', 'caption'}:
            return True
        if not text:
            return True
        return len(text) <= 64 or self.fusion_service._is_caption_candidate(text) or text.lower().startswith('[img')

    def _is_cover_drop_candidate(self, result):
        source = str((result or {}).get('source') or '').strip().lower()
        if source in {'bookmark_proxy', 'body_text_proxy'}:
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        if repair_reason not in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
            'picture_overlap_prune',
        }:
            return False
        matched_unit_count = self._try_parse_int_id(result.get('matched_pdf_unit_count'))
        if matched_unit_count is None:
            matched_unit_count = 0
        return matched_unit_count <= 0

    def _find_same_page_covering_claim(self, result, claimed_rows):
        if not result:
            return None
        bbox = result.get('bbox')
        if not bbox or len(bbox) < 4:
            return None
        result_label = self._get_visual_label(result)

        best_candidate = None
        best_score = None
        for candidate in claimed_rows or []:
            if not candidate or candidate.get('element_id') is None:
                continue
            candidate_bbox = candidate.get('bbox')
            overlap_ratio = self._bbox_overlap_ratio(bbox, candidate_bbox)
            if overlap_ratio < 0.98:
                continue
            candidate_label = self._get_visual_label(candidate)
            labels_compatible = (
                candidate_label == result_label or
                (
                    result_label in {'text', 'paragraph', 'caption'} and
                    candidate_label in {'text', 'paragraph', 'caption', 'section_header'}
                ) or
                (result_label == 'picture' and candidate_label == 'picture') or
                (
                    self._is_table_like_visual_result(result) and
                    self._is_table_like_visual_result(candidate)
                ) or
                (
                    str((result or {}).get('repair_reason') or '').strip().lower() == 'table_lead_inherit' and
                    self._is_table_like_visual_result(candidate)
                )
            )
            if not labels_compatible:
                continue
            candidate_score = (
                overlap_ratio,
                float(candidate.get('alignment_confidence') or 0.0),
                -self._bbox_area(candidate_bbox),
            )
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        return best_candidate

    def _select_valid_same_page_table_claims(self, page_claims):
        valid_claims = [
            claim for claim in (page_claims or [])
            if self._is_table_like_visual_result(claim.get('result') or {})
        ]
        return valid_claims

    def _is_claimed_cover_clear_candidate(self, result):
        if not result:
            return False
        if self._is_synthetic_rescue_result(result):
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        return repair_reason in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
        }

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

    def _assign_null_results_to_unclaimed_targets(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'assigned_body_targets': 0, 'assigned_note_targets': 0}

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        note_targets = self._load_note_targets_for_ref(db, canonical_ref_tipe, ref_id)
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body' and target.get('is_eligible_target')
        }

        claimed_body_ids = set()
        claimed_note_ids = set()
        for payload in (page_vis_payload or {}).values():
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                if element_id in body_by_id:
                    claimed_body_ids.add(element_id)
                else:
                    claimed_note_ids.add(element_id)

        eligible_body_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]
        eligible_note_targets = [
            target for target in note_targets
            if target.get('is_eligible_target')
        ]

        assigned_body_targets = 0
        assigned_note_targets = 0

        for page_num, payload in sorted((page_vis_payload or {}).items()):
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            ordered_anchors = self._iter_page_body_sequence_anchors(fused_results, body_by_id)

            for result in fused_results:
                if not self._result_supports_target_assignment(result):
                    continue

                prev_seq, next_seq = self._find_sequence_anchor_window(result, ordered_anchors)
                result_is_table = self._result_prefers_table_target(result)
                result_is_code = self._result_is_code_like(result)
                result_is_caption = self._result_prefers_caption_target(result)

                best_target = None
                best_score = None
                best_kind = None

                if self._result_is_note_like(result):
                    for target in eligible_note_targets:
                        element_id = self._try_parse_int_id(target.get('element_id'))
                        if element_id is None or element_id in claimed_note_ids:
                            continue
                        score = self._score_note_target_candidate(result, target)
                        if score is None:
                            continue
                        if best_score is None or score > best_score:
                            best_score = score
                            best_target = target
                            best_kind = 'note'
                else:
                    candidates = []
                    for target in eligible_body_targets:
                        element_id = self._try_parse_int_id(target.get('element_id'))
                        if element_id is None:
                            continue
                        if element_id in claimed_body_ids and not (result_is_table and self._is_table_target(target)):
                            continue
                        candidates.append(target)

                    if result_is_table:
                        table_candidates = [target for target in candidates if self._is_table_target(target)]
                        if table_candidates:
                            candidates = table_candidates
                    elif result_is_caption:
                        caption_candidates = [
                            target for target in candidates
                            if str(target.get('block_kind') or '').strip().lower() in {'caption', 'figure'}
                            or 'caption' in str(target.get('element_type') or '').strip().lower()
                        ]
                        if caption_candidates:
                            candidates = caption_candidates
                    elif result_is_code:
                        preferred = [
                            target for target in candidates
                            if self._is_code_like_target(target)
                        ]
                        contextual = [
                            target for target in candidates
                            if (
                                target.get('block_key') and result.get('block_key') and
                                str(target.get('block_key')).strip().lower() == str(result.get('block_key')).strip().lower()
                            ) or (
                                self._try_parse_int_id(target.get('block_order')) is not None and
                                self._try_parse_int_id(target.get('block_order')) == self._try_parse_int_id(result.get('block_order'))
                            )
                        ]
                        deduped = []
                        seen_ids = set()
                        for target in preferred + contextual + candidates:
                            element_id = self._try_parse_int_id(target.get('element_id'))
                            if element_id is None or element_id in seen_ids:
                                continue
                            seen_ids.add(element_id)
                            deduped.append(target)
                        candidates = deduped

                    for target in candidates:
                        score = self._score_body_target_candidate(result, target, prev_seq, next_seq)
                        if score is None:
                            continue
                        if best_score is None or score > best_score:
                            best_score = score
                            best_target = target
                            best_kind = 'body'

                if not best_target:
                    continue

                threshold = 0.90
                if best_kind == 'note':
                    threshold = 1.00
                elif result_is_table:
                    threshold = 0.52
                elif result_is_code:
                    threshold = 0.72
                elif result_is_caption or self._is_picture_result(result):
                    threshold = 0.70

                if best_score is None or best_score < threshold:
                    continue

                if not self._assign_result_to_target(result, best_target, 'document_unclaimed_target'):
                    continue

                if best_kind == 'note':
                    claimed_note_ids.add(best_target['element_id'])
                    assigned_note_targets += 1
                else:
                    if not (result_is_table and self._is_table_target(best_target)):
                        claimed_body_ids.add(best_target['element_id'])
                    assigned_body_targets += 1
                    sequence = self._try_parse_int_id(best_target.get('sequence'))
                    center_y = self._get_bbox_center_y(result.get('bbox'))
                    if sequence is not None and center_y is not None:
                        ordered_anchors.append({
                            'sequence': sequence,
                            'center_y': center_y,
                            'x0': float(result.get('bbox')[0]) if result.get('bbox') else 0.0,
                        })
                        ordered_anchors.sort(key=lambda item: (item['center_y'], item['x0'], item['sequence']))

        return {
            'assigned_body_targets': assigned_body_targets,
            'assigned_note_targets': assigned_note_targets,
        }

    def _repair_document_header_footer_claims(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'reassigned_rows': 0, 'affected_pages': 0}

        header_footer_targets = [
            target for target in self._load_header_footer_targets_for_ref(db, canonical_ref_tipe, ref_id)
            if target.get('is_eligible_target')
        ]
        if not header_footer_targets:
            return {'reassigned_rows': 0, 'affected_pages': 0}

        targets_by_label = {
            'page_header': [target for target in header_footer_targets if target.get('target_kind') == 'header'],
            'page_footer': [target for target in header_footer_targets if target.get('target_kind') == 'footer'],
        }

        claimed_by_label = {'page_header': [], 'page_footer': []}
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            parsed_page_num = self._try_parse_int_id(page_num)
            for row in (payload or {}).get('fused_results') or []:
                if (row or {}).get('_drop_from_output'):
                    continue
                label = self._get_visual_label(row)
                if label not in claimed_by_label:
                    continue
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None:
                    continue
                claimed_by_label[label].append({
                    'page': parsed_page_num,
                    'element_id': element_id,
                })

        reassigned_rows = 0
        affected_pages = set()
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            parsed_page_num = self._try_parse_int_id(page_num)
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            for row in fused_results:
                if (row or {}).get('_drop_from_output'):
                    continue
                label = self._get_visual_label(row)
                if label not in targets_by_label:
                    continue
                if self._try_parse_int_id((row or {}).get('element_id')) is not None:
                    continue

                candidates = list(targets_by_label.get(label) or [])
                neighbor_ids = {
                    item['element_id']
                    for item in claimed_by_label.get(label, [])
                    if item.get('page') is not None and parsed_page_num is not None and abs(item['page'] - parsed_page_num) <= 1
                }
                exact_text = self._normalize_text_value((row or {}).get('text'))
                exact_matches = [
                    target for target in candidates
                    if target.get('text_norm') and target.get('text_norm') == exact_text
                ]
                page_number_candidates = [
                    target for target in candidates
                    if target.get('is_numeric_page_token')
                ]
                global_numeric_candidates = [
                    target for target in header_footer_targets
                    if target.get('is_numeric_page_token')
                ]

                best_target = None
                if len(exact_matches) == 1:
                    best_target = exact_matches[0]
                elif len(neighbor_ids) == 1:
                    best_target = next(
                        (target for target in candidates if self._try_parse_int_id(target.get('element_id')) in neighbor_ids),
                        None,
                    )
                elif len({self._try_parse_int_id(target.get('element_id')) for target in candidates}) == 1:
                    best_target = candidates[0]
                elif self._looks_like_page_number_token((row or {}).get('text')) and len(page_number_candidates) == 1:
                    best_target = page_number_candidates[0]
                elif self._looks_like_page_number_token((row or {}).get('text')):
                    global_numeric_ids = {
                        self._try_parse_int_id(target.get('element_id'))
                        for target in global_numeric_candidates
                        if self._try_parse_int_id(target.get('element_id')) is not None
                    }
                    if len(global_numeric_ids) == 1:
                        only_id = next(iter(global_numeric_ids))
                        best_target = next(
                            (
                                target for target in global_numeric_candidates
                                if self._try_parse_int_id(target.get('element_id')) == only_id
                            ),
                            None,
                        )

                if not best_target:
                    continue
                if not self._assign_result_to_target(row, best_target, 'document_header_footer_fallback'):
                    continue
                reassigned_rows += 1
                affected_pages.add(parsed_page_num if parsed_page_num is not None else page_num)
                claimed_by_label.setdefault(label, []).append({
                    'page': parsed_page_num,
                    'element_id': self._try_parse_int_id(best_target.get('element_id')),
                })

        return {
            'reassigned_rows': reassigned_rows,
            'affected_pages': len(affected_pages),
        }

    def _backfill_document_bookmark_proxies(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return 0

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        bookmark_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'bookmark'
        ]
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body'
        }

        claimed_ids = set()
        claimed_rows = []
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                claimed_ids.add(element_id)
                target = body_by_id.get(element_id)
                if not target:
                    continue
                sequence = self._try_parse_int_id(target.get('sequence'))
                bbox = row.get('bbox')
                if sequence is None or not bbox or len(bbox) < 4:
                    continue
                claimed_rows.append({
                    'page_num': page_num,
                    'result': row,
                    'target': target,
                })

        created_count = 0
        for bookmark in bookmark_targets:
            bookmark_id = self._try_parse_int_id(bookmark.get('element_id'))
            bookmark_seq = self._try_parse_int_id(bookmark.get('sequence'))
            if bookmark_id is None or bookmark_id in claimed_ids or bookmark_seq is None:
                continue

            best_claim = None
            best_score = None
            for claim in claimed_rows:
                target = claim.get('target') or {}
                sequence = self._try_parse_int_id(target.get('sequence'))
                if sequence is None:
                    continue
                gap = abs(sequence - bookmark_seq)
                if gap > 3:
                    continue
                score = 1.0 - (gap * 0.2)
                if sequence <= bookmark_seq:
                    score += 0.12
                if best_score is None or score > best_score:
                    best_score = score
                    best_claim = claim

            if not best_claim:
                continue

            owner_result = best_claim['result']
            proxy_result = {
                'bbox': list(owner_result.get('bbox') or []),
                'label': self._get_visual_label(owner_result) or 'text',
                'text': '',
                'overlap': float(owner_result.get('overlap') or 0.0),
                'source': 'bookmark_proxy',
                'synthetic_proxy_kind': 'bookmark_end',
                'element_id': bookmark_id,
                'element_type': bookmark.get('element_type'),
                'element_sequence': bookmark_seq,
                'block_kind': bookmark.get('block_kind') or owner_result.get('block_kind'),
                'block_key': bookmark.get('block_key') or owner_result.get('block_key'),
                'content_role': bookmark.get('content_role') or owner_result.get('content_role'),
                'block_order': bookmark.get('block_order') or owner_result.get('block_order'),
                'target_kind': 'bookmark',
                'alignment_confidence': 0.99,
                'candidate_source': 'bookmark_backfill',
                'matched_pdf_unit_count': 0,
            }
            page_vis_payload[best_claim['page_num']].setdefault('fused_results', []).append(proxy_result)
            claimed_ids.add(bookmark_id)
            created_count += 1

        return created_count

    def _backfill_document_text_proxies(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return 0

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        eligible_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]

        claimed_ids = set()
        claimed_rows = []
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                claimed_ids.add(element_id)
                bbox = row.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                claimed_rows.append({
                    'page_num': page_num,
                    'result': row,
                })

        created_count = 0
        for target in eligible_targets:
            target_id = self._try_parse_int_id(target.get('element_id'))
            target_seq = self._try_parse_int_id(target.get('sequence'))
            if target_id is None or target_id in claimed_ids or target_seq is None:
                continue

            target_is_code = self._is_code_like_target(target)
            if not target_is_code and str(target.get('content_role') or '').strip().lower() != 'continuation_body':
                continue

            best_claim = None
            best_score = None
            for claim in claimed_rows:
                row = claim.get('result') or {}
                row_seq = self._try_parse_int_id(row.get('element_sequence'))
                if row_seq is None or abs(row_seq - target_seq) > 6:
                    continue
                same_block_key = bool(
                    target.get('block_key') and row.get('block_key') and
                    str(target.get('block_key')).strip().lower() == str(row.get('block_key')).strip().lower()
                )
                same_block_order = (
                    self._try_parse_int_id(target.get('block_order')) is not None and
                    self._try_parse_int_id(target.get('block_order')) == self._try_parse_int_id(row.get('block_order'))
                )
                if not (same_block_key or same_block_order or self._result_is_code_like(row)):
                    continue
                score = 0.8 - (abs(row_seq - target_seq) * 0.08)
                if same_block_key:
                    score += 0.35
                if same_block_order:
                    score += 0.20
                if claim.get('page_num') is not None:
                    score += 0.05
                if best_score is None or score > best_score:
                    best_score = score
                    best_claim = claim

            if not best_claim or best_score is None or best_score < 0.60:
                continue

            owner_result = best_claim['result']
            proxy_result = {
                'bbox': list(owner_result.get('bbox') or []),
                'label': self._get_visual_label(owner_result) or 'text',
                'text': target.get('text') or '',
                'overlap': float(owner_result.get('overlap') or 0.0),
                'source': 'body_text_proxy',
                'synthetic_proxy_kind': 'body_text',
                'element_id': target_id,
                'element_type': target.get('element_type'),
                'element_sequence': target_seq,
                'block_kind': target.get('block_kind'),
                'block_key': target.get('block_key'),
                'content_role': target.get('content_role'),
                'block_order': target.get('block_order'),
                'target_kind': 'body',
                'alignment_confidence': 0.97,
                'candidate_source': 'body_text_backfill',
                'matched_pdf_unit_count': 0,
            }
            page_vis_payload[best_claim['page_num']].setdefault('fused_results', []).append(proxy_result)
            claimed_ids.add(target_id)
            claimed_rows.append({
                'page_num': best_claim['page_num'],
                'result': proxy_result,
            })
            created_count += 1

        return created_count

    def _derive_visual_chain_key(self, result):
        if not result:
            return None
        block_key = str((result or {}).get('block_key') or '').strip().lower()
        if block_key:
            return block_key
        text = self._coerce_text((result or {}).get('text')).strip()
        if not text:
            return None
        metadata = self.alignment_service._derive_block_metadata(
            text,
            elem_type=str((result or {}).get('element_type') or self._get_visual_label(result) or ''),
            is_table=self._is_table_like_visual_result(result),
            is_code_like=False,
            current_block=None,
        )
        derived_key = str((metadata or {}).get('block_key') or '').strip().lower()
        return derived_key or None

    def _is_visual_chain_repair_candidate(self, result):
        if not self._result_supports_target_assignment(result):
            return False
        visual_label = self._get_visual_label(result)
        if visual_label in {'picture', 'caption'}:
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        return repair_reason in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'picture_overlap_prune',
        }

    def _repair_adjacent_page_visual_chains(self, page_vis_payload):
        if not page_vis_payload:
            return {'reassigned_rows': 0, 'dropped_rows': 0, 'affected_pages': 0}

        page_numbers = sorted(
            self._try_parse_int_id(page_num)
            for page_num in page_vis_payload.keys()
            if self._try_parse_int_id(page_num) is not None
        )
        reassigned_rows = 0
        dropped_rows = 0
        affected_pages = set()

        for index, page_num in enumerate(page_numbers):
            payload = page_vis_payload.get(page_num) or page_vis_payload.get(str(page_num)) or {}
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            adjacent_rows = []
            for neighbor_page in (page_numbers[index - 1:index] + page_numbers[index + 1:index + 2]):
                neighbor_payload = page_vis_payload.get(neighbor_page) or page_vis_payload.get(str(neighbor_page)) or {}
                for row in (neighbor_payload or {}).get('fused_results') or []:
                    if self._try_parse_int_id((row or {}).get('element_id')) is None:
                        continue
                    if not self._is_visual_chain_repair_candidate(row):
                        continue
                    adjacent_rows.append(row)

            if not adjacent_rows:
                continue

            for result in fused_results:
                if not self._is_visual_chain_repair_candidate(result):
                    continue
                if self._try_parse_int_id((result or {}).get('element_id')) is not None:
                    continue

                chain_key = self._derive_visual_chain_key(result)
                if not chain_key:
                    continue
                result_bbox = result.get('bbox')
                result_label = self._get_visual_label(result)
                best_owner = None
                best_score = None

                for owner in adjacent_rows:
                    owner_key = self._derive_visual_chain_key(owner)
                    if not owner_key or owner_key != chain_key:
                        continue
                    owner_label = self._get_visual_label(owner)
                    if result_label and owner_label and result_label != owner_label:
                        if {result_label, owner_label} != {'picture', 'caption'}:
                            continue
                    score = 1.1
                    if owner_label == result_label:
                        score += 0.18
                    if result_bbox and owner.get('bbox'):
                        score += self._bbox_x_overlap_ratio(result_bbox, owner.get('bbox')) * 0.25
                    if best_score is None or score > best_score:
                        best_score = score
                        best_owner = owner

                if not best_owner or best_score is None or best_score < 1.10:
                    continue

                if self._assign_result_to_existing_owner(result, best_owner, 'adjacent_page_visual_chain'):
                    reassigned_rows += 1
                    affected_pages.add(page_num)
                elif self._merge_same_page_null_fragment_into_owner(result, best_owner):
                    dropped_rows += 1
                    affected_pages.add(page_num)

        return {
            'reassigned_rows': reassigned_rows,
            'dropped_rows': dropped_rows,
            'affected_pages': len(affected_pages),
        }

    def _score_invalid_duplicate_target_candidate(
        self,
        rows,
        target,
        prev_seq,
        next_seq,
        page_claimed_ids,
        page_table_caption_sequences=None,
        page_element_row_counts=None,
    ):
        if not rows or not target:
            return None
        scores = []
        row_count = 0
        for row in rows:
            score = self._score_body_target_candidate(row, target, prev_seq, next_seq)
            row_count += 1
            if score is not None:
                scores.append(score)
        if not scores:
            return None

        avg_score = sum(scores) / len(scores)
        coverage_bonus = (len(scores) / max(1, row_count)) * 0.45
        candidate_id = self._try_parse_int_id(target.get('element_id'))
        candidate_seq = self._try_parse_int_id(target.get('sequence'))
        target_text_norm = self._normalize_text_value(target.get('text'))
        cluster_is_table = any(self._is_table_like_visual_result(row) for row in rows)
        cluster_is_code = any(self._result_is_code_like(row) for row in rows)
        cluster_text_norm = self._normalize_text_value(
            ' '.join(self._coerce_text((row or {}).get('text')) for row in rows)
        )
        cluster_is_caption = any(
            self._get_visual_label(row) == 'caption' or self._is_table_caption_text((row or {}).get('text'))
            for row in rows
        ) or cluster_text_norm in {'lanjutan', '(lanjutan)'}
        continuation_only_caption = cluster_text_norm in {'lanjutan', '(lanjutan)'}
        claimed_same_page = candidate_id in (page_claimed_ids or set())
        page_claim_count = 0
        if page_element_row_counts and candidate_id is not None:
            page_claim_count = int(page_element_row_counts.get(candidate_id) or 0)
        avg_alignment_confidence = sum(
            float((row or {}).get('alignment_confidence') or 0.0)
            for row in rows
        ) / max(1, len(rows))
        candidate_in_window = self._sequence_within_assignment_window(candidate_seq, prev_seq, next_seq, slack=2)

        total = avg_score + coverage_bonus
        if claimed_same_page:
            total += 0.28
        if cluster_is_table and self._is_table_target(target):
            total += 0.42
        if cluster_is_code and self._is_code_like_target(target):
            total += 0.20
            if candidate_in_window:
                total += 0.72
            elif prev_seq is not None or next_seq is not None:
                total -= 0.95
                if avg_alignment_confidence < 0.60:
                    total -= 0.18
        if candidate_in_window:
            total += 0.18
        if (cluster_is_table or cluster_is_caption) and page_claim_count > 0:
            total += min(0.36, page_claim_count * 0.04)
        if cluster_is_table and candidate_seq is not None and page_table_caption_sequences:
            closest_caption_gap = min(
                abs(candidate_seq - seq)
                for seq in page_table_caption_sequences
                if seq is not None
            )
            if closest_caption_gap <= 1:
                total += 0.72
            elif closest_caption_gap <= 2:
                total += 0.48
            elif closest_caption_gap <= 4:
                total += 0.18
            elif closest_caption_gap >= 8:
                total -= 0.22
        if cluster_is_caption and candidate_seq is not None and page_table_caption_sequences:
            closest_caption_gap = min(
                abs(candidate_seq - seq)
                for seq in page_table_caption_sequences
                if seq is not None
            )
            if closest_caption_gap <= 1:
                total += 0.78
            elif closest_caption_gap <= 2:
                total += 0.44
            elif closest_caption_gap >= 8:
                total -= 0.26
            if continuation_only_caption:
                if target_text_norm in {'lanjutan', '(lanjutan)'}:
                    if closest_caption_gap <= 1:
                        total += 0.30
                    elif closest_caption_gap >= 8:
                        total -= 1.10
                elif self._is_table_caption_text(target.get('text')):
                    if closest_caption_gap <= 1:
                        total += 1.05
        return total

    def _repair_invalid_duplicate_claims_to_local_targets(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'reassigned_rows': 0, 'repaired_elements': 0, 'affected_pages': 0}

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body' and target.get('is_eligible_target')
        }
        candidate_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]

        rows_by_element = {}
        for page_num, payload in (page_vis_payload or {}).items():
            fused_results = list((payload or {}).get('fused_results') or [])
            for row in fused_results:
                if (row or {}).get('_drop_from_output'):
                    continue
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or element_id not in body_by_id:
                    continue
                rows_by_element.setdefault(element_id, []).append({
                    'page': self._try_parse_int_id(page_num),
                    'row': row,
                })

        invalid_groups = {}
        for element_id, entries in rows_by_element.items():
            analysis = analyze_cross_page_entries(
                [{'page': item['page'], 'bbox': (item['row'] or {}).get('bbox')} for item in entries]
            )
            if analysis.get('is_invalid_duplicate'):
                invalid_groups[element_id] = {
                    'analysis': analysis,
                    'entries': entries,
                }

        if not invalid_groups:
            return {'reassigned_rows': 0, 'repaired_elements': 0, 'affected_pages': 0}

        invalid_element_ids = set(invalid_groups)
        reassigned_rows = 0
        repaired_elements = 0
        affected_pages = set()

        for element_id, group in invalid_groups.items():
            current_target = body_by_id.get(element_id)
            if not current_target:
                continue
            entries_by_page = {}
            for item in group['entries']:
                entries_by_page.setdefault(item['page'], []).append(item['row'])

            for page_num, rows in entries_by_page.items():
                payload = page_vis_payload.get(page_num) or page_vis_payload.get(str(page_num)) or {}
                fused_results = list((payload or {}).get('fused_results') or [])
                if not fused_results or not rows:
                    continue

                cluster_bbox = self.alignment_service._merge_bboxes([row.get('bbox') for row in rows if row.get('bbox')])
                if not cluster_bbox:
                    continue
                anchor_rows = [
                    row for row in fused_results
                    if not (row or {}).get('_drop_from_output')
                    and self._try_parse_int_id((row or {}).get('element_id')) in body_by_id
                    and self._try_parse_int_id((row or {}).get('element_id')) not in invalid_element_ids
                ]
                ordered_anchors = self._iter_page_body_sequence_anchors(anchor_rows, body_by_id)
                prev_seq, next_seq = self._find_sequence_anchor_window({'bbox': cluster_bbox}, ordered_anchors)
                page_claimed_ids = {
                    self._try_parse_int_id((row or {}).get('element_id'))
                    for row in fused_results
                    if self._try_parse_int_id((row or {}).get('element_id')) is not None and not (row or {}).get('_drop_from_output')
                }

                cluster_is_table = any(self._is_table_like_visual_result(row) for row in rows)
                cluster_is_code = any(self._result_is_code_like(row) for row in rows)
                cluster_is_caption = any(
                    self._get_visual_label(row) == 'caption' or self._is_table_caption_text((row or {}).get('text'))
                    for row in rows
                ) or self._normalize_text_value(
                    ' '.join(self._coerce_text((row or {}).get('text')) for row in rows)
                ) in {'lanjutan', '(lanjutan)'}
                page_table_caption_sequences = [
                    self._try_parse_int_id((body_by_id.get(self._try_parse_int_id((row or {}).get('element_id'))) or {}).get('sequence'))
                    for row in fused_results
                    if (
                        not (row or {}).get('_drop_from_output')
                        and self._get_visual_label(row) == 'caption'
                        and self._is_table_caption_text((row or {}).get('text'))
                    )
                ]
                page_table_caption_sequences = [
                    seq for seq in page_table_caption_sequences
                    if seq is not None
                ]
                page_element_row_counts = {}
                for row in fused_results:
                    if (row or {}).get('_drop_from_output'):
                        continue
                    row_element_id = self._try_parse_int_id((row or {}).get('element_id'))
                    if row_element_id is None:
                        continue
                    page_element_row_counts[row_element_id] = page_element_row_counts.get(row_element_id, 0) + 1

                target_pool = []
                for target in candidate_targets:
                    target_id = self._try_parse_int_id(target.get('element_id'))
                    if target_id is None or target_id == element_id:
                        continue
                    if cluster_is_table and not self._is_table_target(target):
                        continue
                    if cluster_is_code and not self._is_code_like_target(target):
                        continue
                    if cluster_is_table:
                        target_seq = self._try_parse_int_id(target.get('sequence'))
                        if not self._sequence_within_assignment_window(target_seq, prev_seq, next_seq, slack=2):
                            continue
                    target_pool.append(target)
                if cluster_is_caption:
                    caption_target_pool = [
                        target for target in target_pool
                        if (
                            str(target.get('block_kind') or '').strip().lower() in {'caption', 'figure'}
                            or 'caption' in str(target.get('element_type') or '').strip().lower()
                            or self._try_parse_int_id(target.get('sequence')) in page_table_caption_sequences
                        )
                    ]
                    if caption_target_pool:
                        target_pool = caption_target_pool

                best_target = None
                best_score = None
                current_score = self._score_invalid_duplicate_target_candidate(
                    rows,
                    current_target,
                    prev_seq,
                    next_seq,
                    page_claimed_ids,
                    page_table_caption_sequences=page_table_caption_sequences,
                    page_element_row_counts=page_element_row_counts,
                )

                for target in target_pool:
                    candidate_score = self._score_invalid_duplicate_target_candidate(
                        rows,
                        target,
                        prev_seq,
                        next_seq,
                        page_claimed_ids,
                        page_table_caption_sequences=page_table_caption_sequences,
                        page_element_row_counts=page_element_row_counts,
                    )
                    if candidate_score is None:
                        continue
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_target = target

                if not best_target or best_score is None:
                    continue
                required_delta = 0.25
                if cluster_is_code:
                    current_seq = self._try_parse_int_id(current_target.get('sequence'))
                    current_in_window = self._sequence_within_assignment_window(current_seq, prev_seq, next_seq, slack=2)
                    avg_alignment_confidence = sum(
                        float((row or {}).get('alignment_confidence') or 0.0)
                        for row in rows
                    ) / max(1, len(rows))
                    if not current_in_window:
                        required_delta = 0.06 if avg_alignment_confidence < 0.60 else 0.10
                if current_score is not None and best_score <= current_score + required_delta:
                    continue

                changed_here = 0
                for row in rows:
                    if self._assign_result_to_target(row, best_target, 'invalid_duplicate_local_reassign'):
                        changed_here += 1
                if changed_here > 0:
                    reassigned_rows += changed_here
                    repaired_elements += 1
                    affected_pages.add(page_num)

        return {
            'reassigned_rows': reassigned_rows,
            'repaired_elements': repaired_elements,
            'affected_pages': len(affected_pages),
        }

    def _resolve_document_visual_claims(self, page_vis_payload):
        if not page_vis_payload:
            return {
                'cleared_claims': 0,
                'affected_pages': 0,
                'same_page_cleared': 0,
                'far_gap_cleared': 0,
                'cross_page_rescue_cleared': 0,
            }

        single_page_repair_reasons = set()
        if self._is_env_enabled_default_true("ALIGNMENT_ENABLE_RESCUE_DUPLICATE_PRUNE"):
            single_page_repair_reasons = {
                'caption_suffix_inherit',
                'image_placeholder_neighbor_inherit',
                'caption_fragment_inherit',
                'table_lead_inherit',
            }

        claims_by_element = {}
        for page_num, payload in (page_vis_payload or {}).items():
            parsed_page_num = self._try_parse_int_id(page_num)
            if parsed_page_num is None:
                continue
            for result in payload.get('fused_results') or []:
                elem_id = self._try_parse_int_id((result or {}).get('element_id'))
                if elem_id is None:
                    continue
                visual_label = self._get_visual_label(result)
                if visual_label in ('page_header', 'page_footer'):
                    continue
                claims_by_element.setdefault(elem_id, []).append({
                    'page': parsed_page_num,
                    'result': result,
                    'score': self._visual_result_claim_score(result)
                })

        cleared_claims = 0
        same_page_cleared = 0
        far_gap_cleared = 0
        cross_page_rescue_cleared = 0
        affected_pages = set()

        for elem_id, claims in claims_by_element.items():
            claims_by_page = {}
            for claim in claims:
                claims_by_page.setdefault(claim['page'], []).append(claim)

            for page, page_claims in sorted(claims_by_page.items()):
                allowed_page_claims = self._select_valid_same_page_table_claims(page_claims)
                if not allowed_page_claims:
                    allowed_page_claims = self._select_valid_same_page_chart_caption_claims(page_claims)
                if allowed_page_claims:
                    allowed_ids = {id(claim) for claim in allowed_page_claims}
                    for claim in page_claims:
                        if id(claim) in allowed_ids:
                            continue
                        if self._clear_visual_result_claim(
                            claim.get('result'),
                            'same_page_duplicate',
                            allowed_page_claims[0],
                            drop_from_output=True
                        ):
                            cleared_claims += 1
                            same_page_cleared += 1
                            affected_pages.add(page)
                    continue
                winner_claim = max(page_claims, key=lambda claim: claim['score'])
                for claim in page_claims:
                    if claim is winner_claim:
                        continue
                    if self._clear_visual_result_claim(
                        claim.get('result'),
                        'same_page_duplicate',
                        winner_claim,
                        drop_from_output=True
                    ):
                        cleared_claims += 1
                        same_page_cleared += 1
                        affected_pages.add(page)

            active_claims = [
                claim for claim in claims
                if (claim.get('result') or {}).get('element_id') is not None
            ]
            if len(active_claims) <= 1:
                continue

            repair_claims = [
                claim for claim in active_claims
                if (claim.get('result') or {}).get('repair_reason') in single_page_repair_reasons
            ]
            if repair_claims:
                non_repair_claims = [
                    claim for claim in active_claims
                    if claim not in repair_claims
                ]
                if non_repair_claims:
                    winner_claim = max(non_repair_claims, key=lambda claim: claim['score'])
                else:
                    winner_claim = max(repair_claims, key=lambda claim: claim['score'])

                for claim in repair_claims:
                    if claim is winner_claim:
                        continue
                    page = claim.get('page')
                    result = claim.get('result') or {}
                    if self._clear_visual_result_claim(result, 'cross_page_rescue_duplicate', winner_claim):
                        result['_drop_from_output'] = True
                        cleared_claims += 1
                    cross_page_rescue_cleared += 1
                    if page is not None:
                        affected_pages.add(page)

        return {
            'cleared_claims': cleared_claims,
            'affected_pages': len(affected_pages),
            'same_page_cleared': same_page_cleared,
            'far_gap_cleared': far_gap_cleared,
            'cross_page_rescue_cleared': cross_page_rescue_cleared,
        }
