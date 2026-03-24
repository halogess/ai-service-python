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


class MergingExtractionTargetAssignmentScoringMixin:


    def _result_supports_target_assignment(self, result):
        if not result or self._try_parse_int_id(result.get('element_id')) is not None:
            return False
        if (result or {}).get('_drop_from_output'):
            return False
        source = str((result or {}).get('source') or '').strip().lower()
        if source in {'bookmark_proxy', 'body_text_proxy', 'header_footer'}:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label in {'page_header', 'page_footer'}:
            return False
        bbox = result.get('bbox')
        if not bbox or len(bbox) < 4 or self._bbox_area(bbox) <= 0.0:
            return False
        matched_unit_count = self._try_parse_int_id(result.get('matched_pdf_unit_count'))
        text_norm = self._normalize_text_value(result.get('text'))
        return bool(text_norm or (matched_unit_count is not None and matched_unit_count > 0))

    def _result_is_note_like(self, result):
        if not result:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label == 'footnote':
            return True
        source = str((result or {}).get('source') or '').strip().lower()
        if source == 'note':
            return True
        block_kind = str((result or {}).get('block_kind') or '').strip().lower()
        return block_kind == 'footnote'

    def _result_is_code_like(self, result):
        if not result:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label == 'code':
            return True
        block_kind = str((result or {}).get('block_kind') or '').strip().lower()
        if block_kind in {'code', 'algorithm'}:
            return True
        element_type = str((result or {}).get('element_type') or '').strip().lower()
        text = self._coerce_text((result or {}).get('text')).strip()
        return (
            element_type.startswith('list-item') or
            self._looks_like_code_line_text(text)
        )

    def _result_prefers_table_target(self, result):
        return self._is_table_like_visual_result(result)

    def _result_prefers_caption_target(self, result):
        return self._get_visual_label(result) == 'caption'

    def _iter_page_body_sequence_anchors(self, fused_results, body_by_id):
        ordered = []
        for row in fused_results or []:
            element_id = self._try_parse_int_id((row or {}).get('element_id'))
            target = body_by_id.get(element_id)
            if not target:
                continue
            sequence = self._try_parse_int_id(target.get('sequence'))
            bbox = (row or {}).get('bbox')
            center_y = self._get_bbox_center_y(bbox)
            if sequence is None or center_y is None:
                continue
            ordered.append({
                'sequence': sequence,
                'center_y': center_y,
                'x0': float(bbox[0]) if bbox and len(bbox) >= 4 else 0.0,
            })
        ordered.sort(key=lambda item: (item['center_y'], item['x0'], item['sequence']))
        return ordered

    def _find_sequence_anchor_window(self, result, ordered_anchors):
        center_y = self._get_bbox_center_y((result or {}).get('bbox'))
        if center_y is None or not ordered_anchors:
            return None, None

        prev_seq = None
        next_seq = None
        for anchor in ordered_anchors:
            anchor_y = anchor.get('center_y')
            if anchor_y is None:
                continue
            if anchor_y <= center_y:
                prev_seq = anchor.get('sequence')
            elif next_seq is None:
                next_seq = anchor.get('sequence')
                break
        return prev_seq, next_seq

    def _target_assignment_interval_bonus(self, candidate_seq, prev_seq, next_seq):
        if candidate_seq is None:
            return 0.0
        if prev_seq is not None and next_seq is not None:
            if prev_seq <= candidate_seq <= next_seq:
                return 0.35
            distance = min(abs(candidate_seq - prev_seq), abs(candidate_seq - next_seq))
            return -min(0.40, distance * 0.04)
        if prev_seq is not None:
            if candidate_seq >= prev_seq:
                return 0.12
            return -min(0.30, abs(candidate_seq - prev_seq) * 0.04)
        if next_seq is not None:
            if candidate_seq <= next_seq:
                return 0.12
            return -min(0.30, abs(candidate_seq - next_seq) * 0.04)
        return 0.0

    def _is_table_target(self, target):
        if not target:
            return False
        target_type = str(target.get('element_type') or '').strip().lower()
        target_block_kind = str(target.get('block_kind') or '').strip().lower()
        return 'table' in target_type or target_block_kind == 'table'

    def _is_code_like_target(self, target):
        if not target:
            return False
        target_type = str(target.get('element_type') or '').strip().lower()
        target_block_kind = str(target.get('block_kind') or '').strip().lower()
        if target_block_kind in {'code', 'algorithm'}:
            return True
        if target_type.startswith('list-item'):
            return True
        return self._looks_like_code_line_text(target.get('text'))

    def _sequence_within_assignment_window(self, candidate_seq, prev_seq, next_seq, slack=2):
        candidate_seq = self._try_parse_int_id(candidate_seq)
        prev_seq = self._try_parse_int_id(prev_seq)
        next_seq = self._try_parse_int_id(next_seq)
        if candidate_seq is None:
            return False
        if prev_seq is not None and next_seq is not None:
            return (prev_seq - slack) <= candidate_seq <= (next_seq + slack)
        if prev_seq is not None:
            return candidate_seq >= (prev_seq - slack)
        if next_seq is not None:
            return candidate_seq <= (next_seq + slack)
        return False

    def _extract_leading_code_line_number(self, text):
        coerced = self._coerce_text(text).strip()
        if not coerced:
            return None
        match = self.CODE_LINE_NUMBER_REGEX.match(coerced)
        if not match:
            return None
        number_match = re.match(r'^\s*(\d{1,3})', match.group(0))
        if not number_match:
            return None
        try:
            return int(number_match.group(1))
        except (TypeError, ValueError):
            return None

    def _normalize_code_body_text(self, text):
        normalized = self._normalize_text_value(text)
        if not normalized:
            return ''
        return re.sub(r'^\d{1,3}(?::|\)|\.|-)?', '', normalized).strip()

    def _score_body_target_candidate(self, result, target, prev_seq, next_seq):
        if not result or not target:
            return None

        target_text_norm = self._normalize_text_value(target.get('text'))
        result_text_norm = self._normalize_text_value(result.get('text'))
        result_is_picture = self._is_picture_result(result)
        target_has_image = bool(target.get('has_image_content'))
        target_block_kind = str(target.get('block_kind') or '').strip().lower()
        target_type = str(target.get('element_type') or '').strip().lower()
        target_is_table = self._is_table_target(target)
        target_is_code = self._is_code_like_target(target)

        if not target_text_norm and not (result_is_picture and target_has_image):
            return None

        similarity = self._compute_text_similarity(result_text_norm, target_text_norm)
        contains_match = bool(
            result_text_norm and (
                result_text_norm in target_text_norm or
                target_text_norm in result_text_norm
            )
        )
        exact_match = bool(result_text_norm and result_text_norm == target_text_norm)

        result_sequence = self._try_parse_int_id(result.get('element_sequence'))
        target_sequence = self._try_parse_int_id(target.get('sequence'))
        block_order_match = (
            self._try_parse_int_id(result.get('block_order')) is not None and
            self._try_parse_int_id(result.get('block_order')) == self._try_parse_int_id(target.get('block_order'))
        )
        block_key_match = bool(
            result.get('block_key') and target.get('block_key') and
            str(result.get('block_key')).strip().lower() == str(target.get('block_key')).strip().lower()
        )
        block_kind_match = bool(
            result.get('block_kind') and target.get('block_kind') and
            str(result.get('block_kind')).strip().lower() == str(target.get('block_kind')).strip().lower()
        )
        role_match = bool(
            result.get('content_role') and target.get('content_role') and
            str(result.get('content_role')).strip().lower() == str(target.get('content_role')).strip().lower()
        )

        result_is_code = self._result_is_code_like(result)
        result_is_table = self._result_prefers_table_target(result)
        result_is_caption = self._result_prefers_caption_target(result)
        target_in_window = self._sequence_within_assignment_window(target_sequence, prev_seq, next_seq, slack=2)
        result_line_number = self._extract_leading_code_line_number(result.get('text'))
        target_line_number = self._extract_leading_code_line_number(target.get('text'))
        result_code_body_norm = self._normalize_code_body_text(result.get('text'))
        target_code_body_norm = self._normalize_code_body_text(target.get('text'))
        code_body_similarity = self._compute_text_similarity(result_code_body_norm, target_code_body_norm)
        code_body_contains = bool(
            result_code_body_norm and (
                result_code_body_norm in target_code_body_norm or
                target_code_body_norm in result_code_body_norm
            )
        )
        code_body_exact = bool(result_code_body_norm and result_code_body_norm == target_code_body_norm)

        if result_is_table and not target_is_table:
            return None
        if result_is_picture:
            if not target_has_image and target_block_kind != 'figure':
                return None
            similarity = max(similarity, 0.24 if (block_key_match or target_block_kind == 'figure') else 0.12)
        if result_is_caption and target_block_kind not in {'caption', 'figure'} and 'caption' not in target_type:
            similarity *= 0.8
        if result_is_code and not target_is_code:
            strong_text_match = exact_match or contains_match or similarity >= 0.78
            if not (strong_text_match or block_order_match or block_key_match or target_in_window):
                return None
            if not strong_text_match:
                similarity *= 0.65
            if code_body_similarity > similarity:
                similarity = code_body_similarity

        if not result_text_norm:
            if not (block_order_match or block_key_match):
                return None
            similarity = max(similarity, 0.35)

        if result_text_norm and len(result_text_norm) <= 3 and not (exact_match or contains_match):
            if not (block_order_match or block_key_match):
                return None

        score = similarity * 1.9
        if exact_match:
            score += 0.75
        elif contains_match:
            score += 0.40
        if block_order_match:
            score += 0.35
        if block_key_match:
            score += 0.30
        if block_kind_match:
            score += 0.18
        if role_match:
            score += 0.12
        score += self._target_assignment_interval_bonus(target_sequence, prev_seq, next_seq)
        if result_sequence is not None and target_sequence is not None:
            gap = abs(result_sequence - target_sequence)
            score += max(-0.35, 0.25 - (gap * 0.05))
        if result_is_table and target_is_table:
            score += 0.45
            if target_in_window:
                score += 0.40
            if contains_match:
                score += 0.18
            elif result_text_norm and len(result_text_norm) <= 3:
                score += 0.12
        if result_is_code and target_is_code:
            score += 0.28
            if target_in_window:
                score += 0.24
            if target_type.startswith('list-item'):
                score += 0.10
            score += code_body_similarity * 0.45
            if code_body_exact:
                score += 0.42
            elif code_body_contains:
                score += 0.20
            if result_line_number is not None and target_line_number is not None:
                line_gap = abs(result_line_number - target_line_number)
                score += max(-0.18, 0.12 - (line_gap * 0.01))
        if result_is_caption and target_block_kind == 'caption':
            score += 0.12
        if result_is_picture and target_has_image:
            score += 0.30
            if target_block_kind == 'figure':
                score += 0.18
            if self._is_trivial_inline_text(target.get('text')):
                score += 0.10
        return score

    def _score_note_target_candidate(self, result, target):
        if not result or not target or not self._result_is_note_like(result):
            return None

        result_text_norm = self._normalize_text_value(result.get('text'))
        target_text_norm = self._normalize_text_value(target.get('text'))
        if not result_text_norm or not target_text_norm:
            return None

        similarity = self._compute_text_similarity(result_text_norm, target_text_norm)
        contains_match = result_text_norm in target_text_norm or target_text_norm in result_text_norm
        score = similarity * 2.2
        if contains_match:
            score += 0.45
        if len(result_text_norm) >= 12:
            score += 0.10
        return score

    def _assign_result_to_target(self, result, target, assignment_reason):
        if not result or not target:
            return False
        result['element_id'] = target.get('element_id')
        result['element_type'] = target.get('element_type') or target.get('target_kind')
        result['element_sequence'] = target.get('sequence')
        result['block_kind'] = target.get('block_kind')
        result['block_key'] = target.get('block_key')
        result['content_role'] = target.get('content_role')
        result['block_order'] = target.get('block_order')
        result['target_kind'] = target.get('target_kind')
        result['assigned_unclaimed_target'] = True
        result['assigned_unclaimed_target_reason'] = assignment_reason
        result['alignment_confidence'] = max(
            float(result.get('alignment_confidence') or 0.0),
            0.72 if target.get('target_kind') == 'note' else 0.68
        )
        result['candidate_source'] = (
            'document_note_assignment' if target.get('target_kind') == 'note'
            else 'document_target_assignment'
        )
        result.pop('duplicate_claim_conflict', None)
        result.pop('duplicate_claim_reason', None)
        result.pop('duplicate_claim_winner_page', None)
        result.pop('duplicate_claim_winner_element_id', None)
        result.pop('_drop_from_output', None)
        return True
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized.isdigit():
            code = int(normalized)
            return {
                0: 'left',
                1: 'center',
                2: 'right',
                3: 'both',
                4: 'distribute'
            }.get(code, normalized)
        if normalized in ('start', 'left'):
            return 'left'
        if normalized in ('end', 'right'):
            return 'right'
        if normalized in ('justify', 'both'):
            return 'both'
        if normalized == 'distribute':
            return 'distribute'
        if normalized in ('centercontinuous', 'center_continuous', 'center-continuous'):
            return 'center'
        return normalized
