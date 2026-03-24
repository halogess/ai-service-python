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


class MergingExtractionTargetAssignmentMixin:
    def _parse_json_tree_value(self, raw_tree):
        return self._load_json_tree(raw_tree)

    def _extract_text_from_json_tree_value(self, json_tree):
        if not json_tree:
            return ''
        return self.alignment_service._extract_text_from_json_tree(json_tree)

    def _json_tree_has_visual_bearing_content(self, json_tree):
        if not json_tree:
            return False

        if isinstance(json_tree, dict):
            node_type = str(json_tree.get('type') or '').strip().lower()
            text_value = self._coerce_text(json_tree.get('text')).strip()
            if node_type in {'image', 'chart', 'table'}:
                return True
            if text_value:
                return True
            for key in ('children', 'content', 'items'):
                child = json_tree.get(key)
                if self._json_tree_has_visual_bearing_content(child):
                    return True
            return False

        if isinstance(json_tree, list):
            return any(self._json_tree_has_visual_bearing_content(item) for item in json_tree)

        return bool(self._coerce_text(json_tree).strip())

    def _json_tree_has_image_content(self, json_tree):
        if not json_tree:
            return False
        if isinstance(json_tree, dict):
            node_type = str(json_tree.get('type') or '').strip().lower()
            if node_type in {'image', 'chart'}:
                return True
            for key in ('children', 'content', 'items'):
                child = json_tree.get(key)
                if self._json_tree_has_image_content(child):
                    return True
            return False
        if isinstance(json_tree, list):
            return any(self._json_tree_has_image_content(item) for item in json_tree)
        return False

    def _is_trivial_inline_text(self, text):
        text_norm = self._normalize_text_value(text)
        if not text_norm:
            return True
        if len(text_norm) <= 3:
            return True
        return text_norm in {'img', 'image', 'gambar', 'figure', 'chart', 'table'}

    def _looks_like_page_number_token(self, text):
        text_norm = self._normalize_text_value(text)
        if not text_norm:
            return False
        return bool(re.fullmatch(r'\d{1,4}', text_norm))

    def _is_table_caption_text(self, text):
        text_norm = self._normalize_text_value(text)
        if not text_norm:
            return False
        return text_norm.startswith('tabel') and any(ch.isdigit() for ch in text_norm)

    def _is_figure_caption_text(self, text):
        text_norm = self._normalize_text_value(text)
        if not text_norm:
            return False
        return text_norm.startswith('gambar') and any(ch.isdigit() for ch in text_norm)

    def _is_eligible_body_target(self, element_type, text, json_tree):
        normalized_type = str(element_type or '').strip().lower()
        if normalized_type == 'bookmarkend':
            return True
        text_norm = self._normalize_text_value(text)
        if text_norm:
            return True
        return self._json_tree_has_visual_bearing_content(json_tree)

    def _derive_neighbor_figure_key(self, rows, idx):
        if not rows or idx < 0 or idx >= len(rows):
            return None
        neighbor_rows = []
        if idx > 0:
            neighbor_rows.append(rows[idx - 1])
        if idx + 1 < len(rows):
            neighbor_rows.append(rows[idx + 1])
        for row in neighbor_rows:
            json_tree = self._parse_json_tree_value(getattr(row, 'delemen_json_tree', None))
            text = self._extract_text_from_json_tree_value(json_tree)
            metadata = self.alignment_service._derive_block_metadata(
                text,
                elem_type=str(getattr(row, 'delemen_type', '') or ''),
                is_table=('table' in str(getattr(row, 'delemen_type', '') or '').strip().lower()),
                is_code_like=False,
                current_block=None,
            )
            block_kind = str(metadata.get('block_kind') or '').strip().lower()
            block_key = str(metadata.get('block_key') or '').strip()
            if block_kind in {'caption', 'figure'} and block_key:
                return block_key
        return None

    def _load_body_elements_for_ref(self, db, canonical_ref_tipe, ref_id):
        if not db or ref_id is None:
            return []

        rows = (
            db.query(
                DokumenElemen.delemen_id,
                DokumenElemen.delemen_sequence,
                DokumenElemen.delemen_type,
                DokumenElemen.delemen_json_tree
            )
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)
            .filter(
                DokumenSection.dsec_ref_tipe == canonical_ref_tipe,
                DokumenSection.dsec_ref_id == ref_id,
                DokumenPart.dpart_type == 'body'
            )
            .order_by(DokumenElemen.delemen_sequence.asc(), DokumenElemen.delemen_id.asc())
            .all()
        )

        elements = []
        block_state = {
            'current_block': {},
            'block_order': 0,
            'last_key': None,
            'last_kind': None,
        }

        mapped_rows = [row for row in (rows or []) if row.delemen_id is not None]
        for idx, row in enumerate(mapped_rows):
            element_type = str(row.delemen_type or '')
            normalized_element_type = element_type.strip().lower()
            json_tree = self._parse_json_tree_value(getattr(row, 'delemen_json_tree', None))
            text = self._extract_text_from_json_tree_value(json_tree)
            has_image_content = self._json_tree_has_image_content(json_tree)
            metadata = self.alignment_service._derive_block_metadata(
                text,
                elem_type=element_type,
                is_table=('table' in normalized_element_type),
                is_code_like=(
                    normalized_element_type.startswith('list-item') or
                    self._looks_like_code_line_text(text)
                ),
                current_block=block_state.get('current_block'),
            )
            prev_row = mapped_rows[idx - 1] if idx > 0 else None
            next_row = mapped_rows[idx + 1] if idx + 1 < len(mapped_rows) else None
            prev_type = str(getattr(prev_row, 'delemen_type', '') or '').strip().lower()
            next_type = str(getattr(next_row, 'delemen_type', '') or '').strip().lower()
            active_block = block_state.get('current_block') or {}
            text_norm = self._normalize_text_value(text)
            if (
                metadata.get('block_kind') == 'narrative' and
                active_block.get('kind') in {'code', 'algorithm'} and
                'paragraph' in normalized_element_type and
                text_norm and
                len(text_norm) <= 48 and
                (
                    prev_type.startswith('list-item') or
                    next_type.startswith('list-item') or
                    prev_type == 'code' or
                    next_type == 'code'
                )
            ):
                metadata['block_kind'] = active_block.get('kind')
                metadata['block_key'] = active_block.get('key')
                metadata['content_role'] = 'continuation_body'
                metadata['current_block'] = dict(active_block)

            if has_image_content and self._is_trivial_inline_text(text):
                figure_key = self._derive_neighbor_figure_key(mapped_rows, idx)
                if figure_key:
                    metadata['block_kind'] = 'figure'
                    metadata['block_key'] = figure_key
                    metadata['content_role'] = 'placeholder'
                    metadata['current_block'] = {
                        'kind': 'figure',
                        'key': figure_key,
                    }

            current_key = metadata.get('block_key')
            current_kind = metadata.get('block_kind')
            current_role = metadata.get('content_role')
            start_new_block = False
            if current_role in {'heading', 'continuation_heading'}:
                start_new_block = True
            elif current_key and current_key != block_state.get('last_key'):
                start_new_block = True
            elif current_kind in {'table', 'figure', 'caption'} and current_kind != block_state.get('last_kind'):
                start_new_block = True
            elif block_state.get('block_order', 0) <= 0:
                start_new_block = True

            if start_new_block:
                block_state['block_order'] += 1

            block_state['current_block'] = metadata.get('current_block') or {}
            if current_key:
                block_state['last_key'] = current_key
            if current_kind:
                block_state['last_kind'] = current_kind

            elements.append({
                'element_id': int(row.delemen_id),
                'sequence': self._try_parse_int_id(row.delemen_sequence),
                'element_type': element_type,
                'text': text,
                'block_kind': current_kind,
                'block_key': current_key,
                'content_role': current_role,
                'block_order': block_state['block_order'],
                'target_kind': 'bookmark' if normalized_element_type == 'bookmarkend' else 'body',
                'is_non_visual_proxy': normalized_element_type == 'bookmarkend',
                'is_eligible_target': self._is_eligible_body_target(element_type, text, json_tree),
                'has_image_content': has_image_content,
            })

        return elements

    def _load_note_targets_for_ref(self, db, canonical_ref_tipe, ref_id):
        if canonical_ref_tipe != 'dokumen' or not db or ref_id is None:
            return []

        rows = (
            db.query(
                DokumenNote.dnote_id,
                DokumenNote.dnote_kind,
                DokumenNote.dnote_type,
                DokumenNote.dnote_json_tree,
            )
            .filter(
                DokumenNote.dokumen_id == ref_id,
                DokumenNote.dnote_kind.in_(('footnote', 'endnote'))
            )
            .order_by(DokumenNote.dnote_id.asc())
            .all()
        )

        targets = []
        for row in rows or []:
            note_id = self._try_parse_int_id(getattr(row, 'dnote_id', None))
            if note_id is None:
                continue
            note_kind = str(getattr(row, 'dnote_kind', '') or '').strip().lower() or 'footnote'
            note_type = str(getattr(row, 'dnote_type', '') or '').strip().lower()
            if note_type in {'separator', 'continuationseparator'}:
                continue
            json_tree = self._parse_json_tree_value(getattr(row, 'dnote_json_tree', None))
            text = self._extract_text_from_json_tree_value(json_tree)
            if not self._normalize_text_value(text):
                continue
            targets.append({
                'element_id': note_id,
                'target_kind': 'note',
                'note_kind': note_kind,
                'note_type': note_type,
                'text': text,
                'block_kind': 'footnote',
                'block_key': f'{note_kind}:{note_id}',
                'content_role': 'body',
                'block_order': None,
                'is_non_visual_proxy': False,
                'is_eligible_target': True,
            })

        return targets

    def _load_header_footer_targets_for_ref(self, db, canonical_ref_tipe, ref_id):
        if not db or ref_id is None:
            return []

        rows = (
            db.query(
                DokumenPart.dpart_type,
                DokumenPart.dpart_position,
                DokumenElemen.delemen_id,
                DokumenElemen.delemen_sequence,
                DokumenElemen.delemen_type,
                DokumenElemen.delemen_json_tree,
            )
            .join(DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id)
            .join(DokumenElemen, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .filter(
                DokumenSection.dsec_ref_tipe == canonical_ref_tipe,
                DokumenSection.dsec_ref_id == ref_id,
                DokumenPart.dpart_type.in_(('header', 'footer')),
            )
            .order_by(
                DokumenPart.dpart_type.asc(),
                DokumenPart.dpart_position.asc(),
                DokumenElemen.delemen_sequence.asc(),
                DokumenElemen.delemen_id.asc(),
            )
            .all()
        )

        targets = []
        for row in rows or []:
            element_id = self._try_parse_int_id(getattr(row, 'delemen_id', None))
            if element_id is None:
                continue
            json_tree = self._parse_json_tree_value(getattr(row, 'delemen_json_tree', None))
            text = self._extract_text_from_json_tree_value(json_tree)
            text_norm = self._normalize_text_value(text)
            targets.append({
                'element_id': element_id,
                'sequence': self._try_parse_int_id(getattr(row, 'delemen_sequence', None)),
                'element_type': str(getattr(row, 'delemen_type', '') or ''),
                'target_kind': str(getattr(row, 'dpart_type', '') or '').strip().lower(),
                'text': text,
                'text_norm': text_norm,
                'block_kind': str(getattr(row, 'dpart_type', '') or '').strip().lower(),
                'block_key': f"{str(getattr(row, 'dpart_type', '') or '').strip().lower()}:{self._normalize_part_position(getattr(row, 'dpart_position', None))}",
                'content_role': 'body',
                'block_order': None,
                'is_non_visual_proxy': False,
                'is_eligible_target': bool(text_norm or self._json_tree_has_visual_bearing_content(json_tree)),
                'is_numeric_page_token': self._looks_like_page_number_token(text),
            })
        return targets

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
