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


class MergingExtractionTargetAssignmentLoadingMixin:


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
