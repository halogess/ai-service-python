import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingMetadataBlockMixin:


    @staticmethod
    def _is_env_enabled_default_true(env_name):
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _read_positive_int_env(env_name, default_value):
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else default_value
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _read_float_env(env_name, default_value, min_value=None, max_value=None):
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return default_value
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    @staticmethod
    def _collect_matched_pdf_unit_keys(alignments):
        keys = set()

        def add_unit(unit):
            if not isinstance(unit, dict):
                return
            item_idx = unit.get('item_idx')
            if item_idx is not None:
                keys.add(('item_idx', item_idx))
                return
            pdf_unit_id = unit.get('pdf_unit_id') or unit.get('unit_id')
            if pdf_unit_id:
                keys.add(('pdf_unit_id', str(pdf_unit_id)))

        for alignment in alignments or []:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    for matched_unit in cell.get('matched_pdf_units') or []:
                        add_unit(matched_unit)
            else:
                for matched_unit in alignment.get('matched_pdf_units') or []:
                    add_unit(matched_unit)
        return keys

    @classmethod
    def _count_matched_pdf_units(cls, alignments):
        return len(cls._collect_matched_pdf_unit_keys(alignments))

    @staticmethod
    def _collect_matched_openxml_indices(alignments):
        indices = set()

        def add_index(value):
            if value is None:
                return
            try:
                indices.add(int(value))
            except (TypeError, ValueError):
                return

        for alignment in alignments or []:
            add_index(alignment.get('openxml_idx'))
            for openxml_idx in alignment.get('openxml_indices') or []:
                add_index(openxml_idx)
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    add_index(cell.get('openxml_idx'))
        return indices

    @classmethod
    def _count_matched_openxml_units(cls, alignments):
        return len(cls._collect_matched_openxml_indices(alignments))

    @classmethod
    def _compute_match_coverage(cls, alignments, total_pdf_units):
        if total_pdf_units <= 0:
            return 0.0
        matched = cls._count_matched_pdf_units(alignments)
        return min(1.0, matched / total_pdf_units)

    @classmethod
    def _compute_openxml_diversity(cls, alignments):
        matched_pdf_units = cls._count_matched_pdf_units(alignments)
        if matched_pdf_units <= 0:
            return 0.0
        matched_openxml_units = cls._count_matched_openxml_units(alignments)
        return matched_openxml_units / matched_pdf_units

    @staticmethod
    def _try_parse_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_pointer_text(text):
        if not text:
            return ''
        return re.sub(r'\s+', '', str(text).strip().lower())

    def _is_program_segment_heading_text(self, text):
        heading = self._extract_structured_block_heading(
            text,
            allowed_kinds={'segmen program'},
        )
        return bool(heading)

    def _normalize_block_key(self, key):
        if not key:
            return None
        normalized = re.sub(r'\s+', ' ', str(key).strip().lower())
        return normalized or None

    def _extract_caption_block_heading(self, text):
        if not text:
            return None
        if not hasattr(self, '_extract_figure_key'):
            return None
        key = self._extract_figure_key(text)
        key = self._normalize_block_key(key)
        if not key:
            return None
        prefix = key.split(':', 1)[0]
        parent_kind = 'table' if prefix == 'table' else 'figure'
        return {
            'kind': 'caption',
            'number': key.split(':', 1)[1] if ':' in key else None,
            'key': key,
            'parent_kind': parent_kind,
            'text': str(text),
        }

    def _is_caption_block_text(self, text):
        return bool(self._extract_caption_block_heading(text))

    def _derive_block_metadata(
        self,
        text,
        *,
        item_type=None,
        elem_type=None,
        style_ids=None,
        is_code_like=False,
        is_table=False,
        is_chart=False,
        is_visual_slot=False,
        is_image_part=False,
        is_caption_text=False,
        is_header_footer=False,
        current_block=None,
    ):
        current_block = dict(current_block or {})
        raw_text = str(text or '').strip()
        normalized_text = self._normalize_pointer_text(raw_text)
        normalized_item_type = str(item_type or elem_type or '').strip().lower()
        normalized_elem_type = str(elem_type or '').strip().lower()
        style_tokens = {
            str(style_id or '').strip().lower()
            for style_id in (style_ids or [])
            if style_id is not None
        }

        block_kind = 'narrative'
        block_key = None
        content_role = 'body'
        opens_block = False
        activates_block = False

        structured_heading = self._extract_structured_block_heading(raw_text)
        caption_heading = self._extract_caption_block_heading(raw_text) if (raw_text or is_caption_text) else None

        if is_header_footer:
            block_kind = 'header_footer'
            content_role = 'header_footer'
        elif structured_heading:
            block_kind = 'algorithm' if structured_heading['kind'] == 'algoritma' else 'code'
            block_key = structured_heading['key']
            content_role = 'continuation_heading' if structured_heading['is_continuation'] else 'heading'
            opens_block = True
            activates_block = True
        elif caption_heading or is_caption_text:
            block_kind = 'caption'
            block_key = caption_heading['key'] if caption_heading else None
            content_role = 'caption'
        elif is_table or normalized_item_type in {'table', 'hline_table', 'grid_table'} or 'table' in normalized_elem_type:
            block_kind = 'table'
            content_role = 'body'
            if caption_heading and caption_heading.get('parent_kind') == 'table':
                block_key = caption_heading['key']
        elif is_chart or is_visual_slot or is_image_part or normalized_item_type in {'image', 'shape'}:
            block_kind = 'figure'
            if is_visual_slot or is_image_part or normalized_text == '[img]':
                content_role = 'placeholder'
            else:
                content_role = 'body'
            if caption_heading and caption_heading.get('parent_kind') == 'figure':
                block_key = caption_heading['key']
        elif (
            is_code_like or
            normalized_item_type == 'code' or
            normalized_elem_type == 'code' or
            normalized_elem_type.startswith('list-item')
        ):
            if current_block.get('kind') in {'code', 'algorithm'}:
                block_kind = current_block['kind']
                block_key = current_block.get('key')
            else:
                block_kind = 'code'
            content_role = 'continuation_body' if current_block.get('is_continuation') else 'body'
        elif current_block.get('kind') in {'code', 'algorithm'}:
            looks_like_bridge = (
                normalized_text.startswith('segmenprogram') or
                normalized_text.startswith('algoritma') or
                any(token in normalized_text for token in ('function', 'return', 'class', 'void', 'const', 'public', 'private', 'algoritma'))
            )
            if looks_like_bridge:
                block_kind = current_block['kind']
                block_key = current_block.get('key')
                content_role = 'continuation_body' if current_block.get('is_continuation') else 'body'

        if block_kind == 'narrative' and current_block.get('kind') in {'code', 'algorithm'}:
            # Narrative text breaks active code/algorithm blocks.
            current_block = {}

        if opens_block:
            current_block = {
                'kind': block_kind,
                'key': block_key,
                'is_continuation': content_role == 'continuation_heading',
            }
        elif activates_block and block_kind in {'figure', 'table'}:
            current_block = {
                'kind': block_kind,
                'key': block_key,
                'is_continuation': False,
            }

        return {
            'block_kind': block_kind,
            'block_key': self._normalize_block_key(block_key),
            'content_role': content_role,
            'current_block': current_block,
        }

    def _extract_structured_block_heading(self, text, allowed_kinds=None):
        if not text:
            return None
        match = self.STRUCTURED_BLOCK_HEADING_RE.search(str(text))
        if not match:
            return None
        kind = re.sub(r'\s+', ' ', str(match.group('kind') or '').strip().lower())
        if allowed_kinds and kind not in set(allowed_kinds):
            return None
        number = str(match.group('number') or '').strip()
        if not number:
            return None
        continuation_raw = str(match.group('continuation') or '')
        return {
            'kind': kind,
            'number': number,
            'key': f"{kind}:{number}",
            'is_continuation': 'lanjutan' in continuation_raw.lower(),
            'text': str(text),
        }

    def _is_code_like_openxml_unit(self, unit):
        if not isinstance(unit, dict):
            return False
        if unit.get('is_code_like_openxml') or unit.get('is_code_font') or unit.get('is_code_style'):
            return True
        elem_type = str(unit.get('elem_type') or '').strip().lower()
        if 'list-item' in elem_type or elem_type == 'code':
            return True
        return self._looks_like_code_line_text(
            unit.get('text') or unit.get('text_normalized')
        )

    def _looks_like_code_line_text(self, text):
        text = str(text or '').strip()
        if not text:
            return False
        if self.CODE_LINE_NUMBER_RE.match(text):
            return True
        if self.CODE_TEXT_HINT_RE.search(text):
            return True
        symbol_count = sum(1 for ch in text if ch in '{}[]();=<>:+-*/%#\\')
        return symbol_count >= 3

    def _count_code_like_pdf_units(self, pdf_units):
        count = 0
        for unit in pdf_units or []:
            if not isinstance(unit, dict):
                continue
            if unit.get('is_cell'):
                continue
            if unit.get('item_type') in {'table', 'hline_table', 'shape', 'image'}:
                continue
            if self._looks_like_code_line_text(unit.get('text') or unit.get('text_normalized')):
                count += 1
        return count
