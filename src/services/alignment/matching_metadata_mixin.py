import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

class AlignmentMatchingMetadataMixin:
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

    def _collect_alignment_pdf_text(self, alignment):
        if not isinstance(alignment, dict):
            return ''

        matched_units = []
        if alignment.get('is_table') and alignment.get('cells'):
            for cell in alignment.get('cells') or []:
                matched_units.extend((cell or {}).get('matched_pdf_units') or [])
        else:
            matched_units = list(alignment.get('matched_pdf_units') or [])

        matched_units.sort(key=lambda unit: unit.get('item_idx', 10**9))
        text_parts = []
        for unit in matched_units:
            text = str((unit or {}).get('text') or '').strip()
            if text:
                text_parts.append(text)
        return ' '.join(text_parts).strip()

    def _alignment_has_figure_key_mismatch(self, alignment):
        if not isinstance(alignment, dict):
            return False

        openxml_text = alignment.get('element_text') or alignment.get('text') or ''
        pdf_text = self._collect_alignment_pdf_text(alignment)
        if not openxml_text or not pdf_text:
            return False

        openxml_key = None
        pdf_key = None
        if hasattr(self, '_extract_figure_key'):
            openxml_key = self._extract_figure_key(openxml_text)
            pdf_key = self._extract_figure_key(pdf_text)

        return bool(openxml_key and pdf_key and openxml_key != pdf_key)

    def _is_pointer_safe_alignment(self, alignment):
        if not isinstance(alignment, dict):
            return False
        if alignment.get('late_matched'):
            return False
        if alignment.get('is_synthetic_marker_repair'):
            return False
        if alignment.get('is_table') and alignment.get('cells'):
            return False
        if alignment.get('is_image_part') or alignment.get('is_openxml_visual_slot'):
            return False
        if self._alignment_has_figure_key_mismatch(alignment):
            return False

        if hasattr(self, '_is_caption_like_text'):
            element_text = alignment.get('element_text') or alignment.get('text') or ''
            if self._is_caption_like_text(element_text):
                support = self._compute_alignment_support_metrics(alignment)
                if support['matched_chars'] < 12 and support['match_ratio'] < 0.55:
                    return False

        return True

    def _compute_alignment_confidence(self, alignment):
        support = self._compute_alignment_support_metrics(alignment)
        matched_chars = float(support.get('matched_chars') or 0.0)
        match_ratio = float(support.get('match_ratio') or 0.0)
        unit_count = float(support.get('unit_count') or 0.0)
        score = min(1.0, (match_ratio * 0.55) + (min(matched_chars, 80.0) / 80.0 * 0.35) + (min(unit_count, 4.0) / 4.0 * 0.10))

        if self._alignment_has_figure_key_mismatch(alignment):
            score -= 0.35

        block_kind = str(alignment.get('block_kind') or '').strip().lower()
        content_role = str(alignment.get('content_role') or '').strip().lower()
        if block_kind in {'caption', 'figure'} and matched_chars < 12:
            score -= 0.10
        if content_role == 'placeholder' and unit_count <= 1:
            score -= 0.08

        return max(0.0, min(1.0, round(score, 4)))

    def _annotate_alignment_confidence(self, alignments, candidate_source=None):
        for alignment in alignments or []:
            if not isinstance(alignment, dict):
                continue
            alignment['alignment_confidence'] = self._compute_alignment_confidence(alignment)
            if candidate_source is not None:
                alignment['candidate_source'] = candidate_source
        return alignments

    @staticmethod
    def _alignment_top_y(alignment):
        bbox = (alignment or {}).get('merged_bbox')
        if bbox and len(bbox) >= 4:
            try:
                return float(bbox[1])
            except (TypeError, ValueError):
                pass
        tops = []
        for unit in (alignment or {}).get('matched_pdf_units') or []:
            bbox = unit.get('bbox')
            if bbox and len(bbox) >= 4:
                try:
                    tops.append(float(bbox[1]))
                except (TypeError, ValueError):
                    continue
        return min(tops) if tops else None

    def _is_heading_context_alignment(self, alignment):
        if not isinstance(alignment, dict):
            return False
        if alignment.get('is_table') or alignment.get('is_image_part'):
            return False
        block_kind = str(alignment.get('block_kind') or '').strip().lower()
        content_role = str(alignment.get('content_role') or '').strip().lower()
        if block_kind not in {'code', 'algorithm'}:
            return False
        if content_role not in {'heading', 'continuation_heading'}:
            return False
        support = self._compute_alignment_support_metrics(alignment)
        return int(support.get('matched_chars') or 0) >= 12

    def _is_block_context_body_alignment(self, alignment):
        if not isinstance(alignment, dict):
            return False
        if alignment.get('is_table') or alignment.get('is_image_part'):
            return False
        if self._is_heading_context_alignment(alignment):
            return False
        block_kind = str(alignment.get('block_kind') or '').strip().lower()
        elem_type = str(alignment.get('element_type') or '').strip().lower()
        if block_kind in {'code', 'algorithm'}:
            return True
        if bool(alignment.get('is_code_like_openxml')) or bool(alignment.get('is_code_font')) or bool(alignment.get('is_code_style')):
            return True
        return elem_type.startswith('list-item')

    def _rebind_alignment_to_openxml_unit(self, alignment, openxml_unit, openxml_idx):
        if not isinstance(alignment, dict) or not isinstance(openxml_unit, dict):
            return alignment
        alignment['element_id'] = openxml_unit.get('elem_id')
        alignment['element_sequence'] = openxml_unit.get('elem_seq')
        alignment['element_type'] = openxml_unit.get('elem_type')
        alignment['element_text'] = openxml_unit.get('text')
        alignment['unit_id'] = str(openxml_unit.get('elem_id') or alignment.get('unit_id') or '')
        alignment['openxml_idx'] = openxml_idx
        alignment['openxml_indices'] = [openxml_idx]
        alignment['image_index'] = openxml_unit.get('image_index')
        alignment['font_families'] = openxml_unit.get('font_families', [])
        alignment['style_ids'] = openxml_unit.get('style_ids', [])
        alignment['is_code_font'] = openxml_unit.get('is_code_font', False)
        alignment['is_code_style'] = openxml_unit.get('is_code_style', False)
        alignment['is_code_like_openxml'] = openxml_unit.get('is_code_like_openxml', False)
        alignment['is_openxml_chart'] = openxml_unit.get('is_openxml_chart', False)
        alignment['is_openxml_visual_slot'] = openxml_unit.get('is_openxml_visual_slot', False)
        alignment['is_chart_caption_text'] = openxml_unit.get('is_chart_caption_text', False)
        alignment['block_kind'] = openxml_unit.get('block_kind')
        alignment['block_key'] = openxml_unit.get('block_key')
        alignment['content_role'] = openxml_unit.get('content_role')
        alignment['block_order'] = openxml_unit.get('block_order')
        return alignment

    def _remap_block_context_drift_alignments(self, alignments, openxml_units):
        if not alignments or not openxml_units:
            return alignments, []

        headings = []
        for alignment in alignments:
            if not self._is_heading_context_alignment(alignment):
                continue
            top_y = self._alignment_top_y(alignment)
            block_order = self._try_parse_int(alignment.get('block_order'))
            element_seq = self._try_parse_int(alignment.get('element_sequence'))
            if top_y is None or block_order is None or element_seq is None:
                continue
            headings.append({
                'alignment': alignment,
                'top_y': top_y,
                'block_order': block_order,
                'element_sequence': element_seq,
            })

        if not headings:
            return alignments, []

        headings.sort(key=lambda item: (item['top_y'], item['element_sequence']))
        block_text_lookup = {}
        for openxml_idx, unit in enumerate(openxml_units or []):
            block_order = self._try_parse_int(unit.get('block_order'))
            if block_order is None:
                continue
            block_kind = str(unit.get('block_kind') or '').strip().lower()
            if block_kind not in {'code', 'algorithm'}:
                continue
            text_norm = self._normalize_pointer_text(unit.get('text_normalized') or unit.get('text'))
            if not text_norm:
                continue
            block_text_lookup.setdefault((block_order, text_norm), []).append((openxml_idx, unit))

        if not block_text_lookup:
            return alignments, []

        used_element_ids = {
            alignment.get('element_id')
            for alignment in alignments
            if alignment.get('element_id') is not None
        }
        debug_entries = []

        for alignment in alignments:
            if not self._is_block_context_body_alignment(alignment):
                continue
            top_y = self._alignment_top_y(alignment)
            if top_y is None:
                continue

            heading_idx = None
            for idx, heading in enumerate(headings):
                if top_y > heading['top_y']:
                    heading_idx = idx
                else:
                    break
            if heading_idx is None:
                continue

            heading = headings[heading_idx]
            next_heading = headings[heading_idx + 1] if heading_idx + 1 < len(headings) else None
            if next_heading and top_y >= next_heading['top_y']:
                continue

            target_block_order = heading['block_order']
            current_block_order = self._try_parse_int(alignment.get('block_order'))
            if current_block_order == target_block_order:
                continue

            text_norm = self._normalize_pointer_text(alignment.get('element_text') or '')
            if len(text_norm) < 3:
                continue

            candidates = block_text_lookup.get((target_block_order, text_norm), [])
            if not candidates:
                continue

            heading_seq = heading['element_sequence']
            next_heading_seq = next_heading['element_sequence'] if next_heading else None
            filtered_candidates = []
            for openxml_idx, unit in candidates:
                elem_id = unit.get('elem_id')
                elem_seq = self._try_parse_int(unit.get('elem_seq'))
                if elem_id is None or elem_id in used_element_ids:
                    continue
                if elem_seq is None or elem_seq < heading_seq:
                    continue
                if next_heading_seq is not None and elem_seq >= next_heading_seq:
                    continue
                filtered_candidates.append((openxml_idx, unit))

            if not filtered_candidates:
                continue

            target_openxml_idx, target_unit = min(
                filtered_candidates,
                key=lambda item: abs((self._try_parse_int(item[1].get('elem_seq')) or heading_seq) - heading_seq)
            )
            old_element_id = alignment.get('element_id')
            old_element_sequence = alignment.get('element_sequence')
            self._rebind_alignment_to_openxml_unit(alignment, target_unit, target_openxml_idx)
            used_element_ids.discard(old_element_id)
            used_element_ids.add(alignment.get('element_id'))
            debug_entries.append({
                'from_element_id': old_element_id,
                'from_sequence': old_element_sequence,
                'to_element_id': alignment.get('element_id'),
                'to_sequence': alignment.get('element_sequence'),
                'heading_sequence': heading_seq,
                'target_block_order': target_block_order,
                'text': (alignment.get('element_text') or '')[:120],
            })

        if debug_entries:
            alignments.sort(key=lambda item: item.get('element_sequence') or 0)
        return alignments, debug_entries
