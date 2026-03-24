import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingMetadataConfidenceMixin:


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
