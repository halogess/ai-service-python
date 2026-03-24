from copy import deepcopy
import os
import re


class AlignmentPostprocessParagraphRescueMixin:


    def _rescue_paragraph_alignments(self, rescue_candidates, alignments, unaligned_pdf_indices, pdf_units):
        if not rescue_candidates or not unaligned_pdf_indices:
            return alignments, unaligned_pdf_indices, []

        pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox = self._build_pdf_lookup_maps(pdf_units)
        unaligned_set = set(unaligned_pdf_indices or [])
        existing_element_ids = {
            alignment.get('element_id')
            for alignment in alignments or []
            if alignment.get('element_id') is not None
        }

        min_units = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RESCUE_MIN_UNITS', 2)
        min_chars = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RESCUE_MIN_MATCH_CHARS', 18)
        min_ratio = self._read_float_env(
            'ALIGNMENT_PARAGRAPH_RESCUE_MIN_MATCH_RATIO',
            0.05,
            min_value=0.0,
            max_value=1.0
        )
        min_availability_ratio = self._read_float_env(
            'ALIGNMENT_PARAGRAPH_RESCUE_MIN_AVAILABILITY_RATIO',
            0.6,
            min_value=0.0,
            max_value=1.0
        )
        long_text_len = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_LONG_TEXT_LEN', 160)

        rescue_debug = []
        ordered_candidates = sorted(
            rescue_candidates,
            key=lambda alignment: (
                self._get_alignment_sequence(alignment),
                self._get_alignment_min_item_idx(alignment) or 2**31 - 1
            )
        )

        for candidate in ordered_candidates:
            if not self._is_paragraph_like_alignment(candidate):
                continue

            element_id = candidate.get('element_id')
            if element_id is not None and element_id in existing_element_ids:
                continue

            candidate_units = candidate.get('matched_pdf_units', []) or []
            is_short_caption_fragment = self._is_short_caption_fragment_text(
                candidate.get('element_text')
            )
            candidate_min_units = 1 if is_short_caption_fragment else min_units
            candidate_min_chars = 0 if is_short_caption_fragment else min_chars
            candidate_min_ratio = 0.0 if is_short_caption_fragment else min_ratio

            if len(candidate_units) < candidate_min_units:
                continue

            rescued_units = []
            rescued_indices = []
            seen_indices = set()
            for unit in candidate_units:
                pdf_idx = self._resolve_unit_pdf_index(
                    unit,
                    pdf_idx_by_unit_id,
                    pdf_idx_by_item_idx,
                    pdf_idx_by_bbox
                )
                if pdf_idx is None or pdf_idx not in unaligned_set or pdf_idx in seen_indices:
                    continue
                rescued_units.append(deepcopy(unit))
                rescued_indices.append(pdf_idx)
                seen_indices.add(pdf_idx)

            if len(rescued_units) < candidate_min_units:
                continue

            availability_ratio = len(rescued_units) / max(1, len(candidate_units))
            if (
                availability_ratio < min_availability_ratio and
                self._paragraph_text_len(candidate) < long_text_len
            ):
                continue

            rescued_alignment = deepcopy(candidate)
            rescued_alignment['matched_pdf_units'] = sorted(
                rescued_units,
                key=lambda unit: unit.get('item_idx', -1)
            )
            self._recompute_alignment_bboxes(rescued_alignment)
            if self._paragraph_rescue_conflicts(rescued_alignment, alignments):
                continue

            matched_chars, ratio, norm_len = self._paragraph_match_stats(
                rescued_alignment,
                rescued_alignment.get('matched_pdf_units', [])
            )
            if norm_len > 0 and matched_chars < candidate_min_chars and ratio < candidate_min_ratio:
                continue

            alignments.append(rescued_alignment)
            if element_id is not None:
                existing_element_ids.add(element_id)
            for pdf_idx in rescued_indices:
                unaligned_set.discard(pdf_idx)

            rescue_debug.append({
                'element_id': element_id,
                'element_sequence': rescued_alignment.get('element_sequence'),
                'rescued_units': len(rescued_indices),
                'availability_ratio': availability_ratio,
                'matched_chars': matched_chars,
                'match_ratio': ratio,
            })

        if rescue_debug:
            alignments.sort(key=lambda alignment: alignment.get('element_sequence') or 0)
        return alignments, sorted(unaligned_set), rescue_debug

    def _rescue_fragment_paragraph_alignments(self, openxml_units, alignments, page_sequence_range=None):
        if not openxml_units or not alignments:
            return alignments, []

        seq_min = seq_max = None
        if page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range

        existing_element_ids = {
            alignment.get('element_id')
            for alignment in alignments or []
            if alignment.get('element_id') is not None
        }
        seen_candidate_ids = set()
        rescue_debug = []

        for openxml_idx, openxml_unit in enumerate(openxml_units or []):
            if openxml_unit.get('is_cell'):
                continue

            elem_id = openxml_unit.get('elem_id')
            elem_seq = openxml_unit.get('elem_seq')
            elem_type = str(openxml_unit.get('elem_type') or '').strip().lower()
            text = str(openxml_unit.get('text') or '').strip()
            text_norm = self._normalize_text(text).strip()

            if elem_id is None or elem_id in existing_element_ids or elem_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(elem_id)
            if seq_min is not None and (elem_seq is None or elem_seq < seq_min):
                continue
            if seq_max is not None and (elem_seq is None or elem_seq > seq_max):
                continue
            if 'paragraph' not in elem_type or not text_norm:
                continue

            prev_alignment = self._find_neighbor_alignment_by_sequence(
                alignments,
                elem_seq,
                direction=-1,
                max_gap=2
            )
            next_alignment = self._find_neighbor_alignment_by_sequence(
                alignments,
                elem_seq,
                direction=1,
                max_gap=2
            )

            source_alignment = None
            reason = None

            prev_text_norm = self._normalize_text(
                (prev_alignment or {}).get('element_text') or ''
            ).strip()
            prev_figure_key = self._extract_figure_key((prev_alignment or {}).get('element_text'))
            if (
                prev_alignment is not None and
                prev_text_norm and
                text_norm != prev_text_norm and
                text_norm in prev_text_norm and
                prev_figure_key
            ):
                source_alignment = prev_alignment
                reason = 'caption_suffix_inherit'
            elif self._is_image_placeholder_only_text(text):
                if next_alignment is not None and self._alignment_has_visual_units(next_alignment):
                    source_alignment = next_alignment
                elif prev_alignment is not None and self._alignment_has_visual_units(prev_alignment):
                    source_alignment = prev_alignment
                elif next_alignment is not None:
                    source_alignment = next_alignment
                elif prev_alignment is not None:
                    source_alignment = prev_alignment
                reason = 'image_placeholder_neighbor_inherit'
            elif self._is_short_caption_fragment_text(text):
                if next_alignment is not None:
                    source_alignment = next_alignment
                elif prev_alignment is not None:
                    source_alignment = prev_alignment
                reason = 'caption_fragment_inherit'
            elif next_alignment is not None and next_alignment.get('is_table'):
                _, prev_unit = self._find_prev_meaningful_openxml_unit(
                    openxml_units,
                    openxml_idx,
                    seq_min=seq_min,
                    seq_max=seq_max
                )
                prev_key = self._extract_figure_key((prev_unit or {}).get('text'))
                if prev_key and prev_key.startswith('tabel'):
                    source_alignment = next_alignment
                    reason = 'table_lead_inherit'

            if source_alignment is None or not reason:
                continue

            inherited_alignment = self._build_inherited_alignment(
                source_alignment,
                openxml_unit,
                openxml_idx,
                reason
            )
            if inherited_alignment is None:
                continue
            if (
                not inherited_alignment.get('merged_bbox') and
                not inherited_alignment.get('matched_pdf_units')
            ):
                continue

            alignments.append(inherited_alignment)
            existing_element_ids.add(elem_id)
            rescue_debug.append({
                'element_id': elem_id,
                'element_sequence': elem_seq,
                'reason': reason,
                'source_element_id': source_alignment.get('element_id'),
                'source_sequence': source_alignment.get('element_sequence'),
            })

        if rescue_debug:
            alignments.sort(key=lambda alignment: alignment.get('element_sequence') or 0)
        return alignments, rescue_debug
