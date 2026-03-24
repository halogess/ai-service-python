import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

from .matching_two_pass_selection_mixin import AlignmentMatchingTwoPassSelectionMixin
from .matching_two_pass_pipeline_mixin import AlignmentMatchingTwoPassPipelineMixin


class AlignmentMatchingExecutionMixin(
    AlignmentMatchingTwoPassPipelineMixin,
    AlignmentMatchingTwoPassSelectionMixin,
):

    def _compute_max_openxml_idx_from_alignments(self, alignments, min_openxml_idx):
        max_idx = None
        for alignment in alignments or []:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment['cells']:
                    idx = cell.get('openxml_idx')
                    if idx is not None:
                        max_idx = idx if max_idx is None else max(max_idx, idx)
            else:
                indices = alignment.get('openxml_indices')
                if indices:
                    for idx in indices:
                        if idx is None:
                            continue
                        max_idx = idx if max_idx is None else max(max_idx, idx)
                else:
                    idx = alignment.get('openxml_idx')
                    if idx is not None:
                        max_idx = idx if max_idx is None else max(max_idx, idx)
        if max_idx is None:
            return min_openxml_idx
        return max(min_openxml_idx, max_idx)

    def _is_marker_only_text(self, text):
        if not text:
            return False
        return bool(self.MARKER_ONLY_TEXT_RE.match(str(text).strip()))

    def _is_bookmark_end_unit(self, unit):
        if not unit:
            return False
        return str(unit.get('elem_type') or '').strip().lower() == 'bookmarkend'

    def _repair_marker_only_alignment_gaps(self, alignments, openxml_units):
        if not alignments or not openxml_units:
            return alignments

        seq_to_alignment = {}
        for alignment in alignments:
            seq = alignment.get('element_sequence')
            if seq is None:
                continue
            seq_to_alignment[seq] = alignment

        if not seq_to_alignment:
            return alignments

        seq_to_openxml = {}
        for openxml_idx, unit in enumerate(openxml_units):
            seq = unit.get('elem_seq')
            if seq is None or seq in seq_to_openxml:
                continue
            seq_to_openxml[seq] = (openxml_idx, unit)

        if not seq_to_openxml:
            return alignments

        min_seq = min(seq_to_alignment.keys())
        max_seq = max(seq_to_alignment.keys())
        missing_marker_seqs = [
            seq for seq in range(max(0, min_seq - 2), max_seq + 3)
            if seq not in seq_to_alignment
            and seq in seq_to_openxml
            and (
                self._is_marker_only_text(seq_to_openxml[seq][1].get('text', ''))
                or self._is_bookmark_end_unit(seq_to_openxml[seq][1])
            )
        ]
        if not missing_marker_seqs:
            return alignments

        created = []
        for missing_seq in missing_marker_seqs:
            openxml_idx, openxml_unit = seq_to_openxml[missing_seq]
            is_bookmark_proxy = self._is_bookmark_end_unit(openxml_unit)

            prev_candidates = sorted(
                [
                    a for a in alignments
                    if a.get('element_sequence') is not None
                    and a.get('element_sequence') < missing_seq
                    and (missing_seq - a.get('element_sequence')) <= 2
                    and not a.get('is_table')
                    and not a.get('is_image_part')
                    and a.get('matched_pdf_units')
                ],
                key=lambda a: a.get('element_sequence'),
                reverse=True
            )
            next_candidates = sorted(
                [
                    a for a in alignments
                    if a.get('element_sequence') is not None
                    and a.get('element_sequence') > missing_seq
                    and (a.get('element_sequence') - missing_seq) <= 3
                    and not a.get('is_table')
                    and not a.get('is_image_part')
                ],
                key=lambda a: a.get('element_sequence')
            )
            if not next_candidates:
                continue

            donor_alignment = None
            donor_unit = None

            if is_bookmark_proxy:
                proxy_candidates = []
                for candidate in prev_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if units:
                        proxy_candidates.append((candidate, units[-1]))
                for candidate in next_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if units:
                        proxy_candidates.append((candidate, units[0]))
                if proxy_candidates:
                    donor_alignment, donor_unit = proxy_candidates[0]
            else:
                for candidate in next_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if len(units) < 2:
                        continue

                    leading_markers = []
                    for unit in units:
                        if self._is_marker_only_text(unit.get('text', '')):
                            leading_markers.append(unit)
                            continue
                        break

                    if len(leading_markers) >= 2:
                        donor_alignment = candidate
                        donor_unit = leading_markers[0]
                        break

            if donor_alignment is None or donor_unit is None:
                continue

            if not is_bookmark_proxy:
                donor_key = self._matched_unit_key(donor_unit)
                donor_units = donor_alignment.get('matched_pdf_units', [])
                consumed = False
                kept_units = []
                for unit in donor_units:
                    unit_key = self._matched_unit_key(unit)
                    if not consumed and donor_key is not None and unit_key == donor_key:
                        consumed = True
                        continue
                    kept_units.append(unit)

                if not consumed:
                    continue

                donor_alignment['matched_pdf_units'] = kept_units
                self._recompute_alignment_bboxes(donor_alignment)

            restored = {
                'element_id': openxml_unit['elem_id'],
                'element_sequence': openxml_unit['elem_seq'],
                'element_type': openxml_unit['elem_type'],
                'is_table': False,
                'is_synthetic_marker_repair': True,
                'is_synthetic_bookmark_proxy': is_bookmark_proxy,
                'element_text': openxml_unit.get('text', ''),
                'matched_pdf_units': [donor_unit],
                'merged_bbox': list(donor_unit.get('bbox')) if donor_unit.get('bbox') else None,
                'cells': None,
                'is_text_part': openxml_unit.get('is_text_part', False),
                'is_image_part': False,
                'unit_id': str(openxml_unit['elem_id']),
                'openxml_indices': [openxml_idx],
                'openxml_idx': openxml_idx,
                'image_index': openxml_unit.get('image_index'),
                'font_families': openxml_unit.get('font_families', []),
                'style_ids': openxml_unit.get('style_ids', []),
                'is_code_font': openxml_unit.get('is_code_font', False),
                'is_code_style': openxml_unit.get('is_code_style', False),
                'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
            }
            created.append(restored)
            seq_to_alignment[missing_seq] = restored

        if created:
            alignments.extend(created)
            alignments[:] = [
                a for a in alignments
                if a.get('is_table') or a.get('matched_pdf_units')
            ]
            alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments

    def _rescue_chart_hline_alignments(self, alignments, openxml_units, min_openxml_idx=None):
        if not alignments or not openxml_units:
            return alignments, []

        min_openxml_idx = self._try_parse_int(min_openxml_idx)

        max_seq_gap = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MAX_SEQ_GAP',
            4
        )
        min_width = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MIN_WIDTH',
            160
        )
        min_height = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MIN_HEIGHT',
            100
        )

        def bbox_size_ok(bbox):
            if not bbox or len(bbox) < 4:
                return False
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            return width >= min_width and height >= min_height

        seq_to_alignment = {}
        for alignment in alignments:
            if alignment.get('is_table'):
                continue
            seq = self._try_parse_int(alignment.get('element_sequence'))
            if seq is None:
                continue
            seq_to_alignment[seq] = alignment

        seq_to_openxml = {}
        for openxml_idx, unit in enumerate(openxml_units):
            seq = self._try_parse_int((unit or {}).get('elem_seq'))
            if seq is None or seq in seq_to_openxml:
                continue
            seq_to_openxml[seq] = (openxml_idx, unit)

        if not seq_to_alignment or not seq_to_openxml:
            return alignments, []

        debug = []
        created = []
        touched_alignments = []

        for alignment in list(alignments):
            if alignment.get('is_table') or alignment.get('is_openxml_chart') or alignment.get('is_openxml_visual_slot'):
                continue
            if not self._is_paragraph_like_alignment(alignment):
                continue

            donor_seq = self._try_parse_int(alignment.get('element_sequence'))
            if donor_seq is None:
                continue

            donor_text = self._normalize_text(alignment.get('element_text') or '')
            if not donor_text.startswith('gambar'):
                continue

            units = list(alignment.get('matched_pdf_units') or [])
            moved_units = [
                unit for unit in units
                if unit.get('item_type') == 'hline_table' and bbox_size_ok(unit.get('bbox'))
            ]
            if not moved_units:
                continue

            candidate = None
            for gap in range(1, max_seq_gap + 1):
                prev_seq = donor_seq - gap
                prev_entry = seq_to_openxml.get(prev_seq)
                if prev_entry and prev_seq not in seq_to_alignment:
                    openxml_idx, openxml_unit = prev_entry
                    if min_openxml_idx is not None and openxml_idx < min_openxml_idx:
                        prev_entry = None
                    if not prev_entry:
                        continue
                    if openxml_unit.get('is_openxml_chart') or openxml_unit.get('is_openxml_visual_slot'):
                        candidate = (prev_seq, openxml_idx, openxml_unit)
                        break

                next_seq = donor_seq + gap
                next_entry = seq_to_openxml.get(next_seq)
                if next_entry and next_seq not in seq_to_alignment:
                    openxml_idx, openxml_unit = next_entry
                    if min_openxml_idx is not None and openxml_idx < min_openxml_idx:
                        next_entry = None
                    if not next_entry:
                        continue
                    if openxml_unit.get('is_openxml_chart') or openxml_unit.get('is_openxml_visual_slot'):
                        candidate = (next_seq, openxml_idx, openxml_unit)
                        break

            if not candidate:
                continue

            candidate_seq, openxml_idx, openxml_unit = candidate
            moved_units = sorted(moved_units, key=lambda unit: unit.get('item_idx', -1))
            remaining_units = [unit for unit in units if unit not in moved_units]

            rescued_alignment = {
                'element_id': openxml_unit.get('elem_id'),
                'element_sequence': openxml_unit.get('elem_seq'),
                'element_type': openxml_unit.get('elem_type'),
                'is_table': False,
                'is_chart_rescue': True,
                'element_text': openxml_unit.get('text', ''),
                'matched_pdf_units': moved_units,
                'merged_bbox': self._merge_bboxes([u.get('bbox') for u in moved_units]),
                'cells': None,
                'is_text_part': openxml_unit.get('is_text_part', False),
                'is_image_part': openxml_unit.get('is_image_part', False),
                'unit_id': str(openxml_unit.get('unit_id') or openxml_unit.get('elem_id')),
                'openxml_indices': [openxml_idx],
                'openxml_idx': openxml_idx,
                'image_index': openxml_unit.get('image_index'),
                'font_families': openxml_unit.get('font_families', []),
                'style_ids': openxml_unit.get('style_ids', []),
                'is_code_font': openxml_unit.get('is_code_font', False),
                'is_code_style': openxml_unit.get('is_code_style', False),
                'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
                'chart_rescued_from_element_id': alignment.get('element_id'),
            }

            alignment['matched_pdf_units'] = remaining_units
            touched_alignments.append(alignment)
            created.append(rescued_alignment)
            seq_to_alignment[candidate_seq] = rescued_alignment
            debug.append({
                'from_element_id': alignment.get('element_id'),
                'from_element_sequence': donor_seq,
                'to_element_id': openxml_unit.get('elem_id'),
                'to_element_sequence': candidate_seq,
                'moved_unit_count': len(moved_units),
            })

        if created:
            for alignment in touched_alignments:
                self._recompute_alignment_bboxes(alignment)
            alignments.extend(created)
            alignments[:] = [
                alignment for alignment in alignments
                if alignment.get('is_table') or alignment.get('matched_pdf_units')
            ]
            alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments, debug

    def _drop_far_backward_alignments(self, alignments, unaligned_pdf_indices, pdf_units, min_openxml_idx):
        if not alignments:
            return alignments, list(unaligned_pdf_indices or []), []

        base_min_openxml_idx = self._try_parse_int(min_openxml_idx)
        if base_min_openxml_idx is None or base_min_openxml_idx <= 0:
            return alignments, list(unaligned_pdf_indices or []), []

        max_backward_gap = self._read_positive_int_env(
            'ALIGNMENT_MAX_BACKWARD_ALIGNMENT_GAP',
            24
        )
        cutoff_idx = max(0, base_min_openxml_idx - max_backward_gap)

        kept = []
        dropped = []
        pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox = self._build_pdf_lookup_maps(pdf_units)
        restored_unaligned = set(unaligned_pdf_indices or [])
        for alignment in alignments:
            indices = self._collect_alignment_openxml_indices(
                alignment,
                include_table_cells=True
            )
            if not indices:
                kept.append(alignment)
                continue

            alignment_max_idx = max(indices)
            if alignment_max_idx < cutoff_idx:
                dropped.append({
                    'element_id': alignment.get('element_id'),
                    'element_sequence': alignment.get('element_sequence'),
                    'openxml_max_idx': alignment_max_idx,
                    'cutoff_idx': cutoff_idx,
                })
                for unit in alignment.get('matched_pdf_units', []) or []:
                    pdf_idx = self._resolve_unit_pdf_index(
                        unit,
                        pdf_idx_by_unit_id,
                        pdf_idx_by_item_idx,
                        pdf_idx_by_bbox
                    )
                    if pdf_idx is not None:
                        restored_unaligned.add(pdf_idx)
                for cell in alignment.get('cells', []) or []:
                    for unit in cell.get('matched_pdf_units', []) or []:
                        pdf_idx = self._resolve_unit_pdf_index(
                            unit,
                            pdf_idx_by_unit_id,
                            pdf_idx_by_item_idx,
                            pdf_idx_by_bbox
                        )
                        if pdf_idx is not None:
                            restored_unaligned.add(pdf_idx)
                continue
            kept.append(alignment)

        return kept, sorted(restored_unaligned), dropped
