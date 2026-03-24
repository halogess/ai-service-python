from copy import deepcopy
import os
import re


from .postprocess_visual_shape_mixin import AlignmentPostprocessVisualShapeMixin
from .postprocess_visual_chart_mixin import AlignmentPostprocessVisualChartMixin


class AlignmentPostprocessVisualMixin(
    AlignmentPostprocessVisualShapeMixin,
    AlignmentPostprocessVisualChartMixin,
):


    def _recompute_alignment_bboxes(self, alignment):
        if alignment.get('is_table') and alignment.get('cells'):
            cell_bboxes = [c.get('merged_bbox') for c in alignment.get('cells') if c.get('merged_bbox')]
            alignment['merged_bbox'] = self._merge_bboxes(cell_bboxes) if cell_bboxes else None
            return
        if alignment.get('matched_pdf_units'):
            alignment['merged_bbox'] = self._merge_bboxes(
                [u.get('bbox') for u in alignment.get('matched_pdf_units') if u.get('bbox')]
            )

    def _prune_visual_alignment_text_units(self, alignments, unaligned_pdf_indices, pdf_units):
        if not alignments:
            return alignments, unaligned_pdf_indices, []

        vertical_gap_max = self._read_float_env(
            'ALIGNMENT_VISUAL_CAPTION_BAND_MAX_GAP',
            48.0,
            min_value=0.0
        )
        x_overlap_min = self._read_float_env(
            'ALIGNMENT_VISUAL_CAPTION_BAND_MIN_X_OVERLAP',
            0.5,
            min_value=0.0,
            max_value=1.0
        )

        previous_keys = self._collect_alignment_unit_keys(alignments)
        debug = []

        for alignment in alignments:
            if not self._is_visual_target_alignment(alignment):
                continue

            units = list(alignment.get('matched_pdf_units') or [])
            if not units:
                continue

            visual_bbox = self._alignment_visual_bbox(alignment)
            if not visual_bbox:
                continue

            figure_key = self._extract_figure_key(alignment.get('element_text'))
            kept_units = []
            dropped_units = []
            kept_caption_units = 0

            for unit in units:
                if self._is_visual_pdf_unit(unit):
                    kept_units.append(unit)
                    continue

                unit_text = str(unit.get('text') or '').strip()
                unit_bbox = unit.get('bbox')
                unit_figure_key = self._extract_figure_key(unit_text)

                if figure_key and unit_figure_key and unit_figure_key != figure_key:
                    dropped_units.append(unit)
                    continue

                if unit_figure_key and figure_key and unit_figure_key == figure_key:
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                in_caption_band = self._is_caption_band_bbox(
                    unit_bbox,
                    visual_bbox,
                    vertical_gap_max=vertical_gap_max,
                    x_overlap_min=x_overlap_min
                )
                if in_caption_band:
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                if self._is_caption_like_text(unit_text):
                    kept_units.append(unit)
                    kept_caption_units += 1
                    continue

                dropped_units.append(unit)

            if not dropped_units:
                continue

            alignment['matched_pdf_units'] = sorted(
                kept_units,
                key=lambda unit: unit.get('item_idx', -1)
            )
            self._recompute_alignment_bboxes(alignment)
            alignment['repair_reason'] = 'picture_overlap_prune'
            debug.append({
                'element_id': alignment.get('element_id'),
                'element_sequence': alignment.get('element_sequence'),
                'dropped_unit_count': len(dropped_units),
                'kept_unit_count': len(kept_units),
                'kept_caption_unit_count': kept_caption_units,
                'visual_bbox': list(visual_bbox) if visual_bbox else None,
            })

        if debug:
            unaligned_pdf_indices = self._restore_dropped_alignment_units(
                previous_keys,
                alignments,
                unaligned_pdf_indices,
                pdf_units
            )
        return alignments, unaligned_pdf_indices, debug
