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


class MergingExtractionFusionDuplicateSyncMixin:


    def _merge_duplicate_units_with_neighbors(self, alignments, duplicate_element_ids):
        if not alignments or not duplicate_element_ids:
            return alignments, set()

        def is_visual_alignment(alignment):
            if not alignment:
                return False
            if (
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment') or
                alignment.get('is_image_part')
            ):
                return True
            units = alignment.get('matched_pdf_units', []) or []
            return any(
                unit.get('is_chart_visual') or unit.get('item_type') in ('image', 'shape', 'hline_table')
                for unit in units
            )

        ordered = [
            alignment for alignment in alignments
            if not alignment.get('is_table') and alignment.get('merged_bbox')
        ]
        ordered.sort(key=lambda a: (self._get_alignment_center_y(a) or 0, a.get('merged_bbox')[0]))

        touched = set()
        removed_element_ids = set()
        for idx, alignment in enumerate(ordered):
            if alignment.get('element_id') not in duplicate_element_ids:
                continue
            if not self._is_duplicate_sequence_far(
                alignments,
                alignment,
                self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ):
                continue

            units = list(alignment.get('matched_pdf_units', []))
            if not units:
                continue

            above = ordered[idx - 1] if idx > 0 else None
            below = ordered[idx + 1] if idx + 1 < len(ordered) else None

            remaining_units = []
            for unit in units:
                if unit.get('item_type') != 'group':
                    remaining_units.append(unit)
                    continue
                unit_bbox = unit.get('bbox')
                if not unit_bbox:
                    remaining_units.append(unit)
                    continue
                unit_text = self._normalize_text_value(unit.get('text'))
                if not unit_text:
                    remaining_units.append(unit)
                    continue

                target = None
                above_text = self._normalize_text_value(above.get('element_text')) if above else ''
                below_text = self._normalize_text_value(below.get('element_text')) if below else ''
                above_contains = bool(above_text) and unit_text in above_text
                below_contains = bool(below_text) and unit_text in below_text
                if unit_text and len(unit_text) <= self.SHORT_DUPLICATE_UNIT_LEN:
                    simplified_unit = self._simplify_duplicate_unit_text(unit_text)
                    if simplified_unit:
                        if not above_contains and above_text:
                            simplified_above = self._simplify_duplicate_unit_text(above_text)
                            if simplified_above and simplified_unit in simplified_above:
                                above_contains = True
                        if not below_contains and below_text:
                            simplified_below = self._simplify_duplicate_unit_text(below_text)
                            if simplified_below and simplified_unit in simplified_below:
                                below_contains = True

                if above_contains and not below_contains:
                    target = above
                elif below_contains and not above_contains:
                    target = below
                elif above_contains and below_contains:
                    unit_y = self._get_bbox_center_y(unit_bbox)
                    above_y = self._get_alignment_center_y(above)
                    below_y = self._get_alignment_center_y(below)
                    above_delta = abs(unit_y - above_y) if unit_y is not None and above_y is not None else None
                    below_delta = abs(unit_y - below_y) if unit_y is not None and below_y is not None else None
                    if above_delta is None and below_delta is None:
                        target = below
                    elif above_delta is None:
                        target = below
                    elif below_delta is None:
                        target = above
                    else:
                        target = above if above_delta <= below_delta else below

                if is_visual_alignment(target):
                    target = None

                if not target:
                    remaining_units.append(unit)
                    continue

                unit_key = self._pdf_unit_key(unit)
                target_units = target.setdefault('matched_pdf_units', [])
                target_keys = {
                    self._pdf_unit_key(u)
                    for u in target_units
                    if self._pdf_unit_key(u) is not None
                }
                if unit_key is None or unit_key in target_keys:
                    remaining_units.append(unit)
                    continue

                unit['merged_from_duplicate'] = True
                target_units.append(unit)
                target_units.sort(key=lambda u: u.get('item_idx', -1))
                touched.add(id(target))

            alignment['matched_pdf_units'] = remaining_units
            touched.add(id(alignment))
            if not remaining_units:
                removed_element_ids.add(alignment.get('element_id'))

        if touched:
            for alignment in alignments:
                if id(alignment) in touched:
                    self.alignment_service._recompute_alignment_bboxes(alignment)

        if not removed_element_ids:
            return alignments, set()
        return (
            [alignment for alignment in alignments if alignment.get('element_id') not in removed_element_ids],
            removed_element_ids
        )

    def _sync_fused_bboxes_with_alignments(self, fused_results, alignments, removed_element_ids=None):
        if not fused_results or not alignments:
            return
        if removed_element_ids:
            fused_results[:] = [
                result for result in fused_results
                if not (
                    result.get('source') == 'alignment'
                    and result.get('element_id') in removed_element_ids
                )
            ]

        alignment_by_id = {}
        for alignment in alignments:
            elem_id = alignment.get('element_id')
            if elem_id is None:
                continue
            alignment_by_id.setdefault(elem_id, []).append(alignment)

        updated_results = []
        seen_picture_bboxes = set()

        for result in fused_results:
            if (
                result.get('source') != 'alignment' and
                not (
                    result.get('source') == 'merged' and
                    result.get('element_id') is not None
                )
            ):
                updated_results.append(result)
                continue
            elem_id = result.get('element_id')
            if elem_id is None:
                updated_results.append(result)
                continue

            is_picture = (
                result.get('label') == 'picture'
                or result.get('docling_label') == 'picture'
                or result.get('has_pdf_image')
                or result.get('is_image_part')
            )
            alignments_for_elem = alignment_by_id.get(elem_id, [])
            has_chart_alignment = any(
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment')
                for alignment in alignments_for_elem
            )

            if is_picture and alignments_for_elem and not has_chart_alignment:
                image_units = [
                    unit
                    for alignment in alignments_for_elem
                    for unit in (alignment.get('matched_pdf_units', []) or [])
                    if unit.get('item_type') in ('image', 'shape') or unit.get('text') == '[IMG]'
                ]
                if image_units:
                    for unit in image_units:
                        bbox = unit.get('bbox')
                        if not bbox or len(bbox) < 4:
                            continue
                        key = (elem_id, tuple(bbox))
                        if key in seen_picture_bboxes:
                            continue
                        seen_picture_bboxes.add(key)
                        new_result = dict(result)
                        new_result['bbox'] = list(bbox)
                        updated_results.append(new_result)
                    continue

            candidate_alignments = alignments_for_elem
            if is_picture and alignments_for_elem:
                candidate_alignments = self._select_chart_visual_alignments(alignments_for_elem)
            elif not is_picture and alignments_for_elem:
                if result.get('is_text_part'):
                    candidate_alignments = [
                        alignment for alignment in alignments_for_elem
                        if alignment.get('is_text_part')
                    ]
                elif result.get('is_chart_caption_text'):
                    candidate_alignments = [
                        alignment for alignment in alignments_for_elem
                        if alignment.get('is_chart_caption_text')
                    ]
                elif result.get('is_image_part') is not True:
                    candidate_alignments = [
                        alignment for alignment in alignments_for_elem
                        if not alignment.get('is_image_part')
                    ]
                if not candidate_alignments:
                    candidate_alignments = alignments_for_elem

            align_bboxes = [
                alignment.get('merged_bbox')
                for alignment in candidate_alignments
                if alignment.get('merged_bbox')
            ]
            if not align_bboxes:
                updated_results.append(result)
                continue
            align_bbox = self.alignment_service._merge_bboxes(align_bboxes)
            if not align_bbox:
                updated_results.append(result)
                continue
            if is_picture and has_chart_alignment:
                result['bbox'] = list(align_bbox)
                updated_results.append(result)
                continue
            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                result['bbox'] = list(align_bbox)
                updated_results.append(result)
                continue
            result['bbox'] = [
                min(bbox[0], align_bbox[0]),
                min(bbox[1], align_bbox[1]),
                max(bbox[2], align_bbox[2]),
                max(bbox[3], align_bbox[3])
            ]
            updated_results.append(result)

        fused_results[:] = updated_results

    def _alignment_has_visual_units(self, alignment):
        if not alignment:
            return False
        return any(
            unit.get('is_chart_visual') or unit.get('item_type') in ('image', 'shape', 'hline_table')
            for unit in (alignment.get('matched_pdf_units') or [])
        )

    def _sort_fused_results_in_reading_order(self, fused_results):
        if not fused_results:
            return
        from functools import cmp_to_key

        def sort_key(item):
            return item.get('bbox') or [0, 0, 0, 0]

        def compare(a, b):
            a_bbox = sort_key(a)
            b_bbox = sort_key(b)
            y_diff = a_bbox[1] - b_bbox[1]
            if abs(y_diff) > 10:
                return -1 if y_diff < 0 else 1
            x_diff = a_bbox[0] - b_bbox[0]
            return -1 if x_diff < 0 else (1 if x_diff > 0 else 0)

        fused_results.sort(key=cmp_to_key(compare))
