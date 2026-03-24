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


class MergingExtractionFusionRepairsMixin:
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

    def _find_best_visual_alignment_for_bbox(self, alignments, target_bbox):
        if not alignments or not target_bbox:
            return None

        candidates = []
        for alignment in alignments:
            bbox = alignment.get('merged_bbox')
            if not bbox or not self._alignment_has_visual_units(alignment):
                continue
            if not (
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment')
            ):
                continue
            overlap = self.fusion_service.calculate_overlap(bbox, target_bbox)
            if overlap <= 0:
                continue
            candidates.append((alignment, overlap))

        if not candidates:
            return None

        def candidate_score(entry):
            alignment, overlap = entry
            bbox = alignment.get('merged_bbox')
            return (
                1 if alignment.get('is_openxml_visual_slot') else 0,
                overlap,
                self._bbox_area(bbox),
                alignment.get('element_sequence') or 0,
            )

        return max(candidates, key=candidate_score)[0]

    def _build_picture_result_from_alignment(self, alignment, docling_bbox=None, repair_reason=None):
        if not alignment or not alignment.get('merged_bbox'):
            return None
        matched_units = alignment.get('matched_pdf_units') or []
        has_pdf_image = any(unit.get('item_type') == 'image' for unit in matched_units)
        has_shape_units = any(
            unit.get('item_type') in ('shape', 'hline_table') or unit.get('is_chart_visual')
            for unit in matched_units
        )
        has_table_units = any(unit.get('item_type') in ('table', 'hline_table') for unit in matched_units)
        openxml_indices = alignment.get('openxml_indices') or []
        overlap = 0.0
        if docling_bbox:
            overlap = self.fusion_service.calculate_overlap(alignment.get('merged_bbox'), docling_bbox)
        picture_text = alignment.get('element_text', '')
        if (
            alignment.get('is_openxml_chart') and
            self.fusion_service._is_caption_candidate(self._coerce_text(picture_text))
        ):
            picture_text = ''
        return {
            'bbox': list(alignment.get('merged_bbox')),
            'label': 'picture',
            'text': picture_text,
            'overlap': overlap,
            'source': 'alignment',
            'element_id': alignment.get('element_id'),
            'element_type': alignment.get('element_type'),
            'element_sequence': alignment.get('element_sequence'),
            'openxml_idx': min(openxml_indices) if openxml_indices else alignment.get('openxml_idx'),
            'docling_label': 'picture' if docling_bbox else None,
            'is_text_part': alignment.get('is_text_part'),
            'is_image_part': alignment.get('is_image_part'),
            'unit_id': alignment.get('unit_id'),
            'merged_count': 1,
            'is_picture_area': True,
            'has_shape_units': has_shape_units,
            'has_pdf_image': has_pdf_image,
            'has_table_units': has_table_units,
            'is_text_only_item': False,
            'is_openxml_chart': alignment.get('is_openxml_chart', False),
            'is_openxml_visual_slot': alignment.get('is_openxml_visual_slot', False),
            'is_chart_caption_text': alignment.get('is_chart_caption_text', False),
            'visual_slot_promoted': alignment.get('visual_slot_promoted', False),
            'repair_reason': repair_reason or alignment.get('repair_reason'),
        }

    def _picture_body_text_overlap_ratio(self, picture_result, fused_results):
        if not picture_result or not fused_results:
            return 0.0
        picture_bbox = picture_result.get('bbox')
        if not picture_bbox:
            return 0.0

        max_overlap = 0.0
        picture_elem_id = picture_result.get('element_id')
        for other in fused_results:
            if other is picture_result:
                continue
            if other.get('element_id') == picture_elem_id and picture_elem_id is not None:
                continue
            other_bbox = other.get('bbox')
            if not other_bbox:
                continue
            label = self._get_visual_label(other)
            if label in ('picture', 'caption', 'table', 'page_header', 'page_footer', 'formula', 'code', 'footnote'):
                continue
            if self.fusion_service._is_caption_candidate(self._coerce_text(other.get('text'))):
                continue
            overlap = self.fusion_service.calculate_overlap(picture_bbox, other_bbox)
            if overlap > max_overlap:
                max_overlap = overlap
        return max_overlap

    def _repair_picture_fusion_results(self, alignments, fused_results, docling_predictions=None):
        if not fused_results:
            return fused_results, {
                'missing_picture_repair_count': 0,
                'picture_overlap_prune_count': 0,
            }

        debug = {
            'missing_picture_repair_count': 0,
            'picture_overlap_prune_count': 0,
        }
        raw_picture_preds = [
            pred for pred in (docling_predictions or [])
            if pred.get('label') == 'picture' and pred.get('bbox')
        ]
        picture_results = [result for result in fused_results if self._is_picture_result(result)]

        if raw_picture_preds and not picture_results:
            for pred in raw_picture_preds:
                alignment = self._find_best_visual_alignment_for_bbox(alignments, pred.get('bbox'))
                if not alignment:
                    continue
                replacement = self._build_picture_result_from_alignment(
                    alignment,
                    docling_bbox=pred.get('bbox'),
                    repair_reason='missing_picture_repair'
                )
                if not replacement:
                    continue
                existing_result = None
                for result in fused_results:
                    if result.get('element_id') == alignment.get('element_id'):
                        if self._is_caption_like_visual_result(result) and not self._is_picture_result(result):
                            continue
                        existing_result = result
                        break
                if existing_result is not None:
                    existing_result.update(replacement)
                else:
                    fused_results.append(replacement)
                debug['missing_picture_repair_count'] += 1
            picture_results = [result for result in fused_results if self._is_picture_result(result)]

        alignment_by_element = {}
        for alignment in alignments or []:
            elem_id = alignment.get('element_id')
            if elem_id is None or not alignment.get('merged_bbox'):
                continue
            alignment_by_element.setdefault(elem_id, []).append(alignment)

        picture_overlap_threshold = self._read_float_env(
            'ALIGNMENT_PICTURE_TEXT_OVERLAP_REPAIR_THRESHOLD',
            0.2,
            min_value=0.0,
            max_value=1.0
        )

        for result in picture_results:
            overlap_ratio = self._picture_body_text_overlap_ratio(result, fused_results)
            if overlap_ratio <= picture_overlap_threshold:
                continue
            elem_id = result.get('element_id')
            candidate_alignments = alignment_by_element.get(elem_id) or []
            if not candidate_alignments:
                continue
            best_alignment = max(
                candidate_alignments,
                key=lambda alignment: (
                    1 if alignment.get('is_openxml_visual_slot') else 0,
                    self.fusion_service.calculate_overlap(
                        alignment.get('merged_bbox'),
                        result.get('bbox')
                    ) if alignment.get('merged_bbox') and result.get('bbox') else 0.0,
                    self._bbox_area(alignment.get('merged_bbox')),
                )
            )
            align_bbox = best_alignment.get('merged_bbox')
            if not align_bbox:
                continue
            result['bbox'] = list(align_bbox)
            result['repair_reason'] = 'picture_overlap_prune'
            result['picture_text_overlap_ratio'] = overlap_ratio
            debug['picture_overlap_prune_count'] += 1

        self._sort_fused_results_in_reading_order(fused_results)
        return fused_results, debug

    def _collapse_table_visual_results_for_page(self, fused_results):
        if not fused_results:
            return []

        def is_collapsible_table_row(row):
            if not row or not self._is_table_like_visual_result(row):
                return False
            if self._try_parse_int_id((row or {}).get('element_id')) is None:
                return False
            visual_label = self._get_visual_label(row)
            if visual_label and visual_label != 'table':
                return False
            element_type = str((row or {}).get('element_type') or '').strip().lower()
            if 'caption' in element_type:
                return False
            return True

        def merge_bbox_rows(rows):
            bboxes = [row.get('bbox') for row in rows if row.get('bbox') and len(row.get('bbox')) >= 4]
            if not bboxes:
                return None
            return [
                min(float(bbox[0]) for bbox in bboxes),
                min(float(bbox[1]) for bbox in bboxes),
                max(float(bbox[2]) for bbox in bboxes),
                max(float(bbox[3]) for bbox in bboxes),
            ]

        def first_non_empty(rows, key):
            for row in rows:
                value = row.get(key)
                if value not in (None, ''):
                    return value
            return None

        grouped_rows = {}
        for row in fused_results:
            if not is_collapsible_table_row(row):
                continue
            element_id = self._try_parse_int_id((row or {}).get('element_id'))
            grouped_rows.setdefault(element_id, []).append(row)

        collapsed = []
        seen_element_ids = set()
        for row in fused_results:
            if not is_collapsible_table_row(row):
                collapsed.append(row)
                continue

            element_id = self._try_parse_int_id((row or {}).get('element_id'))
            if element_id in seen_element_ids:
                continue
            seen_element_ids.add(element_id)

            rows = grouped_rows.get(element_id) or []
            if len(rows) <= 1:
                collapsed.append(row)
                continue

            merged_row = dict(row)
            merged_row['bbox'] = merge_bbox_rows(rows)
            merged_row['label'] = 'table'
            merged_row['docling_label'] = 'table'
            merged_row['source'] = 'table_page_merge'
            merged_row['has_table_units'] = True
            merged_row['dev_label_struktural'] = first_non_empty(rows, 'dev_label_struktural') or 'tabel'

            merged_text_parts = [
                self._coerce_text(candidate.get('text')).strip()
                for candidate in rows
                if self._coerce_text(candidate.get('text')).strip()
            ]
            merged_row['text'] = '\n'.join(merged_text_parts)

            merged_row['merged_count'] = sum(
                self._try_parse_int_id(candidate.get('merged_count')) or 1
                for candidate in rows
            )

            overlap_values = [
                float(candidate.get('overlap'))
                for candidate in rows
                if candidate.get('overlap') is not None
            ]
            if overlap_values:
                merged_row['overlap'] = max(overlap_values)

            confidence_values = [
                float(candidate.get('alignment_confidence'))
                for candidate in rows
                if candidate.get('alignment_confidence') is not None
            ]
            if confidence_values:
                merged_row['alignment_confidence'] = max(confidence_values)

            block_orders = [
                self._try_parse_int_id(candidate.get('block_order'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('block_order')) is not None
            ]
            if block_orders:
                merged_row['block_order'] = min(block_orders)

            element_sequences = [
                self._try_parse_int_id(candidate.get('element_sequence'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('element_sequence')) is not None
            ]
            if element_sequences:
                merged_row['element_sequence'] = min(element_sequences)

            openxml_indices = [
                self._try_parse_int_id(candidate.get('openxml_idx'))
                for candidate in rows
                if self._try_parse_int_id(candidate.get('openxml_idx')) is not None
            ]
            if openxml_indices:
                merged_row['openxml_idx'] = min(openxml_indices)

            collapsed.append(merged_row)

        return collapsed
