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


class MergingExtractionFusionPictureRepairMixin:


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
