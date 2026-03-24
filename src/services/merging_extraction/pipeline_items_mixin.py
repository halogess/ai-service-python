
import os
import json
import logging
import difflib
import re
from datetime import datetime
from sqlalchemy.orm import Session
from models import Bab, Dokumen, DokumenSection, DokumenPart, DokumenElemen, DokumenElemenVisual, DokumenNote, DokumenFormatText, DokumenFormatParagraf
from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from services.docling_service import DoclingService
from services.docling_fusion_service import DoclingFusionService
from services.visualization_service import VisualizationService
from utils.cross_page_claims import analyze_cross_page_entries
from database import SessionLocal
from services.merging_extraction import (
    MergingExtractionClaimRepairMixin,
    MergingExtractionFusionRepairsMixin,
    MergingExtractionPersistenceMixin,
    MergingExtractionStructuralLabelsMixin,
    MergingExtractionTargetAssignmentMixin,
)

logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv("VOLUME_BASE_PATH", "/app/storage")
VISUALIZATION_OUTPUT = os.getenv("VISUALIZATION_OUTPUT", "visualization_output")


class MergingExtractionPipelineItemsMixin:


    def _transform_extraction_data_to_items(self, data):
        """Transform extraction dict into list of typed items for alignment."""
        items = []
        
        # Char Groups -> 'group'
        for g in data.get('char_groups', []):
            items.append({
                'type': 'group',
                'bbox': g.get('bbox') or g.get('merged_bbox'),
                'data': {'text': g.get('text', '')}
            })
            
        # Basic Tables -> 'table'
        for t in data.get('basic_tables', []):
            items.append({
                'type': 'table',
                'bbox': t.get('bbox'),
                'data': {'rows': t.get('rows', [])}
            })
            
        # Hline Tables -> 'hline_table'
        for t in data.get('hline_tables', []):
            items.append({
                'type': 'hline_table',
                'bbox': t.get('bbox'),
                'data': {
                    'rows': t.get('rows', []),
                    'cells': [] # Legacy structure might have cells at top too
                }
            })
            
        # Shapes -> 'shape'
        for s in data.get('shapes', []):
            items.append({
                'type': 'shape',
                'bbox': s.get('bbox'),
                'data': {
                    'text': s.get('text', ''),
                    'image_bbox': s.get('image_bbox')
                }
            })
            
        # Images -> 'image'
        for img in data.get('page_images', []):
            items.append({
                'type': 'image',
                'bbox': img.get('bbox'),
                'data': {}
            })

        # Sort by reading order (line-aware) to match legacy frontend
        # Items are on the same line if >=30% Y overlap (based on smaller height)
        from functools import cmp_to_key

        def compare_items(a, b):
            if not a.get('bbox') or not b.get('bbox'):
                return 0

            y_a0, y_a1 = a['bbox'][1], a['bbox'][3]
            y_b0, y_b1 = b['bbox'][1], b['bbox'][3]
            height_a = y_a1 - y_a0
            height_b = y_b1 - y_b0

            overlap_start = max(y_a0, y_b0)
            overlap_end = min(y_a1, y_b1)
            overlap_amount = max(0, overlap_end - overlap_start)
            smaller_height = min(height_a, height_b)

            overlap_ratio = (overlap_amount / smaller_height) if smaller_height > 0 else 0
            is_same_line = overlap_ratio >= 0.30

            if is_same_line:
                return -1 if a['bbox'][0] < b['bbox'][0] else (1 if a['bbox'][0] > b['bbox'][0] else 0)
            return -1 if y_a0 < y_b0 else (1 if y_a0 > y_b0 else 0)

        items.sort(key=cmp_to_key(compare_items))
        return items

    def _annotate_picture_visual_items(self, extraction_items, docling_predictions):
        if not extraction_items or not docling_predictions:
            return extraction_items

        overlap_threshold = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_DOCLING_OVERLAP",
            self.fusion_service.OVERLAP_THRESHOLD,
            min_value=0.0,
            max_value=1.0
        )
        min_width = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_MIN_WIDTH",
            160.0,
            min_value=1.0
        )
        min_height = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_MIN_HEIGHT",
            100.0,
            min_value=1.0
        )
        shape_min_width = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_SHAPE_MIN_WIDTH",
            1.0,
            min_value=0.0
        )
        shape_min_height = self._read_float_env(
            "ALIGNMENT_CHART_VISUAL_SHAPE_MIN_HEIGHT",
            1.0,
            min_value=0.0
        )

        picture_preds = [
            pred for pred in (docling_predictions or [])
            if str(pred.get('label') or '').strip().lower() == 'picture' and pred.get('bbox')
        ]
        if not picture_preds:
            return extraction_items

        for item in extraction_items:
            item_type = str(item.get('type') or '').strip().lower()
            if item_type not in {'hline_table', 'shape'}:
                continue

            bbox = item.get('bbox')
            if not bbox or len(bbox) < 4:
                continue

            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            item_min_width = shape_min_width if item_type == 'shape' else min_width
            item_min_height = shape_min_height if item_type == 'shape' else min_height
            if width < item_min_width or height < item_min_height:
                continue

            best_overlap = 0.0
            best_pred_bbox = None
            for pred in picture_preds:
                pred_bbox = pred.get('bbox')
                overlap = self.fusion_service.calculate_overlap(bbox, pred_bbox)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_pred_bbox = pred_bbox

            if best_overlap < overlap_threshold:
                continue

            item['docling_label'] = 'picture'
            item['docling_picture_overlap'] = round(best_overlap, 4)
            item['docling_picture_bbox'] = list(best_pred_bbox) if best_pred_bbox else None
            item['is_docling_picture_area'] = True
            item['suppress_text_alignment'] = True
            item['is_chart_visual'] = True

        return extraction_items

    def _pdf_unit_key(self, unit):
        unit_id = unit.get('unit_id')
        if unit_id is not None:
            return ('unit_id', unit_id)
        item_idx = unit.get('item_idx')
        if item_idx is not None:
            return ('item_idx', item_idx)
        bbox = unit.get('bbox')
        if bbox and len(bbox) >= 4:
            return ('bbox', tuple(bbox))
        return None

    def _unit_overlaps_fused(self, unit_bbox, fused_results):
        if not unit_bbox or len(unit_bbox) < 4:
            return False
        for result in fused_results or []:
            bbox = result.get('bbox')
            if not bbox or len(bbox) < 4:
                continue
            if self.fusion_service.calculate_overlap(unit_bbox, bbox) > 0:
                return True
        return False

    def _collect_unfused_pdf_units(self, all_pdf_units, fused_results, unaligned_pdf_units):
        unaligned_keys = set()
        for unit in unaligned_pdf_units or []:
            key = self._pdf_unit_key(unit)
            if key is not None:
                unaligned_keys.add(key)

        unfused = []
        seen = set(unaligned_keys)
        for unit in all_pdf_units or []:
            if not unit or not unit.get('bbox'):
                continue
            key = self._pdf_unit_key(unit)
            if key is None or key in seen:
                continue
            if self._unit_overlaps_fused(unit.get('bbox'), fused_results):
                seen.add(key)
                continue
            unfused.append(unit)
            seen.add(key)
        return unfused

    def _collect_duplicate_openxml_element_ids(self, page_vis_payload):
        duplicate_analysis = self._analyze_duplicate_openxml_elements(page_vis_payload)
        return {
            elem_id
            for elem_id, analysis in (duplicate_analysis or {}).items()
            if analysis.get('is_invalid_duplicate')
        }

    def _analyze_duplicate_openxml_elements(self, page_vis_payload):
        element_entries = {}
        page_heights = {}

        for page_num, payload in (page_vis_payload or {}).items():
            parsed_page_num = self._try_parse_int_id(page_num)
            if parsed_page_num is None:
                continue
            page_height = payload.get('page_height')
            if page_height is not None:
                try:
                    page_heights[parsed_page_num] = float(page_height)
                except (TypeError, ValueError):
                    pass
            for alignment in payload.get('alignments') or []:
                elem_id = self._try_parse_int_id((alignment or {}).get('element_id'))
                bbox = (alignment or {}).get('merged_bbox') or (alignment or {}).get('bbox')
                if elem_id is None or not bbox or len(bbox) < 4:
                    continue
                element_entries.setdefault(elem_id, []).append({
                    'page': parsed_page_num,
                    'bbox': bbox,
                })

        duplicate_analysis = {}
        for elem_id, entries in element_entries.items():
            analysis = analyze_cross_page_entries(entries, page_heights=page_heights)
            if analysis.get('is_multi_page'):
                duplicate_analysis[elem_id] = analysis
        return duplicate_analysis
