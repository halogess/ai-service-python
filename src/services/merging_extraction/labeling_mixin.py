
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


class MergingExtractionLabelingMixin:


    def _coerce_text(self, value):
        if value is None:
            return ''
        if isinstance(value, list):
            return ' '.join(str(v) for v in value)
        return str(value)

    def _text_starts_with_bab(self, text):
        if not text:
            return False
        return bool(self.BAB_TITLE_REGEX.match(text))

    def _is_subchapter_title(self, text):
        if not text:
            return False
        return bool(self.SUBCHAPTER_TITLE_REGEX.match(text))

    def _get_text_list_marker(self, text):
        if not text:
            return None
        if self._is_subchapter_title(text):
            return None
        if self.LIST_NUMERIC_REGEX.match(text):
            return 'numeric'
        if self.LIST_ALPHA_REGEX.match(text):
            return 'alpha'
        if self.LIST_TEXTUAL_BULLET_REGEX.match(text):
            return 'bullet_textual'
        if self.LIST_BULLET_REGEX.match(text):
            return 'bullet_symbol'
        return None

    @staticmethod
    def _canonicalize_visual_label(label):
        normalized = str(label or '').strip().lower()
        if normalized == 'paragraph':
            return 'text'
        return normalized

    def _get_visual_label(self, result):
        return self._canonicalize_visual_label(
            result.get('label') or result.get('docling_label')
        )

    def _is_picture_result(self, result):
        if not result:
            return False
        label = str(result.get('label') or '').lower()
        docling_label = str(result.get('docling_label') or '').lower()
        return label == 'picture' or docling_label == 'picture'

    def _is_caption_like_visual_result(self, result):
        if not result:
            return False
        label = self._get_visual_label(result)
        if label == 'caption':
            return True
        if result.get('is_chart_caption_text'):
            return True
        text = self._coerce_text(result.get('text')).strip()
        if not text:
            return False
        return self.fusion_service._is_caption_candidate(text)

    def _select_chart_visual_alignments(self, alignments):
        selected = [
            alignment for alignment in (alignments or [])
            if (
                alignment.get('is_image_part') or
                alignment.get('is_openxml_chart') or
                alignment.get('is_openxml_visual_slot') or
                alignment.get('is_chart_visual_attachment') or
                self._alignment_has_visual_units(alignment)
            )
        ]
        return selected or list(alignments or [])

    def _is_valid_same_page_chart_caption_pair(self, picture_result, caption_result):
        if not self._is_picture_result(picture_result):
            return False
        if not self._is_caption_like_visual_result(caption_result):
            return False
        caption_text = self._coerce_text((caption_result or {}).get('text'))
        generic_picture_caption = (
            self._try_parse_int_id((picture_result or {}).get('matched_pdf_unit_count')) is not None and
            self._try_parse_int_id((picture_result or {}).get('matched_pdf_unit_count')) > 0 and
            bool((picture_result or {}).get('is_picture_area')) and
            self._is_figure_caption_text(caption_text)
        )
        if not (
            picture_result.get('is_openxml_chart') or
            picture_result.get('repair_reason') == 'chart_visual_attach' or
            picture_result.get('is_chart_visual_attachment') or
            generic_picture_caption
        ):
            return False

        picture_bbox = picture_result.get('bbox')
        caption_bbox = caption_result.get('bbox')
        if not picture_bbox or not caption_bbox:
            return False
        if len(picture_bbox) < 4 or len(caption_bbox) < 4:
            return False

        caption_gap = float(caption_bbox[1]) - float(picture_bbox[3])
        if caption_gap < -4 or caption_gap > 80:
            return False
        x_overlap = self._bbox_x_overlap_ratio(picture_bbox, caption_bbox)
        if x_overlap < 0.15:
            return False
        return True

    def _select_valid_same_page_chart_caption_results(self, results):
        if not results:
            return []
        picture_results = [result for result in results if self._is_picture_result(result)]
        caption_results = [result for result in results if self._is_caption_like_visual_result(result)]
        if not picture_results or not caption_results:
            return []

        best_pair = None
        best_gap = None
        for picture_result in picture_results:
            picture_bbox = picture_result.get('bbox')
            if not picture_bbox or len(picture_bbox) < 4:
                continue
            for caption_result in caption_results:
                if caption_result is picture_result:
                    continue
                if not self._is_valid_same_page_chart_caption_pair(picture_result, caption_result):
                    continue
                caption_bbox = caption_result.get('bbox')
                gap = max(0.0, float(caption_bbox[1]) - float(picture_bbox[3]))
                if best_pair is None or gap < best_gap:
                    best_pair = (picture_result, caption_result)
                    best_gap = gap
        if not best_pair:
            return []
        return [best_pair[0], best_pair[1]]

    def _select_valid_same_page_chart_caption_claims(self, claims):
        if not claims:
            return []
        pair_results = self._select_valid_same_page_chart_caption_results(
            [claim.get('result') for claim in claims if claim.get('result')]
        )
        if not pair_results:
            return []
        allowed_result_ids = {id(result) for result in pair_results}
        return [
            claim for claim in claims
            if id(claim.get('result')) in allowed_result_ids
        ]

    def _is_figure_panel_marker_text(self, text):
        if not text:
            return False
        return bool(self.FIGURE_PANEL_MARKER_REGEX.match(text.strip()))

    def _has_adjacent_picture_result(self, fused_results, idx):
        if not fused_results or idx < 0 or idx >= len(fused_results):
            return False

        prev_idx = idx - 1
        while prev_idx >= 0:
            prev_label = self._get_visual_label(fused_results[prev_idx])
            if prev_label not in ('page_header', 'page_footer'):
                break
            prev_idx -= 1

        if prev_idx >= 0 and self._is_picture_result(fused_results[prev_idx]):
            return True

        next_idx = idx + 1
        while next_idx < len(fused_results):
            next_label = self._get_visual_label(fused_results[next_idx])
            if next_label not in ('page_header', 'page_footer'):
                break
            next_idx += 1

        return next_idx < len(fused_results) and self._is_picture_result(fused_results[next_idx])

    def _looks_like_code_line_text(self, text):
        text = self._coerce_text(text).strip()
        if not text:
            return False
        if self.CODE_LINE_NUMBER_REGEX.match(text):
            return True
        if self.CODE_TEXT_HINT_REGEX.search(text):
            return True
        symbol_count = sum(1 for ch in text if ch in '{}[]();=<>:+-*/%#\\')
        return symbol_count >= 3

    def _is_code_title_like_text(self, text):
        text = self._coerce_text(text).strip()
        if not text:
            return False
        if self.CODE_TITLE_HEADER_REGEX.search(text):
            return True
        return bool(self.CODE_TITLE_FLEX_REGEX.match(text))

    def _count_following_code_like_lines(self, fused_results, start_idx, allow_title_bridges=False):
        count = 0
        title_bridge_count = 0
        for i in range(start_idx + 1, len(fused_results)):
            candidate = fused_results[i]
            visual_label = self._get_visual_label(candidate)
            if visual_label in ('page_header', 'page_footer'):
                continue
            if visual_label == 'code':
                count += 1
                continue
            if visual_label == 'text' and self._looks_like_code_line_text(candidate.get('text')):
                count += 1
                continue
            if (
                allow_title_bridges
                and count == 0
                and title_bridge_count < 3
                and visual_label in ('caption', 'text', 'section_header')
                and self._is_code_title_like_text(candidate.get('text'))
            ):
                title_bridge_count += 1
                continue
            break
        return count

    def _load_json_tree(self, raw_tree):
        if raw_tree is None:
            return None
        if isinstance(raw_tree, str):
            try:
                return json.loads(raw_tree)
            except Exception:
                return None
        return raw_tree

    def _get_element_json_tree(self, element, cache):
        if not element:
            return None
        elem_id = element.delemen_id
        if elem_id in cache:
            return cache[elem_id]
        tree = self._load_json_tree(element.delemen_json_tree)
        cache[elem_id] = tree
        return tree

    def _normalize_alignment_value(self, value):
        if value is None:
            return None
        if isinstance(value, dict):
            for key in ('val', 'value', 'align', 'alignment'):
                if key in value:
                    return self._normalize_alignment_value(value.get(key))
            return None
        if isinstance(value, list):
            for item in value:
                normalized = self._normalize_alignment_value(item)
                if normalized:
                    return normalized
            return None
        normalized = str(value).strip().lower()
        if not normalized:
            return None
        if normalized.isdigit():
            code = int(normalized)
            return {
                0: 'left',
                1: 'center',
                2: 'right',
                3: 'both',
                4: 'distribute'
            }.get(code, normalized)
        if normalized in ('start', 'left'):
            return 'left'
        if normalized in ('end', 'right'):
            return 'right'
        if normalized in ('justify', 'both'):
            return 'both'
        if normalized == 'distribute':
            return 'distribute'
        if normalized in ('centercontinuous', 'center_continuous', 'center-continuous'):
            return 'center'
        return normalized
