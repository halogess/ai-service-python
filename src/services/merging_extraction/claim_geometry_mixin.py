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


class MergingExtractionClaimGeometryMixin:


    @staticmethod
    def _bbox_area(bbox):
        if not bbox or len(bbox) < 4:
            return 0.0
        width = max(0.0, float(bbox[2]) - float(bbox[0]))
        height = max(0.0, float(bbox[3]) - float(bbox[1]))
        return width * height

    @staticmethod
    def _bbox_x_overlap_ratio(bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        left = max(float(bbox_a[0]), float(bbox_b[0]))
        right = min(float(bbox_a[2]), float(bbox_b[2]))
        if right <= left:
            return 0.0
        width_a = max(0.0, float(bbox_a[2]) - float(bbox_a[0]))
        width_b = max(0.0, float(bbox_b[2]) - float(bbox_b[0]))
        min_width = min(width_a, width_b)
        if min_width <= 0.0:
            return 0.0
        return (right - left) / min_width

    @staticmethod
    def _bbox_y_overlap_ratio(bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        top = max(float(bbox_a[1]), float(bbox_b[1]))
        bottom = min(float(bbox_a[3]), float(bbox_b[3]))
        if bottom <= top:
            return 0.0
        height_a = max(0.0, float(bbox_a[3]) - float(bbox_a[1]))
        height_b = max(0.0, float(bbox_b[3]) - float(bbox_b[1]))
        min_height = min(height_a, height_b)
        if min_height <= 0.0:
            return 0.0
        return (bottom - top) / min_height

    @classmethod
    def _bbox_overlap_ratio(cls, bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        left = max(float(bbox_a[0]), float(bbox_b[0]))
        top = max(float(bbox_a[1]), float(bbox_b[1]))
        right = min(float(bbox_a[2]), float(bbox_b[2]))
        bottom = min(float(bbox_a[3]), float(bbox_b[3]))
        if right <= left or bottom <= top:
            return 0.0
        overlap_area = (right - left) * (bottom - top)
        min_area = min(cls._bbox_area(bbox_a), cls._bbox_area(bbox_b))
        if min_area <= 0.0:
            return 0.0
        return overlap_area / min_area

    def _visual_result_claim_score(self, result):
        source = str((result or {}).get('source') or '').strip().lower()
        confidence = float((result or {}).get('alignment_confidence') or 0.0)
        if source in {'note', 'bookmark_proxy', 'body_text_proxy'}:
            confidence = max(confidence, 0.98)
        elif source == 'header_footer':
            confidence = max(confidence, 0.95)
        if (result or {}).get('repair_reason'):
            confidence = max(0.0, confidence - 0.05)
        overlap = float((result or {}).get('overlap') or 0.0)
        area = self._bbox_area((result or {}).get('bbox'))
        text_len = len(self._coerce_text((result or {}).get('text')))
        return confidence, overlap, area, text_len

    def _visual_existing_claim_score(self, row):
        bbox = [
            getattr(row, 'dev_bbox_x0', None),
            getattr(row, 'dev_bbox_y0', None),
            getattr(row, 'dev_bbox_x1', None),
            getattr(row, 'dev_bbox_y1', None),
        ]
        area = self._bbox_area(bbox)
        text_len = len(self._coerce_text(getattr(row, 'dev_text', None)))
        # Historical rows do not store overlap, so default to 0.0.
        return 0.0, 0.0, area, text_len

    def _is_table_like_visual_result(self, result):
        if not result:
            return False
        visual_label = self._get_visual_label(result)
        if visual_label == 'table':
            return True
        if result.get('has_table_units'):
            return True
        element_type = str(result.get('element_type') or '').strip().lower()
        return 'table' in element_type

    def _clear_visual_result_claim(self, result, reason, winner_claim=None, drop_from_output=False):
        if not result or result.get('element_id') is None:
            return False
        result['element_id'] = None
        result['duplicate_claim_conflict'] = True
        result['duplicate_claim_reason'] = reason
        source = str((result or {}).get('source') or '').strip().lower()
        synthetic_proxy_kind = str((result or {}).get('synthetic_proxy_kind') or '').strip().lower()
        can_drop_from_output = source in {'bookmark_proxy', 'body_text_proxy'} or synthetic_proxy_kind in {
            'bookmark_end',
            'body_text',
        }
        if drop_from_output and can_drop_from_output:
            result['_drop_from_output'] = True
        else:
            result.pop('_drop_from_output', None)
        if winner_claim:
            result['duplicate_claim_winner_page'] = winner_claim.get('page')
            winner_result = winner_claim.get('result') or {}
            result['duplicate_claim_winner_element_id'] = winner_result.get('element_id')
        return True

    @staticmethod
    def _is_synthetic_repair_reason(reason):
        return str(reason or '').strip().lower() in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
        }

    def _is_synthetic_rescue_result(self, result):
        if not result:
            return False
        source = str(result.get('source') or '').strip().lower()
        if source in {'bookmark_proxy', 'body_text_proxy'}:
            return True
        if not self._is_synthetic_repair_reason(result.get('repair_reason')):
            return False
        matched_unit_count = self._try_parse_int_id(result.get('matched_pdf_unit_count'))
        if matched_unit_count is None:
            matched_unit_count = 0
        return matched_unit_count <= 0

    def _is_short_ambiguous_result(self, result):
        label = self._get_visual_label(result)
        text = self._coerce_text((result or {}).get('text')).strip()
        if label in {'picture', 'caption'}:
            return True
        if not text:
            return True
        return len(text) <= 64 or self.fusion_service._is_caption_candidate(text) or text.lower().startswith('[img')

    def _is_cover_drop_candidate(self, result):
        source = str((result or {}).get('source') or '').strip().lower()
        if source in {'bookmark_proxy', 'body_text_proxy'}:
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        if repair_reason not in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
            'picture_overlap_prune',
        }:
            return False
        matched_unit_count = self._try_parse_int_id(result.get('matched_pdf_unit_count'))
        if matched_unit_count is None:
            matched_unit_count = 0
        return matched_unit_count <= 0

    def _find_same_page_covering_claim(self, result, claimed_rows):
        if not result:
            return None
        bbox = result.get('bbox')
        if not bbox or len(bbox) < 4:
            return None
        result_label = self._get_visual_label(result)

        best_candidate = None
        best_score = None
        for candidate in claimed_rows or []:
            if not candidate or candidate.get('element_id') is None:
                continue
            candidate_bbox = candidate.get('bbox')
            overlap_ratio = self._bbox_overlap_ratio(bbox, candidate_bbox)
            if overlap_ratio < 0.98:
                continue
            candidate_label = self._get_visual_label(candidate)
            labels_compatible = (
                candidate_label == result_label or
                (
                    result_label in {'text', 'paragraph', 'caption'} and
                    candidate_label in {'text', 'paragraph', 'caption', 'section_header'}
                ) or
                (result_label == 'picture' and candidate_label == 'picture') or
                (
                    self._is_table_like_visual_result(result) and
                    self._is_table_like_visual_result(candidate)
                ) or
                (
                    str((result or {}).get('repair_reason') or '').strip().lower() == 'table_lead_inherit' and
                    self._is_table_like_visual_result(candidate)
                )
            )
            if not labels_compatible:
                continue
            candidate_score = (
                overlap_ratio,
                float(candidate.get('alignment_confidence') or 0.0),
                -self._bbox_area(candidate_bbox),
            )
            if best_score is None or candidate_score > best_score:
                best_score = candidate_score
                best_candidate = candidate

        return best_candidate

    def _select_valid_same_page_table_claims(self, page_claims):
        valid_claims = [
            claim for claim in (page_claims or [])
            if self._is_table_like_visual_result(claim.get('result') or {})
        ]
        return valid_claims

    def _is_claimed_cover_clear_candidate(self, result):
        if not result:
            return False
        if self._is_synthetic_rescue_result(result):
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        return repair_reason in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'table_lead_inherit',
        }
