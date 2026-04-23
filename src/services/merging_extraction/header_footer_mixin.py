
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


class MergingExtractionHeaderFooterMixin:


    def _get_alignment_sequence_value(self, alignment):
        seq = alignment.get('element_sequence') if alignment else None
        if seq is None:
            return None
        try:
            return int(seq)
        except (TypeError, ValueError):
            return None

    def _get_alignment_center_y(self, alignment):
        bbox = (alignment or {}).get('merged_bbox') or (alignment or {}).get('bbox')
        if not bbox or len(bbox) < 4:
            return None
        return (bbox[1] + bbox[3]) / 2

    def _normalize_text_value(self, value):
        if value is None:
            return ''
        if isinstance(value, list):
            value = ' '.join(str(v) for v in value)
        return self.alignment_service._normalize_text(str(value))

    def _try_parse_int_id(self, value):
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value if value >= 0 else None
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed.isdigit():
                return int(trimmed)
        return None

    def _normalize_part_position(self, position):
        if not position:
            return 'default'
        normalized = str(position).strip().lower()
        if normalized in ('first', 'even', 'default'):
            return normalized
        return 'default'

    def _resolve_section_start_page(self, db, canonical_ref_tipe, ref_id, section_id):
        if not db or ref_id is None or section_id is None:
            return None

        row = (
            db.query(DokumenElemenVisual.dev_page)
            .join(DokumenElemen, DokumenElemenVisual.dokumen_elemen_id == DokumenElemen.delemen_id)
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .filter(
                DokumenElemenVisual.dev_ref_tipe == canonical_ref_tipe,
                DokumenElemenVisual.dev_ref_id == ref_id,
                DokumenElemenVisual.dev_page.isnot(None),
                DokumenPart.dsec_id == section_id,
                DokumenPart.dpart_type == 'body'
            )
            .order_by(DokumenElemenVisual.dev_page.asc())
            .first()
        )
        if not row or row[0] is None:
            return None
        try:
            return int(row[0])
        except (TypeError, ValueError):
            return None

    def _current_page_has_section_body_element(self, db, fused_results, section_id):
        if not db or not fused_results or section_id is None:
            return False

        candidate_ids = set()
        for result in fused_results:
            label = self._get_visual_label(result)
            if label in ('page_header', 'page_footer'):
                continue
            elem_id = self._try_parse_int_id(result.get('element_id'))
            if elem_id is not None:
                candidate_ids.add(elem_id)

        if not candidate_ids:
            return False

        row = (
            db.query(DokumenElemen.delemen_id)
            .join(DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id)
            .filter(
                DokumenElemen.delemen_id.in_(candidate_ids),
                DokumenPart.dsec_id == section_id,
                DokumenPart.dpart_type == 'body'
            )
            .first()
        )
        return row is not None

    def _select_header_footer_part(self, part_infos, preference_order):
        if not part_infos:
            return None

        order_index = {pos: idx for idx, pos in enumerate(preference_order)}

        for position in preference_order:
            matches = [
                info for info in part_infos
                if info.get('position') == position and info.get('elements')
            ]
            if matches:
                return sorted(matches, key=lambda item: item.get('part_id') or 0)[0]

        candidates = [info for info in part_infos if info.get('elements')]
        if not candidates:
            return None

        def key_fn(item):
            position = item.get('position') or 'default'
            return (order_index.get(position, len(order_index)), item.get('part_id') or 0)

        return sorted(candidates, key=key_fn)[0]

    def _extract_element_text_norm(self, element):
        if not element:
            return ''
        tree = self._load_json_tree(element.delemen_json_tree)
        text = self.alignment_service._extract_text_from_json_tree(tree)
        return self._normalize_text_value(text)

    def _build_header_footer_mapping_context(
        self,
        db,
        canonical_ref_tipe,
        ref_id,
        page_num,
        fused_results,
        section_data
    ):
        section_id = self._try_parse_int_id((section_data or {}).get('dsec_id')) if isinstance(section_data, dict) else None
        if section_id is None:
            return None

        section = db.query(DokumenSection).filter(DokumenSection.dsec_id == section_id).first()
        if section is None:
            return None

        section_start_page = self._resolve_section_start_page(db, canonical_ref_tipe, ref_id, section_id)
        if section_start_page is None and self._current_page_has_section_body_element(db, fused_results, section_id):
            section_start_page = page_num

        is_first_page = section_start_page is not None and int(page_num) == int(section_start_page)
        is_even_page = int(page_num) % 2 == 0

        part_rows = db.query(DokumenPart).filter(
            DokumenPart.dsec_id == section_id,
            DokumenPart.dpart_type.in_(('header', 'footer'))
        ).all()
        if not part_rows:
            return None

        part_ids = [part.dpart_id for part in part_rows if part.dpart_id is not None]
        elements_by_part = {}
        if part_ids:
            part_elements = db.query(DokumenElemen).filter(DokumenElemen.dpart_id.in_(part_ids)).all()
            for element in part_elements:
                elements_by_part.setdefault(element.dpart_id, []).append(element)

        context = {}
        for part_type in ('header', 'footer'):
            type_parts = [part for part in part_rows if (part.dpart_type or '').lower() == part_type]
            part_infos = []
            for part in type_parts:
                position = self._normalize_part_position(part.dpart_position)
                raw_elements = elements_by_part.get(part.dpart_id, [])
                ordered_elements = sorted(
                    raw_elements,
                    key=lambda elem: (
                        0 if str(elem.delemen_type or '').lower() == 'paragraph' else 1,
                        elem.delemen_sequence if elem.delemen_sequence is not None else 2**31 - 1,
                        elem.delemen_id
                    )
                )
                part_infos.append({
                    'part_id': part.dpart_id,
                    'position': position,
                    'elements': [
                        {
                            'id': int(elem.delemen_id),
                            'text_norm': self._extract_element_text_norm(elem)
                        }
                        for elem in ordered_elements
                    ]
                })

            if is_first_page and bool(section.dsec_has_title_page):
                preference_order = ['first', 'default', 'even']
            elif bool(section.dsec_different_odd_even) and is_even_page:
                preference_order = ['even', 'default', 'first']
            else:
                preference_order = ['default', 'first', 'even']

            context[part_type] = {
                'selected_part': self._select_header_footer_part(part_infos, preference_order),
                'parts': part_infos
            }

        return context

    def _resolve_header_footer_element_id(self, result, visual_label, header_footer_context):
        if visual_label not in ('page_header', 'page_footer') or not header_footer_context:
            return None

        part_type = 'header' if visual_label == 'page_header' else 'footer'
        selected_part = (header_footer_context.get(part_type) or {}).get('selected_part')
        if not selected_part:
            return None

        candidates = selected_part.get('elements') or []
        if not candidates:
            return None

        text_norm = self._normalize_text_value(result.get('text'))
        if text_norm:
            matches = [
                candidate for candidate in candidates
                if candidate.get('text_norm') and candidate.get('text_norm') == text_norm
            ]
            if len(matches) == 1:
                return matches[0].get('id')
            if len(matches) > 1:
                return None

        return candidates[0].get('id')

    def _simplify_duplicate_unit_text(self, text):
        if not text:
            return ''
        return re.sub(r'[\W_]+', '', text, flags=re.UNICODE)

    def _get_bbox_center_y(self, bbox):
        if not bbox or len(bbox) < 4:
            return None
        return (bbox[1] + bbox[3]) / 2

    def _get_caption_structural_label(self, bbox, fused_results):
        if not bbox:
            return 'caption'
        best_label = None
        best_gap = None
        for result in fused_results or []:
            label = str(result.get('label') or result.get('docling_label') or '').lower()
            if label not in ('picture', 'table'):
                continue
            cand_bbox = result.get('bbox')
            if not cand_bbox or len(cand_bbox) < 4:
                continue
            if bbox[1] >= cand_bbox[3]:
                gap = bbox[1] - cand_bbox[3]
            elif cand_bbox[1] >= bbox[3]:
                gap = cand_bbox[1] - bbox[3]
            else:
                gap = 0
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_label = label
        if best_label == 'table':
            return 'caption_tabel'
        if best_label == 'picture':
            return 'caption_gambar'
        return 'caption'
