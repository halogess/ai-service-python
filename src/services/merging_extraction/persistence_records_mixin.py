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


class MergingExtractionPersistenceRecordsMixin:


    def _collect_existing_claims_by_element(self, db, canonical_ref_tipe, ref_id, page_num, element_ids):
        if not db or ref_id is None or page_num is None or not element_ids:
            return {}

        query = db.query(DokumenElemenVisual).filter(
            DokumenElemenVisual.dev_ref_id == ref_id,
            DokumenElemenVisual.dokumen_elemen_id.in_(list(element_ids)),
            DokumenElemenVisual.dev_page.isnot(None),
            DokumenElemenVisual.dev_page != page_num
        )
        if canonical_ref_tipe == 'bab':
            query = query.filter(DokumenElemenVisual.dev_ref_tipe.in_(('bab', 'buku')))
        else:
            query = query.filter(DokumenElemenVisual.dev_ref_tipe == canonical_ref_tipe)

        existing_rows = list(query.all() or [])
        claims_by_element = {}
        for row in existing_rows:
            elem_id = self._try_parse_int_id(getattr(row, 'dokumen_elemen_id', None))
            page = self._try_parse_int_id(getattr(row, 'dev_page', None))
            if elem_id is None or page is None:
                continue
            claims_by_element.setdefault(elem_id, []).append({
                'dev_id': self._try_parse_int_id(getattr(row, 'dev_id', None)),
                'page': page,
                'score': self._visual_existing_claim_score(row),
                'label': getattr(row, 'dev_label', None),
                'is_table_like': str(getattr(row, 'dev_label', '') or '').strip().lower() == 'table'
            })
        return claims_by_element

    def _prune_far_gap_duplicate_claims(self, fused_results, page_num, existing_claims_by_element):
        if not fused_results:
            return fused_results, 0, set()

        current_page = self._try_parse_int_id(page_num)
        if current_page is None:
            return fused_results, 0, set()

        claim_result_indices = {}
        for idx, result in enumerate(fused_results):
            elem_id = self._try_parse_int_id((result or {}).get('element_id'))
            if elem_id is None:
                continue
            visual_label = self._get_visual_label(result)
            if visual_label in ('page_header', 'page_footer'):
                continue
            claim_result_indices.setdefault(elem_id, []).append(idx)

        if not claim_result_indices:
            return fused_results, 0, set()

        cleared_current_claims = 0
        clear_existing_ids = set()

        for elem_id, indices in claim_result_indices.items():
            current_results = [fused_results[idx] for idx in indices]
            if any(self._is_table_like_visual_result(result) for result in current_results):
                continue

            existing_claims = existing_claims_by_element.get(elem_id) or []
            far_claims = [
                claim for claim in existing_claims
                if abs((claim.get('page') or current_page) - current_page) > self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ]
            if not far_claims:
                continue
            if any(bool(claim.get('is_table_like')) for claim in far_claims):
                continue

            allowed_pair_results = self._select_valid_same_page_chart_caption_results(current_results)
            allowed_pair_result_ids = {id(result) for result in allowed_pair_results}
            scoring_indices = [
                idx for idx in indices
                if id(fused_results[idx]) not in allowed_pair_result_ids
            ]
            if not scoring_indices:
                picture_indices = [
                    idx for idx in indices
                    if self._is_picture_result(fused_results[idx])
                ]
                scoring_indices = picture_indices or list(indices)

            best_current_idx = max(scoring_indices, key=lambda i: self._visual_result_claim_score(fused_results[i]))
            best_current_score = self._visual_result_claim_score(fused_results[best_current_idx])
            best_existing_score = max((claim.get('score') or (0.0, 0.0, 0)) for claim in far_claims)

            if best_current_score > best_existing_score:
                for idx in indices:
                    if (
                        idx == best_current_idx or
                        id(fused_results[idx]) in allowed_pair_result_ids or
                        fused_results[idx].get('element_id') is None
                    ):
                        continue
                    fused_results[idx]['element_id'] = None
                    fused_results[idx]['duplicate_claim_conflict'] = True
                    cleared_current_claims += 1
                for claim in far_claims:
                    dev_id = claim.get('dev_id')
                    if dev_id is not None:
                        clear_existing_ids.add(dev_id)
            else:
                for idx in indices:
                    if fused_results[idx].get('element_id') is None:
                        continue
                    fused_results[idx]['element_id'] = None
                    fused_results[idx]['duplicate_claim_conflict'] = True
                    cleared_current_claims += 1

        return fused_results, cleared_current_claims, clear_existing_ids

    def _replace_visual_records(self, db, ref_tipe, ref_id, page_num, fused_results, structural_state=None, section_data=None, apply_duplicate_claim_guard=True):
        if not db or ref_id is None or page_num is None:
            return list(fused_results or [])
        canonical_ref_tipe = self._canonical_ref_tipe(ref_tipe)
        fused_results = [
            result for result in (fused_results or [])
            if not (result or {}).get('_drop_from_output')
        ]

        if apply_duplicate_claim_guard:
            claimed_element_ids = set()
            for result in fused_results:
                elem_id = self._try_parse_int_id(result.get('element_id'))
                if elem_id is None:
                    continue
                visual_label = self._get_visual_label(result)
                if visual_label in ('page_header', 'page_footer'):
                    continue
                claimed_element_ids.add(elem_id)

            existing_claims = self._collect_existing_claims_by_element(
                db,
                canonical_ref_tipe,
                ref_id,
                page_num,
                claimed_element_ids
            )
            fused_results, cleared_claim_rows, clear_existing_ids = self._prune_far_gap_duplicate_claims(
                fused_results,
                page_num,
                existing_claims
            )
            if clear_existing_ids:
                db.query(DokumenElemenVisual).filter(
                    DokumenElemenVisual.dev_id.in_(list(clear_existing_ids))
                ).update(
                    {DokumenElemenVisual.dokumen_elemen_id: None},
                    synchronize_session=False
                )

            logger.debug(
                "Page %s: far-gap duplicate claim guard cleared_current=%s cleared_existing=%s",
                page_num,
                cleared_claim_rows,
                len(clear_existing_ids)
            )

        if fused_results:
            self._apply_structural_labels(
                db,
                fused_results,
                structural_state=structural_state,
                skip_if_labeled=True
            )
        delete_query = db.query(DokumenElemenVisual).filter(
            DokumenElemenVisual.dev_ref_id == ref_id,
            DokumenElemenVisual.dev_page == page_num
        )
        if canonical_ref_tipe == 'bab':
            delete_query = delete_query.filter(DokumenElemenVisual.dev_ref_tipe.in_(('bab', 'buku')))
        else:
            delete_query = delete_query.filter(DokumenElemenVisual.dev_ref_tipe == canonical_ref_tipe)
        delete_query.delete(synchronize_session=False)

        has_header_footer_rows = any(
            self._get_visual_label(result) in ('page_header', 'page_footer')
            for result in (fused_results or [])
        )
        header_footer_context = None
        if has_header_footer_rows:
            header_footer_context = self._build_header_footer_mapping_context(
                db,
                canonical_ref_tipe,
                ref_id,
                page_num,
                fused_results,
                section_data
            )

        header_footer_total = 0
        header_footer_mapped = 0
        header_footer_null = 0

        for result in fused_results or []:
            text_content = result.get('text', '')
            if isinstance(text_content, list):
                text_content = " ".join(text_content)
            elif text_content is None:
                text_content = ""

            bbox = result.get('bbox')
            x0 = y0 = x1 = y1 = 0
            if bbox and len(bbox) == 4:
                x0, y0, x1, y1 = bbox

            visual_label = self._get_visual_label(result)
            final_element_id = result.get('element_id')
            if visual_label in ('page_header', 'page_footer'):
                header_footer_total += 1
                parsed_element_id = self._try_parse_int_id(final_element_id)
                if parsed_element_id is None:
                    parsed_element_id = self._resolve_header_footer_element_id(
                        result,
                        visual_label,
                        header_footer_context
                    )
                final_element_id = parsed_element_id
                result['element_id'] = final_element_id

                if final_element_id is None:
                    header_footer_null += 1
                else:
                    header_footer_mapped += 1

            dev = DokumenElemenVisual(
                dev_ref_tipe=canonical_ref_tipe,
                dev_ref_id=ref_id,
                dev_page=page_num,
                dokumen_elemen_id=final_element_id,
                dev_bbox_x0=float(x0),
                dev_bbox_y0=float(y0),
                dev_bbox_x1=float(x1),
                dev_bbox_y1=float(y1),
                dev_label=result.get('label') or result.get('docling_label'),
                dev_label_struktural=result.get('dev_label_struktural'),
                dev_text=text_content
            )
            db.add(dev)

        if header_footer_total > 0:
            logger.info(
                "Page %s: header/footer rows total=%s mapped=%s null=%s",
                page_num,
                header_footer_total,
                header_footer_mapped,
                header_footer_null
            )

        # SessionLocal is configured with autoflush=False, so flush explicitly to
        # make this page's claims visible to duplicate-claim guard on next pages.
        db.flush()
        return fused_results

    def _is_duplicate_sequence_far(self, alignments, alignment, threshold):
        seq = self._get_alignment_sequence_value(alignment)
        if seq is None:
            return False

        target_y = self._get_alignment_center_y(alignment)
        if target_y is None:
            return False

        prev_seq = None
        next_seq = None
        best_above_delta = None
        best_below_delta = None
        for candidate in alignments or []:
            if candidate is alignment:
                continue
            cand_seq = self._get_alignment_sequence_value(candidate)
            if cand_seq is None:
                continue
            cand_y = self._get_alignment_center_y(candidate)
            if cand_y is None:
                continue
            delta = cand_y - target_y
            if delta < 0:
                delta = abs(delta)
                if best_above_delta is None or delta < best_above_delta:
                    best_above_delta = delta
                    prev_seq = cand_seq
            elif delta > 0:
                if best_below_delta is None or delta < best_below_delta:
                    best_below_delta = delta
                    next_seq = cand_seq

        if prev_seq is None and next_seq is None:
            return False
        if prev_seq is None:
            return (next_seq - seq) > threshold
        if next_seq is None:
            return (seq - prev_seq) > threshold
        return (seq - prev_seq) > threshold or (next_seq - seq) > threshold

    def _collect_duplicate_units_for_page(self, alignments, duplicate_analysis, page_num):
        if not alignments or not duplicate_analysis:
            return []
        current_page = self._try_parse_int_id(page_num)
        if current_page is None:
            return []
        duplicates = []
        for alignment in alignments:
            elem_id = self._try_parse_int_id(alignment.get('element_id'))
            if elem_id is None:
                continue
            analysis = (duplicate_analysis or {}).get(elem_id) or {}
            if not analysis.get('is_invalid_duplicate'):
                continue
            if current_page not in set(analysis.get('invalid_pages') or []):
                continue
            if not self._is_duplicate_sequence_far(
                alignments,
                alignment,
                self.DUPLICATE_SEQUENCE_GAP_THRESHOLD
            ):
                continue
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    duplicates.extend(
                        unit for unit in cell.get('matched_pdf_units', [])
                        if unit.get('item_type') == 'group'
                    )
            else:
                duplicates.extend(
                    unit for unit in alignment.get('matched_pdf_units', [])
                    if unit.get('item_type') == 'group'
                )
        return duplicates
