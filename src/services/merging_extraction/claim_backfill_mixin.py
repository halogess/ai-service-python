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


class MergingExtractionClaimBackfillMixin:


    def _repair_document_header_footer_claims(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'reassigned_rows': 0, 'affected_pages': 0}

        header_footer_targets = [
            target for target in self._load_header_footer_targets_for_ref(db, canonical_ref_tipe, ref_id)
            if target.get('is_eligible_target')
        ]
        if not header_footer_targets:
            return {'reassigned_rows': 0, 'affected_pages': 0}

        targets_by_label = {
            'page_header': [target for target in header_footer_targets if target.get('target_kind') == 'header'],
            'page_footer': [target for target in header_footer_targets if target.get('target_kind') == 'footer'],
        }

        claimed_by_label = {'page_header': [], 'page_footer': []}
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            parsed_page_num = self._try_parse_int_id(page_num)
            for row in (payload or {}).get('fused_results') or []:
                if (row or {}).get('_drop_from_output'):
                    continue
                label = self._get_visual_label(row)
                if label not in claimed_by_label:
                    continue
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None:
                    continue
                claimed_by_label[label].append({
                    'page': parsed_page_num,
                    'element_id': element_id,
                })

        reassigned_rows = 0
        affected_pages = set()
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            parsed_page_num = self._try_parse_int_id(page_num)
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            for row in fused_results:
                if (row or {}).get('_drop_from_output'):
                    continue
                label = self._get_visual_label(row)
                if label not in targets_by_label:
                    continue
                if self._try_parse_int_id((row or {}).get('element_id')) is not None:
                    continue

                candidates = list(targets_by_label.get(label) or [])
                neighbor_ids = {
                    item['element_id']
                    for item in claimed_by_label.get(label, [])
                    if item.get('page') is not None and parsed_page_num is not None and abs(item['page'] - parsed_page_num) <= 1
                }
                exact_text = self._normalize_text_value((row or {}).get('text'))
                exact_matches = [
                    target for target in candidates
                    if target.get('text_norm') and target.get('text_norm') == exact_text
                ]
                page_number_candidates = [
                    target for target in candidates
                    if target.get('is_numeric_page_token')
                ]
                global_numeric_candidates = [
                    target for target in header_footer_targets
                    if target.get('is_numeric_page_token')
                ]

                best_target = None
                if len(exact_matches) == 1:
                    best_target = exact_matches[0]
                elif len(neighbor_ids) == 1:
                    best_target = next(
                        (target for target in candidates if self._try_parse_int_id(target.get('element_id')) in neighbor_ids),
                        None,
                    )
                elif len({self._try_parse_int_id(target.get('element_id')) for target in candidates}) == 1:
                    best_target = candidates[0]
                elif self._looks_like_page_number_token((row or {}).get('text')) and len(page_number_candidates) == 1:
                    best_target = page_number_candidates[0]
                elif self._looks_like_page_number_token((row or {}).get('text')):
                    global_numeric_ids = {
                        self._try_parse_int_id(target.get('element_id'))
                        for target in global_numeric_candidates
                        if self._try_parse_int_id(target.get('element_id')) is not None
                    }
                    if len(global_numeric_ids) == 1:
                        only_id = next(iter(global_numeric_ids))
                        best_target = next(
                            (
                                target for target in global_numeric_candidates
                                if self._try_parse_int_id(target.get('element_id')) == only_id
                            ),
                            None,
                        )

                if not best_target:
                    continue
                if not self._assign_result_to_target(row, best_target, 'document_header_footer_fallback'):
                    continue
                reassigned_rows += 1
                affected_pages.add(parsed_page_num if parsed_page_num is not None else page_num)
                claimed_by_label.setdefault(label, []).append({
                    'page': parsed_page_num,
                    'element_id': self._try_parse_int_id(best_target.get('element_id')),
                })

        return {
            'reassigned_rows': reassigned_rows,
            'affected_pages': len(affected_pages),
        }

    def _backfill_document_bookmark_proxies(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return 0

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        bookmark_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'bookmark'
        ]
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body'
        }

        claimed_ids = set()
        claimed_rows = []
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                claimed_ids.add(element_id)
                target = body_by_id.get(element_id)
                if not target:
                    continue
                sequence = self._try_parse_int_id(target.get('sequence'))
                bbox = row.get('bbox')
                if sequence is None or not bbox or len(bbox) < 4:
                    continue
                claimed_rows.append({
                    'page_num': page_num,
                    'result': row,
                    'target': target,
                })

        created_count = 0
        for bookmark in bookmark_targets:
            bookmark_id = self._try_parse_int_id(bookmark.get('element_id'))
            bookmark_seq = self._try_parse_int_id(bookmark.get('sequence'))
            if bookmark_id is None or bookmark_id in claimed_ids or bookmark_seq is None:
                continue

            best_claim = None
            best_score = None
            for claim in claimed_rows:
                target = claim.get('target') or {}
                sequence = self._try_parse_int_id(target.get('sequence'))
                if sequence is None:
                    continue
                gap = abs(sequence - bookmark_seq)
                if gap > 3:
                    continue
                score = 1.0 - (gap * 0.2)
                if sequence <= bookmark_seq:
                    score += 0.12
                if best_score is None or score > best_score:
                    best_score = score
                    best_claim = claim

            if not best_claim:
                continue

            owner_result = best_claim['result']
            proxy_result = {
                'bbox': list(owner_result.get('bbox') or []),
                'label': self._get_visual_label(owner_result) or 'text',
                'text': '',
                'overlap': float(owner_result.get('overlap') or 0.0),
                'source': 'bookmark_proxy',
                'synthetic_proxy_kind': 'bookmark_end',
                'element_id': bookmark_id,
                'element_type': bookmark.get('element_type'),
                'element_sequence': bookmark_seq,
                'block_kind': bookmark.get('block_kind') or owner_result.get('block_kind'),
                'block_key': bookmark.get('block_key') or owner_result.get('block_key'),
                'content_role': bookmark.get('content_role') or owner_result.get('content_role'),
                'block_order': bookmark.get('block_order') or owner_result.get('block_order'),
                'target_kind': 'bookmark',
                'alignment_confidence': 0.99,
                'candidate_source': 'bookmark_backfill',
                'matched_pdf_unit_count': 0,
            }
            page_vis_payload[best_claim['page_num']].setdefault('fused_results', []).append(proxy_result)
            claimed_ids.add(bookmark_id)
            created_count += 1

        return created_count

    def _backfill_document_text_proxies(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return 0

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        eligible_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]

        claimed_ids = set()
        claimed_rows = []
        for page_num, payload in sorted((page_vis_payload or {}).items()):
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                claimed_ids.add(element_id)
                bbox = row.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                claimed_rows.append({
                    'page_num': page_num,
                    'result': row,
                })

        created_count = 0
        for target in eligible_targets:
            target_id = self._try_parse_int_id(target.get('element_id'))
            target_seq = self._try_parse_int_id(target.get('sequence'))
            if target_id is None or target_id in claimed_ids or target_seq is None:
                continue

            target_is_code = self._is_code_like_target(target)
            if not target_is_code and str(target.get('content_role') or '').strip().lower() != 'continuation_body':
                continue

            best_claim = None
            best_score = None
            for claim in claimed_rows:
                row = claim.get('result') or {}
                row_seq = self._try_parse_int_id(row.get('element_sequence'))
                if row_seq is None or abs(row_seq - target_seq) > 6:
                    continue
                same_block_key = bool(
                    target.get('block_key') and row.get('block_key') and
                    str(target.get('block_key')).strip().lower() == str(row.get('block_key')).strip().lower()
                )
                same_block_order = (
                    self._try_parse_int_id(target.get('block_order')) is not None and
                    self._try_parse_int_id(target.get('block_order')) == self._try_parse_int_id(row.get('block_order'))
                )
                if not (same_block_key or same_block_order or self._result_is_code_like(row)):
                    continue
                score = 0.8 - (abs(row_seq - target_seq) * 0.08)
                if same_block_key:
                    score += 0.35
                if same_block_order:
                    score += 0.20
                if claim.get('page_num') is not None:
                    score += 0.05
                if best_score is None or score > best_score:
                    best_score = score
                    best_claim = claim

            if not best_claim or best_score is None or best_score < 0.60:
                continue

            owner_result = best_claim['result']
            proxy_result = {
                'bbox': list(owner_result.get('bbox') or []),
                'label': self._get_visual_label(owner_result) or 'text',
                'text': target.get('text') or '',
                'overlap': float(owner_result.get('overlap') or 0.0),
                'source': 'body_text_proxy',
                'synthetic_proxy_kind': 'body_text',
                'element_id': target_id,
                'element_type': target.get('element_type'),
                'element_sequence': target_seq,
                'block_kind': target.get('block_kind'),
                'block_key': target.get('block_key'),
                'content_role': target.get('content_role'),
                'block_order': target.get('block_order'),
                'target_kind': 'body',
                'alignment_confidence': 0.97,
                'candidate_source': 'body_text_backfill',
                'matched_pdf_unit_count': 0,
            }
            page_vis_payload[best_claim['page_num']].setdefault('fused_results', []).append(proxy_result)
            claimed_ids.add(target_id)
            claimed_rows.append({
                'page_num': best_claim['page_num'],
                'result': proxy_result,
            })
            created_count += 1

        return created_count

    def _derive_visual_chain_key(self, result):
        if not result:
            return None
        block_key = str((result or {}).get('block_key') or '').strip().lower()
        if block_key:
            return block_key
        text = self._coerce_text((result or {}).get('text')).strip()
        if not text:
            return None
        metadata = self.alignment_service._derive_block_metadata(
            text,
            elem_type=str((result or {}).get('element_type') or self._get_visual_label(result) or ''),
            is_table=self._is_table_like_visual_result(result),
            is_code_like=False,
            current_block=None,
        )
        derived_key = str((metadata or {}).get('block_key') or '').strip().lower()
        return derived_key or None

    def _is_visual_chain_repair_candidate(self, result):
        if not self._result_supports_target_assignment(result):
            return False
        visual_label = self._get_visual_label(result)
        if visual_label in {'picture', 'caption'}:
            return True
        repair_reason = str((result or {}).get('repair_reason') or '').strip().lower()
        return repair_reason in {
            'caption_suffix_inherit',
            'image_placeholder_neighbor_inherit',
            'caption_fragment_inherit',
            'picture_overlap_prune',
        }

    def _repair_adjacent_page_visual_chains(self, page_vis_payload):
        if not page_vis_payload:
            return {'reassigned_rows': 0, 'dropped_rows': 0, 'affected_pages': 0}

        page_numbers = sorted(
            self._try_parse_int_id(page_num)
            for page_num in page_vis_payload.keys()
            if self._try_parse_int_id(page_num) is not None
        )
        reassigned_rows = 0
        dropped_rows = 0
        affected_pages = set()

        for index, page_num in enumerate(page_numbers):
            payload = page_vis_payload.get(page_num) or page_vis_payload.get(str(page_num)) or {}
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            adjacent_rows = []
            for neighbor_page in (page_numbers[index - 1:index] + page_numbers[index + 1:index + 2]):
                neighbor_payload = page_vis_payload.get(neighbor_page) or page_vis_payload.get(str(neighbor_page)) or {}
                for row in (neighbor_payload or {}).get('fused_results') or []:
                    if self._try_parse_int_id((row or {}).get('element_id')) is None:
                        continue
                    if not self._is_visual_chain_repair_candidate(row):
                        continue
                    adjacent_rows.append(row)

            if not adjacent_rows:
                continue

            for result in fused_results:
                if not self._is_visual_chain_repair_candidate(result):
                    continue
                if self._try_parse_int_id((result or {}).get('element_id')) is not None:
                    continue

                chain_key = self._derive_visual_chain_key(result)
                if not chain_key:
                    continue
                result_bbox = result.get('bbox')
                result_label = self._get_visual_label(result)
                best_owner = None
                best_score = None

                for owner in adjacent_rows:
                    owner_key = self._derive_visual_chain_key(owner)
                    if not owner_key or owner_key != chain_key:
                        continue
                    owner_label = self._get_visual_label(owner)
                    if result_label and owner_label and result_label != owner_label:
                        if {result_label, owner_label} != {'picture', 'caption'}:
                            continue
                    score = 1.1
                    if owner_label == result_label:
                        score += 0.18
                    if result_bbox and owner.get('bbox'):
                        score += self._bbox_x_overlap_ratio(result_bbox, owner.get('bbox')) * 0.25
                    if best_score is None or score > best_score:
                        best_score = score
                        best_owner = owner

                if not best_owner or best_score is None or best_score < 1.10:
                    continue

                if self._assign_result_to_existing_owner(result, best_owner, 'adjacent_page_visual_chain'):
                    reassigned_rows += 1
                    affected_pages.add(page_num)
                elif self._merge_same_page_null_fragment_into_owner(result, best_owner):
                    dropped_rows += 1
                    affected_pages.add(page_num)

        return {
            'reassigned_rows': reassigned_rows,
            'dropped_rows': dropped_rows,
            'affected_pages': len(affected_pages),
        }
