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


class MergingExtractionClaimDuplicateMixin:


    def _score_invalid_duplicate_target_candidate(
        self,
        rows,
        target,
        prev_seq,
        next_seq,
        page_claimed_ids,
        page_table_caption_sequences=None,
        page_element_row_counts=None,
    ):
        if not rows or not target:
            return None
        scores = []
        row_count = 0
        for row in rows:
            score = self._score_body_target_candidate(row, target, prev_seq, next_seq)
            row_count += 1
            if score is not None:
                scores.append(score)
        if not scores:
            return None

        avg_score = sum(scores) / len(scores)
        coverage_bonus = (len(scores) / max(1, row_count)) * 0.45
        candidate_id = self._try_parse_int_id(target.get('element_id'))
        candidate_seq = self._try_parse_int_id(target.get('sequence'))
        target_text_norm = self._normalize_text_value(target.get('text'))
        cluster_is_table = any(self._is_table_like_visual_result(row) for row in rows)
        cluster_is_code = any(self._result_is_code_like(row) for row in rows)
        cluster_text_norm = self._normalize_text_value(
            ' '.join(self._coerce_text((row or {}).get('text')) for row in rows)
        )
        cluster_is_caption = any(
            self._get_visual_label(row) == 'caption' or self._is_table_caption_text((row or {}).get('text'))
            for row in rows
        ) or cluster_text_norm in {'lanjutan', '(lanjutan)'}
        continuation_only_caption = cluster_text_norm in {'lanjutan', '(lanjutan)'}
        claimed_same_page = candidate_id in (page_claimed_ids or set())
        page_claim_count = 0
        if page_element_row_counts and candidate_id is not None:
            page_claim_count = int(page_element_row_counts.get(candidate_id) or 0)
        avg_alignment_confidence = sum(
            float((row or {}).get('alignment_confidence') or 0.0)
            for row in rows
        ) / max(1, len(rows))
        candidate_in_window = self._sequence_within_assignment_window(candidate_seq, prev_seq, next_seq, slack=2)

        total = avg_score + coverage_bonus
        if claimed_same_page:
            total += 0.28
        if cluster_is_table and self._is_table_target(target):
            total += 0.42
        if cluster_is_code and self._is_code_like_target(target):
            total += 0.20
            if candidate_in_window:
                total += 0.72
            elif prev_seq is not None or next_seq is not None:
                total -= 0.95
                if avg_alignment_confidence < 0.60:
                    total -= 0.18
        if candidate_in_window:
            total += 0.18
        if (cluster_is_table or cluster_is_caption) and page_claim_count > 0:
            total += min(0.36, page_claim_count * 0.04)
        if cluster_is_table and candidate_seq is not None and page_table_caption_sequences:
            closest_caption_gap = min(
                abs(candidate_seq - seq)
                for seq in page_table_caption_sequences
                if seq is not None
            )
            if closest_caption_gap <= 1:
                total += 0.72
            elif closest_caption_gap <= 2:
                total += 0.48
            elif closest_caption_gap <= 4:
                total += 0.18
            elif closest_caption_gap >= 8:
                total -= 0.22
        if cluster_is_caption and candidate_seq is not None and page_table_caption_sequences:
            closest_caption_gap = min(
                abs(candidate_seq - seq)
                for seq in page_table_caption_sequences
                if seq is not None
            )
            if closest_caption_gap <= 1:
                total += 0.78
            elif closest_caption_gap <= 2:
                total += 0.44
            elif closest_caption_gap >= 8:
                total -= 0.26
            if continuation_only_caption:
                if target_text_norm in {'lanjutan', '(lanjutan)'}:
                    if closest_caption_gap <= 1:
                        total += 0.30
                    elif closest_caption_gap >= 8:
                        total -= 1.10
                elif self._is_table_caption_text(target.get('text')):
                    if closest_caption_gap <= 1:
                        total += 1.05
        return total

    def _repair_invalid_duplicate_claims_to_local_targets(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'reassigned_rows': 0, 'repaired_elements': 0, 'affected_pages': 0}

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body' and target.get('is_eligible_target')
        }
        candidate_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]

        rows_by_element = {}
        for page_num, payload in (page_vis_payload or {}).items():
            fused_results = list((payload or {}).get('fused_results') or [])
            for row in fused_results:
                if (row or {}).get('_drop_from_output'):
                    continue
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or element_id not in body_by_id:
                    continue
                rows_by_element.setdefault(element_id, []).append({
                    'page': self._try_parse_int_id(page_num),
                    'row': row,
                })

        invalid_groups = {}
        for element_id, entries in rows_by_element.items():
            analysis = analyze_cross_page_entries(
                [{'page': item['page'], 'bbox': (item['row'] or {}).get('bbox')} for item in entries]
            )
            if analysis.get('is_invalid_duplicate'):
                invalid_groups[element_id] = {
                    'analysis': analysis,
                    'entries': entries,
                }

        if not invalid_groups:
            return {'reassigned_rows': 0, 'repaired_elements': 0, 'affected_pages': 0}

        invalid_element_ids = set(invalid_groups)
        reassigned_rows = 0
        repaired_elements = 0
        affected_pages = set()

        for element_id, group in invalid_groups.items():
            current_target = body_by_id.get(element_id)
            if not current_target:
                continue
            entries_by_page = {}
            for item in group['entries']:
                entries_by_page.setdefault(item['page'], []).append(item['row'])

            for page_num, rows in entries_by_page.items():
                payload = page_vis_payload.get(page_num) or page_vis_payload.get(str(page_num)) or {}
                fused_results = list((payload or {}).get('fused_results') or [])
                if not fused_results or not rows:
                    continue

                cluster_bbox = self.alignment_service._merge_bboxes([row.get('bbox') for row in rows if row.get('bbox')])
                if not cluster_bbox:
                    continue
                anchor_rows = [
                    row for row in fused_results
                    if not (row or {}).get('_drop_from_output')
                    and self._try_parse_int_id((row or {}).get('element_id')) in body_by_id
                    and self._try_parse_int_id((row or {}).get('element_id')) not in invalid_element_ids
                ]
                ordered_anchors = self._iter_page_body_sequence_anchors(anchor_rows, body_by_id)
                prev_seq, next_seq = self._find_sequence_anchor_window({'bbox': cluster_bbox}, ordered_anchors)
                page_claimed_ids = {
                    self._try_parse_int_id((row or {}).get('element_id'))
                    for row in fused_results
                    if self._try_parse_int_id((row or {}).get('element_id')) is not None and not (row or {}).get('_drop_from_output')
                }

                cluster_is_table = any(self._is_table_like_visual_result(row) for row in rows)
                cluster_is_code = any(self._result_is_code_like(row) for row in rows)
                cluster_is_caption = any(
                    self._get_visual_label(row) == 'caption' or self._is_table_caption_text((row or {}).get('text'))
                    for row in rows
                ) or self._normalize_text_value(
                    ' '.join(self._coerce_text((row or {}).get('text')) for row in rows)
                ) in {'lanjutan', '(lanjutan)'}
                page_table_caption_sequences = [
                    self._try_parse_int_id((body_by_id.get(self._try_parse_int_id((row or {}).get('element_id'))) or {}).get('sequence'))
                    for row in fused_results
                    if (
                        not (row or {}).get('_drop_from_output')
                        and self._get_visual_label(row) == 'caption'
                        and self._is_table_caption_text((row or {}).get('text'))
                    )
                ]
                page_table_caption_sequences = [
                    seq for seq in page_table_caption_sequences
                    if seq is not None
                ]
                page_element_row_counts = {}
                for row in fused_results:
                    if (row or {}).get('_drop_from_output'):
                        continue
                    row_element_id = self._try_parse_int_id((row or {}).get('element_id'))
                    if row_element_id is None:
                        continue
                    page_element_row_counts[row_element_id] = page_element_row_counts.get(row_element_id, 0) + 1

                target_pool = []
                for target in candidate_targets:
                    target_id = self._try_parse_int_id(target.get('element_id'))
                    if target_id is None or target_id == element_id:
                        continue
                    if cluster_is_table and not self._is_table_target(target):
                        continue
                    if cluster_is_code and not self._is_code_like_target(target):
                        continue
                    if cluster_is_table:
                        target_seq = self._try_parse_int_id(target.get('sequence'))
                        if not self._sequence_within_assignment_window(target_seq, prev_seq, next_seq, slack=2):
                            continue
                    target_pool.append(target)
                if cluster_is_caption:
                    caption_target_pool = [
                        target for target in target_pool
                        if (
                            str(target.get('block_kind') or '').strip().lower() in {'caption', 'figure'}
                            or 'caption' in str(target.get('element_type') or '').strip().lower()
                            or self._try_parse_int_id(target.get('sequence')) in page_table_caption_sequences
                        )
                    ]
                    if caption_target_pool:
                        target_pool = caption_target_pool

                best_target = None
                best_score = None
                current_score = self._score_invalid_duplicate_target_candidate(
                    rows,
                    current_target,
                    prev_seq,
                    next_seq,
                    page_claimed_ids,
                    page_table_caption_sequences=page_table_caption_sequences,
                    page_element_row_counts=page_element_row_counts,
                )

                for target in target_pool:
                    candidate_score = self._score_invalid_duplicate_target_candidate(
                        rows,
                        target,
                        prev_seq,
                        next_seq,
                        page_claimed_ids,
                        page_table_caption_sequences=page_table_caption_sequences,
                        page_element_row_counts=page_element_row_counts,
                    )
                    if candidate_score is None:
                        continue
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_target = target

                if not best_target or best_score is None:
                    continue
                required_delta = 0.25
                if cluster_is_code:
                    current_seq = self._try_parse_int_id(current_target.get('sequence'))
                    current_in_window = self._sequence_within_assignment_window(current_seq, prev_seq, next_seq, slack=2)
                    avg_alignment_confidence = sum(
                        float((row or {}).get('alignment_confidence') or 0.0)
                        for row in rows
                    ) / max(1, len(rows))
                    if not current_in_window:
                        required_delta = 0.06 if avg_alignment_confidence < 0.60 else 0.10
                if current_score is not None and best_score <= current_score + required_delta:
                    continue

                changed_here = 0
                for row in rows:
                    if self._assign_result_to_target(row, best_target, 'invalid_duplicate_local_reassign'):
                        changed_here += 1
                if changed_here > 0:
                    reassigned_rows += changed_here
                    repaired_elements += 1
                    affected_pages.add(page_num)

        return {
            'reassigned_rows': reassigned_rows,
            'repaired_elements': repaired_elements,
            'affected_pages': len(affected_pages),
        }

    def _resolve_document_visual_claims(self, page_vis_payload):
        if not page_vis_payload:
            return {
                'cleared_claims': 0,
                'affected_pages': 0,
                'same_page_cleared': 0,
                'far_gap_cleared': 0,
                'cross_page_rescue_cleared': 0,
            }

        single_page_repair_reasons = set()
        if self._is_env_enabled_default_true("ALIGNMENT_ENABLE_RESCUE_DUPLICATE_PRUNE"):
            single_page_repair_reasons = {
                'caption_suffix_inherit',
                'image_placeholder_neighbor_inherit',
                'caption_fragment_inherit',
                'table_lead_inherit',
            }

        claims_by_element = {}
        for page_num, payload in (page_vis_payload or {}).items():
            parsed_page_num = self._try_parse_int_id(page_num)
            if parsed_page_num is None:
                continue
            for result in payload.get('fused_results') or []:
                elem_id = self._try_parse_int_id((result or {}).get('element_id'))
                if elem_id is None:
                    continue
                visual_label = self._get_visual_label(result)
                if visual_label in ('page_header', 'page_footer'):
                    continue
                claims_by_element.setdefault(elem_id, []).append({
                    'page': parsed_page_num,
                    'result': result,
                    'score': self._visual_result_claim_score(result)
                })

        cleared_claims = 0
        same_page_cleared = 0
        far_gap_cleared = 0
        cross_page_rescue_cleared = 0
        affected_pages = set()

        for elem_id, claims in claims_by_element.items():
            claims_by_page = {}
            for claim in claims:
                claims_by_page.setdefault(claim['page'], []).append(claim)

            for page, page_claims in sorted(claims_by_page.items()):
                allowed_page_claims = self._select_valid_same_page_table_claims(page_claims)
                if not allowed_page_claims:
                    allowed_page_claims = self._select_valid_same_page_chart_caption_claims(page_claims)
                if allowed_page_claims:
                    allowed_ids = {id(claim) for claim in allowed_page_claims}
                    for claim in page_claims:
                        if id(claim) in allowed_ids:
                            continue
                        if self._clear_visual_result_claim(
                            claim.get('result'),
                            'same_page_duplicate',
                            allowed_page_claims[0],
                            drop_from_output=True
                        ):
                            cleared_claims += 1
                            same_page_cleared += 1
                            affected_pages.add(page)
                    continue
                winner_claim = max(page_claims, key=lambda claim: claim['score'])
                for claim in page_claims:
                    if claim is winner_claim:
                        continue
                    if self._clear_visual_result_claim(
                        claim.get('result'),
                        'same_page_duplicate',
                        winner_claim,
                        drop_from_output=True
                    ):
                        cleared_claims += 1
                        same_page_cleared += 1
                        affected_pages.add(page)

            active_claims = [
                claim for claim in claims
                if (claim.get('result') or {}).get('element_id') is not None
            ]
            if len(active_claims) <= 1:
                continue

            repair_claims = [
                claim for claim in active_claims
                if (claim.get('result') or {}).get('repair_reason') in single_page_repair_reasons
            ]
            if repair_claims:
                non_repair_claims = [
                    claim for claim in active_claims
                    if claim not in repair_claims
                ]
                if non_repair_claims:
                    winner_claim = max(non_repair_claims, key=lambda claim: claim['score'])
                else:
                    winner_claim = max(repair_claims, key=lambda claim: claim['score'])

                for claim in repair_claims:
                    if claim is winner_claim:
                        continue
                    page = claim.get('page')
                    result = claim.get('result') or {}
                    if self._clear_visual_result_claim(result, 'cross_page_rescue_duplicate', winner_claim):
                        result['_drop_from_output'] = True
                        cleared_claims += 1
                    cross_page_rescue_cleared += 1
                    if page is not None:
                        affected_pages.add(page)

        return {
            'cleared_claims': cleared_claims,
            'affected_pages': len(affected_pages),
            'same_page_cleared': same_page_cleared,
            'far_gap_cleared': far_gap_cleared,
            'cross_page_rescue_cleared': cross_page_rescue_cleared,
        }
