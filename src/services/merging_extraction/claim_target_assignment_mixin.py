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


class MergingExtractionClaimTargetAssignmentMixin:


    def _assign_null_results_to_unclaimed_targets(self, db, canonical_ref_tipe, ref_id, page_vis_payload):
        if not db or ref_id is None or not page_vis_payload:
            return {'assigned_body_targets': 0, 'assigned_note_targets': 0}

        body_targets = self._load_body_elements_for_ref(db, canonical_ref_tipe, ref_id)
        note_targets = self._load_note_targets_for_ref(db, canonical_ref_tipe, ref_id)
        body_by_id = {
            target['element_id']: target
            for target in body_targets
            if target.get('target_kind') == 'body' and target.get('is_eligible_target')
        }

        claimed_body_ids = set()
        claimed_note_ids = set()
        for payload in (page_vis_payload or {}).values():
            for row in (payload or {}).get('fused_results') or []:
                element_id = self._try_parse_int_id((row or {}).get('element_id'))
                if element_id is None or (row or {}).get('_drop_from_output'):
                    continue
                if element_id in body_by_id:
                    claimed_body_ids.add(element_id)
                else:
                    claimed_note_ids.add(element_id)

        eligible_body_targets = [
            target for target in body_targets
            if target.get('target_kind') == 'body'
            and target.get('is_eligible_target')
            and not target.get('is_non_visual_proxy')
        ]
        eligible_note_targets = [
            target for target in note_targets
            if target.get('is_eligible_target')
        ]

        assigned_body_targets = 0
        assigned_note_targets = 0

        for page_num, payload in sorted((page_vis_payload or {}).items()):
            fused_results = list((payload or {}).get('fused_results') or [])
            if not fused_results:
                continue

            ordered_anchors = self._iter_page_body_sequence_anchors(fused_results, body_by_id)

            for result in fused_results:
                if not self._result_supports_target_assignment(result):
                    continue

                prev_seq, next_seq = self._find_sequence_anchor_window(result, ordered_anchors)
                result_is_table = self._result_prefers_table_target(result)
                result_is_code = self._result_is_code_like(result)
                result_is_caption = self._result_prefers_caption_target(result)

                best_target = None
                best_score = None
                best_kind = None

                if self._result_is_note_like(result):
                    for target in eligible_note_targets:
                        element_id = self._try_parse_int_id(target.get('element_id'))
                        if element_id is None or element_id in claimed_note_ids:
                            continue
                        score = self._score_note_target_candidate(result, target)
                        if score is None:
                            continue
                        if best_score is None or score > best_score:
                            best_score = score
                            best_target = target
                            best_kind = 'note'
                else:
                    candidates = []
                    for target in eligible_body_targets:
                        element_id = self._try_parse_int_id(target.get('element_id'))
                        if element_id is None:
                            continue
                        if element_id in claimed_body_ids and not (result_is_table and self._is_table_target(target)):
                            continue
                        candidates.append(target)

                    if result_is_table:
                        table_candidates = [target for target in candidates if self._is_table_target(target)]
                        if table_candidates:
                            candidates = table_candidates
                    elif result_is_caption:
                        caption_candidates = [
                            target for target in candidates
                            if str(target.get('block_kind') or '').strip().lower() in {'caption', 'figure'}
                            or 'caption' in str(target.get('element_type') or '').strip().lower()
                        ]
                        if caption_candidates:
                            candidates = caption_candidates
                    elif result_is_code:
                        preferred = [
                            target for target in candidates
                            if self._is_code_like_target(target)
                        ]
                        contextual = [
                            target for target in candidates
                            if (
                                target.get('block_key') and result.get('block_key') and
                                str(target.get('block_key')).strip().lower() == str(result.get('block_key')).strip().lower()
                            ) or (
                                self._try_parse_int_id(target.get('block_order')) is not None and
                                self._try_parse_int_id(target.get('block_order')) == self._try_parse_int_id(result.get('block_order'))
                            )
                        ]
                        deduped = []
                        seen_ids = set()
                        for target in preferred + contextual + candidates:
                            element_id = self._try_parse_int_id(target.get('element_id'))
                            if element_id is None or element_id in seen_ids:
                                continue
                            seen_ids.add(element_id)
                            deduped.append(target)
                        candidates = deduped

                    for target in candidates:
                        score = self._score_body_target_candidate(result, target, prev_seq, next_seq)
                        if score is None:
                            continue
                        if best_score is None or score > best_score:
                            best_score = score
                            best_target = target
                            best_kind = 'body'

                if not best_target:
                    continue

                threshold = 0.90
                if best_kind == 'note':
                    threshold = 1.00
                elif result_is_table:
                    threshold = 0.52
                elif result_is_code:
                    threshold = 0.72
                elif result_is_caption or self._is_picture_result(result):
                    threshold = 0.70

                if best_score is None or best_score < threshold:
                    continue

                if not self._assign_result_to_target(result, best_target, 'document_unclaimed_target'):
                    continue

                if best_kind == 'note':
                    claimed_note_ids.add(best_target['element_id'])
                    assigned_note_targets += 1
                else:
                    if not (result_is_table and self._is_table_target(best_target)):
                        claimed_body_ids.add(best_target['element_id'])
                    assigned_body_targets += 1
                    sequence = self._try_parse_int_id(best_target.get('sequence'))
                    center_y = self._get_bbox_center_y(result.get('bbox'))
                    if sequence is not None and center_y is not None:
                        ordered_anchors.append({
                            'sequence': sequence,
                            'center_y': center_y,
                            'x0': float(result.get('bbox')[0]) if result.get('bbox') else 0.0,
                        })
                        ordered_anchors.sort(key=lambda item: (item['center_y'], item['x0'], item['sequence']))

        return {
            'assigned_body_targets': assigned_body_targets,
            'assigned_note_targets': assigned_note_targets,
        }
