import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingRetryAnchorMixin:


    @staticmethod
    def _extract_anchor_tokens(text):
        if not text:
            return []
        return [
            token for token in re.findall(r'[a-z0-9]+', str(text).lower())
            if len(token) >= 3
        ]

    def _is_pdf_anchor_candidate(self, unit):
        if not isinstance(unit, dict):
            return False
        if unit.get('is_cell'):
            return False
        if unit.get('item_type') in {'table', 'hline_table', 'shape', 'image'}:
            return False
        text = self._normalize_pointer_text(unit.get('text_normalized') or unit.get('text'))
        min_len = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MIN_ANCHOR_TEXT_LEN', 12)
        return len(text) >= min_len

    def _is_openxml_anchor_candidate(self, unit):
        if not isinstance(unit, dict):
            return False
        if unit.get('is_cell'):
            return False
        if unit.get('elem_type') in {'table', 'grid_table'}:
            return False
        if unit.get('is_image_part'):
            return False
        text = self._normalize_pointer_text(unit.get('text_normalized') or unit.get('text'))
        min_len = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MIN_ANCHOR_TEXT_LEN', 12)
        if len(text) < min_len:
            return False
        return text != '[img]'

    def _collect_pdf_anchor_blocks(self, pdf_units):
        if not pdf_units:
            return []

        max_anchors = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MAX_ANCHORS', 8)
        early_anchors = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_EARLY_ANCHORS', 3)
        candidates = []
        for local_idx, unit in enumerate(pdf_units):
            if not self._is_pdf_anchor_candidate(unit):
                continue
            normalized_text = self._normalize_pointer_text(unit.get('text_normalized') or unit.get('text'))
            candidates.append({
                'local_idx': local_idx,
                'item_idx': unit.get('item_idx'),
                'item_type': unit.get('item_type'),
                'text': unit.get('text') or '',
                'text_normalized': normalized_text,
                'tokens': self._extract_anchor_tokens(normalized_text),
                'bbox': unit.get('bbox'),
                'text_len': len(normalized_text),
                'block_kind': unit.get('block_kind'),
                'block_key': unit.get('block_key'),
                'content_role': unit.get('content_role'),
                'block_order': unit.get('block_order'),
            })

        if not candidates:
            return []

        selected = []
        seen_local_idx = set()

        for candidate in candidates[:early_anchors]:
            selected.append(candidate)
            seen_local_idx.add(candidate['local_idx'])
            if len(selected) >= max_anchors:
                return selected

        for candidate in sorted(candidates, key=lambda item: (-item['text_len'], item['local_idx'])):
            if candidate['local_idx'] in seen_local_idx:
                continue
            selected.append(candidate)
            seen_local_idx.add(candidate['local_idx'])
            if len(selected) >= max_anchors:
                break

        block_max_units = min(
            3,
            self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MAX_BLOCK_UNITS', 2)
        )
        block_candidates = []
        block_seen = set()
        for candidate in selected:
            base_local_idx = candidate['local_idx']
            for block_size in range(1, block_max_units + 1):
                block_units = []
                for offset in range(block_size):
                    next_local_idx = base_local_idx + offset
                    if next_local_idx >= len(pdf_units):
                        break
                    next_unit = pdf_units[next_local_idx]
                    if not self._is_pdf_anchor_candidate(next_unit):
                        break
                    block_units.append(next_unit)

                if len(block_units) != block_size:
                    continue

                key = (base_local_idx, block_size)
                if key in block_seen:
                    continue
                block_seen.add(key)

                block_text_parts = [str(unit.get('text') or '').strip() for unit in block_units if unit.get('text')]
                block_text = ' '.join(part for part in block_text_parts if part).strip()
                block_norm = self._normalize_pointer_text(block_text)
                if not block_norm:
                    continue

                block_candidates.append({
                    'local_idx': base_local_idx,
                    'item_idx': block_units[0].get('item_idx'),
                    'item_type': block_units[0].get('item_type'),
                    'text': block_text,
                    'text_normalized': block_norm,
                    'tokens': self._extract_anchor_tokens(block_norm),
                    'bbox': candidate.get('bbox'),
                    'text_len': len(block_norm),
                    'block_size': block_size,
                    'block_kind': block_units[0].get('block_kind'),
                    'block_key': block_units[0].get('block_key'),
                    'content_role': block_units[0].get('content_role'),
                    'block_order': block_units[0].get('block_order'),
                })

        block_candidates.sort(key=lambda item: (-item['block_size'], -item['text_len'], item['local_idx']))
        return block_candidates[:max_anchors]

    def _compose_openxml_anchor_block(self, openxml_units, start_idx, max_units=2, max_chars=400):
        if not openxml_units or start_idx is None or start_idx < 0 or start_idx >= len(openxml_units):
            return {'text': '', 'text_normalized': '', 'end_idx': start_idx}

        text_parts = []
        end_idx = start_idx
        added_units = 0
        block_kind = None
        block_key = None
        content_role = None
        block_order = None
        for idx in range(start_idx, len(openxml_units)):
            unit = openxml_units[idx]
            if not self._is_openxml_anchor_candidate(unit):
                if added_units > 0:
                    break
                continue
            text = str(unit.get('text') or '').strip()
            if text:
                text_parts.append(text)
            added_units += 1
            end_idx = idx
            if block_kind is None:
                block_kind = unit.get('block_kind')
            if block_key is None:
                block_key = unit.get('block_key')
            if content_role is None:
                content_role = unit.get('content_role')
            if block_order is None:
                block_order = unit.get('block_order')
            if added_units >= max_units:
                break
            if sum(len(part) for part in text_parts) >= max_chars:
                break

        block_text = ' '.join(part for part in text_parts if part).strip()
        block_norm = self._normalize_pointer_text(block_text)
        return {
            'text': block_text,
            'text_normalized': block_norm,
            'end_idx': end_idx,
            'block_kind': block_kind,
            'block_key': block_key,
            'content_role': content_role,
            'block_order': block_order,
        }

    def _score_anchor_similarity(self, pdf_text, openxml_text):
        pdf_norm = self._normalize_pointer_text(pdf_text)
        openxml_norm = self._normalize_pointer_text(openxml_text)
        if not pdf_norm or not openxml_norm:
            return 0.0

        pdf_sample = pdf_norm[:320]
        openxml_sample = openxml_norm[:320]
        char_ratio = difflib.SequenceMatcher(None, pdf_sample, openxml_sample, autojunk=False).ratio()

        pdf_tokens = set(self._extract_anchor_tokens(pdf_sample))
        openxml_tokens = set(self._extract_anchor_tokens(openxml_sample))
        token_overlap = 0.0
        if pdf_tokens and openxml_tokens:
            token_overlap = len(pdf_tokens & openxml_tokens) / max(1, min(len(pdf_tokens), len(openxml_tokens)))

        containment_bonus = 0.0
        if pdf_sample in openxml_sample or openxml_sample in pdf_sample:
            containment_bonus = 0.10

        return (char_ratio * 0.65) + (token_overlap * 0.35) + containment_bonus

    def _score_anchor_candidate(self, anchor, openxml_unit, openxml_block):
        base_score = self._score_anchor_similarity(
            (anchor or {}).get('text_normalized') or (anchor or {}).get('text'),
            (openxml_block or {}).get('text_normalized') or (openxml_block or {}).get('text'),
        )
        if base_score <= 0:
            return 0.0

        score = base_score
        anchor_kind = str((anchor or {}).get('block_kind') or '').strip().lower()
        openxml_kind = str((openxml_block or {}).get('block_kind') or (openxml_unit or {}).get('block_kind') or '').strip().lower()
        anchor_key = self._normalize_block_key((anchor or {}).get('block_key'))
        openxml_key = self._normalize_block_key((openxml_block or {}).get('block_key') or (openxml_unit or {}).get('block_key'))
        anchor_role = str((anchor or {}).get('content_role') or '').strip().lower()
        openxml_role = str((openxml_block or {}).get('content_role') or (openxml_unit or {}).get('content_role') or '').strip().lower()

        if anchor_kind and openxml_kind:
            if anchor_kind == openxml_kind:
                score += 0.10
            elif {anchor_kind, openxml_kind} <= {'caption', 'figure'}:
                score += 0.04
            elif {anchor_kind, openxml_kind} <= {'caption', 'table'}:
                score += 0.04
            else:
                score -= 0.05

        if anchor_key and openxml_key:
            if anchor_key == openxml_key:
                score += 0.30
            else:
                score -= 0.18

        if anchor_role and openxml_role:
            if anchor_role == openxml_role:
                score += 0.06
            elif 'heading' in anchor_role and 'heading' in openxml_role:
                score += 0.04

        return max(0.0, score)
