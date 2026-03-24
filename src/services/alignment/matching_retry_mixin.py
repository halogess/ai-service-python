import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

class AlignmentMatchingRetryMixin:
    @classmethod
    def _estimate_unit_match_chars(cls, unit):
        if not isinstance(unit, dict):
            return 0
        matched_count = cls._try_parse_int(unit.get('matched_count'))
        if matched_count is not None and matched_count > 0:
            return matched_count
        normalized_text = cls._normalize_pointer_text(
            unit.get('text') or unit.get('text_normalized') or unit.get('normalized_text')
        )
        if normalized_text:
            return len(normalized_text)
        return 1

    @classmethod
    def _collect_alignment_openxml_indices(cls, alignment, include_table_cells=True):
        indices = []
        if not isinstance(alignment, dict):
            return indices

        openxml_idx = cls._try_parse_int(alignment.get('openxml_idx'))
        if openxml_idx is not None:
            indices.append(openxml_idx)

        for value in alignment.get('openxml_indices') or []:
            parsed = cls._try_parse_int(value)
            if parsed is not None:
                indices.append(parsed)

        if include_table_cells and alignment.get('is_table') and alignment.get('cells'):
            for cell in alignment.get('cells') or []:
                parsed = cls._try_parse_int((cell or {}).get('openxml_idx'))
                if parsed is not None:
                    indices.append(parsed)

        return sorted(set(indices))

    @classmethod
    def _compute_alignment_support_metrics(cls, alignment):
        if not isinstance(alignment, dict):
            return {
                'matched_chars': 0,
                'match_ratio': 0.0,
                'unit_count': 0,
            }

        matched_units = []
        if alignment.get('is_table') and alignment.get('cells'):
            for cell in alignment.get('cells') or []:
                matched_units.extend((cell or {}).get('matched_pdf_units') or [])
        else:
            matched_units = list(alignment.get('matched_pdf_units') or [])

        matched_chars = sum(cls._estimate_unit_match_chars(unit) for unit in matched_units)
        element_text = cls._normalize_pointer_text(alignment.get('element_text') or alignment.get('text'))
        if not element_text:
            element_text = ''.join(
                cls._normalize_pointer_text((unit or {}).get('text'))
                for unit in matched_units
                if isinstance(unit, dict)
            )
        norm_len = len(element_text)
        match_ratio = (matched_chars / norm_len) if norm_len > 0 else 0.0
        return {
            'matched_chars': matched_chars,
            'match_ratio': match_ratio,
            'unit_count': len(matched_units),
        }

    @classmethod
    def _compute_alignment_max_openxml_idx(cls, alignments):
        max_idx = None
        for alignment in alignments or []:
            for idx in cls._collect_alignment_openxml_indices(alignment, include_table_cells=True):
                max_idx = idx if max_idx is None else max(max_idx, idx)
        return max_idx

    @classmethod
    def _extract_cross_page_skip_metrics(cls, traversal_log, total_skip_count=None):
        skip_entries = []
        early_cross_page_skip_count = 0
        for step_idx, entry in enumerate(traversal_log or []):
            if entry.get('action') != 'SKIP':
                continue
            if not str(entry.get('reason') or '').startswith('cross_page_backward'):
                continue
            openxml_idx = cls._try_parse_int(entry.get('openxml_unit'))
            if openxml_idx is None or openxml_idx < 0:
                continue
            skip_entries.append(openxml_idx)
            if step_idx < 128:
                early_cross_page_skip_count += 1

        if not skip_entries:
            return {
                'cross_page_backward_skip_count': 0,
                'cross_page_backward_skip_ratio': 0.0,
                'first_cross_page_skip_openxml_idx': None,
                'min_cross_page_skip_openxml_idx': None,
                'median_cross_page_skip_openxml_idx': None,
                'early_cross_page_skip_count': 0,
                'early_cross_page_skip_ratio': 0.0,
            }

        ordered = sorted(skip_entries)
        midpoint = len(ordered) // 2
        if len(ordered) % 2 == 1:
            median_skip = ordered[midpoint]
        else:
            median_skip = int(round((ordered[midpoint - 1] + ordered[midpoint]) / 2))

        total_skips = len(skip_entries)
        ratio_denominator = total_skip_count if total_skip_count is not None else total_skips
        return {
            'cross_page_backward_skip_count': total_skips,
            'cross_page_backward_skip_ratio': (
                total_skips / ratio_denominator if ratio_denominator else 0.0
            ),
            'first_cross_page_skip_openxml_idx': skip_entries[0],
            'min_cross_page_skip_openxml_idx': ordered[0],
            'median_cross_page_skip_openxml_idx': median_skip,
            'early_cross_page_skip_count': early_cross_page_skip_count,
            'early_cross_page_skip_ratio': (
                early_cross_page_skip_count / total_skips if total_skips > 0 else 0.0
            ),
        }

    def _compute_stable_pass1_pointer(self, alignments, min_openxml_idx):
        raw_max_openxml_idx = self._compute_alignment_max_openxml_idx(alignments)
        cluster_gap = self._read_positive_int_env('ALIGNMENT_STABLE_POINTER_CLUSTER_GAP', 40)
        min_match_chars = self._read_positive_int_env('ALIGNMENT_STABLE_POINTER_MIN_MATCH_CHARS', 12)
        min_match_ratio = self._read_float_env(
            'ALIGNMENT_STABLE_POINTER_MIN_MATCH_RATIO',
            0.10,
            min_value=0.0,
            max_value=1.0
        )

        index_support = {}
        for alignment in alignments or []:
            if not self._is_pointer_safe_alignment(alignment):
                continue

            support = self._compute_alignment_support_metrics(alignment)
            if (
                support['matched_chars'] < min_match_chars and
                support['match_ratio'] < min_match_ratio
            ):
                continue

            for idx in self._collect_alignment_openxml_indices(alignment, include_table_cells=False):
                existing = index_support.get(idx)
                candidate_support = (
                    support['matched_chars'],
                    support['unit_count'],
                    support['match_ratio'],
                )
                if existing is None or candidate_support > existing['score']:
                    index_support[idx] = {
                        'score': candidate_support,
                        'matched_chars': support['matched_chars'],
                    }

        if not index_support:
            frozen_pointer = max(0, int(min_openxml_idx or 0))
            return {
                'source': 'frozen',
                'cluster_min': None,
                'cluster_max': frozen_pointer,
                'cluster_size': 0,
                'cluster_total_matched_chars': 0,
                'raw_max_openxml_idx': raw_max_openxml_idx,
                'max_openxml_idx': frozen_pointer,
            }

        sorted_indices = sorted(index_support)
        clusters = []
        current_cluster = [sorted_indices[0]]
        for idx in sorted_indices[1:]:
            if (idx - current_cluster[-1]) > cluster_gap:
                clusters.append(current_cluster)
                current_cluster = [idx]
            else:
                current_cluster.append(idx)
        if current_cluster:
            clusters.append(current_cluster)

        def cluster_score(cluster_items):
            cluster_min = min(cluster_items)
            total_chars = sum(index_support[idx]['matched_chars'] for idx in cluster_items)
            distance = abs(cluster_min - int(min_openxml_idx or 0))
            return (
                len(cluster_items),
                total_chars,
                -distance,
            )

        winner_cluster = max(clusters, key=cluster_score)
        cluster_min = min(winner_cluster)
        cluster_max = max(winner_cluster)
        return {
            'source': 'stable_cluster',
            'cluster_min': cluster_min,
            'cluster_max': cluster_max,
            'cluster_size': len(winner_cluster),
            'cluster_total_matched_chars': sum(index_support[idx]['matched_chars'] for idx in winner_cluster),
            'raw_max_openxml_idx': raw_max_openxml_idx,
            'max_openxml_idx': max(cluster_max, int(min_openxml_idx or 0)),
        }

    def _build_pass1_retry_candidates(
        self,
        base_min_openxml_idx,
        initial_debug_info,
        max_retries
    ):
        if max_retries <= 0:
            return []

        median_skip_idx = self._try_parse_int((initial_debug_info or {}).get('median_cross_page_skip_openxml_idx'))
        min_skip_idx = self._try_parse_int((initial_debug_info or {}).get('min_cross_page_skip_openxml_idx'))
        retry_values = []

        for raw_candidate in (
            median_skip_idx - 8 if median_skip_idx is not None else None,
            min_skip_idx - 8 if min_skip_idx is not None else None,
            base_min_openxml_idx - 120,
            base_min_openxml_idx - 240,
            0,
        ):
            candidate = self._try_parse_int(raw_candidate)
            if candidate is None:
                continue
            candidate = max(0, candidate)
            if candidate >= base_min_openxml_idx:
                continue
            if candidate in retry_values:
                continue
            retry_values.append(candidate)
            if len(retry_values) >= max_retries:
                break

        return retry_values

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

    @staticmethod
    def _normalize_sequence_range(seq_range):
        if not seq_range or len(seq_range) != 2:
            return None
        try:
            seq_min = int(seq_range[0])
            seq_max = int(seq_range[1])
        except (TypeError, ValueError):
            return None
        if seq_min > seq_max:
            seq_min, seq_max = seq_max, seq_min
        return (seq_min, seq_max)

    def _expand_sequence_range(self, seq_range, buffer_size):
        normalized = self._normalize_sequence_range(seq_range)
        if normalized is None:
            return None
        seq_min, seq_max = normalized
        return (seq_min - int(buffer_size or 0), seq_max + int(buffer_size or 0))

    def _collect_openxml_indices_for_sequence_range(self, openxml_units, seq_range):
        normalized = self._normalize_sequence_range(seq_range)
        if normalized is None:
            return []
        seq_min, seq_max = normalized
        return [
            idx for idx, unit in enumerate(openxml_units or [])
            if unit.get('elem_seq') is not None and seq_min <= unit.get('elem_seq') <= seq_max
        ]

    def _select_sequence_local_openxml_indices(
        self,
        pdf_units,
        openxml_units,
        min_openxml_idx=0,
        page_sequence_range=None
    ):
        total_openxml_units = len(openxml_units or [])
        if total_openxml_units <= 0:
            return {
                'indices': [],
                'source': 'empty_openxml',
                'anchor_hits': [],
                'anchor_hit_count': 0,
                'anchor_count': 0,
                'preferred_seq_range': None,
                'selected_seq_range': None,
                'search_floor': 0,
            }

        preferred_range = self._normalize_sequence_range(page_sequence_range)
        preferred_span = 0
        if preferred_range is not None:
            preferred_span = max(1, preferred_range[1] - preferred_range[0])

        seq_buffer = min(
            self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MAX_SEQ_BUFFER', 160),
            max(
                self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MIN_SEQ_BUFFER', 18),
                preferred_span // 2 if preferred_span > 0 else max(18, len(pdf_units or []) * 3)
            )
        )
        expanded_preferred_range = self._expand_sequence_range(preferred_range, seq_buffer)
        preferred_indices = self._collect_openxml_indices_for_sequence_range(openxml_units, expanded_preferred_range)
        preferred_index_set = set(preferred_indices)

        search_backtrack = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_BACKTRACK_BUFFER', 240)
        search_floor = max(0, int(min_openxml_idx or 0) - search_backtrack)
        pointer_bonus_window = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_POINTER_BONUS_WINDOW', 160)
        preferred_bonus = self._read_float_env(
            'ALIGNMENT_PAGE_LOCAL_PREFERRED_SEQ_BONUS',
            0.12,
            min_value=0.0,
            max_value=1.0
        )
        outside_penalty = self._read_float_env(
            'ALIGNMENT_PAGE_LOCAL_OUTSIDE_SEQ_PENALTY',
            0.04,
            min_value=0.0,
            max_value=1.0
        )
        pointer_bonus = self._read_float_env(
            'ALIGNMENT_PAGE_LOCAL_POINTER_BONUS',
            0.05,
            min_value=0.0,
            max_value=1.0
        )
        anchor_score_threshold = self._read_float_env(
            'ALIGNMENT_PAGE_LOCAL_MIN_ANCHOR_SCORE',
            0.40,
            min_value=0.0,
            max_value=2.0
        )

        candidate_openxml_units = []
        for idx, unit in enumerate(openxml_units):
            if idx < search_floor and idx not in preferred_index_set:
                continue
            if not self._is_openxml_anchor_candidate(unit):
                continue
            candidate_openxml_units.append((idx, unit))

        anchors = self._collect_pdf_anchor_blocks(pdf_units)
        anchor_hits = []
        openxml_block_units = min(
            4,
            max(
                1,
                self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_OPENXML_BLOCK_UNITS', 3)
            )
        )
        for anchor in anchors:
            best_hit = None
            for openxml_idx, unit in candidate_openxml_units:
                block_unit_limit = max(1, min(openxml_block_units, int(anchor.get('block_size') or 1) + 1))
                openxml_block = self._compose_openxml_anchor_block(
                    openxml_units,
                    openxml_idx,
                    max_units=block_unit_limit
                )
                base_score = self._score_anchor_candidate(
                    anchor,
                    unit,
                    openxml_block,
                )
                if base_score <= 0:
                    continue

                seq = self._try_parse_int(unit.get('elem_seq'))
                in_preferred_range = (
                    expanded_preferred_range is not None and
                    seq is not None and
                    expanded_preferred_range[0] <= seq <= expanded_preferred_range[1]
                )
                score = base_score
                if expanded_preferred_range is not None:
                    score += preferred_bonus if in_preferred_range else (-outside_penalty)
                if min_openxml_idx is not None and openxml_idx >= int(min_openxml_idx or 0):
                    if openxml_idx <= int(min_openxml_idx or 0) + pointer_bonus_window:
                        score += pointer_bonus

                if score < anchor_score_threshold:
                    continue

                hit = {
                    'pdf_local_idx': anchor.get('local_idx'),
                    'pdf_item_idx': anchor.get('item_idx'),
                    'openxml_idx': openxml_idx,
                    'openxml_end_idx': openxml_block.get('end_idx'),
                    'elem_seq': seq,
                    'score': score,
                    'in_preferred_range': in_preferred_range,
                    'pdf_text': (anchor.get('text') or '')[:80],
                    'openxml_text': (openxml_block.get('text') or unit.get('text') or '')[:80],
                    'openxml_block_size': block_unit_limit,
                    'block_kind': openxml_block.get('block_kind') or unit.get('block_kind'),
                    'block_key': openxml_block.get('block_key') or unit.get('block_key'),
                    'content_role': openxml_block.get('content_role') or unit.get('content_role'),
                    'block_order': openxml_block.get('block_order') or unit.get('block_order'),
                    'exact_block_key_match': bool(
                        self._normalize_block_key(anchor.get('block_key')) and
                        self._normalize_block_key(anchor.get('block_key')) ==
                        self._normalize_block_key(openxml_block.get('block_key') or unit.get('block_key'))
                    ),
                }
                if best_hit is None or hit['score'] > best_hit['score']:
                    best_hit = hit

            if best_hit is not None:
                anchor_hits.append(best_hit)

        selected_seq_range = None
        selected_source = None
        anchor_cluster_min_idx = None
        anchor_cluster_max_idx = None

        if anchor_hits:
            cluster_gap = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_SEQ_CLUSTER_GAP', 24)
            sortable_hits = sorted(
                anchor_hits,
                key=lambda hit: (
                    hit['elem_seq'] if hit.get('elem_seq') is not None else 10 ** 9,
                    hit['openxml_idx']
                )
            )
            clusters = []
            current_cluster = [sortable_hits[0]]
            for hit in sortable_hits[1:]:
                prev_hit = current_cluster[-1]
                prev_seq = prev_hit.get('elem_seq')
                hit_seq = hit.get('elem_seq')
                same_cluster = False
                if prev_seq is not None and hit_seq is not None:
                    same_cluster = abs(hit_seq - prev_seq) <= cluster_gap
                else:
                    same_cluster = abs(hit.get('openxml_idx', 0) - prev_hit.get('openxml_idx', 0)) <= cluster_gap
                if same_cluster:
                    current_cluster.append(hit)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [hit]
            if current_cluster:
                clusters.append(current_cluster)

            def cluster_score(cluster):
                preferred_hits = sum(1 for hit in cluster if hit.get('in_preferred_range'))
                exact_block_hits = sum(1 for hit in cluster if hit.get('block_key'))
                min_distance = min(
                    abs((hit.get('openxml_idx') or 0) - int(min_openxml_idx or 0))
                    for hit in cluster
                ) if cluster else 10 ** 9
                return (
                    round(sum(hit.get('score', 0.0) for hit in cluster), 6),
                    exact_block_hits,
                    len(cluster),
                    preferred_hits,
                    -min_distance
                )

            winner_cluster = max(clusters, key=cluster_score)
            seq_hits = [hit.get('elem_seq') for hit in winner_cluster if hit.get('elem_seq') is not None]
            if seq_hits:
                cluster_buffer = min(
                    self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MAX_CLUSTER_SEQ_BUFFER', 96),
                    max(
                        self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_MIN_CLUSTER_SEQ_BUFFER', 12),
                        preferred_span if preferred_span > 0 else len(anchors) * 6
                    )
                )
                selected_seq_range = self._expand_sequence_range((min(seq_hits), max(seq_hits)), cluster_buffer)
                selected_source = 'sequence_anchor_cluster'
            anchor_cluster_min_idx = min(hit.get('openxml_idx') or 0 for hit in winner_cluster)
            anchor_cluster_max_idx = max(
                hit.get('openxml_end_idx')
                if hit.get('openxml_end_idx') is not None
                else (hit.get('openxml_idx') or 0)
                for hit in winner_cluster
            )

        candidate_indices = []
        if selected_seq_range is not None:
            candidate_indices = self._collect_openxml_indices_for_sequence_range(openxml_units, selected_seq_range)
        if not candidate_indices and preferred_indices:
            candidate_indices = preferred_indices
            selected_seq_range = expanded_preferred_range
            selected_source = 'page_sequence_range'
        if not candidate_indices:
            if min_openxml_idx and min_openxml_idx > 0:
                fallback_forward = self._read_positive_int_env('ALIGNMENT_PAGE_LOCAL_FALLBACK_FORWARD', 480)
                start_idx = search_floor
                end_idx = min(total_openxml_units - 1, max(int(min_openxml_idx or 0), start_idx) + fallback_forward)
                candidate_indices = list(range(start_idx, end_idx + 1))
                selected_source = 'pointer_local_fallback'
            else:
                candidate_indices = list(range(total_openxml_units))
                selected_source = 'global_fallback'

        candidate_indices = sorted(set(candidate_indices))
        return {
            'indices': candidate_indices,
            'source': selected_source or 'global_fallback',
            'anchor_hits': anchor_hits[:8],
            'anchor_hit_count': len(anchor_hits),
            'anchor_count': len(anchors),
            'preferred_seq_range': expanded_preferred_range,
            'selected_seq_range': selected_seq_range,
            'search_floor': search_floor,
            'anchor_cluster_min_idx': anchor_cluster_min_idx,
            'anchor_cluster_max_idx': anchor_cluster_max_idx,
        }

    def _build_retry_page_sequence_range(
        self,
        initial_candidate,
        page_sequence_range,
        openxml_units,
        pdf_units
    ):
        preferred_range = self._normalize_sequence_range(page_sequence_range)
        observed_skip_idx = self._try_parse_int((initial_candidate or {}).get('median_cross_page_skip_openxml_idx'))
        if observed_skip_idx is None:
            observed_skip_idx = self._try_parse_int((initial_candidate or {}).get('min_cross_page_skip_openxml_idx'))
        if observed_skip_idx is None:
            observed_skip_idx = self._try_parse_int((initial_candidate or {}).get('first_cross_page_skip_openxml_idx'))

        skip_seq = None
        if observed_skip_idx is not None and 0 <= observed_skip_idx < len(openxml_units or []):
            skip_seq = self._try_parse_int((openxml_units[observed_skip_idx] or {}).get('elem_seq'))

        if skip_seq is None:
            return preferred_range

        preferred_span = 0
        if preferred_range is not None:
            preferred_span = max(1, preferred_range[1] - preferred_range[0])
        retry_span = max(
            self._read_positive_int_env('ALIGNMENT_RETRY_SEQUENCE_MIN_SPAN', 24),
            preferred_span,
            len(pdf_units or []) * 4
        )
        retry_buffer = min(
            self._read_positive_int_env('ALIGNMENT_RETRY_SEQUENCE_MAX_BUFFER', 96),
            max(
                self._read_positive_int_env('ALIGNMENT_RETRY_SEQUENCE_MIN_BUFFER', 12),
                retry_span // 3
            )
        )
        half_span = max(12, retry_span // 2)
        return (
            skip_seq - half_span - retry_buffer,
            skip_seq + half_span + retry_buffer,
        )

    def _compute_candidate_band_metrics(self, alignments, openxml_units, seq_range=None, idx_range=None):
        normalized_seq_range = self._normalize_sequence_range(seq_range)
        normalized_idx_range = self._normalize_sequence_range(idx_range)

        total_alignments = 0
        in_band_alignments = 0
        total_chars = 0
        in_band_chars = 0

        for alignment in alignments or []:
            indices = self._collect_alignment_openxml_indices(alignment, include_table_cells=True)
            if not indices:
                continue

            support = self._compute_alignment_support_metrics(alignment)
            matched_chars = int(support.get('matched_chars') or 0)
            total_alignments += 1
            total_chars += matched_chars

            in_band = False
            if normalized_seq_range is not None:
                seq_min, seq_max = normalized_seq_range
                for idx in indices:
                    if idx < 0 or idx >= len(openxml_units or []):
                        continue
                    elem_seq = self._try_parse_int((openxml_units[idx] or {}).get('elem_seq'))
                    if elem_seq is not None and seq_min <= elem_seq <= seq_max:
                        in_band = True
                        break

            if not in_band and normalized_idx_range is not None:
                idx_min, idx_max = normalized_idx_range
                in_band = any(idx_min <= idx <= idx_max for idx in indices)

            if in_band:
                in_band_alignments += 1
                in_band_chars += matched_chars

        return {
            'alignment_ratio': (in_band_alignments / total_alignments) if total_alignments else 0.0,
            'char_ratio': (in_band_chars / total_chars) if total_chars else 0.0,
            'alignment_count': in_band_alignments,
            'total_alignment_count': total_alignments,
        }

    @staticmethod
    def _score_pass1_candidate(candidate):
        retry_lock_candidate = bool(candidate.get('retry_lock_candidate'))
        backward_retry_penalty = bool(candidate.get('backward_retry_penalty'))
        skip_anchor_penalty = bool(candidate.get('skip_anchor_penalty'))
        local_anchor_source = 1 if candidate.get('candidate_openxml_source') == 'sequence_anchor_cluster' else 0
        anchor_hit_count = int(candidate.get('candidate_anchor_hit_count') or 0)
        band_alignment_ratio = float(candidate.get('candidate_band_alignment_ratio') or 0.0)
        band_char_ratio = float(candidate.get('candidate_band_char_ratio') or 0.0)
        cpb_ratio = float(candidate.get('cross_page_backward_skip_ratio') or 0.0)
        coverage = float(candidate.get('match_coverage') or 0.0)
        openxml_diversity = float(candidate.get('openxml_diversity') or 0.0)
        matched_pdf_units = int(candidate.get('matched_pdf_units') or 0)
        return (
            0 if retry_lock_candidate else 1,
            0 if backward_retry_penalty else 1,
            0 if skip_anchor_penalty else 1,
            -cpb_ratio,
            coverage,
            band_alignment_ratio,
            band_char_ratio,
            openxml_diversity,
            matched_pdf_units,
            local_anchor_source,
            anchor_hit_count,
        )
