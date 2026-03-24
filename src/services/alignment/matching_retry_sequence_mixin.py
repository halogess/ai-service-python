import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingRetrySequenceMixin:


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
