import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingRetrySupportMixin:


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
