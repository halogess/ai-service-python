import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

class AlignmentMatchingExecutionMixin:
    def _perform_two_pass_alignment(
        self,
        pdf_units,
        openxml_units,
        min_openxml_idx,
        trace_context=None,
        page_sequence_range=None
    ):
        trace_pass1 = dict(trace_context or {})
        trace_pass1['phase'] = 'pass1'
        p1_align, p1_un_pdf, _, p1_debug = self._perform_char_alignment(
            pdf_units,
            openxml_units,
            min_openxml_idx,
            trace_context=trace_pass1,
            page_sequence_range=page_sequence_range
        )
        p1_debug = dict(p1_debug or {})
        p1_debug.setdefault('pass1_retry_used', False)
        p1_debug.setdefault('pass1_retry_min_openxml_idx', None)
        total_pdf_units = len(pdf_units or [])
        base_min_openxml_idx = int(min_openxml_idx or 0)

        def build_candidate(
            phase_name,
            attempt_num,
            candidate_min_openxml_idx,
            alignments,
            unaligned_pdf,
            debug_info,
            fallback_skip_anchor=None,
            retry_page_sequence_range=None
        ):
            debug_info = dict(debug_info or {})
            cross_page_ratio = float(debug_info.get('cross_page_backward_skip_ratio') or 0.0)
            matched_pdf_units = self._count_matched_pdf_units(alignments)
            match_coverage = self._compute_match_coverage(alignments, total_pdf_units)
            openxml_diversity = self._compute_openxml_diversity(alignments)
            stable_pointer = self._compute_stable_pass1_pointer(alignments, candidate_min_openxml_idx)
            observed_skip_anchor = self._try_parse_int(debug_info.get('median_cross_page_skip_openxml_idx'))
            if observed_skip_anchor is None:
                observed_skip_anchor = self._try_parse_int(debug_info.get('min_cross_page_skip_openxml_idx'))
            if observed_skip_anchor is None:
                observed_skip_anchor = self._try_parse_int(debug_info.get('first_cross_page_skip_openxml_idx'))
            if observed_skip_anchor is None:
                observed_skip_anchor = self._try_parse_int(fallback_skip_anchor)
            stable_pointer_max = self._try_parse_int(stable_pointer.get('max_openxml_idx'))
            rollback_gap = max(
                0,
                base_min_openxml_idx - (
                    stable_pointer_max
                    if stable_pointer_max is not None
                    else candidate_min_openxml_idx
                )
            )
            skip_anchor_distance = (
                abs(candidate_min_openxml_idx - observed_skip_anchor)
                if observed_skip_anchor is not None
                else None
            )
            candidate_seq_range = self._normalize_sequence_range((
                debug_info.get('candidate_openxml_seq_min'),
                debug_info.get('candidate_openxml_seq_max'),
            ))
            candidate_idx_range = self._normalize_sequence_range((
                debug_info.get('candidate_openxml_band_idx_min'),
                debug_info.get('candidate_openxml_band_idx_max'),
            ))
            band_metrics = self._compute_candidate_band_metrics(
                alignments,
                openxml_units,
                seq_range=candidate_seq_range,
                idx_range=candidate_idx_range
            )
            return {
                'phase': phase_name,
                'attempt': attempt_num,
                'min_openxml_idx': candidate_min_openxml_idx,
                'alignments': alignments,
                'unaligned_pdf': unaligned_pdf,
                'debug': debug_info,
                'matched_pdf_units': matched_pdf_units,
                'match_coverage': match_coverage,
                'openxml_diversity': openxml_diversity,
                'cross_page_backward_skip_ratio': cross_page_ratio,
                'first_cross_page_skip_openxml_idx': self._try_parse_int(
                    debug_info.get('first_cross_page_skip_openxml_idx')
                ),
                'min_cross_page_skip_openxml_idx': self._try_parse_int(
                    debug_info.get('min_cross_page_skip_openxml_idx')
                ),
                'median_cross_page_skip_openxml_idx': self._try_parse_int(
                    debug_info.get('median_cross_page_skip_openxml_idx')
                ),
                'early_cross_page_skip_count': int(debug_info.get('early_cross_page_skip_count') or 0),
                'early_cross_page_skip_ratio': float(debug_info.get('early_cross_page_skip_ratio') or 0.0),
                'observed_skip_anchor': observed_skip_anchor,
                'skip_anchor_distance': skip_anchor_distance,
                'stable_pointer_max': stable_pointer_max,
                'rollback_gap': rollback_gap,
                'candidate_openxml_source': debug_info.get('candidate_openxml_source'),
                'candidate_anchor_hit_count': int(debug_info.get('candidate_openxml_anchor_hit_count') or 0),
                'candidate_seq_min': self._try_parse_int(debug_info.get('candidate_openxml_seq_min')),
                'candidate_seq_max': self._try_parse_int(debug_info.get('candidate_openxml_seq_max')),
                'candidate_band_idx_min': self._try_parse_int(debug_info.get('candidate_openxml_band_idx_min')),
                'candidate_band_idx_max': self._try_parse_int(debug_info.get('candidate_openxml_band_idx_max')),
                'candidate_band_alignment_ratio': band_metrics.get('alignment_ratio', 0.0),
                'candidate_band_char_ratio': band_metrics.get('char_ratio', 0.0),
                'candidate_band_alignment_count': band_metrics.get('alignment_count', 0),
                'candidate_band_total_alignment_count': band_metrics.get('total_alignment_count', 0),
                'retry_page_sequence_range': self._normalize_sequence_range(retry_page_sequence_range),
            }

        candidate_runs = [
            build_candidate(
                'pass1',
                0,
                base_min_openxml_idx,
                p1_align,
                p1_un_pdf,
                p1_debug
            )
        ]

        retry_cpb_ratio = self._read_float_env("ALIGNMENT_PASS1_RETRY_CPB_RATIO", 0.9, min_value=0.0, max_value=1.0)
        retry_min_coverage = self._read_float_env("ALIGNMENT_PASS1_RETRY_MIN_COVERAGE", 0.25, min_value=0.0, max_value=1.0)
        retry_min_matched_units = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_MATCHED_UNITS", 5)
        retry_min_openxml_diversity = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_MIN_OPENXML_DIVERSITY",
            0.35,
            min_value=0.0
        )
        retry_min_skip_gap = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_SKIP_GAP", 24)
        retry_min_early_skip_count = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_EARLY_SKIP_COUNT", 8)
        retry_min_early_skip_ratio = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_MIN_EARLY_SKIP_RATIO",
            0.5,
            min_value=0.0,
            max_value=1.0
        )
        retry_anchor_cpb_ratio = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_ANCHOR_CPB_RATIO",
            0.4,
            min_value=0.0,
            max_value=1.0
        )
        max_retries = self._read_positive_int_env("ALIGNMENT_PASS1_MAX_RETRIES", 2)

        initial_candidate = candidate_runs[0]
        initial_skip_anchor = initial_candidate.get('observed_skip_anchor')
        initial_anchor_gap = (
            max(0, base_min_openxml_idx - initial_skip_anchor)
            if initial_skip_anchor is not None
            else 0
        )
        initial_early_skip_count = int(initial_candidate.get('early_cross_page_skip_count') or 0)
        initial_early_skip_ratio = float(initial_candidate.get('debug', {}).get('early_cross_page_skip_ratio') or 0.0)
        strong_skip_anchor_signal = (
            initial_skip_anchor is not None and
            initial_anchor_gap > retry_min_skip_gap and
            initial_early_skip_count >= retry_min_early_skip_count and
            (
                initial_early_skip_ratio >= retry_min_early_skip_ratio or
                initial_candidate['cross_page_backward_skip_ratio'] >= retry_anchor_cpb_ratio
            )
        )
        is_retry_candidate = (
            strong_skip_anchor_signal or
            (
                initial_candidate['cross_page_backward_skip_ratio'] >= retry_cpb_ratio and
                (
                    initial_candidate['match_coverage'] <= retry_min_coverage or
                    initial_candidate['matched_pdf_units'] <= retry_min_matched_units or
                    initial_candidate['openxml_diversity'] <= retry_min_openxml_diversity
                )
            )
        )
        skip_anchor_penalty_cpb_ratio = self._read_float_env(
            "ALIGNMENT_PASS1_SKIP_ANCHOR_PENALTY_CPB_RATIO",
            0.4,
            min_value=0.0,
            max_value=1.0
        )

        attempted_min_openxml_idx = set()
        if is_retry_candidate:
            retry_min_openxml_idx_candidates = self._build_pass1_retry_candidates(
                base_min_openxml_idx,
                initial_candidate.get('debug'),
                max_retries
            )
            for attempt_num, retry_min_openxml_idx in enumerate(retry_min_openxml_idx_candidates, start=1):
                if retry_min_openxml_idx in attempted_min_openxml_idx:
                    continue
                attempted_min_openxml_idx.add(retry_min_openxml_idx)
                retry_page_seq_range = self._build_retry_page_sequence_range(
                    initial_candidate,
                    page_sequence_range,
                    openxml_units,
                    pdf_units
                )

                trace_pass1_retry = dict(trace_context or {})
                trace_pass1_retry['phase'] = f'pass1_retry_{attempt_num}'
                retry_align, retry_un_pdf, _, retry_debug = self._perform_char_alignment(
                    pdf_units,
                    openxml_units,
                    retry_min_openxml_idx,
                    trace_context=trace_pass1_retry,
                    page_sequence_range=retry_page_seq_range
                )
                candidate_runs.append(
                    build_candidate(
                        f'pass1_retry_{attempt_num}',
                        attempt_num,
                        retry_min_openxml_idx,
                        retry_align,
                        retry_un_pdf,
                        retry_debug,
                        fallback_skip_anchor=initial_skip_anchor,
                        retry_page_sequence_range=retry_page_seq_range
                    )
                )
                if retry_min_openxml_idx == 0:
                    break

        def mark_retry_lock_candidate(candidate):
            retry_max_backward_gap = self._read_positive_int_env(
                "ALIGNMENT_PASS1_RETRY_MAX_BACKWARD_GAP",
                48
            )
            candidate['retry_lock_candidate'] = (
                candidate['cross_page_backward_skip_ratio'] >= retry_cpb_ratio and
                (
                    candidate['match_coverage'] <= retry_min_coverage or
                    candidate['matched_pdf_units'] <= retry_min_matched_units or
                    candidate['openxml_diversity'] <= retry_min_openxml_diversity
                )
            )
            candidate['backward_retry_penalty'] = (
                candidate.get('attempt', 0) > 0 and
                int(candidate.get('rollback_gap') or 0) > retry_max_backward_gap
            )
            observed_skip_anchor = candidate.get('observed_skip_anchor')
            candidate['skip_anchor_penalty'] = (
                observed_skip_anchor is not None and
                (
                    candidate['cross_page_backward_skip_ratio'] >= skip_anchor_penalty_cpb_ratio or
                    int(candidate.get('early_cross_page_skip_count') or 0) >= retry_min_early_skip_count
                ) and
                candidate['min_openxml_idx'] > (observed_skip_anchor + 16)
            )
            return candidate

        candidate_runs = [mark_retry_lock_candidate(candidate) for candidate in candidate_runs]
        selected_candidate = max(candidate_runs, key=self._score_pass1_candidate)
        p1_align = selected_candidate['alignments']
        p1_un_pdf = selected_candidate['unaligned_pdf']
        p1_debug = dict(selected_candidate['debug'] or {})
        self._annotate_alignment_confidence(
            p1_align,
            candidate_source=selected_candidate.get('candidate_openxml_source'),
        )
        p1_debug['pass1_retry_used'] = selected_candidate['attempt'] > 0
        p1_debug['pass1_retry_min_openxml_idx'] = (
            selected_candidate['min_openxml_idx'] if selected_candidate['attempt'] > 0 else None
        )
        p1_debug['pass1_retry_attempts'] = max(0, len(candidate_runs) - 1)
        p1_debug['pass1_retry_candidate_scores'] = [
            {
                'phase': candidate['phase'],
                'attempt': candidate['attempt'],
                'min_openxml_idx': candidate['min_openxml_idx'],
                'matched_pdf_units': candidate['matched_pdf_units'],
                'match_coverage': candidate['match_coverage'],
                'openxml_diversity': candidate['openxml_diversity'],
                'cross_page_backward_skip_ratio': candidate['cross_page_backward_skip_ratio'],
                'retry_lock_candidate': candidate.get('retry_lock_candidate', False),
                'backward_retry_penalty': candidate.get('backward_retry_penalty', False),
                'skip_anchor_penalty': candidate.get('skip_anchor_penalty', False),
                'skip_anchor_distance': candidate.get('skip_anchor_distance'),
                'observed_skip_anchor': candidate.get('observed_skip_anchor'),
                'stable_pointer_max': candidate.get('stable_pointer_max'),
                'rollback_gap': candidate.get('rollback_gap'),
                'candidate_openxml_source': candidate.get('candidate_openxml_source'),
                'candidate_anchor_hit_count': candidate.get('candidate_anchor_hit_count'),
                'candidate_seq_min': candidate.get('candidate_seq_min'),
                'candidate_seq_max': candidate.get('candidate_seq_max'),
                'candidate_band_idx_min': candidate.get('candidate_band_idx_min'),
                'candidate_band_idx_max': candidate.get('candidate_band_idx_max'),
                'candidate_band_alignment_ratio': candidate.get('candidate_band_alignment_ratio'),
                'candidate_band_char_ratio': candidate.get('candidate_band_char_ratio'),
                'retry_page_sequence_range': candidate.get('retry_page_sequence_range'),
            }
            for candidate in candidate_runs
        ]
        p1_debug['pass1_selected_attempt'] = selected_candidate['attempt']
        p1_debug['pass1_matched_pdf_units'] = selected_candidate['matched_pdf_units']
        p1_debug['pass1_matched_openxml_units'] = self._count_matched_openxml_units(p1_align)
        p1_debug['pass1_match_coverage'] = selected_candidate['match_coverage']
        p1_debug['pass1_openxml_diversity'] = selected_candidate['openxml_diversity']
        p1_debug['pass1_observed_skip_anchor'] = selected_candidate.get('observed_skip_anchor')
        p1_debug['pass1_skip_anchor_distance'] = selected_candidate.get('skip_anchor_distance')
        p1_debug['pass1_candidate_openxml_source'] = selected_candidate.get('candidate_openxml_source')
        p1_debug['pass1_candidate_anchor_hit_count'] = selected_candidate.get('candidate_anchor_hit_count', 0)
        p1_debug['pass1_candidate_seq_min'] = selected_candidate.get('candidate_seq_min')
        p1_debug['pass1_candidate_seq_max'] = selected_candidate.get('candidate_seq_max')
        p1_debug['pass1_candidate_band_idx_min'] = selected_candidate.get('candidate_band_idx_min')
        p1_debug['pass1_candidate_band_idx_max'] = selected_candidate.get('candidate_band_idx_max')
        p1_debug['pass1_candidate_band_alignment_ratio'] = selected_candidate.get('candidate_band_alignment_ratio', 0.0)
        p1_debug['pass1_candidate_band_char_ratio'] = selected_candidate.get('candidate_band_char_ratio', 0.0)
        p1_debug['initial_pass_cross_page_backward_skip_ratio'] = initial_candidate.get('cross_page_backward_skip_ratio', 0.0)
        p1_debug['initial_pass_first_cross_page_skip_openxml_idx'] = initial_candidate.get('first_cross_page_skip_openxml_idx')
        p1_debug['initial_pass_min_cross_page_skip_openxml_idx'] = initial_candidate.get('min_cross_page_skip_openxml_idx')
        p1_debug['initial_pass_median_cross_page_skip_openxml_idx'] = initial_candidate.get('median_cross_page_skip_openxml_idx')
        p1_debug['initial_pass_early_cross_page_skip_count'] = initial_candidate.get('early_cross_page_skip_count', 0)
        p1_debug['initial_pass_early_cross_page_skip_ratio'] = initial_candidate.get('early_cross_page_skip_ratio', 0.0)
        p1_debug['initial_pass_observed_skip_anchor'] = initial_candidate.get('observed_skip_anchor')
        p1_debug['selected_candidate_cross_page_backward_skip_ratio'] = selected_candidate.get('cross_page_backward_skip_ratio', 0.0)
        p1_debug['selected_candidate_first_cross_page_skip_openxml_idx'] = selected_candidate.get('first_cross_page_skip_openxml_idx')
        p1_debug['selected_candidate_min_cross_page_skip_openxml_idx'] = selected_candidate.get('min_cross_page_skip_openxml_idx')
        p1_debug['selected_candidate_median_cross_page_skip_openxml_idx'] = selected_candidate.get('median_cross_page_skip_openxml_idx')
        p1_debug['selected_candidate_early_cross_page_skip_count'] = selected_candidate.get('early_cross_page_skip_count', 0)
        p1_debug['selected_candidate_early_cross_page_skip_ratio'] = selected_candidate.get('early_cross_page_skip_ratio', 0.0)
        p1_debug['selected_candidate_stable_pointer_max'] = selected_candidate.get('stable_pointer_max')
        selected_retry_seq_range = selected_candidate.get('retry_page_sequence_range')
        p1_debug['selected_candidate_retry_seq_min'] = (
            selected_retry_seq_range[0] if selected_retry_seq_range else None
        )
        p1_debug['selected_candidate_retry_seq_max'] = (
            selected_retry_seq_range[1] if selected_retry_seq_range else None
        )
        stable_pointer = self._compute_stable_pass1_pointer(p1_align, min_openxml_idx)
        p1_align, p1_un_pdf, backward_alignment_prune_debug = self._drop_far_backward_alignments(
            p1_align,
            p1_un_pdf,
            pdf_units,
            min_openxml_idx
        )

        final_align = list(p1_align)
        final_align.sort(key=lambda x: x.get('element_sequence') or 0)
        final_un_pdf = list(p1_un_pdf)

        final_align, final_un_pdf = self._absorb_unaligned_into_alignments(final_align, final_un_pdf, pdf_units)

        p1_un_ox = p1_debug.get('unaligned_openxml_indices', [])
        p1_debug['pass2_openxml_source'] = 'selected_candidate_unaligned_openxml'
        final_align, final_un_pdf, _ = self._match_remaining_with_unaligned_openxml(
            final_align,
            final_un_pdf,
            p1_un_ox,
            pdf_units,
            openxml_units,
            min_openxml_idx=min_openxml_idx,
            trace_context=trace_context,
            page_sequence_range=page_sequence_range
        )
        final_align, block_context_remap_debug = self._remap_block_context_drift_alignments(
            final_align,
            openxml_units
        )

        pre_cleanup_keys = self._collect_alignment_unit_keys(final_align)

        final_align = self._cleanup_punctuation_alignments(final_align)
        final_align, shape_conflict_debug = self._resolve_shape_alignment_conflicts(final_align, pdf_units)
        final_align, final_un_pdf, chart_visual_attach_debug = self._attach_chart_visuals_to_chart_alignments(
            final_align,
            final_un_pdf,
            pdf_units,
            openxml_units,
            min_openxml_idx=min_openxml_idx,
            page_sequence_range=page_sequence_range
        )
        final_align, final_un_pdf, picture_overlap_prune_debug = self._prune_visual_alignment_text_units(
            final_align,
            final_un_pdf,
            pdf_units
        )
        final_align, final_un_pdf, shape_attach_debug = self._attach_shape_clusters_to_next_alignment(final_align, final_un_pdf, pdf_units)
        final_align = self._merge_line_overlap_alignments(final_align)
        final_align = self._repair_marker_only_alignment_gaps(final_align, openxml_units)
        final_align, chart_rescue_debug = self._rescue_chart_hline_alignments(
            final_align,
            openxml_units,
            min_openxml_idx=min_openxml_idx
        )
        pre_filter_alignments = deepcopy(final_align)
        final_align, final_un_pdf = self._filter_sparse_matched_units(final_align, final_un_pdf, pdf_units)
        final_align, final_un_pdf = self._filter_low_match_alignments(final_align, final_un_pdf, pdf_units)
        final_un_pdf = self._restore_dropped_alignment_units(
            pre_cleanup_keys,
            final_align,
            final_un_pdf,
            pdf_units
        )
        paragraph_rescue_debug = []
        rescue_paragraph_alignments = getattr(self, '_rescue_paragraph_alignments', None)
        if (
            self._is_env_enabled_default_true("ALIGNMENT_ENABLE_PARAGRAPH_RESCUE")
            and callable(rescue_paragraph_alignments)
        ):
            final_align, final_un_pdf, paragraph_rescue_debug = rescue_paragraph_alignments(
                pre_filter_alignments,
                final_align,
                final_un_pdf,
                pdf_units
            )
        fragment_rescue_debug = []
        rescue_fragment_alignments = getattr(self, '_rescue_fragment_paragraph_alignments', None)
        if (
            self._is_env_enabled_default_true("ALIGNMENT_ENABLE_FRAGMENT_RESCUE")
            and callable(rescue_fragment_alignments)
        ):
            final_align, fragment_rescue_debug = rescue_fragment_alignments(
                openxml_units,
                final_align,
                page_sequence_range=page_sequence_range
            )
        if self._is_env_enabled_default_true("ALIGNMENT_ENABLE_Y_OVERLAP_ABSORB"):
            final_align, final_un_pdf = self._absorb_unaligned_by_y_overlap(final_align, final_un_pdf, pdf_units)

        self._annotate_alignment_confidence(
            final_align,
            candidate_source=selected_candidate.get('candidate_openxml_source'),
        )

        # Legacy debug fields
        p1_debug['pass2_shape_debug'] = []
        p1_debug['pass2_shape_matched'] = 0
        p1_debug['pass2_consumed_pdf'] = []
        p1_debug['shape_openxml_count'] = 0
        p1_debug['non_shape_openxml_count'] = len(openxml_units)
        p1_debug['shape_conflict_debug'] = shape_conflict_debug
        p1_debug['shape_conflict_count'] = len(shape_conflict_debug)
        p1_debug['chart_visual_attach_debug'] = chart_visual_attach_debug
        p1_debug['chart_visual_attach_count'] = len(chart_visual_attach_debug)
        p1_debug['visual_slot_attach_count'] = sum(
            1 for entry in (chart_visual_attach_debug or [])
            if entry.get('target_type') == 'visual_slot'
        )
        p1_debug['picture_overlap_prune_debug'] = picture_overlap_prune_debug
        p1_debug['picture_overlap_prune_count'] = len(picture_overlap_prune_debug)
        p1_debug['shape_attach_debug'] = shape_attach_debug
        p1_debug['shape_attach_count'] = len(shape_attach_debug)
        p1_debug['backward_alignment_prune_debug'] = backward_alignment_prune_debug
        p1_debug['backward_alignment_prune_count'] = len(backward_alignment_prune_debug)
        p1_debug['chart_rescue_debug'] = chart_rescue_debug
        p1_debug['chart_rescue_count'] = len(chart_rescue_debug)
        p1_debug['block_context_remap_debug'] = block_context_remap_debug
        p1_debug['block_context_remap_count'] = len(block_context_remap_debug)
        p1_debug['paragraph_rescue_debug'] = paragraph_rescue_debug
        p1_debug['paragraph_rescue_count'] = len(paragraph_rescue_debug)
        p1_debug['fragment_rescue_debug'] = fragment_rescue_debug
        p1_debug['fragment_rescue_count'] = len(fragment_rescue_debug)
        orphan_chart_visual_indices = [
            pdf_idx for pdf_idx in final_un_pdf
            if 0 <= pdf_idx < len(pdf_units) and pdf_units[pdf_idx].get('is_chart_visual')
        ]
        p1_debug['orphan_chart_visual_count'] = len(orphan_chart_visual_indices)
        p1_debug['orphan_chart_visual_indices'] = orphan_chart_visual_indices
        pass2_max_idx = self._compute_alignment_max_openxml_idx(
            [alignment for alignment in final_align if alignment.get('late_matched')]
        )
        max_idx = stable_pointer.get('max_openxml_idx', min_openxml_idx)
        p1_debug['pass1_pointer_cluster_min'] = stable_pointer.get('cluster_min')
        p1_debug['pass1_pointer_cluster_max'] = stable_pointer.get('cluster_max')
        p1_debug['pass1_pointer_cluster_size'] = stable_pointer.get('cluster_size', 0)
        p1_debug['pass1_pointer_cluster_total_matched_chars'] = stable_pointer.get('cluster_total_matched_chars', 0)
        p1_debug['pass1_pointer_source'] = stable_pointer.get('source', 'frozen')
        p1_debug['pass1_max_openxml_idx_raw'] = stable_pointer.get('raw_max_openxml_idx')
        p1_debug['pass2_max_openxml_idx'] = pass2_max_idx
        p1_debug['final_max_openxml_idx'] = max_idx

        # Filter unaligned OpenXML to only those within this page's sequence range
        page_unaligned_openxml = p1_un_ox
        if p1_align and p1_un_ox:
            aligned_sequences = [a.get('element_sequence') or 0 for a in p1_align]
            min_seq = min(aligned_sequences)
            max_seq = max(aligned_sequences)
            page_unaligned_openxml = [
                idx for idx in p1_un_ox
                if min_seq <= (openxml_units[idx].get('elem_seq') or 0) <= max_seq
            ]

        return {
            'phase1_alignments': p1_align,
            'final_alignments': final_align,
            'shape_alignments': [],
            'unaligned_after_phase1': p1_un_pdf,
            'unaligned_final': final_un_pdf,
            'unaligned_openxml': page_unaligned_openxml,
            'debug_info': p1_debug,
            'max_openxml_idx': max_idx
        }

    def _compute_max_openxml_idx_from_alignments(self, alignments, min_openxml_idx):
        max_idx = None
        for alignment in alignments or []:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment['cells']:
                    idx = cell.get('openxml_idx')
                    if idx is not None:
                        max_idx = idx if max_idx is None else max(max_idx, idx)
            else:
                indices = alignment.get('openxml_indices')
                if indices:
                    for idx in indices:
                        if idx is None:
                            continue
                        max_idx = idx if max_idx is None else max(max_idx, idx)
                else:
                    idx = alignment.get('openxml_idx')
                    if idx is not None:
                        max_idx = idx if max_idx is None else max(max_idx, idx)
        if max_idx is None:
            return min_openxml_idx
        return max(min_openxml_idx, max_idx)

    def _is_marker_only_text(self, text):
        if not text:
            return False
        return bool(self.MARKER_ONLY_TEXT_RE.match(str(text).strip()))

    def _is_bookmark_end_unit(self, unit):
        if not unit:
            return False
        return str(unit.get('elem_type') or '').strip().lower() == 'bookmarkend'

    def _repair_marker_only_alignment_gaps(self, alignments, openxml_units):
        if not alignments or not openxml_units:
            return alignments

        seq_to_alignment = {}
        for alignment in alignments:
            seq = alignment.get('element_sequence')
            if seq is None:
                continue
            seq_to_alignment[seq] = alignment

        if not seq_to_alignment:
            return alignments

        seq_to_openxml = {}
        for openxml_idx, unit in enumerate(openxml_units):
            seq = unit.get('elem_seq')
            if seq is None or seq in seq_to_openxml:
                continue
            seq_to_openxml[seq] = (openxml_idx, unit)

        if not seq_to_openxml:
            return alignments

        min_seq = min(seq_to_alignment.keys())
        max_seq = max(seq_to_alignment.keys())
        missing_marker_seqs = [
            seq for seq in range(max(0, min_seq - 2), max_seq + 3)
            if seq not in seq_to_alignment
            and seq in seq_to_openxml
            and (
                self._is_marker_only_text(seq_to_openxml[seq][1].get('text', ''))
                or self._is_bookmark_end_unit(seq_to_openxml[seq][1])
            )
        ]
        if not missing_marker_seqs:
            return alignments

        created = []
        for missing_seq in missing_marker_seqs:
            openxml_idx, openxml_unit = seq_to_openxml[missing_seq]
            is_bookmark_proxy = self._is_bookmark_end_unit(openxml_unit)

            prev_candidates = sorted(
                [
                    a for a in alignments
                    if a.get('element_sequence') is not None
                    and a.get('element_sequence') < missing_seq
                    and (missing_seq - a.get('element_sequence')) <= 2
                    and not a.get('is_table')
                    and not a.get('is_image_part')
                    and a.get('matched_pdf_units')
                ],
                key=lambda a: a.get('element_sequence'),
                reverse=True
            )
            next_candidates = sorted(
                [
                    a for a in alignments
                    if a.get('element_sequence') is not None
                    and a.get('element_sequence') > missing_seq
                    and (a.get('element_sequence') - missing_seq) <= 3
                    and not a.get('is_table')
                    and not a.get('is_image_part')
                ],
                key=lambda a: a.get('element_sequence')
            )
            if not next_candidates:
                continue

            donor_alignment = None
            donor_unit = None

            if is_bookmark_proxy:
                proxy_candidates = []
                for candidate in prev_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if units:
                        proxy_candidates.append((candidate, units[-1]))
                for candidate in next_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if units:
                        proxy_candidates.append((candidate, units[0]))
                if proxy_candidates:
                    donor_alignment, donor_unit = proxy_candidates[0]
            else:
                for candidate in next_candidates:
                    units = sorted(
                        candidate.get('matched_pdf_units', []),
                        key=lambda u: u.get('item_idx', -1)
                    )
                    if len(units) < 2:
                        continue

                    leading_markers = []
                    for unit in units:
                        if self._is_marker_only_text(unit.get('text', '')):
                            leading_markers.append(unit)
                            continue
                        break

                    if len(leading_markers) >= 2:
                        donor_alignment = candidate
                        donor_unit = leading_markers[0]
                        break

            if donor_alignment is None or donor_unit is None:
                continue

            if not is_bookmark_proxy:
                donor_key = self._matched_unit_key(donor_unit)
                donor_units = donor_alignment.get('matched_pdf_units', [])
                consumed = False
                kept_units = []
                for unit in donor_units:
                    unit_key = self._matched_unit_key(unit)
                    if not consumed and donor_key is not None and unit_key == donor_key:
                        consumed = True
                        continue
                    kept_units.append(unit)

                if not consumed:
                    continue

                donor_alignment['matched_pdf_units'] = kept_units
                self._recompute_alignment_bboxes(donor_alignment)

            restored = {
                'element_id': openxml_unit['elem_id'],
                'element_sequence': openxml_unit['elem_seq'],
                'element_type': openxml_unit['elem_type'],
                'is_table': False,
                'is_synthetic_marker_repair': True,
                'is_synthetic_bookmark_proxy': is_bookmark_proxy,
                'element_text': openxml_unit.get('text', ''),
                'matched_pdf_units': [donor_unit],
                'merged_bbox': list(donor_unit.get('bbox')) if donor_unit.get('bbox') else None,
                'cells': None,
                'is_text_part': openxml_unit.get('is_text_part', False),
                'is_image_part': False,
                'unit_id': str(openxml_unit['elem_id']),
                'openxml_indices': [openxml_idx],
                'openxml_idx': openxml_idx,
                'image_index': openxml_unit.get('image_index'),
                'font_families': openxml_unit.get('font_families', []),
                'style_ids': openxml_unit.get('style_ids', []),
                'is_code_font': openxml_unit.get('is_code_font', False),
                'is_code_style': openxml_unit.get('is_code_style', False),
                'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
            }
            created.append(restored)
            seq_to_alignment[missing_seq] = restored

        if created:
            alignments.extend(created)
            alignments[:] = [
                a for a in alignments
                if a.get('is_table') or a.get('matched_pdf_units')
            ]
            alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments

    def _rescue_chart_hline_alignments(self, alignments, openxml_units, min_openxml_idx=None):
        if not alignments or not openxml_units:
            return alignments, []

        min_openxml_idx = self._try_parse_int(min_openxml_idx)

        max_seq_gap = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MAX_SEQ_GAP',
            4
        )
        min_width = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MIN_WIDTH',
            160
        )
        min_height = self._read_positive_int_env(
            'ALIGNMENT_CHART_RESCUE_MIN_HEIGHT',
            100
        )

        def bbox_size_ok(bbox):
            if not bbox or len(bbox) < 4:
                return False
            width = max(0.0, float(bbox[2]) - float(bbox[0]))
            height = max(0.0, float(bbox[3]) - float(bbox[1]))
            return width >= min_width and height >= min_height

        seq_to_alignment = {}
        for alignment in alignments:
            if alignment.get('is_table'):
                continue
            seq = self._try_parse_int(alignment.get('element_sequence'))
            if seq is None:
                continue
            seq_to_alignment[seq] = alignment

        seq_to_openxml = {}
        for openxml_idx, unit in enumerate(openxml_units):
            seq = self._try_parse_int((unit or {}).get('elem_seq'))
            if seq is None or seq in seq_to_openxml:
                continue
            seq_to_openxml[seq] = (openxml_idx, unit)

        if not seq_to_alignment or not seq_to_openxml:
            return alignments, []

        debug = []
        created = []
        touched_alignments = []

        for alignment in list(alignments):
            if alignment.get('is_table') or alignment.get('is_openxml_chart') or alignment.get('is_openxml_visual_slot'):
                continue
            if not self._is_paragraph_like_alignment(alignment):
                continue

            donor_seq = self._try_parse_int(alignment.get('element_sequence'))
            if donor_seq is None:
                continue

            donor_text = self._normalize_text(alignment.get('element_text') or '')
            if not donor_text.startswith('gambar'):
                continue

            units = list(alignment.get('matched_pdf_units') or [])
            moved_units = [
                unit for unit in units
                if unit.get('item_type') == 'hline_table' and bbox_size_ok(unit.get('bbox'))
            ]
            if not moved_units:
                continue

            candidate = None
            for gap in range(1, max_seq_gap + 1):
                prev_seq = donor_seq - gap
                prev_entry = seq_to_openxml.get(prev_seq)
                if prev_entry and prev_seq not in seq_to_alignment:
                    openxml_idx, openxml_unit = prev_entry
                    if min_openxml_idx is not None and openxml_idx < min_openxml_idx:
                        prev_entry = None
                    if not prev_entry:
                        continue
                    if openxml_unit.get('is_openxml_chart') or openxml_unit.get('is_openxml_visual_slot'):
                        candidate = (prev_seq, openxml_idx, openxml_unit)
                        break

                next_seq = donor_seq + gap
                next_entry = seq_to_openxml.get(next_seq)
                if next_entry and next_seq not in seq_to_alignment:
                    openxml_idx, openxml_unit = next_entry
                    if min_openxml_idx is not None and openxml_idx < min_openxml_idx:
                        next_entry = None
                    if not next_entry:
                        continue
                    if openxml_unit.get('is_openxml_chart') or openxml_unit.get('is_openxml_visual_slot'):
                        candidate = (next_seq, openxml_idx, openxml_unit)
                        break

            if not candidate:
                continue

            candidate_seq, openxml_idx, openxml_unit = candidate
            moved_units = sorted(moved_units, key=lambda unit: unit.get('item_idx', -1))
            remaining_units = [unit for unit in units if unit not in moved_units]

            rescued_alignment = {
                'element_id': openxml_unit.get('elem_id'),
                'element_sequence': openxml_unit.get('elem_seq'),
                'element_type': openxml_unit.get('elem_type'),
                'is_table': False,
                'is_chart_rescue': True,
                'element_text': openxml_unit.get('text', ''),
                'matched_pdf_units': moved_units,
                'merged_bbox': self._merge_bboxes([u.get('bbox') for u in moved_units]),
                'cells': None,
                'is_text_part': openxml_unit.get('is_text_part', False),
                'is_image_part': openxml_unit.get('is_image_part', False),
                'unit_id': str(openxml_unit.get('unit_id') or openxml_unit.get('elem_id')),
                'openxml_indices': [openxml_idx],
                'openxml_idx': openxml_idx,
                'image_index': openxml_unit.get('image_index'),
                'font_families': openxml_unit.get('font_families', []),
                'style_ids': openxml_unit.get('style_ids', []),
                'is_code_font': openxml_unit.get('is_code_font', False),
                'is_code_style': openxml_unit.get('is_code_style', False),
                'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
                'chart_rescued_from_element_id': alignment.get('element_id'),
            }

            alignment['matched_pdf_units'] = remaining_units
            touched_alignments.append(alignment)
            created.append(rescued_alignment)
            seq_to_alignment[candidate_seq] = rescued_alignment
            debug.append({
                'from_element_id': alignment.get('element_id'),
                'from_element_sequence': donor_seq,
                'to_element_id': openxml_unit.get('elem_id'),
                'to_element_sequence': candidate_seq,
                'moved_unit_count': len(moved_units),
            })

        if created:
            for alignment in touched_alignments:
                self._recompute_alignment_bboxes(alignment)
            alignments.extend(created)
            alignments[:] = [
                alignment for alignment in alignments
                if alignment.get('is_table') or alignment.get('matched_pdf_units')
            ]
            alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments, debug

    def _drop_far_backward_alignments(self, alignments, unaligned_pdf_indices, pdf_units, min_openxml_idx):
        if not alignments:
            return alignments, list(unaligned_pdf_indices or []), []

        base_min_openxml_idx = self._try_parse_int(min_openxml_idx)
        if base_min_openxml_idx is None or base_min_openxml_idx <= 0:
            return alignments, list(unaligned_pdf_indices or []), []

        max_backward_gap = self._read_positive_int_env(
            'ALIGNMENT_MAX_BACKWARD_ALIGNMENT_GAP',
            24
        )
        cutoff_idx = max(0, base_min_openxml_idx - max_backward_gap)

        kept = []
        dropped = []
        pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox = self._build_pdf_lookup_maps(pdf_units)
        restored_unaligned = set(unaligned_pdf_indices or [])
        for alignment in alignments:
            indices = self._collect_alignment_openxml_indices(
                alignment,
                include_table_cells=True
            )
            if not indices:
                kept.append(alignment)
                continue

            alignment_max_idx = max(indices)
            if alignment_max_idx < cutoff_idx:
                dropped.append({
                    'element_id': alignment.get('element_id'),
                    'element_sequence': alignment.get('element_sequence'),
                    'openxml_max_idx': alignment_max_idx,
                    'cutoff_idx': cutoff_idx,
                })
                for unit in alignment.get('matched_pdf_units', []) or []:
                    pdf_idx = self._resolve_unit_pdf_index(
                        unit,
                        pdf_idx_by_unit_id,
                        pdf_idx_by_item_idx,
                        pdf_idx_by_bbox
                    )
                    if pdf_idx is not None:
                        restored_unaligned.add(pdf_idx)
                for cell in alignment.get('cells', []) or []:
                    for unit in cell.get('matched_pdf_units', []) or []:
                        pdf_idx = self._resolve_unit_pdf_index(
                            unit,
                            pdf_idx_by_unit_id,
                            pdf_idx_by_item_idx,
                            pdf_idx_by_bbox
                        )
                        if pdf_idx is not None:
                            restored_unaligned.add(pdf_idx)
                continue
            kept.append(alignment)

        return kept, sorted(restored_unaligned), dropped
