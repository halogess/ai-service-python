class AlignmentMatchingTwoPassSelectionMixin:
    def _build_pass1_candidate(
        self,
        phase_name,
        attempt_num,
        candidate_min_openxml_idx,
        alignments,
        unaligned_pdf,
        debug_info,
        *,
        total_pdf_units,
        base_min_openxml_idx,
        openxml_units,
        fallback_skip_anchor=None,
        retry_page_sequence_range=None,
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
            ),
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
            idx_range=candidate_idx_range,
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

    def _mark_retry_lock_candidate(
        self,
        candidate,
        *,
        retry_cpb_ratio,
        retry_min_coverage,
        retry_min_matched_units,
        retry_min_openxml_diversity,
        skip_anchor_penalty_cpb_ratio,
        retry_min_early_skip_count,
    ):
        retry_max_backward_gap = self._read_positive_int_env(
            "ALIGNMENT_PASS1_RETRY_MAX_BACKWARD_GAP",
            48,
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

    def _select_pass1_candidate_runs(
        self,
        pdf_units,
        openxml_units,
        min_openxml_idx,
        *,
        trace_context=None,
        page_sequence_range=None,
    ):
        trace_pass1 = dict(trace_context or {})
        trace_pass1['phase'] = 'pass1'
        p1_align, p1_un_pdf, _, p1_debug = self._perform_char_alignment(
            pdf_units,
            openxml_units,
            min_openxml_idx,
            trace_context=trace_pass1,
            page_sequence_range=page_sequence_range,
        )
        p1_debug = dict(p1_debug or {})
        p1_debug.setdefault('pass1_retry_used', False)
        p1_debug.setdefault('pass1_retry_min_openxml_idx', None)
        total_pdf_units = len(pdf_units or [])
        base_min_openxml_idx = int(min_openxml_idx or 0)

        candidate_runs = [
            self._build_pass1_candidate(
                'pass1',
                0,
                base_min_openxml_idx,
                p1_align,
                p1_un_pdf,
                p1_debug,
                total_pdf_units=total_pdf_units,
                base_min_openxml_idx=base_min_openxml_idx,
                openxml_units=openxml_units,
            )
        ]

        retry_cpb_ratio = self._read_float_env("ALIGNMENT_PASS1_RETRY_CPB_RATIO", 0.9, min_value=0.0, max_value=1.0)
        retry_min_coverage = self._read_float_env("ALIGNMENT_PASS1_RETRY_MIN_COVERAGE", 0.25, min_value=0.0, max_value=1.0)
        retry_min_matched_units = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_MATCHED_UNITS", 5)
        retry_min_openxml_diversity = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_MIN_OPENXML_DIVERSITY",
            0.35,
            min_value=0.0,
        )
        retry_min_skip_gap = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_SKIP_GAP", 24)
        retry_min_early_skip_count = self._read_positive_int_env("ALIGNMENT_PASS1_RETRY_MIN_EARLY_SKIP_COUNT", 8)
        retry_min_early_skip_ratio = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_MIN_EARLY_SKIP_RATIO",
            0.5,
            min_value=0.0,
            max_value=1.0,
        )
        retry_anchor_cpb_ratio = self._read_float_env(
            "ALIGNMENT_PASS1_RETRY_ANCHOR_CPB_RATIO",
            0.4,
            min_value=0.0,
            max_value=1.0,
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
            max_value=1.0,
        )

        attempted_min_openxml_idx = set()
        if is_retry_candidate:
            retry_min_openxml_idx_candidates = self._build_pass1_retry_candidates(
                base_min_openxml_idx,
                initial_candidate.get('debug'),
                max_retries,
            )
            for attempt_num, retry_min_openxml_idx in enumerate(retry_min_openxml_idx_candidates, start=1):
                if retry_min_openxml_idx in attempted_min_openxml_idx:
                    continue
                attempted_min_openxml_idx.add(retry_min_openxml_idx)
                retry_page_seq_range = self._build_retry_page_sequence_range(
                    initial_candidate,
                    page_sequence_range,
                    openxml_units,
                    pdf_units,
                )

                trace_pass1_retry = dict(trace_context or {})
                trace_pass1_retry['phase'] = f'pass1_retry_{attempt_num}'
                retry_align, retry_un_pdf, _, retry_debug = self._perform_char_alignment(
                    pdf_units,
                    openxml_units,
                    retry_min_openxml_idx,
                    trace_context=trace_pass1_retry,
                    page_sequence_range=retry_page_seq_range,
                )
                candidate_runs.append(
                    self._build_pass1_candidate(
                        f'pass1_retry_{attempt_num}',
                        attempt_num,
                        retry_min_openxml_idx,
                        retry_align,
                        retry_un_pdf,
                        retry_debug,
                        total_pdf_units=total_pdf_units,
                        base_min_openxml_idx=base_min_openxml_idx,
                        openxml_units=openxml_units,
                        fallback_skip_anchor=initial_skip_anchor,
                        retry_page_sequence_range=retry_page_seq_range,
                    )
                )
                if retry_min_openxml_idx == 0:
                    break

        candidate_runs = [
            self._mark_retry_lock_candidate(
                candidate,
                retry_cpb_ratio=retry_cpb_ratio,
                retry_min_coverage=retry_min_coverage,
                retry_min_matched_units=retry_min_matched_units,
                retry_min_openxml_diversity=retry_min_openxml_diversity,
                skip_anchor_penalty_cpb_ratio=skip_anchor_penalty_cpb_ratio,
                retry_min_early_skip_count=retry_min_early_skip_count,
            )
            for candidate in candidate_runs
        ]
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

        return selected_candidate, initial_candidate, p1_align, p1_un_pdf, p1_debug
