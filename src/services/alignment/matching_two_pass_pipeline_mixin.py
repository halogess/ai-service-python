from copy import deepcopy


class AlignmentMatchingTwoPassPipelineMixin:
    def _compute_page_unaligned_openxml(self, phase1_alignments, unaligned_openxml_indices, openxml_units):
        page_unaligned_openxml = unaligned_openxml_indices
        if phase1_alignments and unaligned_openxml_indices:
            aligned_sequences = [a.get('element_sequence') or 0 for a in phase1_alignments]
            min_seq = min(aligned_sequences)
            max_seq = max(aligned_sequences)
            page_unaligned_openxml = [
                idx for idx in unaligned_openxml_indices
                if min_seq <= (openxml_units[idx].get('elem_seq') or 0) <= max_seq
            ]
        return page_unaligned_openxml

    def _run_selected_candidate_postprocess(
        self,
        selected_candidate,
        p1_align,
        p1_un_pdf,
        p1_debug,
        pdf_units,
        openxml_units,
        min_openxml_idx,
        *,
        trace_context=None,
        page_sequence_range=None,
    ):
        stable_pointer = self._compute_stable_pass1_pointer(p1_align, min_openxml_idx)
        p1_align, p1_un_pdf, backward_alignment_prune_debug = self._drop_far_backward_alignments(
            p1_align,
            p1_un_pdf,
            pdf_units,
            min_openxml_idx,
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
            page_sequence_range=page_sequence_range,
        )
        final_align, block_context_remap_debug = self._remap_block_context_drift_alignments(
            final_align,
            openxml_units,
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
            page_sequence_range=page_sequence_range,
        )
        final_align, final_un_pdf, picture_overlap_prune_debug = self._prune_visual_alignment_text_units(
            final_align,
            final_un_pdf,
            pdf_units,
        )
        final_align, final_un_pdf, shape_attach_debug = self._attach_shape_clusters_to_next_alignment(
            final_align,
            final_un_pdf,
            pdf_units,
        )
        final_align = self._merge_line_overlap_alignments(final_align)
        final_align = self._repair_marker_only_alignment_gaps(final_align, openxml_units)
        final_align, chart_rescue_debug = self._rescue_chart_hline_alignments(
            final_align,
            openxml_units,
            min_openxml_idx=min_openxml_idx,
        )
        pre_filter_alignments = deepcopy(final_align)
        final_align, final_un_pdf = self._filter_sparse_matched_units(final_align, final_un_pdf, pdf_units)
        final_align, final_un_pdf = self._filter_low_match_alignments(final_align, final_un_pdf, pdf_units)
        final_un_pdf = self._restore_dropped_alignment_units(
            pre_cleanup_keys,
            final_align,
            final_un_pdf,
            pdf_units,
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
                pdf_units,
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
                page_sequence_range=page_sequence_range,
            )
        if self._is_env_enabled_default_true("ALIGNMENT_ENABLE_Y_OVERLAP_ABSORB"):
            final_align, final_un_pdf = self._absorb_unaligned_by_y_overlap(final_align, final_un_pdf, pdf_units)

        self._annotate_alignment_confidence(
            final_align,
            candidate_source=selected_candidate.get('candidate_openxml_source'),
        )

        orphan_chart_visual_indices = [
            pdf_idx for pdf_idx in final_un_pdf
            if 0 <= pdf_idx < len(pdf_units) and pdf_units[pdf_idx].get('is_chart_visual')
        ]
        pass2_max_idx = self._compute_alignment_max_openxml_idx(
            [alignment for alignment in final_align if alignment.get('late_matched')]
        )
        max_idx = stable_pointer.get('max_openxml_idx', min_openxml_idx)
        page_unaligned_openxml = self._compute_page_unaligned_openxml(p1_align, p1_un_ox, openxml_units)

        return {
            'phase1_alignments': p1_align,
            'unaligned_after_phase1': p1_un_pdf,
            'final_alignments': final_align,
            'unaligned_final': final_un_pdf,
            'page_unaligned_openxml': page_unaligned_openxml,
            'openxml_count': len(openxml_units),
            'shape_conflict_debug': shape_conflict_debug,
            'chart_visual_attach_debug': chart_visual_attach_debug,
            'picture_overlap_prune_debug': picture_overlap_prune_debug,
            'shape_attach_debug': shape_attach_debug,
            'backward_alignment_prune_debug': backward_alignment_prune_debug,
            'chart_rescue_debug': chart_rescue_debug,
            'block_context_remap_debug': block_context_remap_debug,
            'paragraph_rescue_debug': paragraph_rescue_debug,
            'fragment_rescue_debug': fragment_rescue_debug,
            'orphan_chart_visual_indices': orphan_chart_visual_indices,
            'pass2_max_idx': pass2_max_idx,
            'stable_pointer': stable_pointer,
            'max_idx': max_idx,
        }

    def _append_two_pass_postprocess_debug(self, p1_debug, postprocess):
        p1_debug['pass2_shape_debug'] = []
        p1_debug['pass2_shape_matched'] = 0
        p1_debug['pass2_consumed_pdf'] = []
        p1_debug['shape_openxml_count'] = 0
        p1_debug['non_shape_openxml_count'] = postprocess['openxml_count']
        p1_debug['shape_conflict_debug'] = postprocess['shape_conflict_debug']
        p1_debug['shape_conflict_count'] = len(postprocess['shape_conflict_debug'])
        p1_debug['chart_visual_attach_debug'] = postprocess['chart_visual_attach_debug']
        p1_debug['chart_visual_attach_count'] = len(postprocess['chart_visual_attach_debug'])
        p1_debug['visual_slot_attach_count'] = sum(
            1 for entry in (postprocess['chart_visual_attach_debug'] or [])
            if entry.get('target_type') == 'visual_slot'
        )
        p1_debug['picture_overlap_prune_debug'] = postprocess['picture_overlap_prune_debug']
        p1_debug['picture_overlap_prune_count'] = len(postprocess['picture_overlap_prune_debug'])
        p1_debug['shape_attach_debug'] = postprocess['shape_attach_debug']
        p1_debug['shape_attach_count'] = len(postprocess['shape_attach_debug'])
        p1_debug['backward_alignment_prune_debug'] = postprocess['backward_alignment_prune_debug']
        p1_debug['backward_alignment_prune_count'] = len(postprocess['backward_alignment_prune_debug'])
        p1_debug['chart_rescue_debug'] = postprocess['chart_rescue_debug']
        p1_debug['chart_rescue_count'] = len(postprocess['chart_rescue_debug'])
        p1_debug['block_context_remap_debug'] = postprocess['block_context_remap_debug']
        p1_debug['block_context_remap_count'] = len(postprocess['block_context_remap_debug'])
        p1_debug['paragraph_rescue_debug'] = postprocess['paragraph_rescue_debug']
        p1_debug['paragraph_rescue_count'] = len(postprocess['paragraph_rescue_debug'])
        p1_debug['fragment_rescue_debug'] = postprocess['fragment_rescue_debug']
        p1_debug['fragment_rescue_count'] = len(postprocess['fragment_rescue_debug'])
        p1_debug['orphan_chart_visual_count'] = len(postprocess['orphan_chart_visual_indices'])
        p1_debug['orphan_chart_visual_indices'] = postprocess['orphan_chart_visual_indices']
        stable_pointer = postprocess['stable_pointer']
        p1_debug['pass1_pointer_cluster_min'] = stable_pointer.get('cluster_min')
        p1_debug['pass1_pointer_cluster_max'] = stable_pointer.get('cluster_max')
        p1_debug['pass1_pointer_cluster_size'] = stable_pointer.get('cluster_size', 0)
        p1_debug['pass1_pointer_cluster_total_matched_chars'] = stable_pointer.get('cluster_total_matched_chars', 0)
        p1_debug['pass1_pointer_source'] = stable_pointer.get('source', 'frozen')
        p1_debug['pass1_max_openxml_idx_raw'] = stable_pointer.get('raw_max_openxml_idx')
        p1_debug['pass2_max_openxml_idx'] = postprocess['pass2_max_idx']
        p1_debug['final_max_openxml_idx'] = postprocess['max_idx']

    def _perform_two_pass_alignment(
        self,
        pdf_units,
        openxml_units,
        min_openxml_idx,
        trace_context=None,
        page_sequence_range=None,
    ):
        selected_candidate, _, p1_align, p1_un_pdf, p1_debug = self._select_pass1_candidate_runs(
            pdf_units,
            openxml_units,
            min_openxml_idx,
            trace_context=trace_context,
            page_sequence_range=page_sequence_range,
        )
        postprocess = self._run_selected_candidate_postprocess(
            selected_candidate,
            p1_align,
            p1_un_pdf,
            p1_debug,
            pdf_units,
            openxml_units,
            min_openxml_idx,
            trace_context=trace_context,
            page_sequence_range=page_sequence_range,
        )
        self._append_two_pass_postprocess_debug(p1_debug, postprocess)

        return {
            'phase1_alignments': postprocess['phase1_alignments'],
            'final_alignments': postprocess['final_alignments'],
            'shape_alignments': [],
            'unaligned_after_phase1': postprocess['unaligned_after_phase1'],
            'unaligned_final': postprocess['unaligned_final'],
            'unaligned_openxml': postprocess['page_unaligned_openxml'],
            'debug_info': p1_debug,
            'max_openxml_idx': postprocess['max_idx'],
        }
