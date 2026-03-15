import os
from typing import List, Dict, Any

from database import SessionLocal
from services.alignment import (
    AlignmentOpenXmlMixin,
    AlignmentPreprocessMixin,
    AlignmentMatchingMixin,
    AlignmentPostprocessMixin,
)


class AlignmentService(
    AlignmentOpenXmlMixin,
    AlignmentPreprocessMixin,
    AlignmentMatchingMixin,
    AlignmentPostprocessMixin
):
    TRACE_DIR = 'logs'
    TRACE_PREFIX = 'alignment_trace'
    MATCHED_UNIT_MAX_ITEM_GAP = 10
    MATCHED_UNIT_MIN_CLUSTER_SIZE = 2
    LINE_OVERLAP_MIN_RATIO = 0.30

    def __init__(self):
        pass

    @staticmethod
    def _read_positive_int_env(env_name: str, default_value: int) -> int:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else default_value
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _read_float_env(env_name: str, default_value: float, min_value: float = None, max_value: float = None) -> float:
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return default_value
        if min_value is not None:
            parsed = max(min_value, parsed)
        if max_value is not None:
            parsed = min(max_value, parsed)
        return parsed

    @staticmethod
    def _is_env_enabled_default_true(env_name: str) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _safe_int(value, default=None):
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_float(value, default=0.0):
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _resolve_next_page_pointer(self, current_min_openxml_idx: int, alignment_result: Dict[str, Any], pointer_state: Dict[str, Any] = None) -> Dict[str, Any]:
        current_min_openxml_idx = max(0, int(current_min_openxml_idx or 0))
        pointer_state = dict(pointer_state or {})
        jump_lock_streak = self._safe_int(pointer_state.get('jump_lock_streak'), 0) or 0
        pointer_freeze_streak = self._safe_int(pointer_state.get('pointer_freeze_streak'), 0) or 0
        last_stable_pointer_max = self._safe_int(pointer_state.get('last_stable_pointer_max'), 0)
        if last_stable_pointer_max is None:
            last_stable_pointer_max = 0

        page_debug = alignment_result.get('page_debug')
        if not isinstance(page_debug, dict):
            page_debug = {}
            alignment_result['page_debug'] = page_debug

        requested_max_openxml_idx = self._safe_int(alignment_result.get('max_openxml_idx'))
        if not alignment_result.get('success') or requested_max_openxml_idx is None:
            return {
                'next_min_openxml_idx': current_min_openxml_idx,
                'effective_max_openxml_idx': current_min_openxml_idx,
                'pointer_state': {
                    'jump_lock_streak': jump_lock_streak,
                    'pointer_freeze_streak': pointer_freeze_streak,
                    'last_stable_pointer_max': last_stable_pointer_max,
                },
                'page_debug': page_debug,
            }

        initial_cpb_ratio = self._safe_float(
            page_debug.get('initial_pass_cross_page_backward_skip_ratio'),
            self._safe_float(page_debug.get('cross_page_backward_skip_ratio'), 0.0)
        )
        cpb_ratio = initial_cpb_ratio
        pass1_coverage = self._safe_float(page_debug.get('pass1_match_coverage'), 0.0)
        pass1_openxml_diversity = self._safe_float(page_debug.get('pass1_openxml_diversity'), 1.0)
        pass1_matched_openxml_units = self._safe_int(page_debug.get('pass1_matched_openxml_units'))
        pass1_retry_attempts = self._safe_int(page_debug.get('pass1_retry_attempts'), 0) or 0
        pass1_selected_attempt = self._safe_int(page_debug.get('pass1_selected_attempt'), 0) or 0
        pass1_retry_used = bool(page_debug.get('pass1_retry_used'))
        pass1_pointer_source = str(page_debug.get('pass1_pointer_source') or 'unknown')
        pass1_pointer_cluster_max = self._safe_int(page_debug.get('pass1_pointer_cluster_max'))
        if pass1_pointer_cluster_max is None:
            pass1_pointer_cluster_max = requested_max_openxml_idx

        median_skip_idx = self._safe_int(
            page_debug.get('initial_pass_median_cross_page_skip_openxml_idx'),
            self._safe_int(page_debug.get('median_cross_page_skip_openxml_idx'))
        )
        early_skip_count = self._safe_int(
            page_debug.get('initial_pass_early_cross_page_skip_count'),
            self._safe_int(page_debug.get('early_cross_page_skip_count'), 0)
        ) or 0
        early_skip_ratio = self._safe_float(
            page_debug.get('initial_pass_early_cross_page_skip_ratio'),
            self._safe_float(page_debug.get('early_cross_page_skip_ratio'), 0.0)
        )

        clamp_cpb_ratio = self._read_float_env(
            'ALIGNMENT_JUMP_CLAMP_CPB_RATIO',
            0.9,
            min_value=0.0,
            max_value=1.0
        )
        clamp_min_coverage = self._read_float_env(
            'ALIGNMENT_JUMP_CLAMP_MIN_COVERAGE',
            0.25,
            min_value=0.0,
            max_value=1.0
        )
        clamp_min_openxml_diversity = self._read_float_env(
            'ALIGNMENT_JUMP_CLAMP_MIN_OPENXML_DIVERSITY',
            0.35,
            min_value=0.0
        )
        max_openxml_jump = self._read_positive_int_env(
            'ALIGNMENT_MAX_OPENXML_JUMP_PER_PAGE',
            250
        )
        force_max_openxml_jump = self._is_env_enabled_default_true(
            'ALIGNMENT_FORCE_MAX_OPENXML_JUMP'
        )
        pointer_freeze_required_streak = self._read_positive_int_env(
            'ALIGNMENT_POINTER_FREEZE_REQUIRED_STREAK',
            2
        )
        pointer_freeze_min_early_skip_count = self._read_positive_int_env(
            'ALIGNMENT_POINTER_FREEZE_MIN_EARLY_SKIP_COUNT',
            8
        )
        pointer_freeze_min_skip_gap = self._read_positive_int_env(
            'ALIGNMENT_POINTER_FREEZE_MIN_SKIP_GAP',
            24
        )
        pointer_freeze_min_early_skip_ratio = self._read_float_env(
            'ALIGNMENT_POINTER_FREEZE_MIN_EARLY_SKIP_RATIO',
            0.5,
            min_value=0.0,
            max_value=1.0
        )
        pointer_freeze_min_cpb_ratio = self._read_float_env(
            'ALIGNMENT_POINTER_FREEZE_MIN_CPB_RATIO',
            0.4,
            min_value=0.0,
            max_value=1.0
        )
        pointer_freeze_force_early_skip_count = self._read_positive_int_env(
            'ALIGNMENT_POINTER_FREEZE_FORCE_EARLY_SKIP_COUNT',
            32
        )

        requested_jump = max(0, requested_max_openxml_idx - current_min_openxml_idx)
        jump_clamp_lock_signal = (
            cpb_ratio >= clamp_cpb_ratio and
            (
                pass1_coverage <= clamp_min_coverage or
                pass1_openxml_diversity <= clamp_min_openxml_diversity
            )
        )
        if jump_clamp_lock_signal:
            jump_lock_streak += 1
        else:
            jump_lock_streak = 0

        pointer_freeze_signal = (
            early_skip_count >= pointer_freeze_min_early_skip_count and
            early_skip_ratio >= pointer_freeze_min_early_skip_ratio and
            median_skip_idx is not None and
            (
                cpb_ratio >= pointer_freeze_min_cpb_ratio or
                early_skip_count >= pointer_freeze_force_early_skip_count
            ) and
            median_skip_idx < (current_min_openxml_idx - pointer_freeze_min_skip_gap)
        )
        if pointer_freeze_signal:
            pointer_freeze_streak += 1
        else:
            pointer_freeze_streak = 0

        jump_clamp_failed_retry_signal = (
            jump_clamp_lock_signal and
            pass1_retry_attempts > 0 and
            pass1_selected_attempt == 0
        )
        jump_clamp_retry_jump_signal = (
            pass1_retry_used and
            requested_jump > max_openxml_jump
        )
        jump_clamp_lock_streak_signal = (
            jump_lock_streak > 0 and
            requested_jump > max_openxml_jump
        )
        jump_clamp_hard_cap_signal = (
            force_max_openxml_jump and
            requested_jump > max_openxml_jump
        )
        jump_clamp_suspicious = (
            requested_jump > max_openxml_jump and
            (
                jump_clamp_hard_cap_signal or
                jump_clamp_failed_retry_signal or
                jump_clamp_retry_jump_signal or
                jump_clamp_lock_streak_signal or
                pointer_freeze_signal
            )
        )

        pointer_freeze_target = max(0, int(last_stable_pointer_max or 0))
        pointer_freeze_applied = (
            pointer_freeze_signal and
            pointer_freeze_streak >= pointer_freeze_required_streak and
            pointer_freeze_target < current_min_openxml_idx
        )

        effective_max_openxml_idx = requested_max_openxml_idx
        jump_clamp_applied = False
        if pointer_freeze_applied:
            effective_max_openxml_idx = pointer_freeze_target
            jump_clamp_applied = effective_max_openxml_idx != current_min_openxml_idx
        elif jump_clamp_suspicious:
            allowed_max_openxml_idx = current_min_openxml_idx + max_openxml_jump
            if effective_max_openxml_idx > allowed_max_openxml_idx:
                effective_max_openxml_idx = allowed_max_openxml_idx
                jump_clamp_applied = True

        if pointer_freeze_applied:
            next_min_openxml_idx = effective_max_openxml_idx
        else:
            next_min_openxml_idx = max(current_min_openxml_idx, effective_max_openxml_idx)

        stable_page_signal = (
            not pointer_freeze_signal and
            not jump_clamp_lock_signal and
            pass1_pointer_source == 'stable_cluster'
        )
        if stable_page_signal:
            last_stable_pointer_max = max(0, int(effective_max_openxml_idx))

        page_debug['jump_clamp_suspicious'] = jump_clamp_suspicious
        page_debug['jump_clamp_applied'] = jump_clamp_applied
        page_debug['jump_clamp_requested_max_openxml_idx'] = requested_max_openxml_idx
        page_debug['jump_clamp_final_max_openxml_idx'] = effective_max_openxml_idx
        page_debug['jump_clamp_max_openxml_jump'] = max_openxml_jump
        page_debug['jump_clamp_requested_jump'] = requested_jump
        page_debug['jump_clamp_lock_signal'] = jump_clamp_lock_signal
        page_debug['jump_clamp_lock_streak'] = jump_lock_streak
        page_debug['jump_clamp_failed_retry_signal'] = jump_clamp_failed_retry_signal
        page_debug['jump_clamp_retry_jump_signal'] = jump_clamp_retry_jump_signal
        page_debug['jump_clamp_lock_streak_signal'] = jump_clamp_lock_streak_signal
        page_debug['jump_clamp_hard_cap_signal'] = jump_clamp_hard_cap_signal
        page_debug['jump_clamp_force_enabled'] = force_max_openxml_jump
        page_debug['jump_clamp_freeze_signal'] = pointer_freeze_signal
        page_debug['jump_clamp_pointer_freeze_signal'] = pointer_freeze_signal
        page_debug['jump_clamp_pointer_freeze_applied'] = pointer_freeze_applied
        page_debug['jump_clamp_pointer_freeze_streak'] = pointer_freeze_streak
        page_debug['jump_clamp_pointer_freeze_required_streak'] = pointer_freeze_required_streak
        page_debug['jump_clamp_pointer_freeze_target'] = pointer_freeze_target
        page_debug['jump_clamp_pointer_freeze_min_skip_gap'] = pointer_freeze_min_skip_gap
        page_debug['jump_clamp_pointer_freeze_min_early_skip_count'] = pointer_freeze_min_early_skip_count
        page_debug['jump_clamp_pointer_freeze_min_early_skip_ratio'] = pointer_freeze_min_early_skip_ratio
        page_debug['jump_clamp_pointer_freeze_min_cpb_ratio'] = pointer_freeze_min_cpb_ratio
        page_debug['jump_clamp_pointer_freeze_force_early_skip_count'] = pointer_freeze_force_early_skip_count
        page_debug['jump_clamp_last_stable_pointer'] = last_stable_pointer_max
        page_debug['jump_clamp_effective_max_openxml_idx'] = effective_max_openxml_idx
        page_debug['jump_clamp_next_min_openxml_idx'] = next_min_openxml_idx
        page_debug['jump_clamp_pass1_pointer_cluster_max'] = pass1_pointer_cluster_max
        page_debug['jump_clamp_pass1_matched_openxml_units'] = pass1_matched_openxml_units
        page_debug['jump_clamp_lock_metrics_source'] = (
            'initial_pass' if 'initial_pass_cross_page_backward_skip_ratio' in page_debug else 'selected_candidate'
        )

        return {
            'next_min_openxml_idx': next_min_openxml_idx,
            'effective_max_openxml_idx': effective_max_openxml_idx,
            'pointer_state': {
                'jump_lock_streak': jump_lock_streak,
                'pointer_freeze_streak': pointer_freeze_streak,
                'last_stable_pointer_max': last_stable_pointer_max,
            },
            'page_debug': page_debug,
        }

    def align_document(self, extraction_results: List[Dict], doc_id: int, ref_tipe: str = 'dokumen') -> List[Dict]:
        """
        Align all pages of a document.

        Args:
            extraction_results: List of extraction results per page
            doc_id: Reference ID (dokumen_id or bab_id)
            ref_tipe: Reference type ('dokumen' or 'bab'; legacy 'buku' is also supported)

        Returns:
            List of alignment results per page
        """
        results = []
        min_openxml_idx = 0
        pointer_state = {
            'jump_lock_streak': 0,
            'pointer_freeze_streak': 0,
            'last_stable_pointer_max': 0,
        }

        for page_data in extraction_results:
            page_num = page_data.get('page', 1)
            items = page_data.get('items', [])
            page_width = page_data.get('page_width', 595)
            page_height = page_data.get('page_height', 842)
            total_pages = len(extraction_results)

            result = self.align(
                doc_id, page_num, items,
                page_width, page_height, total_pages,
                min_openxml_idx,
                ref_tipe=ref_tipe
            )

            effective_max_openxml_idx = result.get('max_openxml_idx', 0)
            if result.get('success'):
                pointer_update = self._resolve_next_page_pointer(
                    min_openxml_idx,
                    result,
                    pointer_state=pointer_state
                )
                min_openxml_idx = pointer_update['next_min_openxml_idx']
                pointer_state = pointer_update['pointer_state']
                effective_max_openxml_idx = pointer_update['effective_max_openxml_idx']

            results.append({
                'success': result.get('success', False),
                'page': page_num,
                'alignments': result.get('alignments', []),
                'unaligned_pdf_units': result.get('unaligned_pdf_units', []),
                'header_footer_units': result.get('header_footer_units', []),
                'max_openxml_idx': effective_max_openxml_idx,
                'page_debug': result.get('page_debug', {}),
                'stats': {
                    'aligned_count': len(result.get('alignments', [])),
                    'unaligned_count': len(result.get('unaligned_pdf_units', []))
                }
            })

        return results

    def align(self, doc_id: int, page_num: int, extraction_items: List[Dict],
              page_width: float, page_height: float, total_pages: int,
              min_openxml_idx: int = 0, ref_tipe: str = 'dokumen') -> Dict[str, Any]:
        """
        Main entry point for alignment.
        Orchestrates extraction flattening, OpenXML retrieval, alignment, and full post-processing.
        Matches `api_merging_alignment` + `perform_two_pass_alignment` from legacy.
        """
        db = SessionLocal()
        try:
            # 1. Get Section Data for margin logic
            sections = self._get_doc_sections(db, doc_id, ref_tipe=ref_tipe)
            current_section = self._get_section_for_page(sections, page_width, page_height)
            if not current_section and sections:
                current_section = sections[0]

            # Build section data (matches legacy /dokumen-elemen-api/sections payload)
            section_data = None
            if current_section:
                twips_per_point = 20
                section_data = {
                    'dsec_id': current_section.dsec_id,
                    'dsec_index': current_section.dsec_index,
                    'page_width_twips': current_section.dsec_page_width_twips,
                    'page_height_twips': current_section.dsec_page_height_twips,
                    'page_width_pt': current_section.dsec_page_width_twips / twips_per_point if current_section.dsec_page_width_twips else None,
                    'page_height_pt': current_section.dsec_page_height_twips / twips_per_point if current_section.dsec_page_height_twips else None,
                    'orientation': current_section.dsec_orientation,
                    'margin_top_twips': current_section.dsec_margin_top_twips,
                    'margin_bottom_twips': current_section.dsec_margin_bottom_twips,
                    'margin_left_twips': current_section.dsec_margin_left_twips,
                    'margin_right_twips': current_section.dsec_margin_right_twips,
                    'margin_top_pt': current_section.dsec_margin_top_twips / twips_per_point if current_section.dsec_margin_top_twips else None,
                    'margin_bottom_pt': current_section.dsec_margin_bottom_twips / twips_per_point if current_section.dsec_margin_bottom_twips else None,
                    'margin_left_pt': current_section.dsec_margin_left_twips / twips_per_point if current_section.dsec_margin_left_twips else None,
                    'margin_right_pt': current_section.dsec_margin_right_twips / twips_per_point if current_section.dsec_margin_right_twips else None,
                    'header_margin_twips': current_section.dsec_header_margin_twips,
                    'footer_margin_twips': current_section.dsec_footer_margin_twips,
                    'header_margin_pt': current_section.dsec_header_margin_twips / twips_per_point if current_section.dsec_header_margin_twips else None,
                    'footer_margin_pt': current_section.dsec_footer_margin_twips / twips_per_point if current_section.dsec_footer_margin_twips else None,
                    'gutter_twips': current_section.dsec_gutter_twips,
                    'gutter_position': current_section.dsec_gutter_position
                }

            # 2. Flatten Extraction Items (PDF Units)
            all_pdf_units = self._flatten_extraction_items(extraction_items)

            # 3. Filter Header/Footer units
            pdf_units, header_footer_units = self._filter_header_footer_items(all_pdf_units, current_section, page_height)

            # 4. Get OpenXML elements (body parts only)
            elements = self._get_openxml_elements(db, doc_id, ref_tipe=ref_tipe)

            # 5. Build OpenXML Units (with image numbering logic)
            # Estimate sequence range for page to number images correctly
            page_sequence_range = self._estimate_page_sequence_range(elements, page_num, total_pages)
            openxml_units, table_debug = self._build_openxml_units(
                elements,
                page_sequence_range,
                db_session=db
            )

            # 6. Perform Two-Pass Alignment (Feature Complete)
            trace_context = {'doc_id': doc_id, 'ref_tipe': ref_tipe, 'page_num': page_num}
            alignment_result = self._perform_two_pass_alignment(
                pdf_units,
                openxml_units,
                min_openxml_idx,
                trace_context=trace_context,
                page_sequence_range=page_sequence_range
            )

            # Add table debug info to page debug
            alignment_result['debug_info']['table_processing'] = table_debug
            alignment_result['debug_info']['section_data'] = section_data

            return {
                'success': True,
                'phase1_alignments': alignment_result['phase1_alignments'],
                'alignments': alignment_result['final_alignments'],
                'final_alignments': alignment_result['final_alignments'],
                'shape_alignments': alignment_result['shape_alignments'],
                'unaligned_pdf_units': [pdf_units[i] for i in alignment_result['unaligned_final'] if i < len(pdf_units)],
                'unaligned_pdf_units_phase1': alignment_result['unaligned_after_phase1'],
                'unaligned_openxml_units': self._format_unaligned_openxml(openxml_units, alignment_result['unaligned_openxml']),
                'header_footer_units': header_footer_units,
                'max_openxml_idx': alignment_result.get('max_openxml_idx', 0),
                'page_debug': alignment_result['debug_info']
            }
        finally:
            db.close()

    def _normalize_text(self, text):
        if not text: return ''
        result = []
        for char in text:
            code = ord(char)
            normalized = None
            if 0x1D400 <= code <= 0x1D419: normalized = chr(ord('A') + (code - 0x1D400))
            elif 0x1D41A <= code <= 0x1D433: normalized = chr(ord('a') + (code - 0x1D41A))
            elif 0x1D434 <= code <= 0x1D44D: normalized = chr(ord('A') + (code - 0x1D434))
            elif 0x1D44E <= code <= 0x1D467:
                 if code == 0x1D455: normalized = 'h'
                 else: normalized = chr(ord('a') + (code - 0x1D44E))
            elif 0x1D468 <= code <= 0x1D49B: normalized = chr(ord('A') + (code - 0x1D468)) if code <= 0x1D481 else chr(ord('a') + (code - 0x1D482))
            elif 0x1D49C <= code <= 0x1D4CF: normalized = chr(ord('A') + (code - 0x1D49C)) if code <= 0x1D4B5 else chr(ord('a') + (code - 0x1D4B6))
            elif 0x1D4D0 <= code <= 0x1D503: normalized = chr(ord('A') + (code - 0x1D4D0)) if code <= 0x1D4E9 else chr(ord('a') + (code - 0x1D4EA))
            elif 0x1D504 <= code <= 0x1D537: normalized = chr(ord('A') + (code - 0x1D504)) if code <= 0x1D51C else chr(ord('a') + (code - 0x1D51E))
            elif 0x1D538 <= code <= 0x1D56B: normalized = chr(ord('A') + (code - 0x1D538)) if code <= 0x1D550 else chr(ord('a') + (code - 0x1D552))
            elif 0x1D56C <= code <= 0x1D59F: normalized = chr(ord('A') + (code - 0x1D56C)) if code <= 0x1D585 else chr(ord('a') + (code - 0x1D586))
            elif 0x1D5A0 <= code <= 0x1D5D3: normalized = chr(ord('A') + (code - 0x1D5A0)) if code <= 0x1D5B9 else chr(ord('a') + (code - 0x1D5BA))
            elif 0x1D5D4 <= code <= 0x1D607: normalized = chr(ord('A') + (code - 0x1D5D4)) if code <= 0x1D5ED else chr(ord('a') + (code - 0x1D5EE))
            elif 0x1D608 <= code <= 0x1D63B: normalized = chr(ord('A') + (code - 0x1D608)) if code <= 0x1D621 else chr(ord('a') + (code - 0x1D622))
            elif 0x1D63C <= code <= 0x1D66F: normalized = chr(ord('A') + (code - 0x1D63C)) if code <= 0x1D655 else chr(ord('a') + (code - 0x1D656))
            elif 0x1D670 <= code <= 0x1D6A3: normalized = chr(ord('A') + (code - 0x1D670)) if code <= 0x1D689 else chr(ord('a') + (code - 0x1D68A))
            elif 0x1D6A8 <= code <= 0x1D6E1: normalized = chr(0x0391 + (code - 0x1D6A8)) if code <= 0x1D6C0 else chr(0x03B1 + (code - 0x1D6C2))
            elif 0x1D6E2 <= code <= 0x1D71B: normalized = chr(0x0391 + (code - 0x1D6E2)) if code <= 0x1D6FA else chr(0x03B1 + (code - 0x1D6FC))
            elif 0x1D71C <= code <= 0x1D755: normalized = chr(0x0391 + (code - 0x1D71C)) if code <= 0x1D734 else chr(0x03B1 + (code - 0x1D736))
            elif 0x1D756 <= code <= 0x1D78F: normalized = chr(0x0391 + (code - 0x1D756)) if code <= 0x1D76E else chr(0x03B1 + (code - 0x1D770))
            elif 0x1D790 <= code <= 0x1D7C9: normalized = chr(0x0391 + (code - 0x1D790)) if code <= 0x1D7A8 else chr(0x03B1 + (code - 0x1D7AA))
            elif code in [0x1D715, 0x1D6DB, 0x1D74F, 0x1D789, 0x1D7C3, 0x2202]: normalized = 'âˆ‚'
            elif 0x1D7CE <= code <= 0x1D7FF: normalized = chr(ord('0') + (code - 0x1D7CE)) if code <= 0x1D7D7 else chr(ord('0') + (code - 0x1D7D8)) if code <= 0x1D7E1 else chr(ord('0') + (code - 0x1D7E2)) if code <= 0x1D7EB else chr(ord('0') + (code - 0x1D7EC)) if code <= 0x1D7F5 else chr(ord('0') + (code - 0x1D7F6))
            elif code == 0x2070: normalized = '0'
            elif code == 0x00B9: normalized = '1'
            elif code == 0x00B2: normalized = '2'
            elif code == 0x00B3: normalized = '3'
            elif 0x2074 <= code <= 0x2079: normalized = chr(ord('0') + (code - 0x2070))
            # Superscript letters
            elif code == 0x1D43: normalized = 'a'   # áµƒ
            elif code == 0x1D47: normalized = 'b'   # áµ‡
            elif code == 0x1D9C: normalized = 'c'   # á¶œ
            elif code == 0x1D48: normalized = 'd'   # áµˆ
            elif code == 0x1D49: normalized = 'e'   # áµ‰
            elif code == 0x1DA0: normalized = 'f'   # á¶ 
            elif code == 0x1D4D: normalized = 'g'   # áµ
            elif code == 0x02B0: normalized = 'h'   # Ê°
            elif code == 0x2071: normalized = 'i'   # â±
            elif code == 0x02B2: normalized = 'j'   # Ê²
            elif code == 0x1D4F: normalized = 'k'   # áµ
            elif code == 0x02E1: normalized = 'l'   # Ë¡
            elif code == 0x1D50: normalized = 'm'   # áµ
            elif code == 0x207F: normalized = 'n'   # â¿
            elif code == 0x1D52: normalized = 'o'   # áµ’
            elif code == 0x1D56: normalized = 'p'   # áµ–
            elif code == 0x02B3: normalized = 'r'   # Ê³
            elif code == 0x02E2: normalized = 's'   # Ë¢
            elif code == 0x1D57: normalized = 't'   # áµ—
            elif code == 0x1D58: normalized = 'u'   # áµ˜
            elif code == 0x1D5B: normalized = 'v'   # áµ›
            elif code == 0x02B7: normalized = 'w'   # Ê·
            elif code == 0x02E3: normalized = 'x'   # Ë£
            elif code == 0x02B8: normalized = 'y'   # Ê¸
            elif code == 0x1DBB: normalized = 'z'   # á¶»
            # Subscript digits
            elif 0x2080 <= code <= 0x2089: normalized = chr(ord('0') + (code - 0x2080))
            # Subscript letters
            elif code == 0x2090: normalized = 'a'   # â‚
            elif code == 0x2091: normalized = 'e'   # â‚‘
            elif code == 0x2095: normalized = 'h'   # â‚•
            elif code == 0x1D62: normalized = 'i'   # áµ¢
            elif code == 0x2C7C: normalized = 'j'   # â±¼
            elif code == 0x2096: normalized = 'k'   # â‚–
            elif code == 0x2097: normalized = 'l'   # â‚—
            elif code == 0x2098: normalized = 'm'   # â‚˜
            elif code == 0x2099: normalized = 'n'   # â‚™
            elif code == 0x2092: normalized = 'o'   # â‚’
            elif code == 0x209A: normalized = 'p'   # â‚š
            elif code == 0x1D63: normalized = 'r'   # áµ£
            elif code == 0x209B: normalized = 's'   # â‚›
            elif code == 0x209C: normalized = 't'   # â‚œ
            elif code == 0x1D64: normalized = 'u'   # áµ¤
            elif code == 0x1D65: normalized = 'v'   # áµ¥
            elif code == 0x2093: normalized = 'x'   # â‚“
            # Math operators
            elif char in 'âˆ’â€“â€”â€â€‘â€’â€•': normalized = '-'
            elif char in 'Ã—âˆ™Â·â€¢â‹…': normalized = '*'
            elif char in 'Ã·âˆ•': normalized = '/'
            elif char == 'Â±': normalized = '+-'
            elif char == 'âˆ“': normalized = '-+'
            elif char in 'ï¼â¼â‚Œ': normalized = '='
            elif char in 'ï¼œâ€¹ã€ˆâŸ¨': normalized = '<'
            elif char in 'ï¼žâ€ºã€‰âŸ©': normalized = '>'
            elif char in 'â‰¤â‰¦â©½': normalized = '<='
            elif char in 'â‰¥â‰§â©¾': normalized = '>='
            elif char in 'â†’â†â†‘â†“â†”â†•â‡’â‡â‡‘â‡“â‡”': normalized = ''
            elif char in 'â€²': normalized = "'"
            elif char in 'â€³': normalized = "''"
            elif char == 'Â½': normalized = '1/2'
            elif char == 'â…“': normalized = '1/3'
            elif char == 'Â¼': normalized = '1/4'
            elif char == 'â…”': normalized = '2/3'
            elif char == 'Â¾': normalized = '3/4'
            elif 0xFF01 <= code <= 0xFF5E: normalized = chr(code - 0xFF00 + 0x20)

            result.append(normalized if normalized is not None else char)
        return ''.join(''.join(result).lower().split())
