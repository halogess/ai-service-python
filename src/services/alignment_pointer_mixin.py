import os
from typing import List, Dict, Any

from database import SessionLocal
from services.alignment import (
    AlignmentOpenXmlMixin,
    AlignmentPreprocessMixin,
    AlignmentMatchingMixin,
    AlignmentPostprocessMixin,
)


class AlignmentPointerMixin:


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
