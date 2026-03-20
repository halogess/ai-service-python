import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingMixin:
    MARKER_ONLY_TEXT_RE = re.compile(r'^\s*\d+(?:\.\d+)*\s*[:.)]?\s*$')
    PROGRAM_SEGMENT_HEADING_RE = re.compile(
        r'\bsegmen\s*program\s*\d+(?:\.\d+)*(?:\s*\(\s*lanjutan\s*\))?',
        re.IGNORECASE,
    )
    STRUCTURED_BLOCK_HEADING_RE = re.compile(
        r'\b(?P<kind>segmen\s*program|algoritma)\s*'
        r'(?P<number>\d+(?:\.\d+)*)'
        r'(?P<continuation>\s*\(\s*lanjutan\s*\))?',
        re.IGNORECASE,
    )
    CODE_LINE_NUMBER_RE = re.compile(r'^\s*\d{1,3}\s*[:.)]\s*')
    CODE_TEXT_HINT_RE = re.compile(
        r'\b(?:const|let|var|function|return|if|else|await|async|class|'
        r'import|from|public|private|static|void|final|map|jsondecode|http|get|post|'
        r'emit|console|socket|response|statuscode)\b',
        re.IGNORECASE,
    )

    @staticmethod
    def _is_env_enabled_default_true(env_name):
        value = os.getenv(env_name)
        if value is None:
            return True
        return str(value).strip().lower() not in ("0", "false", "no", "off")

    @staticmethod
    def _read_positive_int_env(env_name, default_value):
        value = os.getenv(env_name)
        if value is None:
            return default_value
        try:
            parsed = int(str(value).strip())
            return parsed if parsed > 0 else default_value
        except (TypeError, ValueError):
            return default_value

    @staticmethod
    def _read_float_env(env_name, default_value, min_value=None, max_value=None):
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
    def _collect_matched_pdf_unit_keys(alignments):
        keys = set()

        def add_unit(unit):
            if not isinstance(unit, dict):
                return
            item_idx = unit.get('item_idx')
            if item_idx is not None:
                keys.add(('item_idx', item_idx))
                return
            pdf_unit_id = unit.get('pdf_unit_id') or unit.get('unit_id')
            if pdf_unit_id:
                keys.add(('pdf_unit_id', str(pdf_unit_id)))

        for alignment in alignments or []:
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    for matched_unit in cell.get('matched_pdf_units') or []:
                        add_unit(matched_unit)
            else:
                for matched_unit in alignment.get('matched_pdf_units') or []:
                    add_unit(matched_unit)
        return keys

    @classmethod
    def _count_matched_pdf_units(cls, alignments):
        return len(cls._collect_matched_pdf_unit_keys(alignments))

    @staticmethod
    def _collect_matched_openxml_indices(alignments):
        indices = set()

        def add_index(value):
            if value is None:
                return
            try:
                indices.add(int(value))
            except (TypeError, ValueError):
                return

        for alignment in alignments or []:
            add_index(alignment.get('openxml_idx'))
            for openxml_idx in alignment.get('openxml_indices') or []:
                add_index(openxml_idx)
            if alignment.get('is_table') and alignment.get('cells'):
                for cell in alignment.get('cells') or []:
                    add_index(cell.get('openxml_idx'))
        return indices

    @classmethod
    def _count_matched_openxml_units(cls, alignments):
        return len(cls._collect_matched_openxml_indices(alignments))

    @classmethod
    def _compute_match_coverage(cls, alignments, total_pdf_units):
        if total_pdf_units <= 0:
            return 0.0
        matched = cls._count_matched_pdf_units(alignments)
        return min(1.0, matched / total_pdf_units)

    @classmethod
    def _compute_openxml_diversity(cls, alignments):
        matched_pdf_units = cls._count_matched_pdf_units(alignments)
        if matched_pdf_units <= 0:
            return 0.0
        matched_openxml_units = cls._count_matched_openxml_units(alignments)
        return matched_openxml_units / matched_pdf_units

    @staticmethod
    def _try_parse_int(value):
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_pointer_text(text):
        if not text:
            return ''
        return re.sub(r'\s+', '', str(text).strip().lower())

    def _is_program_segment_heading_text(self, text):
        heading = self._extract_structured_block_heading(
            text,
            allowed_kinds={'segmen program'},
        )
        return bool(heading)

    def _extract_structured_block_heading(self, text, allowed_kinds=None):
        if not text:
            return None
        match = self.STRUCTURED_BLOCK_HEADING_RE.search(str(text))
        if not match:
            return None
        kind = re.sub(r'\s+', ' ', str(match.group('kind') or '').strip().lower())
        if allowed_kinds and kind not in set(allowed_kinds):
            return None
        number = str(match.group('number') or '').strip()
        if not number:
            return None
        continuation_raw = str(match.group('continuation') or '')
        return {
            'kind': kind,
            'number': number,
            'key': f"{kind}:{number}",
            'is_continuation': 'lanjutan' in continuation_raw.lower(),
            'text': str(text),
        }


    def _is_code_like_openxml_unit(self, unit):
        if not isinstance(unit, dict):
            return False
        if unit.get('is_code_like_openxml') or unit.get('is_code_font') or unit.get('is_code_style'):
            return True
        elem_type = str(unit.get('elem_type') or '').strip().lower()
        if 'list-item' in elem_type or elem_type == 'code':
            return True
        return self._looks_like_code_line_text(
            unit.get('text') or unit.get('text_normalized')
        )

    def _looks_like_code_line_text(self, text):
        text = str(text or '').strip()
        if not text:
            return False
        if self.CODE_LINE_NUMBER_RE.match(text):
            return True
        if self.CODE_TEXT_HINT_RE.search(text):
            return True
        symbol_count = sum(1 for ch in text if ch in '{}[]();=<>:+-*/%#\\')
        return symbol_count >= 3

    def _count_code_like_pdf_units(self, pdf_units):
        count = 0
        for unit in pdf_units or []:
            if not isinstance(unit, dict):
                continue
            if unit.get('is_cell'):
                continue
            if unit.get('item_type') in {'table', 'hline_table', 'shape', 'image'}:
                continue
            if self._looks_like_code_line_text(unit.get('text') or unit.get('text_normalized')):
                count += 1
        return count

    def _candidate_context_has_program_heading(self, pdf_units, candidate_context):
        for unit in pdf_units or []:
            if self._is_program_segment_heading_text(unit.get('text') or unit.get('text_normalized')):
                return True
        for hit in (candidate_context or {}).get('anchor_hits') or []:
            if self._is_program_segment_heading_text(hit.get('pdf_text')) or self._is_program_segment_heading_text(hit.get('openxml_text')):
                return True
        return False

    def _should_use_program_segment_local_band(self, pdf_units, candidate_context):
        if not candidate_context or not candidate_context.get('indices'):
            return False
        if candidate_context.get('source') not in {'sequence_anchor_cluster', 'page_sequence_range'}:
            return False
        if not self._candidate_context_has_program_heading(pdf_units, candidate_context):
            return False
        min_code_like_lines = self._read_positive_int_env(
            'ALIGNMENT_PROGRAM_SEGMENT_MIN_CODE_LINES',
            8,
        )
        return self._count_code_like_pdf_units(pdf_units) >= min_code_like_lines

    def _collect_alignment_pdf_text(self, alignment):
        if not isinstance(alignment, dict):
            return ''

        matched_units = []
        if alignment.get('is_table') and alignment.get('cells'):
            for cell in alignment.get('cells') or []:
                matched_units.extend((cell or {}).get('matched_pdf_units') or [])
        else:
            matched_units = list(alignment.get('matched_pdf_units') or [])

        matched_units.sort(key=lambda unit: unit.get('item_idx', 10**9))
        text_parts = []
        for unit in matched_units:
            text = str((unit or {}).get('text') or '').strip()
            if text:
                text_parts.append(text)
        return ' '.join(text_parts).strip()

    def _alignment_has_figure_key_mismatch(self, alignment):
        if not isinstance(alignment, dict):
            return False

        openxml_text = alignment.get('element_text') or alignment.get('text') or ''
        pdf_text = self._collect_alignment_pdf_text(alignment)
        if not openxml_text or not pdf_text:
            return False

        openxml_key = None
        pdf_key = None
        if hasattr(self, '_extract_figure_key'):
            openxml_key = self._extract_figure_key(openxml_text)
            pdf_key = self._extract_figure_key(pdf_text)

        return bool(openxml_key and pdf_key and openxml_key != pdf_key)

    def _is_pointer_safe_alignment(self, alignment):
        if not isinstance(alignment, dict):
            return False
        if alignment.get('late_matched'):
            return False
        if alignment.get('is_synthetic_marker_repair'):
            return False
        if alignment.get('is_table') and alignment.get('cells'):
            return False
        if alignment.get('is_image_part') or alignment.get('is_openxml_visual_slot'):
            return False
        if self._alignment_has_figure_key_mismatch(alignment):
            return False

        if hasattr(self, '_is_caption_like_text'):
            element_text = alignment.get('element_text') or alignment.get('text') or ''
            if self._is_caption_like_text(element_text):
                support = self._compute_alignment_support_metrics(alignment)
                if support['matched_chars'] < 12 and support['match_ratio'] < 0.55:
                    return False

        return True

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
                })

        block_candidates.sort(key=lambda item: (-item['block_size'], -item['text_len'], item['local_idx']))
        return block_candidates[:max_anchors]

    def _compose_openxml_anchor_block(self, openxml_units, start_idx, max_units=2, max_chars=400):
        if not openxml_units or start_idx is None or start_idx < 0 or start_idx >= len(openxml_units):
            return {'text': '', 'text_normalized': '', 'end_idx': start_idx}

        text_parts = []
        end_idx = start_idx
        added_units = 0
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
                base_score = self._score_anchor_similarity(
                    anchor.get('text_normalized'),
                    openxml_block.get('text_normalized')
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
                min_distance = min(
                    abs((hit.get('openxml_idx') or 0) - int(min_openxml_idx or 0))
                    for hit in cluster
                ) if cluster else 10 ** 9
                return (
                    round(sum(hit.get('score', 0.0) for hit in cluster), 6),
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
            seq for seq in range(min_seq, max_seq + 1)
            if seq not in seq_to_alignment
            and seq in seq_to_openxml
            and self._is_marker_only_text(seq_to_openxml[seq][1].get('text', ''))
        ]
        if not missing_marker_seqs:
            return alignments

        created = []
        for missing_seq in missing_marker_seqs:
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

            openxml_idx, openxml_unit = seq_to_openxml[missing_seq]
            restored = {
                'element_id': openxml_unit['elem_id'],
                'element_sequence': openxml_unit['elem_seq'],
                'element_type': openxml_unit['elem_type'],
                'is_table': False,
                'is_synthetic_marker_repair': True,
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

    def _perform_char_alignment(
        self,
        pdf_units,
        openxml_units,
        min_openxml_idx=0,
        trace_context=None,
        page_sequence_range=None
    ):
        if not pdf_units or not openxml_units:
            return [], list(range(len(pdf_units))), list(range(len(openxml_units))), {
                'max_openxml_idx': min_openxml_idx,
                'unaligned_openxml_indices': list(range(len(openxml_units))),
                'cross_page_backward_skip_count': 0,
                'cross_page_backward_skip_ratio': 0.0
            }

        filter_by_seq_range = os.getenv("ALIGNMENT_FILTER_BY_SEQ_RANGE", "").lower() in ("1", "true", "yes", "on")
        seq_min = seq_max = None
        if filter_by_seq_range and page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range
        candidate_context = self._select_sequence_local_openxml_indices(
            pdf_units,
            openxml_units,
            min_openxml_idx=min_openxml_idx,
            page_sequence_range=page_sequence_range
        )
        suggested_openxml_indices = sorted(set(candidate_context.get('indices') or []))
        if filter_by_seq_range and seq_min is not None and seq_max is not None:
            candidate_openxml_indices = [
                idx for idx, unit in enumerate(openxml_units)
                if unit.get('elem_seq') is not None and seq_min <= unit.get('elem_seq') <= seq_max
            ]
        elif (
            suggested_openxml_indices
            and self._should_use_program_segment_local_band(pdf_units, candidate_context)
        ):
            # Repeated code blocks across "Segmen Program" sections are highly ambiguous
            # at full-document scope. On code-heavy pages with an explicit program heading,
            # prefer the page-local anchor band instead of letting identical lines match
            # against an earlier segment.
            candidate_openxml_indices = suggested_openxml_indices
        else:
            candidate_openxml_indices = list(range(len(openxml_units)))

        if not candidate_openxml_indices:
            candidate_openxml_indices = list(range(len(openxml_units)))

        pdf_concat = ''
        pdf_char_map = []
        pdf_unit_ranges = []

        for i in range(len(pdf_units)):
            u = pdf_units[i]
            text = u['text_normalized']
            start = len(pdf_concat)
            for _ in text:
                pdf_char_map.append(i)
            pdf_concat += text
            if text:
                pdf_unit_ranges.append({
                    'unit_idx': i,
                    'unit_id': u['unit_id'],
                    'start': start,
                    'end': len(pdf_concat),
                    'text': u['text'][:50],
                    'text_normalized': text[:50],
                    'item_type': u['item_type']
                })

        openxml_concat = ''
        openxml_char_map = []
        openxml_unit_ranges = []

        for i in candidate_openxml_indices:
            u = openxml_units[i]
            text = u['text_normalized']
            start = len(openxml_concat)
            for _ in text:
                openxml_char_map.append(i)
            openxml_concat += text
            if text:
                openxml_unit_ranges.append({
                    'unit_idx': i,
                    'unit_id': u['unit_id'],
                    'start': start,
                    'end': len(openxml_concat),
                    'text': u['text'][:50],
                    'text_normalized': text[:50],
                    'elem_type': u['elem_type']
                })

        sm = difflib.SequenceMatcher(None, pdf_concat, openxml_concat, autojunk=False)
        matching_blocks = sm.get_matching_blocks()
        sorted_blocks = sorted(matching_blocks, key=lambda x: x.b)

        # Log gap analysis to file (legacy behavior)
        with open('gap_analysis.log', 'w', encoding='utf-8') as gap_log:
            gap_log.write("=" * 80 + "\n")
            gap_log.write("GAP ANALYSIS - What OpenXML content is NOT being matched\n")
            gap_log.write("=" * 80 + "\n\n")

            prev_end_ox = 0
            for i, block in enumerate(sorted_blocks):
                if block.size == 0:
                    continue
                gap = block.b - prev_end_ox
                if gap > 50:
                    gap_log.write(f"\n[GAP {i}] OX positions {prev_end_ox} to {block.b} (size: {gap} chars)\n")
                    gap_content = openxml_concat[prev_end_ox:block.b]
                    gap_log.write(f"  Content: \"{gap_content[:200]}...\"\n")
                    gap_units = []
                    for unit_range in openxml_unit_ranges:
                        if unit_range['start'] < block.b and unit_range['end'] > prev_end_ox:
                            gap_units.append(unit_range)
                    gap_log.write(f"  Units in gap: {len(gap_units)}\n")
                    for u in gap_units[:5]:
                        gap_log.write(f"    U{u['unit_idx']}: {u['elem_type']} \"{u['text'][:40]}...\"\n")
                gap_log.write(f"\nBlock {i}: OX[{block.b}], PDF[{block.a}], size={block.size}\n")
                gap_log.write(f"  Matched text: \"{pdf_concat[block.a:block.a + min(block.size, 50)]}...\"\n")
                prev_end_ox = block.b + block.size

        consumed_openxml_positions = set()
        pdf_unit_assignment = {}
        openxml_to_pdf = {}
        match_debug = {}
        matching_log = []
        traversal_log = []

        for block_idx, block in enumerate(sorted_blocks):
            if block.size == 0:
                continue

            block_log = {
                'block_num': block_idx,
                'pdf_start': block.a,
                'openxml_start': block.b,
                'size': block.size,
                'matched_text': pdf_concat[block.a:block.a + min(block.size, 30)],
                'matches': []
            }

            for offset in range(block.size):
                pdf_char_idx = block.a + offset
                openxml_char_idx = block.b + offset

                char = pdf_concat[pdf_char_idx] if pdf_char_idx < len(pdf_concat) else '?'
                pdf_idx = pdf_char_map[pdf_char_idx] if pdf_char_idx < len(pdf_char_map) else -1
                openxml_idx = openxml_char_map[openxml_char_idx] if openxml_char_idx < len(openxml_char_map) else -1

                log_entry = {
                    'step': len(traversal_log),
                    'block': block_idx,
                    'offset': offset,
                    'char': char,
                    'pdf_char_idx': pdf_char_idx,
                    'openxml_char_idx': openxml_char_idx,
                    'pdf_unit': pdf_idx,
                    'openxml_unit': openxml_idx,
                    'pdf_unit_id': pdf_units[pdf_idx]['unit_id'] if 0 <= pdf_idx < len(pdf_units) else None,
                    'openxml_unit_id': openxml_units[openxml_idx]['unit_id'] if 0 <= openxml_idx < len(openxml_units) else None,
                    'action': None,
                    'reason': None
                }

                if openxml_char_idx in consumed_openxml_positions:
                    log_entry['action'] = 'SKIP'
                    log_entry['reason'] = 'openxml_pos_consumed'
                    traversal_log.append(log_entry)
                    continue

                if pdf_char_idx < len(pdf_char_map) and openxml_char_idx < len(openxml_char_map):
                    is_shape_pdf = False
                    if 0 <= pdf_idx < len(pdf_units):
                        is_shape_pdf = pdf_units[pdf_idx].get('item_type') == 'shape'

                    if pdf_idx in pdf_unit_assignment and not is_shape_pdf:
                        if pdf_unit_assignment[pdf_idx] != openxml_idx:
                            log_entry['action'] = 'SKIP'
                            log_entry['reason'] = f'pdf_assigned_to_different: {pdf_unit_assignment[pdf_idx]}'
                            traversal_log.append(log_entry)
                            continue
                        log_entry['reason'] = 'continue_existing_assignment'
                    else:
                        if openxml_idx < min_openxml_idx:
                            log_entry['action'] = 'SKIP'
                            log_entry['reason'] = f'cross_page_backward: openxml_idx={openxml_idx} < min_from_prev_page={min_openxml_idx}'
                            traversal_log.append(log_entry)
                            continue

                        if filter_by_seq_range and seq_min is not None and seq_max is not None:
                            openxml_seq = openxml_units[openxml_idx].get('elem_seq')
                            if openxml_seq is None or openxml_seq < seq_min or openxml_seq > seq_max:
                                log_entry['action'] = 'SKIP'
                                log_entry['reason'] = f'seq_out_of_range: seq={openxml_seq} not in [{seq_min}, {seq_max}]'
                                traversal_log.append(log_entry)
                                continue

                        if not is_shape_pdf:
                            backward_violation = False
                            violation_reason = None

                            for other_pdf_idx, other_openxml_idx in pdf_unit_assignment.items():
                                if pdf_idx > other_pdf_idx and openxml_idx < other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] > pdf[{other_pdf_idx}] but openxml[{openxml_idx}] < openxml[{other_openxml_idx}]'
                                    break
                                if pdf_idx < other_pdf_idx and openxml_idx > other_openxml_idx:
                                    backward_violation = True
                                    violation_reason = f'pdf[{pdf_idx}] < pdf[{other_pdf_idx}] but openxml[{openxml_idx}] > openxml[{other_openxml_idx}]'
                                    break

                            if backward_violation:
                                log_entry['action'] = 'SKIP'
                                log_entry['reason'] = f'backward_match_prevented: {violation_reason}'
                                traversal_log.append(log_entry)
                                continue

                            pdf_unit_assignment[pdf_idx] = openxml_idx
                            log_entry['reason'] = 'new_assignment'
                        else:
                            log_entry['reason'] = 'shape_multi_match'

                    consumed_openxml_positions.add(openxml_char_idx)

                    if openxml_idx not in openxml_to_pdf:
                        openxml_to_pdf[openxml_idx] = {}
                    if pdf_idx not in openxml_to_pdf[openxml_idx]:
                        openxml_to_pdf[openxml_idx][pdf_idx] = 0
                    openxml_to_pdf[openxml_idx][pdf_idx] += 1

                    log_entry['action'] = 'MATCH'
                    log_entry['matched_count'] = openxml_to_pdf[openxml_idx][pdf_idx]
                    traversal_log.append(log_entry)

                    debug_key = (openxml_idx, pdf_idx)
                    if debug_key not in match_debug:
                        match_debug[debug_key] = {'matched_chars': []}
                    match_debug[debug_key]['matched_chars'].append(pdf_concat[pdf_char_idx])

                    if len(block_log['matches']) < 5:
                        block_log['matches'].append({
                            'char': pdf_concat[pdf_char_idx],
                            'pdf_unit': pdf_idx,
                            'openxml_unit': openxml_idx
                        })

            if block_log['matches']:
                matching_log.append(block_log)

        unit_matching_summary = []
        for i in range(len(pdf_units)):
            u = pdf_units[i]
            matched_to = []
            for openxml_idx, pdf_counts in openxml_to_pdf.items():
                if i in pdf_counts:
                    matched_to.append({
                        'openxml_unit_idx': openxml_idx,
                        'openxml_unit_id': openxml_units[openxml_idx]['unit_id'],
                        'matched_chars': pdf_counts[i]
                    })

            unit_matching_summary.append({
                'pdf_unit_idx': i,
                'unit_id': u['unit_id'],
                'item_type': u['item_type'],
                'text': u['text'][:30],
                'consumed': i in pdf_unit_assignment,
                'matched_to': matched_to
            })

        alignments = self._build_alignments_from_matching(
            openxml_to_pdf, pdf_units, openxml_units, match_debug
        )

        unaligned_pdf_indices = [
            i for i in range(len(pdf_units))
            if i not in pdf_unit_assignment
        ]

        unaligned_openxml_indices = [
            i for i in candidate_openxml_indices
            if i not in openxml_to_pdf
        ]

        if filter_by_seq_range and seq_min is not None and seq_max is not None:
            unaligned_openxml_indices = [
                i for i in unaligned_openxml_indices
                if openxml_units[i].get('elem_seq') is not None
                and seq_min <= openxml_units[i].get('elem_seq') <= seq_max
            ]

        skip_entries = [entry for entry in traversal_log if entry.get('action') == 'SKIP']
        cross_page_skip_metrics = self._extract_cross_page_skip_metrics(
            traversal_log,
            total_skip_count=len(skip_entries)
        )

        debug_info = {
            'pdf_concat_len': len(pdf_concat),
            'openxml_concat_len': len(openxml_concat),
            'pdf_concat_sample': pdf_concat[:200],
            'openxml_concat_sample': openxml_concat[:200],
            'pdf_unit_ranges': pdf_unit_ranges,
            'openxml_unit_ranges': openxml_unit_ranges,
            'matching_blocks_count': len(matching_blocks),
            'matching_blocks': [
                {
                    'block_num': i,
                    'pdf_pos': b.a,
                    'openxml_pos': b.b,
                    'size': b.size,
                    'text': pdf_concat[b.a:b.a + min(b.size, 50)]
                }
                for i, b in enumerate(matching_blocks) if b.size > 0
            ][:30],
            'matching_log': matching_log[:20],
            'traversal_log': traversal_log,
            'traversal_log_count': len(traversal_log),
            'unit_matching_summary': unit_matching_summary,
            'consumed_pdf_count': len(pdf_unit_assignment),
            'considered_openxml_count': len(candidate_openxml_indices),
            'suggested_openxml_count': len(suggested_openxml_indices),
            'unaligned_pdf_count': len(unaligned_pdf_indices),
            'unaligned_openxml_count': len(unaligned_openxml_indices),
            'unaligned_openxml_indices': unaligned_openxml_indices,
            'cross_page_backward_skip_count': cross_page_skip_metrics['cross_page_backward_skip_count'],
            'cross_page_backward_skip_ratio': cross_page_skip_metrics['cross_page_backward_skip_ratio'],
            'first_cross_page_skip_openxml_idx': cross_page_skip_metrics['first_cross_page_skip_openxml_idx'],
            'min_cross_page_skip_openxml_idx': cross_page_skip_metrics['min_cross_page_skip_openxml_idx'],
            'median_cross_page_skip_openxml_idx': cross_page_skip_metrics['median_cross_page_skip_openxml_idx'],
            'early_cross_page_skip_count': cross_page_skip_metrics['early_cross_page_skip_count'],
            'early_cross_page_skip_ratio': cross_page_skip_metrics['early_cross_page_skip_ratio'],
            'candidate_openxml_source': candidate_context.get('source'),
            'candidate_openxml_anchor_hit_count': candidate_context.get('anchor_hit_count', 0),
            'candidate_openxml_anchor_count': candidate_context.get('anchor_count', 0),
            'candidate_openxml_anchor_hits': candidate_context.get('anchor_hits', []),
            'candidate_openxml_search_floor': candidate_context.get('search_floor'),
            'candidate_openxml_preferred_seq_min': (
                (candidate_context.get('preferred_seq_range') or (None, None))[0]
                if candidate_context.get('preferred_seq_range')
                else None
            ),
            'candidate_openxml_preferred_seq_max': (
                (candidate_context.get('preferred_seq_range') or (None, None))[1]
                if candidate_context.get('preferred_seq_range')
                else None
            ),
            'candidate_openxml_seq_min': (
                (candidate_context.get('selected_seq_range') or (None, None))[0]
                if candidate_context.get('selected_seq_range')
                else None
            ),
            'candidate_openxml_seq_max': (
                (candidate_context.get('selected_seq_range') or (None, None))[1]
                if candidate_context.get('selected_seq_range')
                else None
            ),
            'candidate_openxml_band_idx_min': (
                min(suggested_openxml_indices) if suggested_openxml_indices else None
            ),
            'candidate_openxml_band_idx_max': (
                max(suggested_openxml_indices) if suggested_openxml_indices else None
            ),
            'candidate_openxml_cluster_min_idx': candidate_context.get('anchor_cluster_min_idx'),
            'candidate_openxml_cluster_max_idx': candidate_context.get('anchor_cluster_max_idx'),
            'max_openxml_idx': max(pdf_unit_assignment.values()) if pdf_unit_assignment else min_openxml_idx
        }

        if trace_context:
            self._append_alignment_trace(
                trace_context,
                traversal_log,
                min_openxml_idx,
                len(pdf_units),
                len(candidate_openxml_indices)
            )

        return alignments, unaligned_pdf_indices, unaligned_openxml_indices, debug_info

    def _build_alignments_from_matching(self, openxml_to_pdf, pdf_units, openxml_units, match_debug):
        """
        Build alignment structure organized by OpenXML element.
        Groups table cells under parent element and keeps text/image parts separate.
        """
        elem_alignments = {}
        non_table_units = {}

        for openxml_idx, pdf_counts in openxml_to_pdf.items():
            if not pdf_counts:
                continue

            openxml_unit = openxml_units[openxml_idx]
            elem_id = openxml_unit['elem_id']
            unit_id = openxml_unit['unit_id']

            matched_pdf = []
            for pdf_idx, matched_count in pdf_counts.items():
                pdf_unit = pdf_units[pdf_idx]
                score = matched_count / len(pdf_unit['text_normalized']) if pdf_unit['text_normalized'] else 0

                debug_key = (openxml_idx, pdf_idx)
                debug_info = match_debug.get(debug_key, {})

                matched_pdf.append({
                    'pdf_unit_id': pdf_unit['unit_id'],
                    'item_idx': pdf_unit['item_idx'],
                    'item_type': pdf_unit['item_type'],
                    'text': pdf_unit['text'],
                    'bbox': pdf_unit['bbox'],
                    'matched_count': matched_count,
                    'score': round(score, 3),
                    'is_cell': pdf_unit['is_cell'],
                    'is_hline_table_unit': pdf_unit.get('is_hline_table_unit', False),
                    'row': pdf_unit.get('row'),
                    'col': pdf_unit.get('col'),
                    'debug': {
                        'matched_str': ''.join(debug_info.get('matched_chars', []))
                    } if debug_info else {}
                })

            matched_pdf.sort(key=lambda x: x['item_idx'])

            is_image_part = openxml_unit.get('is_image_part', False)

            if is_image_part:
                for mp_idx, mp in enumerate(matched_pdf):
                    bbox = mp.get('bbox')
                    individual_unit_id = f"{unit_id}_m{mp_idx}"
                    non_table_units[individual_unit_id] = {
                        'element_id': elem_id,
                        'element_sequence': openxml_unit['elem_seq'],
                        'element_type': openxml_unit['elem_type'],
                        'is_table': False,
                        'element_text': openxml_unit['text'],
                        'matched_pdf_units': [mp],
                        'merged_bbox': list(bbox) if bbox and len(bbox) >= 4 else None,
                        'cells': None,
                        'is_text_part': False,
                        'is_image_part': True,
                        'unit_id': individual_unit_id,
                        'openxml_indices': [openxml_idx],
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
                continue

            merged_bbox = self._merge_bboxes([mp.get('bbox') for mp in matched_pdf])

            if openxml_unit['is_cell']:
                if elem_id not in elem_alignments:
                    elem_alignments[elem_id] = {
                        'element_id': elem_id,
                        'element_sequence': openxml_unit['elem_seq'],
                        'element_type': openxml_unit['elem_type'],
                        'is_table': True,
                        'element_text': openxml_unit['text'],
                        'matched_pdf_units': [],
                        'merged_bbox': None,
                        'cells': [],
                        'is_text_part': False,
                        'is_image_part': False,
                        'unit_id': str(elem_id),
                        'openxml_indices': [],
                        'openxml_idx': openxml_idx,
                        'font_families': openxml_unit.get('font_families', []),
                        'style_ids': openxml_unit.get('style_ids', []),
                        'is_code_font': openxml_unit.get('is_code_font', False),
                        'is_code_style': openxml_unit.get('is_code_style', False),
                        'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                        'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                        'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                        'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
                    }

                cell = {
                    'row': openxml_unit.get('row'),
                    'col': openxml_unit.get('col'),
                    'text': openxml_unit.get('text', ''),
                    'matched_pdf_units': matched_pdf,
                    'merged_bbox': merged_bbox,
                    'openxml_idx': openxml_idx,
                    'font_families': openxml_unit.get('font_families', []),
                    'style_ids': openxml_unit.get('style_ids', []),
                    'is_code_font': openxml_unit.get('is_code_font', False),
                    'is_code_style': openxml_unit.get('is_code_style', False),
                    'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
                    'is_openxml_chart': openxml_unit.get('is_openxml_chart', False),
                    'is_openxml_visual_slot': openxml_unit.get('is_openxml_visual_slot', False),
                    'is_chart_caption_text': openxml_unit.get('is_chart_caption_text', False),
                }
                elem_alignments[elem_id]['cells'].append(cell)
                elem_alignments[elem_id]['openxml_indices'].append(openxml_idx)
            else:
                elem_alignments[elem_id] = {
                    'element_id': elem_id,
                    'element_sequence': openxml_unit['elem_seq'],
                    'element_type': openxml_unit['elem_type'],
                    'is_table': False,
                    'element_text': openxml_unit['text'],
                    'matched_pdf_units': matched_pdf,
                    'merged_bbox': merged_bbox,
                    'cells': None,
                    'is_text_part': openxml_unit.get('is_text_part', False),
                    'is_image_part': False,
                    'unit_id': str(elem_id),
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

        alignments = list(elem_alignments.values()) + list(non_table_units.values())

        # Merge cell bboxes
        for align in alignments:
            if align.get('is_table') and align.get('cells'):
                cell_bboxes = [c.get('merged_bbox') for c in align['cells'] if c.get('merged_bbox')]
                if cell_bboxes:
                    align['merged_bbox'] = self._merge_bboxes(cell_bboxes)

        alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        return alignments

    def _match_remaining_with_unaligned_openxml(
        self,
        alignments,
        un_pdf_idx,
        un_ox_idx,
        pdf_units,
        openxml_units,
        min_openxml_idx=None,
        trace_context=None,
        page_sequence_range=None
    ):
        if not un_pdf_idx or not un_ox_idx:
            return alignments, un_pdf_idx, un_ox_idx

        filter_by_seq_range = os.getenv("ALIGNMENT_FILTER_BY_SEQ_RANGE", "").lower() in ("1", "true", "yes", "on")
        seq_min = seq_max = None
        if filter_by_seq_range and page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range

        if filter_by_seq_range and seq_min is not None and seq_max is not None:
            un_ox_idx = [
                idx for idx in un_ox_idx
                if openxml_units[idx].get('elem_seq') is not None
                and seq_min <= openxml_units[idx].get('elem_seq') <= seq_max
            ]
            if not un_ox_idx:
                return alignments, un_pdf_idx, un_ox_idx

        if min_openxml_idx is not None and min_openxml_idx > 0:
            filtered_un_ox_idx = [idx for idx in un_ox_idx if idx >= min_openxml_idx]
            if not filtered_un_ox_idx:
                return alignments, un_pdf_idx, un_ox_idx
            un_ox_idx = filtered_un_ox_idx

        sub_pdf = [pdf_units[i] for i in un_pdf_idx]
        sub_ox = [openxml_units[i] for i in un_ox_idx]

        trace_pass2 = dict(trace_context or {})
        trace_pass2['phase'] = 'pass2'
        late_align, l_un_pdf_local, l_un_ox_local, _ = self._perform_char_alignment(
            sub_pdf,
            sub_ox,
            trace_context=trace_pass2,
            page_sequence_range=page_sequence_range
        )

        remap_pass2 = self._is_env_enabled_default_true("ALIGNMENT_FIX_PASS2_REMAP")
        if remap_pass2 and un_ox_idx:
            def remap_idx(local_idx):
                if local_idx is None:
                    return None
                if 0 <= local_idx < len(un_ox_idx):
                    return un_ox_idx[local_idx]
                return local_idx

            for la in late_align:
                if la.get('openxml_idx') is not None:
                    la['openxml_idx'] = remap_idx(la.get('openxml_idx'))
                if la.get('openxml_indices'):
                    mapped = [remap_idx(i) for i in la.get('openxml_indices') if i is not None]
                    la['openxml_indices'] = sorted({i for i in mapped if i is not None})
                if la.get('is_table') and la.get('cells'):
                    for cell in la.get('cells') or []:
                        if cell.get('openxml_idx') is not None:
                            cell['openxml_idx'] = remap_idx(cell.get('openxml_idx'))

        ex_map = {a['element_id']: a for a in alignments}
        for la in late_align:
            eid = la['element_id']
            for u in la['matched_pdf_units']:
                u['late_matched'] = True

            if eid in ex_map:
                ex = ex_map[eid]
                ex['late_matched'] = True
                ex['matched_pdf_units'].extend(la['matched_pdf_units'])
                ex['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
                ex_fonts = set(ex.get('font_families') or [])
                ex_fonts.update(la.get('font_families') or [])
                ex['font_families'] = sorted(ex_fonts)

                ex_styles = set(ex.get('style_ids') or [])
                ex_styles.update(la.get('style_ids') or [])
                ex['style_ids'] = sorted(ex_styles)

                ex['is_code_font'] = bool(ex.get('is_code_font')) or bool(la.get('is_code_font'))
                ex['is_code_style'] = bool(ex.get('is_code_style')) or bool(la.get('is_code_style'))
                ex['is_code_like_openxml'] = (
                    bool(ex.get('is_code_like_openxml')) or
                    bool(la.get('is_code_like_openxml')) or
                    bool(ex.get('is_code_font')) or
                    bool(ex.get('is_code_style'))
                )
                ex['is_openxml_chart'] = bool(ex.get('is_openxml_chart')) or bool(la.get('is_openxml_chart'))
                ex['is_openxml_visual_slot'] = (
                    bool(ex.get('is_openxml_visual_slot')) or
                    bool(la.get('is_openxml_visual_slot'))
                )
                ex['is_chart_caption_text'] = (
                    bool(ex.get('is_chart_caption_text')) or
                    bool(la.get('is_chart_caption_text'))
                )
                la_indices = la.get('openxml_indices') or []
                if la_indices:
                    ex_indices = ex.setdefault('openxml_indices', [])
                    for idx in la_indices:
                        if idx not in ex_indices:
                            ex_indices.append(idx)
                if la.get('merged_bbox'):
                    if ex.get('merged_bbox'):
                        ex['merged_bbox'] = self._merge_bboxes([ex['merged_bbox'], la['merged_bbox']])
                    else:
                        ex['merged_bbox'] = la['merged_bbox']
            else:
                la['late_matched'] = True
                alignments.append(la)

        alignments.sort(key=lambda x: x.get('element_sequence') or 0)

        final_un_pdf = [un_pdf_idx[i] for i in l_un_pdf_local]
        final_un_ox = [un_ox_idx[i] for i in l_un_ox_local]
        return alignments, final_un_pdf, final_un_ox

    def _append_alignment_trace(self, trace_context, traversal_log, min_openxml_idx, pdf_units_count, openxml_units_count):
        doc_id = trace_context.get('doc_id')
        page_num = trace_context.get('page_num')
        if doc_id is None or page_num is None:
            return
        if not traversal_log:
            return

        phase = trace_context.get('phase', 'pass1')
        os.makedirs(self.TRACE_DIR, exist_ok=True)
        path = os.path.join(
            self.TRACE_DIR,
            f"{self.TRACE_PREFIX}_doc_{doc_id}_page_{page_num}.txt"
        )

        timestamp = datetime.now().isoformat(timespec='seconds')
        header = (
            f"=== {timestamp} doc_id={doc_id} page={page_num} phase={phase} "
            f"steps={len(traversal_log)} min_openxml_idx={min_openxml_idx} "
            f"pdf_units={pdf_units_count} openxml_units={openxml_units_count} ===\n"
        )

        def sanitize_char(value):
            if value is None:
                return ''
            text = str(value)
            return text.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        try:
            with open(path, 'a', encoding='utf-8') as log_file:
                log_file.write(header)
                for entry in traversal_log:
                    char = sanitize_char(entry.get('char'))
                    action = entry.get('action') or ''
                    reason = entry.get('reason') or ''
                    matched_count = entry.get('matched_count')
                    matched_part = f" cnt:{matched_count}" if matched_count is not None else ''
                    log_file.write(
                        f"[{entry.get('step')}] "
                        f"Block{entry.get('block')} Char=\"{char}\" "
                        f"PDF[{entry.get('pdf_char_idx')}] -> U{entry.get('pdf_unit')}({entry.get('pdf_unit_id')}) "
                        f"OX[{entry.get('openxml_char_idx')}] -> U{entry.get('openxml_unit')}({entry.get('openxml_unit_id')}) "
                        f"| {action} {reason}{matched_part}\n"
                    )
                log_file.write("\n")
        except OSError:
            # Trace logging is best-effort and should never break document processing.
            return
