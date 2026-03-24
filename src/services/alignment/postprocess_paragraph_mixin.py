from copy import deepcopy
import os
import re

class AlignmentPostprocessParagraphMixin:
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

    def _is_paragraph_like_alignment(self, alignment):
        if (
            not alignment or
            alignment.get('is_table') or
            alignment.get('is_openxml_chart') or
            alignment.get('is_openxml_visual_slot')
        ):
            return False
        element_type = str(alignment.get('element_type') or '').strip().lower()
        if not element_type:
            return False
        if any(marker in element_type for marker in ('table', 'image', 'picture', 'caption', 'chart')):
            return False
        if 'paragraph' in element_type:
            return True
        return element_type in {
            'text',
            'p',
            'list_item',
            'listitem',
            'title',
            'subtitle',
            'heading',
            'section_header',
        }

    def _extract_figure_key(self, text):
        if not text:
            return None
        match = self.FIGURE_KEY_RE.search(str(text))
        if not match:
            return None
        prefix_match = re.search(r'(gambar|figure|fig\.?|tabel|table)', str(text), re.IGNORECASE)
        prefix = prefix_match.group(1).lower().rstrip('.') if prefix_match else 'gambar'
        return f"{prefix}:{match.group(1)}"

    def _is_caption_like_text(self, text):
        if not text:
            return False
        return bool(self.VISUAL_CAPTION_RE.match(str(text).strip()))

    def _is_image_placeholder_only_text(self, text):
        if not text:
            return False
        stripped = str(text).strip()
        return bool(re.fullmatch(r'(?:\[img(?::\d+)?\]\s*)+', stripped, re.IGNORECASE))

    def _is_short_caption_fragment_text(self, text):
        if not text:
            return False
        stripped = str(text).strip()
        if not stripped or self._is_image_placeholder_only_text(stripped):
            return False
        norm = self._normalize_text(stripped).strip()
        if not norm or len(norm) > 64:
            return False
        if self._is_caption_like_text(stripped):
            return True
        return bool(self.CAPTION_FRAGMENT_LEAD_RE.match(stripped))

    def _bbox_x_overlap_ratio(self, bbox_a, bbox_b):
        if not bbox_a or not bbox_b or len(bbox_a) < 4 or len(bbox_b) < 4:
            return 0.0
        overlap_start = max(bbox_a[0], bbox_b[0])
        overlap_end = min(bbox_a[2], bbox_b[2])
        overlap = max(0.0, overlap_end - overlap_start)
        width_a = max(0.0, bbox_a[2] - bbox_a[0])
        width_b = max(0.0, bbox_b[2] - bbox_b[0])
        denom = min(width_a, width_b)
        if denom <= 0:
            return 0.0
        return overlap / denom

    def _is_visual_target_alignment(self, alignment):
        if not alignment or alignment.get('is_table'):
            return False
        return bool(
            alignment.get('is_openxml_chart') or
            alignment.get('is_openxml_visual_slot') or
            alignment.get('is_chart_visual_attachment')
        )

    def _is_visual_pdf_unit(self, unit):
        if not unit:
            return False
        if unit.get('is_chart_visual'):
            return True
        return unit.get('item_type') in {'image', 'shape', 'hline_table'}

    def _alignment_has_visual_units(self, alignment):
        if not alignment:
            return False
        if self._is_visual_target_alignment(alignment):
            return True
        return any(
            self._is_visual_pdf_unit(unit)
            for unit in (alignment.get('matched_pdf_units') or [])
        )

    def _alignment_visual_bbox(self, alignment):
        visual_bboxes = [
            unit.get('bbox')
            for unit in (alignment or {}).get('matched_pdf_units', []) or []
            if self._is_visual_pdf_unit(unit) and unit.get('bbox')
        ]
        if not visual_bboxes:
            return None
        return self._merge_bboxes(visual_bboxes)

    def _is_caption_band_bbox(
        self,
        unit_bbox,
        visual_bbox,
        vertical_gap_max=48.0,
        x_overlap_min=0.5
    ):
        if not unit_bbox or not visual_bbox or len(unit_bbox) < 4 or len(visual_bbox) < 4:
            return False
        if self._bbox_x_overlap_ratio(unit_bbox, visual_bbox) < x_overlap_min:
            return False
        if unit_bbox[3] <= visual_bbox[1]:
            gap = visual_bbox[1] - unit_bbox[3]
        elif unit_bbox[1] >= visual_bbox[3]:
            gap = unit_bbox[1] - visual_bbox[3]
        else:
            return False
        return gap <= vertical_gap_max

    def _find_prev_meaningful_openxml_unit(self, openxml_units, start_idx, seq_min=None, seq_max=None):
        for idx in range(int(start_idx or 0) - 1, -1, -1):
            unit = openxml_units[idx] or {}
            seq = unit.get('elem_seq')
            if seq_min is not None and (seq is None or seq < seq_min):
                break
            if seq_max is not None and seq is not None and seq > seq_max:
                continue
            text = self._normalize_text(unit.get('text') or '').strip()
            if text:
                return idx, unit
        return None, None

    def _find_next_meaningful_openxml_unit(
        self,
        openxml_units,
        start_idx,
        seq_min=None,
        seq_max=None,
        skip_short_caption_fragments=False
    ):
        for idx in range(int(start_idx or 0) + 1, len(openxml_units or [])):
            unit = openxml_units[idx] or {}
            seq = unit.get('elem_seq')
            if seq_max is not None and (seq is None or seq > seq_max):
                break
            if seq_min is not None and seq is not None and seq < seq_min:
                continue
            if unit.get('is_openxml_visual_slot'):
                continue
            text = self._normalize_text(unit.get('text') or '').strip()
            if text:
                if skip_short_caption_fragments and self._is_short_caption_fragment_text(unit.get('text')):
                    continue
                return idx, unit
        return None, None

    def _is_valid_visual_slot_target(
        self,
        openxml_units,
        openxml_idx,
        page_sequence_range=None
    ):
        if not openxml_units or openxml_idx is None or openxml_idx < 0 or openxml_idx >= len(openxml_units):
            return False
        unit = openxml_units[openxml_idx] or {}
        if not unit.get('is_openxml_visual_slot'):
            return False

        seq_min = seq_max = None
        if page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range

        unit_seq = unit.get('elem_seq')
        if seq_min is not None and (unit_seq is None or unit_seq < seq_min):
            return False
        if seq_max is not None and (unit_seq is None or unit_seq > seq_max):
            return False

        prev_idx, prev_unit = self._find_prev_meaningful_openxml_unit(
            openxml_units,
            openxml_idx,
            seq_min=seq_min,
            seq_max=seq_max
        )
        next_idx, next_unit = self._find_next_meaningful_openxml_unit(
            openxml_units,
            openxml_idx,
            seq_min=seq_min,
            seq_max=seq_max
        )
        bridge_idx = None
        bridge_unit = None
        if next_unit is not None and self._is_short_caption_fragment_text(next_unit.get('text')):
            bridge_idx = next_idx
            bridge_unit = next_unit
            next_idx, next_unit = self._find_next_meaningful_openxml_unit(
                openxml_units,
                bridge_idx,
                seq_min=seq_min,
                seq_max=seq_max,
                skip_short_caption_fragments=True
            )
            if next_unit is None:
                bridge_idx = None
                bridge_unit = None
        if prev_unit is None or next_unit is None:
            return False

        if bridge_unit is not None:
            bridge_seq = bridge_unit.get('elem_seq')
            next_seq = next_unit.get('elem_seq')
            if (
                unit_seq is not None and
                bridge_seq is not None and
                next_seq is not None and
                unit_seq < bridge_seq < next_seq
            ):
                return True

        prev_key = self._extract_figure_key(prev_unit.get('text'))
        if not prev_key:
            return False

        next_key = self._extract_figure_key(next_unit.get('text'))
        if next_key and next_key != prev_key:
            return False

        next_text = str(next_unit.get('text') or '').strip()
        if not next_text:
            return False
        if self._is_caption_like_text(next_text):
            next_word_count = len([part for part in re.split(r'\s+', next_text) if part])
            if next_word_count <= 8:
                return False

        prev_seq = prev_unit.get('elem_seq')
        next_seq = next_unit.get('elem_seq')
        if prev_seq is None or next_seq is None or unit_seq is None:
            return False
        return prev_seq < unit_seq < next_seq and prev_idx < openxml_idx < next_idx

    def _paragraph_text_len(self, alignment):
        if not alignment:
            return 0
        return len(self._normalize_text(alignment.get('element_text') or ''))

    def _paragraph_gap_threshold(self, alignment, units):
        threshold = self.MATCHED_UNIT_MAX_ITEM_GAP
        if not self._is_paragraph_like_alignment(alignment):
            return threshold

        threshold = max(
            threshold,
            self._read_positive_int_env('ALIGNMENT_PARAGRAPH_MAX_ITEM_GAP', 24)
        )
        text_len = self._paragraph_text_len(alignment)
        if text_len >= self._read_positive_int_env('ALIGNMENT_PARAGRAPH_LONG_TEXT_LEN', 160) or len(units or []) >= 6:
            threshold = max(
                threshold,
                self._read_positive_int_env('ALIGNMENT_PARAGRAPH_LONG_MAX_ITEM_GAP', 36)
            )
        return threshold

    def _should_relax_gap_filter(self, alignment, units):
        if not self._is_paragraph_like_alignment(alignment):
            return False
        text_len = self._paragraph_text_len(alignment)
        min_units = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RELAX_MIN_UNITS', 5)
        min_text_len = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RELAX_TEXT_LEN', 100)
        return len(units or []) >= min_units or text_len >= min_text_len

    def _build_pdf_lookup_maps(self, pdf_units):
        pdf_idx_by_unit_id = {
            unit.get('unit_id'): idx
            for idx, unit in enumerate(pdf_units or [])
            if unit.get('unit_id')
        }
        pdf_idx_by_item_idx = {}
        pdf_idx_by_bbox = {}
        for idx, unit in enumerate(pdf_units or []):
            item_idx = unit.get('item_idx')
            if item_idx is not None:
                pdf_idx_by_item_idx.setdefault(item_idx, []).append(idx)
            bbox = unit.get('bbox')
            if bbox and len(bbox) >= 4:
                pdf_idx_by_bbox.setdefault(tuple(bbox), []).append(idx)
        return pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox

    def _resolve_unit_pdf_index(self, unit, pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox):
        unit_id = unit.get('pdf_unit_id') or unit.get('unit_id')
        if unit_id and unit_id in pdf_idx_by_unit_id:
            return pdf_idx_by_unit_id[unit_id]

        item_idx = unit.get('item_idx')
        if item_idx is not None:
            matches = pdf_idx_by_item_idx.get(item_idx) or []
            if len(matches) == 1:
                return matches[0]

        bbox = unit.get('bbox')
        if bbox and len(bbox) >= 4:
            matches = pdf_idx_by_bbox.get(tuple(bbox)) or []
            if len(matches) == 1:
                return matches[0]
        return None

    def _paragraph_match_stats(self, alignment, units):
        matched_chars = sum(unit.get('matched_count') or 0 for unit in units or [])
        norm_len = len(self._normalize_text((alignment or {}).get('element_text') or ''))
        ratio = (matched_chars / norm_len) if norm_len > 0 else 0.0
        return matched_chars, ratio, norm_len

    def _paragraph_rescue_conflicts(self, candidate_alignment, alignments):
        candidate_bbox = candidate_alignment.get('merged_bbox')
        if not candidate_bbox or len(candidate_bbox) < 4:
            return True

        candidate_element_id = candidate_alignment.get('element_id')
        candidate_sequence = self._get_alignment_sequence(candidate_alignment)
        for alignment in alignments or []:
            if alignment.get('is_table') or alignment.get('is_image_part'):
                continue
            if (
                candidate_element_id is not None and
                alignment.get('element_id') is not None and
                alignment.get('element_id') == candidate_element_id
            ):
                return True

            bbox = alignment.get('merged_bbox')
            if not bbox or len(bbox) < 4:
                continue
            overlap = self._bbox_y_overlap_ratio(candidate_bbox, bbox)
            if overlap < 0.8:
                continue

            sequence_delta = abs(self._get_alignment_sequence(alignment) - candidate_sequence)
            if sequence_delta <= 1:
                if self._is_bbox_fully_contained(candidate_bbox, bbox, tolerance=4):
                    return True
                if self._is_bbox_fully_contained(bbox, candidate_bbox, tolerance=4):
                    return True
        return False

    def _find_neighbor_alignment_by_sequence(self, alignments, target_sequence, direction=1, max_gap=2):
        if target_sequence is None:
            return None

        best_alignment = None
        best_gap = None
        for alignment in alignments or []:
            seq = self._get_alignment_sequence(alignment)
            if seq is None:
                continue
            delta = seq - target_sequence
            if direction < 0:
                if delta >= 0:
                    continue
                gap = abs(delta)
            else:
                if delta <= 0:
                    continue
                gap = delta
            if gap > max_gap:
                continue

            if best_alignment is None or gap < best_gap:
                best_alignment = alignment
                best_gap = gap
                continue

            if gap == best_gap:
                if self._alignment_has_visual_units(alignment) and not self._alignment_has_visual_units(best_alignment):
                    best_alignment = alignment
                    best_gap = gap
        return best_alignment

    def _build_inherited_alignment(self, source_alignment, openxml_unit, openxml_idx, reason):
        if not source_alignment or not openxml_unit:
            return None

        inherited = {
            'element_id': openxml_unit.get('elem_id'),
            'element_sequence': openxml_unit.get('elem_seq'),
            'element_type': openxml_unit.get('elem_type'),
            'is_table': False,
            'element_text': openxml_unit.get('text', ''),
            'matched_pdf_units': deepcopy(source_alignment.get('matched_pdf_units', []) or []),
            'merged_bbox': deepcopy(source_alignment.get('merged_bbox')),
            'cells': None,
            'is_text_part': bool(openxml_unit.get('is_text_part', False)),
            'is_image_part': bool(openxml_unit.get('is_image_part', False)),
            'unit_id': str(openxml_unit.get('elem_id')),
            'openxml_indices': [openxml_idx] if openxml_idx is not None else [],
            'openxml_idx': openxml_idx,
            'image_index': openxml_unit.get('image_index'),
            'font_families': openxml_unit.get('font_families', []),
            'style_ids': openxml_unit.get('style_ids', []),
            'is_code_font': openxml_unit.get('is_code_font', False),
            'is_code_style': openxml_unit.get('is_code_style', False),
            'is_code_like_openxml': openxml_unit.get('is_code_like_openxml', False),
            'is_openxml_chart': bool(openxml_unit.get('is_openxml_chart', False)),
            'is_openxml_visual_slot': bool(openxml_unit.get('is_openxml_visual_slot', False)),
            'is_chart_caption_text': bool(openxml_unit.get('is_chart_caption_text', False)),
            'is_chart_visual_attachment': bool(source_alignment.get('is_chart_visual_attachment', False)),
            'matched_by_visual_only': bool(source_alignment.get('matched_by_visual_only', False)),
            'repair_reason': reason,
            'inherited_from_element_id': source_alignment.get('element_id'),
            'inherited_from_sequence': source_alignment.get('element_sequence'),
        }
        if inherited.get('matched_pdf_units'):
            self._recompute_alignment_bboxes(inherited)
        return inherited

    def _rescue_paragraph_alignments(self, rescue_candidates, alignments, unaligned_pdf_indices, pdf_units):
        if not rescue_candidates or not unaligned_pdf_indices:
            return alignments, unaligned_pdf_indices, []

        pdf_idx_by_unit_id, pdf_idx_by_item_idx, pdf_idx_by_bbox = self._build_pdf_lookup_maps(pdf_units)
        unaligned_set = set(unaligned_pdf_indices or [])
        existing_element_ids = {
            alignment.get('element_id')
            for alignment in alignments or []
            if alignment.get('element_id') is not None
        }

        min_units = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RESCUE_MIN_UNITS', 2)
        min_chars = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_RESCUE_MIN_MATCH_CHARS', 18)
        min_ratio = self._read_float_env(
            'ALIGNMENT_PARAGRAPH_RESCUE_MIN_MATCH_RATIO',
            0.05,
            min_value=0.0,
            max_value=1.0
        )
        min_availability_ratio = self._read_float_env(
            'ALIGNMENT_PARAGRAPH_RESCUE_MIN_AVAILABILITY_RATIO',
            0.6,
            min_value=0.0,
            max_value=1.0
        )
        long_text_len = self._read_positive_int_env('ALIGNMENT_PARAGRAPH_LONG_TEXT_LEN', 160)

        rescue_debug = []
        ordered_candidates = sorted(
            rescue_candidates,
            key=lambda alignment: (
                self._get_alignment_sequence(alignment),
                self._get_alignment_min_item_idx(alignment) or 2**31 - 1
            )
        )

        for candidate in ordered_candidates:
            if not self._is_paragraph_like_alignment(candidate):
                continue

            element_id = candidate.get('element_id')
            if element_id is not None and element_id in existing_element_ids:
                continue

            candidate_units = candidate.get('matched_pdf_units', []) or []
            is_short_caption_fragment = self._is_short_caption_fragment_text(
                candidate.get('element_text')
            )
            candidate_min_units = 1 if is_short_caption_fragment else min_units
            candidate_min_chars = 0 if is_short_caption_fragment else min_chars
            candidate_min_ratio = 0.0 if is_short_caption_fragment else min_ratio

            if len(candidate_units) < candidate_min_units:
                continue

            rescued_units = []
            rescued_indices = []
            seen_indices = set()
            for unit in candidate_units:
                pdf_idx = self._resolve_unit_pdf_index(
                    unit,
                    pdf_idx_by_unit_id,
                    pdf_idx_by_item_idx,
                    pdf_idx_by_bbox
                )
                if pdf_idx is None or pdf_idx not in unaligned_set or pdf_idx in seen_indices:
                    continue
                rescued_units.append(deepcopy(unit))
                rescued_indices.append(pdf_idx)
                seen_indices.add(pdf_idx)

            if len(rescued_units) < candidate_min_units:
                continue

            availability_ratio = len(rescued_units) / max(1, len(candidate_units))
            if (
                availability_ratio < min_availability_ratio and
                self._paragraph_text_len(candidate) < long_text_len
            ):
                continue

            rescued_alignment = deepcopy(candidate)
            rescued_alignment['matched_pdf_units'] = sorted(
                rescued_units,
                key=lambda unit: unit.get('item_idx', -1)
            )
            self._recompute_alignment_bboxes(rescued_alignment)
            if self._paragraph_rescue_conflicts(rescued_alignment, alignments):
                continue

            matched_chars, ratio, norm_len = self._paragraph_match_stats(
                rescued_alignment,
                rescued_alignment.get('matched_pdf_units', [])
            )
            if norm_len > 0 and matched_chars < candidate_min_chars and ratio < candidate_min_ratio:
                continue

            alignments.append(rescued_alignment)
            if element_id is not None:
                existing_element_ids.add(element_id)
            for pdf_idx in rescued_indices:
                unaligned_set.discard(pdf_idx)

            rescue_debug.append({
                'element_id': element_id,
                'element_sequence': rescued_alignment.get('element_sequence'),
                'rescued_units': len(rescued_indices),
                'availability_ratio': availability_ratio,
                'matched_chars': matched_chars,
                'match_ratio': ratio,
            })

        if rescue_debug:
            alignments.sort(key=lambda alignment: alignment.get('element_sequence') or 0)
        return alignments, sorted(unaligned_set), rescue_debug

    def _rescue_fragment_paragraph_alignments(self, openxml_units, alignments, page_sequence_range=None):
        if not openxml_units or not alignments:
            return alignments, []

        seq_min = seq_max = None
        if page_sequence_range and len(page_sequence_range) == 2:
            seq_min, seq_max = page_sequence_range

        existing_element_ids = {
            alignment.get('element_id')
            for alignment in alignments or []
            if alignment.get('element_id') is not None
        }
        seen_candidate_ids = set()
        rescue_debug = []

        for openxml_idx, openxml_unit in enumerate(openxml_units or []):
            if openxml_unit.get('is_cell'):
                continue

            elem_id = openxml_unit.get('elem_id')
            elem_seq = openxml_unit.get('elem_seq')
            elem_type = str(openxml_unit.get('elem_type') or '').strip().lower()
            text = str(openxml_unit.get('text') or '').strip()
            text_norm = self._normalize_text(text).strip()

            if elem_id is None or elem_id in existing_element_ids or elem_id in seen_candidate_ids:
                continue
            seen_candidate_ids.add(elem_id)
            if seq_min is not None and (elem_seq is None or elem_seq < seq_min):
                continue
            if seq_max is not None and (elem_seq is None or elem_seq > seq_max):
                continue
            if 'paragraph' not in elem_type or not text_norm:
                continue

            prev_alignment = self._find_neighbor_alignment_by_sequence(
                alignments,
                elem_seq,
                direction=-1,
                max_gap=2
            )
            next_alignment = self._find_neighbor_alignment_by_sequence(
                alignments,
                elem_seq,
                direction=1,
                max_gap=2
            )

            source_alignment = None
            reason = None

            prev_text_norm = self._normalize_text(
                (prev_alignment or {}).get('element_text') or ''
            ).strip()
            prev_figure_key = self._extract_figure_key((prev_alignment or {}).get('element_text'))
            if (
                prev_alignment is not None and
                prev_text_norm and
                text_norm != prev_text_norm and
                text_norm in prev_text_norm and
                prev_figure_key
            ):
                source_alignment = prev_alignment
                reason = 'caption_suffix_inherit'
            elif self._is_image_placeholder_only_text(text):
                if next_alignment is not None and self._alignment_has_visual_units(next_alignment):
                    source_alignment = next_alignment
                elif prev_alignment is not None and self._alignment_has_visual_units(prev_alignment):
                    source_alignment = prev_alignment
                elif next_alignment is not None:
                    source_alignment = next_alignment
                elif prev_alignment is not None:
                    source_alignment = prev_alignment
                reason = 'image_placeholder_neighbor_inherit'
            elif self._is_short_caption_fragment_text(text):
                if next_alignment is not None:
                    source_alignment = next_alignment
                elif prev_alignment is not None:
                    source_alignment = prev_alignment
                reason = 'caption_fragment_inherit'
            elif next_alignment is not None and next_alignment.get('is_table'):
                _, prev_unit = self._find_prev_meaningful_openxml_unit(
                    openxml_units,
                    openxml_idx,
                    seq_min=seq_min,
                    seq_max=seq_max
                )
                prev_key = self._extract_figure_key((prev_unit or {}).get('text'))
                if prev_key and prev_key.startswith('tabel'):
                    source_alignment = next_alignment
                    reason = 'table_lead_inherit'

            if source_alignment is None or not reason:
                continue

            inherited_alignment = self._build_inherited_alignment(
                source_alignment,
                openxml_unit,
                openxml_idx,
                reason
            )
            if inherited_alignment is None:
                continue
            if (
                not inherited_alignment.get('merged_bbox') and
                not inherited_alignment.get('matched_pdf_units')
            ):
                continue

            alignments.append(inherited_alignment)
            existing_element_ids.add(elem_id)
            rescue_debug.append({
                'element_id': elem_id,
                'element_sequence': elem_seq,
                'reason': reason,
                'source_element_id': source_alignment.get('element_id'),
                'source_sequence': source_alignment.get('element_sequence'),
            })

        if rescue_debug:
            alignments.sort(key=lambda alignment: alignment.get('element_sequence') or 0)
        return alignments, rescue_debug
