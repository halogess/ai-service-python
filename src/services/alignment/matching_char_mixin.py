import difflib
import os
import re
from copy import deepcopy
from datetime import datetime

class AlignmentMatchingCharMixin:
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
        candidate_openxml_source = candidate_context.get('source')
        if filter_by_seq_range and seq_min is not None and seq_max is not None:
            candidate_openxml_indices = [
                idx for idx, unit in enumerate(openxml_units)
                if unit.get('elem_seq') is not None and seq_min <= unit.get('elem_seq') <= seq_max
            ]
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

        gap_analysis_path = trace_context.get('gap_analysis_path') if trace_context else None
        if trace_context and gap_analysis_path:
            with open(gap_analysis_path, 'w', encoding='utf-8') as gap_log:
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
            'candidate_openxml_source': candidate_openxml_source,
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
                        'block_kind': openxml_unit.get('block_kind'),
                        'block_key': openxml_unit.get('block_key'),
                        'content_role': openxml_unit.get('content_role'),
                        'block_order': openxml_unit.get('block_order'),
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
                        'block_kind': openxml_unit.get('block_kind'),
                        'block_key': openxml_unit.get('block_key'),
                        'content_role': openxml_unit.get('content_role'),
                        'block_order': openxml_unit.get('block_order'),
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
                    'block_kind': openxml_unit.get('block_kind'),
                    'block_key': openxml_unit.get('block_key'),
                    'content_role': openxml_unit.get('content_role'),
                    'block_order': openxml_unit.get('block_order'),
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
                    'block_kind': openxml_unit.get('block_kind'),
                    'block_key': openxml_unit.get('block_key'),
                    'content_role': openxml_unit.get('content_role'),
                    'block_order': openxml_unit.get('block_order'),
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
