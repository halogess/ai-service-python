import difflib
import os
from datetime import datetime


class AlignmentMatchingMixin:
    def _perform_two_pass_alignment(self, pdf_units, openxml_units, min_openxml_idx, trace_context=None):
        trace_pass1 = dict(trace_context or {})
        trace_pass1['phase'] = 'pass1'
        p1_align, p1_un_pdf, _, p1_debug = self._perform_char_alignment(
            pdf_units, openxml_units, min_openxml_idx, trace_context=trace_pass1
        )

        final_align = list(p1_align)
        final_align.sort(key=lambda x: x.get('element_sequence') or 0)
        final_un_pdf = list(p1_un_pdf)

        final_align, final_un_pdf = self._absorb_unaligned_into_alignments(final_align, final_un_pdf, pdf_units)

        p1_un_ox = p1_debug.get('unaligned_openxml_indices', [])
        final_align, final_un_pdf, _ = self._match_remaining_with_unaligned_openxml(
            final_align,
            final_un_pdf,
            p1_un_ox,
            pdf_units,
            openxml_units,
            min_openxml_idx=min_openxml_idx,
            trace_context=trace_context
        )

        pre_cleanup_keys = self._collect_alignment_unit_keys(final_align)

        final_align = self._cleanup_punctuation_alignments(final_align)
        final_align, shape_conflict_debug = self._resolve_shape_alignment_conflicts(final_align, pdf_units)
        final_align, final_un_pdf, shape_attach_debug = self._attach_shape_clusters_to_next_alignment(final_align, final_un_pdf, pdf_units)
        final_align = self._merge_line_overlap_alignments(final_align)
        final_align, final_un_pdf = self._filter_sparse_matched_units(final_align, final_un_pdf, pdf_units)
        final_un_pdf = self._restore_dropped_alignment_units(
            pre_cleanup_keys,
            final_align,
            final_un_pdf,
            pdf_units
        )
        final_align, final_un_pdf = self._absorb_unaligned_by_y_overlap(final_align, final_un_pdf, pdf_units)

        # Legacy debug fields
        p1_debug['pass2_shape_debug'] = []
        p1_debug['pass2_shape_matched'] = 0
        p1_debug['pass2_consumed_pdf'] = []
        p1_debug['shape_openxml_count'] = 0
        p1_debug['non_shape_openxml_count'] = len(openxml_units)
        p1_debug['shape_conflict_debug'] = shape_conflict_debug
        p1_debug['shape_conflict_count'] = len(shape_conflict_debug)
        p1_debug['shape_attach_debug'] = shape_attach_debug
        p1_debug['shape_attach_count'] = len(shape_attach_debug)

        max_idx = self._compute_max_openxml_idx_from_alignments(final_align, min_openxml_idx)
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
        return max_idx

    def _perform_char_alignment(self, pdf_units, openxml_units, min_openxml_idx=0, trace_context=None):
        if not pdf_units or not openxml_units:
            return [], list(range(len(pdf_units))), list(range(len(openxml_units))), {
                'max_openxml_idx': min_openxml_idx,
                'unaligned_openxml_indices': list(range(len(openxml_units)))
            }

        pdf_concat = ''
        pdf_char_map = []
        pdf_unit_ranges = []

        for i, u in enumerate(pdf_units):
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

        for i, u in enumerate(openxml_units):
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
        for i, u in enumerate(pdf_units):
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
            i for i in range(len(openxml_units))
            if i not in openxml_to_pdf
        ]

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
            'unaligned_pdf_count': len(unaligned_pdf_indices),
            'unaligned_openxml_count': len(unaligned_openxml_indices),
            'unaligned_openxml_indices': unaligned_openxml_indices,
            'max_openxml_idx': max(pdf_unit_assignment.values()) if pdf_unit_assignment else min_openxml_idx
        }

        if trace_context:
            self._append_alignment_trace(
                trace_context,
                traversal_log,
                min_openxml_idx,
                len(pdf_units),
                len(openxml_units)
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
                        'image_index': openxml_unit.get('image_index')
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
                        'openxml_idx': openxml_idx
                    }

                cell = {
                    'row': openxml_unit.get('row'),
                    'col': openxml_unit.get('col'),
                    'text': openxml_unit.get('text', ''),
                    'matched_pdf_units': matched_pdf,
                    'merged_bbox': merged_bbox,
                    'openxml_idx': openxml_idx
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
                    'image_index': openxml_unit.get('image_index')
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
        trace_context=None
    ):
        if not un_pdf_idx or not un_ox_idx:
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
            sub_pdf, sub_ox, trace_context=trace_pass2
        )

        ex_map = {a['element_id']: a for a in alignments}
        for la in late_align:
            eid = la['element_id']
            for u in la['matched_pdf_units']:
                u['late_matched'] = True

            if eid in ex_map:
                ex = ex_map[eid]
                ex['matched_pdf_units'].extend(la['matched_pdf_units'])
                ex['matched_pdf_units'].sort(key=lambda x: x['item_idx'])
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
