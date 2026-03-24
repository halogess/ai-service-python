import difflib
import os
import re
from copy import deepcopy
from datetime import datetime


class AlignmentMatchingCharRemainderMixin:


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
