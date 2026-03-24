from typing import Dict, List


class DoclingFusionCollectMixin:
    def _collect_aligned_items(self, alignments: List[Dict], header_footer_units: List[Dict]) -> List[Dict]:
        all_aligned_items = []

        for alignment in (alignments or []):
            is_table_alignment = alignment.get('is_table') and alignment.get('cells')
            parent_openxml_indices = alignment.get('openxml_indices') or []
            if is_table_alignment:
                parent_openxml_idx = alignment.get('element_sequence')
                for cell in alignment['cells']:
                    item = self._build_cell_fusion_item(alignment, cell, parent_openxml_idx)
                    if item is not None:
                        all_aligned_items.append(item)
            else:
                parent_openxml_idx = (
                    min(parent_openxml_indices)
                    if parent_openxml_indices
                    else alignment.get('openxml_idx')
                )
                item = self._build_alignment_fusion_item(alignment, parent_openxml_idx)
                if item is not None:
                    all_aligned_items.append(item)

        for unit in (header_footer_units or []):
            item = self._build_header_footer_fusion_item(unit)
            if item is not None:
                all_aligned_items.append(item)

        return all_aligned_items

    def _build_cell_fusion_item(self, alignment: Dict, cell: Dict, parent_openxml_idx):
        if not cell.get('merged_bbox'):
            return None

        matched_units = cell.get('matched_pdf_units', [])
        has_image = any(u.get('item_type') == 'image' for u in matched_units)
        has_shape = any(u.get('item_type') == 'shape' for u in matched_units)

        return {
            'bbox': cell['merged_bbox'],
            'text': cell.get('text', ''),
            'source': 'cell',
            'element_id': alignment.get('element_id'),
            'element_sequence': alignment.get('element_sequence'),
            'element_type': alignment.get('element_type'),
            'row': cell.get('row'),
            'col': cell.get('col'),
            'openxml_idx': parent_openxml_idx,
            'has_pdf_image': has_image,
            'has_shape_units': has_shape,
            'has_table_units': True,
            'is_picture_area': has_image or has_shape,
            'font_families': cell.get('font_families', []),
            'style_ids': cell.get('style_ids', []),
            'is_code_font': cell.get('is_code_font', False),
            'is_code_style': cell.get('is_code_style', False),
            'is_code_like_openxml': cell.get('is_code_like_openxml', False),
            'is_openxml_chart': cell.get(
                'is_openxml_chart',
                alignment.get('is_openxml_chart', False),
            ),
            'is_openxml_visual_slot': cell.get(
                'is_openxml_visual_slot',
                alignment.get('is_openxml_visual_slot', False),
            ),
            'is_chart_caption_text': bool(cell.get('is_chart_caption_text', False)),
            'visual_slot_promoted': alignment.get('visual_slot_promoted', False),
            'repair_reason': alignment.get('repair_reason'),
            'block_kind': cell.get('block_kind', alignment.get('block_kind')),
            'block_key': cell.get('block_key', alignment.get('block_key')),
            'content_role': cell.get('content_role', alignment.get('content_role')),
            'block_order': cell.get('block_order', alignment.get('block_order')),
            'alignment_confidence': alignment.get('alignment_confidence'),
            'candidate_source': alignment.get('candidate_source'),
            'matched_pdf_unit_count': len(matched_units),
        }

    def _build_alignment_fusion_item(self, alignment: Dict, parent_openxml_idx):
        if not alignment.get('merged_bbox'):
            return None

        matched_units = alignment.get('matched_pdf_units', [])
        has_shape = any(u.get('item_type') == 'shape' for u in matched_units)
        has_image = any(u.get('item_type') == 'image' for u in matched_units)
        has_table_units = any(u.get('item_type') in ('table', 'hline_table') for u in matched_units)
        has_chart_visual_units = any(u.get('is_chart_visual') for u in matched_units)
        is_picture_area = bool(
            alignment.get('is_image_part') or
            has_shape or
            has_image or
            has_chart_visual_units or
            alignment.get('is_openxml_chart') or
            alignment.get('is_openxml_visual_slot')
        )

        return {
            'bbox': alignment['merged_bbox'],
            'text': alignment.get('element_text', ''),
            'source': 'alignment',
            'element_id': alignment.get('element_id'),
            'element_type': alignment.get('element_type'),
            'element_sequence': alignment.get('element_sequence'),
            'openxml_idx': parent_openxml_idx,
            'is_text_part': alignment.get('is_text_part', False),
            'is_image_part': alignment.get('is_image_part', False),
            'has_shape_units': has_shape,
            'has_pdf_image': has_image,
            'has_table_units': has_table_units,
            'unit_id': alignment.get('unit_id'),
            'is_picture_area': is_picture_area,
            'font_families': alignment.get('font_families', []),
            'style_ids': alignment.get('style_ids', []),
            'is_code_font': alignment.get('is_code_font', False),
            'is_code_style': alignment.get('is_code_style', False),
            'is_code_like_openxml': alignment.get('is_code_like_openxml', False),
            'is_openxml_chart': alignment.get('is_openxml_chart', False),
            'is_openxml_visual_slot': alignment.get('is_openxml_visual_slot', False),
            'is_chart_caption_text': alignment.get('is_chart_caption_text', False),
            'visual_slot_promoted': alignment.get('visual_slot_promoted', False),
            'repair_reason': alignment.get('repair_reason'),
            'block_kind': alignment.get('block_kind'),
            'block_key': alignment.get('block_key'),
            'content_role': alignment.get('content_role'),
            'block_order': alignment.get('block_order'),
            'alignment_confidence': alignment.get('alignment_confidence'),
            'candidate_source': alignment.get('candidate_source'),
            'matched_pdf_unit_count': len(matched_units),
        }

    def _build_header_footer_fusion_item(self, unit: Dict):
        if not unit.get('bbox'):
            return None

        return {
            'bbox': unit['bbox'],
            'text': unit.get('text', ''),
            'source': 'header_footer',
            'zone': unit.get('zone'),
            'has_pdf_image': False,
            'has_shape_units': False,
            'has_table_units': False,
            'is_picture_area': False,
            'is_openxml_chart': False,
            'is_openxml_visual_slot': False,
            'visual_slot_promoted': False,
            'repair_reason': None,
            'block_kind': 'header_footer',
            'block_key': None,
            'content_role': 'header_footer',
            'block_order': None,
            'alignment_confidence': 1.0,
            'candidate_source': 'header_footer_unit',
            'matched_pdf_unit_count': 1,
        }
