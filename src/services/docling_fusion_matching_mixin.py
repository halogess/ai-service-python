from typing import Dict, List


class DoclingFusionMatchingMixin:
    def _collect_docling_matching_items(self, all_aligned_items: List[Dict], doc_bbox, used_indices) -> List[Dict]:
        matching_items = []
        for idx, item in enumerate(all_aligned_items):
            if idx in used_indices:
                continue
            overlap = self.calculate_overlap(item['bbox'], doc_bbox)
            if overlap >= self.OVERLAP_THRESHOLD:
                matching_items.append({'item': item, 'idx': idx, 'overlap': overlap})
        return matching_items

    def _summarize_docling_matching_items(self, doc_item: Dict, matching_items: List[Dict]) -> Dict:
        is_picture_label = doc_item.get('label') == 'picture'
        is_table_label = doc_item.get('label') == 'table'
        element_ids = list(set(
            m['item'].get('element_id')
            for m in matching_items
            if m['item'].get('element_id')
        ))
        all_same_element = len(element_ids) <= 1
        has_text_part = any(m['item'].get('is_text_part') for m in matching_items)
        has_image_part = any(m['item'].get('is_image_part') for m in matching_items)
        has_mixed_parts = has_text_part and has_image_part
        has_shape_units = any(m['item'].get('has_shape_units') for m in matching_items)
        has_pdf_image = any(m['item'].get('has_pdf_image') for m in matching_items)
        has_table_units = any(m['item'].get('has_table_units') for m in matching_items)
        has_openxml_chart = any(m['item'].get('is_openxml_chart') for m in matching_items)
        has_openxml_visual_slot = any(m['item'].get('is_openxml_visual_slot') for m in matching_items)
        has_chart_caption_text = any(m['item'].get('is_chart_caption_text') for m in matching_items)
        all_text_only = all(self._is_text_only_item(m['item']) for m in matching_items)
        should_merge_picture = is_picture_label and len(matching_items) > 1 and has_shape_units
        should_merge = (
            (
                len(matching_items) > 1 and
                all_same_element and
                not has_mixed_parts and
                not has_image_part
            ) or
            should_merge_picture
        )
        if is_picture_label and (has_table_units or has_image_part or has_pdf_image):
            should_merge = False
        if is_table_label:
            should_merge = False

        return {
            'should_merge': should_merge,
            'should_merge_picture': should_merge_picture,
            'has_shape_units': has_shape_units,
            'has_pdf_image': has_pdf_image,
            'has_table_units': has_table_units,
            'has_openxml_chart': has_openxml_chart,
            'has_openxml_visual_slot': has_openxml_visual_slot,
            'has_chart_caption_text': has_chart_caption_text,
            'all_text_only': all_text_only,
        }

    def _build_merged_docling_result(self, matching_items: List[Dict], doc_item: Dict, context: Dict) -> Dict:
        merged_bbox = None
        merged_text = []
        avg_overlap = 0
        sequences = []
        elem_ids = []
        elem_types = []
        openxml_indices = []
        block_kinds = []
        block_keys = []
        content_roles = []
        block_orders = []
        alignment_confidences = []
        candidate_sources = []

        for matched in matching_items:
            item = matched['item']
            merged_bbox = self.merge_bboxes(merged_bbox, item['bbox'])
            if item.get('text'):
                merged_text.append(item['text'])
            avg_overlap += matched['overlap']
            if item.get('element_sequence'):
                sequences.append(item['element_sequence'])
            if item.get('element_id'):
                elem_ids.append(item['element_id'])
            if item.get('element_type'):
                elem_types.append(item['element_type'])
            if item.get('openxml_idx') is not None:
                openxml_indices.append(item['openxml_idx'])
            if item.get('block_kind'):
                block_kinds.append(item.get('block_kind'))
            if item.get('block_key'):
                block_keys.append(item.get('block_key'))
            if item.get('content_role'):
                content_roles.append(item.get('content_role'))
            if item.get('block_order') is not None:
                block_orders.append(item.get('block_order'))
            if item.get('alignment_confidence') is not None:
                alignment_confidences.append(float(item.get('alignment_confidence') or 0.0))
            if item.get('candidate_source'):
                candidate_sources.append(item.get('candidate_source'))

        avg_overlap /= len(matching_items)
        merged_label = doc_item.get('label')
        if merged_label == 'picture':
            if context['has_chart_caption_text']:
                merged_label = 'caption'
            elif not context['has_pdf_image'] and not context['has_openxml_chart'] and not context['has_openxml_visual_slot']:
                merged_label = 'text'
            elif not any(m['item'].get('is_picture_area') for m in matching_items) and not any(
                m['item'].get('has_shape_units') for m in matching_items
            ):
                if context['all_text_only']:
                    merged_label = 'caption'
        if merged_label == 'table':
            merged_label = self._resolve_table_prediction_label(matching_items)
        merged_label = self.correct_header_footer_label(merged_label, merged_bbox)

        ref_elem_id = None
        ref_elem_seq = None
        ref_elem_type = None
        ref_openxml_idx = min(openxml_indices) if openxml_indices else None

        if context['should_merge_picture'] and sequences:
            max_seq = max(sequences)
            max_idx = sequences.index(max_seq)
            ref_elem_seq = max_seq
            ref_elem_id = elem_ids[max_idx] if max_idx < len(elem_ids) else None
            ref_elem_type = elem_types[max_idx] if max_idx < len(elem_types) else None

        return {
            'bbox': merged_bbox,
            'label': merged_label,
            'text': ' '.join(merged_text),
            'overlap': avg_overlap,
            'source': 'merged',
            'merged_count': len(matching_items),
            'element_sequences': ', '.join(str(s) for s in sequences) if sequences else None,
            'element_id': ref_elem_id,
            'element_sequence': ref_elem_seq,
            'element_type': ref_elem_type,
            'openxml_idx': ref_openxml_idx,
            'is_picture_merge': context['should_merge_picture'],
            'docling_label': doc_item.get('label'),
            'is_picture_area': any(m['item'].get('is_picture_area') for m in matching_items),
            'has_shape_units': context['has_shape_units'],
            'has_pdf_image': context['has_pdf_image'],
            'has_table_units': context['has_table_units'],
            'is_text_only_item': context['all_text_only'],
            'is_openxml_chart': context['has_openxml_chart'],
            'is_openxml_visual_slot': context['has_openxml_visual_slot'],
            'is_chart_caption_text': context['has_chart_caption_text'],
            'visual_slot_promoted': any(m['item'].get('visual_slot_promoted') for m in matching_items),
            'repair_reason': next(
                (m['item'].get('repair_reason') for m in matching_items if m['item'].get('repair_reason')),
                None,
            ),
            'block_kind': block_kinds[0] if block_kinds else None,
            'block_key': block_keys[0] if block_keys else None,
            'content_role': content_roles[0] if content_roles else None,
            'block_order': min(block_orders) if block_orders else None,
            'alignment_confidence': max(alignment_confidences) if alignment_confidences else None,
            'candidate_source': candidate_sources[0] if candidate_sources else None,
            'matched_pdf_unit_count': sum(
                int(m['item'].get('matched_pdf_unit_count') or 0)
                for m in matching_items
            ),
        }

    def _resolve_item_docling_label(self, doc_item: Dict, item: Dict) -> str:
        final_label = doc_item.get('label')
        if final_label == 'picture':
            if item.get('is_chart_caption_text'):
                final_label = 'caption'
            elif (
                not item.get('has_pdf_image') and
                not item.get('is_openxml_chart') and
                not item.get('is_openxml_visual_slot')
            ):
                final_label = 'text'
            elif not item.get('is_picture_area') and not item.get('has_shape_units'):
                if self._is_text_only_item(item):
                    final_label = 'caption'
        if final_label == 'table':
            final_label = self._resolve_table_prediction_label([item])
        return self.correct_header_footer_label(final_label, item['bbox'])

    def _build_individual_docling_results(self, matching_items: List[Dict], doc_item: Dict) -> List[Dict]:
        results = []
        for matched in matching_items:
            item = matched['item']
            results.append({
                'bbox': item['bbox'],
                'label': self._resolve_item_docling_label(doc_item, item),
                'text': item.get('text', ''),
                'overlap': matched['overlap'],
                'source': item.get('source'),
                'element_id': item.get('element_id'),
                'element_type': item.get('element_type'),
                'element_sequence': item.get('element_sequence'),
                'openxml_idx': item.get('openxml_idx'),
                'zone': item.get('zone'),
                'docling_label': doc_item.get('label'),
                'is_text_part': item.get('is_text_part'),
                'is_image_part': item.get('is_image_part'),
                'unit_id': item.get('unit_id'),
                'merged_count': 1,
                'is_picture_area': item.get('is_picture_area', False),
                'has_shape_units': item.get('has_shape_units'),
                'has_pdf_image': item.get('has_pdf_image'),
                'has_table_units': item.get('has_table_units'),
                'is_text_only_item': self._is_text_only_item(item),
                'is_openxml_chart': item.get('is_openxml_chart', False),
                'is_openxml_visual_slot': item.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': item.get('is_chart_caption_text', False),
                'visual_slot_promoted': item.get('visual_slot_promoted', False),
                'repair_reason': item.get('repair_reason'),
                'table_canonical_from_element_id': item.get('table_canonical_from_element_id'),
                'table_canonical_from_sequence': item.get('table_canonical_from_sequence'),
                'block_kind': item.get('block_kind'),
                'block_key': item.get('block_key'),
                'content_role': item.get('content_role'),
                'block_order': item.get('block_order'),
                'alignment_confidence': item.get('alignment_confidence'),
                'candidate_source': item.get('candidate_source'),
                'matched_pdf_unit_count': item.get('matched_pdf_unit_count'),
            })
        return results

    def _append_unmatched_fusion_items(self, fused_results: List[Dict], all_aligned_items: List[Dict], used_indices):
        for idx, item in enumerate(all_aligned_items):
            if idx in used_indices:
                continue

            label = 'unknown'
            if item.get('source') == 'header_footer' and item.get('zone'):
                label = 'page_header' if item['zone'] == 'header' else 'page_footer'
            elif item.get('is_image_part') and (
                item.get('has_pdf_image') or
                item.get('is_openxml_chart') or
                item.get('is_openxml_visual_slot')
            ):
                label = 'picture'
            elif item.get('is_image_part'):
                label = self.fallback_label(item)
            elif item.get('is_chart_caption_text') and self._is_caption_candidate(item.get('text')):
                label = 'caption'

            label = self.correct_header_footer_label(label, item['bbox'])
            fused_results.append({
                'bbox': item['bbox'],
                'label': label,
                'text': item.get('text'),
                'overlap': 0,
                'source': item.get('source'),
                'element_id': item.get('element_id'),
                'element_type': item.get('element_type'),
                'element_sequence': item.get('element_sequence'),
                'openxml_idx': item.get('openxml_idx'),
                'zone': item.get('zone'),
                'docling_label': None,
                'is_image_part': item.get('is_image_part'),
                'merged_count': 1,
                'is_picture_area': item.get('is_picture_area', False),
                'has_shape_units': item.get('has_shape_units'),
                'has_pdf_image': item.get('has_pdf_image'),
                'has_table_units': item.get('has_table_units'),
                'is_text_only_item': self._is_text_only_item(item),
                'is_openxml_chart': item.get('is_openxml_chart', False),
                'is_openxml_visual_slot': item.get('is_openxml_visual_slot', False),
                'is_chart_caption_text': item.get('is_chart_caption_text', False),
                'visual_slot_promoted': item.get('visual_slot_promoted', False),
                'repair_reason': item.get('repair_reason'),
                'block_kind': item.get('block_kind'),
                'block_key': item.get('block_key'),
                'content_role': item.get('content_role'),
                'block_order': item.get('block_order'),
                'alignment_confidence': item.get('alignment_confidence'),
                'candidate_source': item.get('candidate_source'),
                'matched_pdf_unit_count': item.get('matched_pdf_unit_count'),
            })
