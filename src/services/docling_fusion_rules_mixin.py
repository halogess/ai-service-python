import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DoclingFusionRulesMixin:
    def fuse_alignments_with_docling(
        self,
        alignments: List[Dict],
        header_footer_units: List[Dict],
        docling_predictions: List[Dict]
    ) -> List[Dict]:
        """
        Main fusion function. Combines alignment bboxes with Docling labels.
        
        Args:
            alignments: List of alignment results with element_id, merged_bbox, etc.
            header_footer_units: List of header/footer PDF units with bbox, text, zone
            docling_predictions: List of Docling predictions with bbox, label
            
        Returns:
            List of fused results with:
            - bbox: Final bbox (from alignment or merged)
            - label: Docling label (corrected)
            - element_id, element_sequence
            - overlap: Match quality (0-1)
            - merged_count: Number of alignments merged
            - is_picture_merge: True if multiple shapes merged
        """
        fused_results = []
        has_docling = docling_predictions and len(docling_predictions) > 0
        
        # Collect all aligned items (body + header/footer) with their bboxes
        all_aligned_items = []
        
        # Add body alignments
        for alignment in (alignments or []):
            is_table_alignment = alignment.get('is_table') and alignment.get('cells')
            parent_openxml_indices = alignment.get('openxml_indices') or []
            if is_table_alignment:
                parent_openxml_idx = alignment.get('element_sequence')
            else:
                parent_openxml_idx = min(parent_openxml_indices) if parent_openxml_indices else alignment.get('openxml_idx')
            if is_table_alignment:
                # Table cells
                for cell in alignment['cells']:
                    if cell.get('merged_bbox'):
                        matched_units = cell.get('matched_pdf_units', [])
                        has_image = any(u.get('item_type') == 'image' for u in matched_units)
                        has_shape = any(u.get('item_type') == 'shape' for u in matched_units)
                        has_table_units = any(u.get('item_type') in ('table', 'hline_table') for u in matched_units)
                        all_aligned_items.append({
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
                                alignment.get('is_openxml_chart', False)
                            ),
                            'is_openxml_visual_slot': cell.get(
                                'is_openxml_visual_slot',
                                alignment.get('is_openxml_visual_slot', False)
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
                        })
            elif alignment.get('merged_bbox'):
                # Non-table elements
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
                
                all_aligned_items.append({
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
                })
        
        # Add header/footer units
        for unit in (header_footer_units or []):
            if unit.get('bbox'):
                all_aligned_items.append({
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
                })
        
        # Track which aligned items have been used
        used_indices = set()
        
        # If we have Docling classifications, find matching alignments
        if has_docling:
            for doc_item in docling_predictions:
                doc_bbox = doc_item.get('bbox')
                if not doc_bbox:
                    continue
                    
                is_picture_label = doc_item.get('label') == 'picture'
                is_table_label = doc_item.get('label') == 'table'
                
                # Find ALL aligned items that overlap with this Docling element
                matching_items = []
                for idx, item in enumerate(all_aligned_items):
                    if idx in used_indices:
                        continue
                    
                    overlap = self.calculate_overlap(item['bbox'], doc_bbox)
                    if overlap >= self.OVERLAP_THRESHOLD:
                        # Allow matching text-only items, but label picture only when PDF image exists.
                        matching_items.append({'item': item, 'idx': idx, 'overlap': overlap})
                
                if matching_items:
                    if is_table_label:
                        matching_items = self._canonicalize_table_matches(matching_items)

                    # Mark all matching items as used
                    for m in matching_items:
                        used_indices.add(m['idx'])
                    
                    # Check if all matching items have the same element_id
                    element_ids = list(set(m['item'].get('element_id') for m in matching_items if m['item'].get('element_id')))
                    all_same_element = len(element_ids) <= 1
                    
                    # Check for mixed text/image parts
                    has_text_part = any(m['item'].get('is_text_part') for m in matching_items)
                    has_image_part = any(m['item'].get('is_image_part') for m in matching_items)
                    has_mixed_parts = has_text_part and has_image_part
                    
                    # Check for shape units
                    has_shape_units = any(m['item'].get('has_shape_units') for m in matching_items)
                    has_pdf_image = any(m['item'].get('has_pdf_image') for m in matching_items)
                    has_table_units = any(m['item'].get('has_table_units') for m in matching_items)
                    has_openxml_chart = any(m['item'].get('is_openxml_chart') for m in matching_items)
                    has_openxml_visual_slot = any(
                        m['item'].get('is_openxml_visual_slot') for m in matching_items
                    )
                    has_chart_caption_text = any(
                        m['item'].get('is_chart_caption_text') for m in matching_items
                    )
                    all_text_only = all(self._is_text_only_item(m['item']) for m in matching_items)
                    
                    # Special case: Docling 'picture' with multiple shapes
                    should_merge_picture = is_picture_label and len(matching_items) > 1 and has_shape_units
                    
                    # Merge if: multiple items AND same element AND not mixed parts
                    # OR: picture label with shapes
                    should_merge = (len(matching_items) > 1 and all_same_element and not has_mixed_parts and not has_image_part) or should_merge_picture
                    if is_picture_label and (has_table_units or has_image_part or has_pdf_image):
                        # Keep picture parts split by cell/image to avoid oversized bboxes.
                        should_merge = False
                    if is_table_label:
                        # Preserve per-cell output for tables while using canonical element mapping.
                        should_merge = False
                    
                    if should_merge:
                        # Merge all matching bboxes
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
                        
                        for m in matching_items:
                            merged_bbox = self.merge_bboxes(merged_bbox, m['item']['bbox'])
                            if m['item'].get('text'):
                                merged_text.append(m['item']['text'])
                            avg_overlap += m['overlap']
                            if m['item'].get('element_sequence'):
                                sequences.append(m['item']['element_sequence'])
                            if m['item'].get('element_id'):
                                elem_ids.append(m['item']['element_id'])
                            if m['item'].get('element_type'):
                                elem_types.append(m['item']['element_type'])
                            if m['item'].get('openxml_idx') is not None:
                                openxml_indices.append(m['item']['openxml_idx'])
                            if m['item'].get('block_kind'):
                                block_kinds.append(m['item'].get('block_kind'))
                            if m['item'].get('block_key'):
                                block_keys.append(m['item'].get('block_key'))
                            if m['item'].get('content_role'):
                                content_roles.append(m['item'].get('content_role'))
                            if m['item'].get('block_order') is not None:
                                block_orders.append(m['item'].get('block_order'))
                            if m['item'].get('alignment_confidence') is not None:
                                alignment_confidences.append(float(m['item'].get('alignment_confidence') or 0.0))
                            if m['item'].get('candidate_source'):
                                candidate_sources.append(m['item'].get('candidate_source'))
                        
                        avg_overlap /= len(matching_items)
                        
                        # Determine label
                        merged_label = doc_item.get('label')
                        if merged_label == 'picture':
                            if has_chart_caption_text:
                                merged_label = 'caption'
                            elif not has_pdf_image and not has_openxml_chart and not has_openxml_visual_slot:
                                merged_label = 'text'
                            elif not any(
                                m['item'].get('is_picture_area') for m in matching_items
                            ) and not any(
                                m['item'].get('has_shape_units') for m in matching_items
                            ):
                                if all_text_only:
                                    merged_label = 'caption'
                        if merged_label == 'table':
                            merged_label = self._resolve_table_prediction_label(matching_items)
                        
                        # Correct header/footer labels
                        merged_label = self.correct_header_footer_label(merged_label, merged_bbox)
                        
                        # For picture merges, use highest sequence (closest to next element)
                        ref_elem_id = None
                        ref_elem_seq = None
                        ref_elem_type = None
                        ref_openxml_idx = min(openxml_indices) if openxml_indices else None
                        
                        if should_merge_picture and sequences:
                            max_seq = max(sequences)
                            max_idx = sequences.index(max_seq)
                            ref_elem_seq = max_seq
                            ref_elem_id = elem_ids[max_idx] if max_idx < len(elem_ids) else None
                            ref_elem_type = elem_types[max_idx] if max_idx < len(elem_types) else None
                        
                        fused_results.append({
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
                            'is_picture_merge': should_merge_picture,
                            'docling_label': doc_item.get('label'),
                            'is_picture_area': any(m['item'].get('is_picture_area') for m in matching_items),
                            'has_shape_units': has_shape_units,
                            'has_pdf_image': has_pdf_image,
                            'has_table_units': has_table_units,
                            'is_text_only_item': all_text_only,
                            'is_openxml_chart': has_openxml_chart,
                            'is_openxml_visual_slot': has_openxml_visual_slot,
                            'is_chart_caption_text': has_chart_caption_text,
                            'visual_slot_promoted': any(
                                m['item'].get('visual_slot_promoted') for m in matching_items
                            ),
                            'repair_reason': next(
                                (m['item'].get('repair_reason') for m in matching_items if m['item'].get('repair_reason')),
                                None
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
                        })
                    else:
                        # Don't merge - add each item separately
                        for m in matching_items:
                            item = m['item']
                            
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
                            
                            final_label = self.correct_header_footer_label(final_label, item['bbox'])
                            
                            fused_results.append({
                                'bbox': item['bbox'],
                                'label': final_label,
                                'text': item.get('text', ''),
                                'overlap': m['overlap'],
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
        
        # Add remaining unmatched aligned items (no Docling match)
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
            
            # Correct header/footer labels
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
        
        # Post-pass: force picture label for image/shape areas that overlap any picture prediction
        if has_docling:
            picture_preds = [
                d for d in docling_predictions
                if d.get('label') == 'picture' and d.get('bbox')
            ]
            if picture_preds:
                for result in fused_results:
                    if result.get('label') == 'picture':
                        continue
                    if (
                        not result.get('has_pdf_image') and
                        not result.get('is_openxml_chart') and
                        not result.get('is_openxml_visual_slot')
                    ):
                        continue
                    bbox = result.get('bbox')
                    if not bbox:
                        continue
                    if any(self.calculate_overlap(bbox, d['bbox']) > 0 for d in picture_preds):
                        result['label'] = 'picture'
                        result['docling_label'] = 'picture'

        picture_results = [r for r in fused_results if r.get('label') == 'picture' and r.get('bbox')]

        # Promote explicit caption text (e.g., "Gambar 2.2") when close to a picture.
        if picture_results:
            for result in fused_results:
                if result.get('label') not in ('text', 'paragraph', 'unknown'):
                    continue
                text = (result.get('text') or '').strip()
                if not self._is_caption_candidate(text):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                if (bbox[3] - bbox[1]) > self.CAPTION_LINE_MAX_HEIGHT:
                    continue
                if self._has_item_above(bbox, picture_results) or self._has_item_below(bbox, picture_results):
                    result['label'] = 'caption'

        caption_results = [r for r in fused_results if r.get('label') == 'caption' and r.get('bbox')]
        if picture_results and caption_results:
            for result in fused_results:
                if result.get('label') not in ('text', 'paragraph', 'unknown'):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                if (bbox[3] - bbox[1]) > self.CAPTION_LINE_MAX_HEIGHT:
                    continue
                above_picture = self._has_item_above(bbox, picture_results)
                below_picture = self._has_item_below(bbox, picture_results)
                # Allow caption proximity without x-overlap to catch (a)/(b) markers.
                above_caption = self._has_item_above(
                    bbox,
                    caption_results,
                    require_x_overlap=False
                )
                below_caption = self._has_item_below(
                    bbox,
                    caption_results,
                    require_x_overlap=False
                )
                if (above_picture and below_caption) or (above_caption and below_picture):
                    result['label'] = 'caption'

        if picture_results:
            for result in fused_results:
                if result.get('label') != 'caption' or not result.get('is_openxml_chart'):
                    continue
                bbox = result.get('bbox')
                if not bbox or len(bbox) < 4:
                    continue
                caption_key = self._extract_figure_key(result.get('text'))
                if not caption_key:
                    continue

                neighbor_picture = None
                for picture in picture_results:
                    picture_bbox = picture.get('bbox')
                    if not picture_bbox or len(picture_bbox) < 4:
                        continue
                    if (
                        self._has_item_above(bbox, [picture], require_x_overlap=False) or
                        self._has_item_below(bbox, [picture], require_x_overlap=False)
                    ):
                        neighbor_picture = picture
                        break

                if not neighbor_picture:
                    continue

                picture_key = self._extract_figure_key(neighbor_picture.get('text'))
                if picture_key and picture_key != caption_key:
                    result['label'] = 'text'
                    result['element_id'] = None

        # Downgrade obvious Docling label noise when the bbox contains narrative text.
        for result in fused_results:
            label = result.get('label')
            text = (result.get('text') or '').strip()
            bbox = result.get('bbox')

            if label == 'table':
                if (
                    self._is_narrative_text(text) and
                    not result.get('has_table_units') and
                    not self._is_table_element_item(result)
                ):
                    result['label'] = 'text'
            elif label == 'picture':
                if (
                    self._is_narrative_text(text) and
                    not result.get('has_pdf_image') and
                    not result.get('has_shape_units') and
                    not result.get('is_openxml_chart') and
                    not result.get('is_openxml_visual_slot')
                ):
                    result['label'] = 'text'
            elif label == 'caption':
                has_picture_neighbor = False
                if bbox and picture_results:
                    has_picture_neighbor = (
                        self._has_item_above(bbox, picture_results) or
                        self._has_item_below(bbox, picture_results)
                    )
                if (
                    self._is_narrative_text(text, min_chars=100, min_words=12) and
                    not has_picture_neighbor
                ):
                    result['label'] = 'text'

        # Sort by reading order (line-aware)
        def sort_key(item):
            return item.get('bbox') or [0, 0, 0, 0]

        def compare(a, b):
            a_bbox = sort_key(a)
            b_bbox = sort_key(b)
            y_diff = a_bbox[1] - b_bbox[1]
            if abs(y_diff) > 10:
                return -1 if y_diff < 0 else 1
            x_diff = a_bbox[0] - b_bbox[0]
            return -1 if x_diff < 0 else (1 if x_diff > 0 else 0)

        from functools import cmp_to_key
        fused_results.sort(key=cmp_to_key(compare))
        
        return fused_results
