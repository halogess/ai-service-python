"""
Docling Fusion Service
Handles fusion of Docling predictions with alignment bboxes.
Ports logic from classification.html's fuseAlignmentWithDocling() function.
"""

import re
from typing import Dict, Any, List, Optional, Tuple


class DoclingFusionService:
    """
    Service to fuse Docling classification predictions with alignment bboxes.
    
    Key operations:
    1. Calculate overlap between alignment bboxes and Docling predictions
    2. Correct page_header/page_footer labels based on margin zones
    3. Merge multiple shapes under same Docling 'picture' label
    4. Generate fallback labels from element_type when no Docling match
    """
    
    OVERLAP_THRESHOLD = 0.3  # 30% overlap required for matching
    CAPTION_LINE_MAX_HEIGHT = 24
    CAPTION_VERTICAL_GAP_MAX = 60
    CAPTION_X_OVERLAP_MIN = 0.3
    CAPTION_TEXT_REGEX = re.compile(
        r'^\s*(?:gambar|figure|fig\.?|tabel|table)\s*\d',
        re.IGNORECASE
    )
    
    def __init__(self, section_data: Optional[Dict] = None):
        """
        Initialize fusion service.
        
        Args:
            section_data: Dict with page margins and dimensions:
                - page_height_pt: Page height in PDF points (default 842)
                - margin_top_pt: Top margin in points (default 72)
                - margin_bottom_pt: Bottom margin in points (default 72)
        """
        self.section_data = section_data or {}
    
    def calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate overlap ratio between two bboxes.
        Uses intersection-over-minimum-area ratio.
        
        Args:
            bbox1: [x0, y0, x1, y1]
            bbox2: [x0, y0, x1, y1]
            
        Returns:
            Overlap ratio (0.0 to 1.0)
        """
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # Calculate intersection
        x0 = max(bbox1[0], bbox2[0])
        y0 = max(bbox1[1], bbox2[1])
        x1 = min(bbox1[2], bbox2[2])
        y1 = min(bbox1[3], bbox2[3])
        
        if x0 >= x1 or y0 >= y1:
            return 0.0
        
        intersection_area = (x1 - x0) * (y1 - y0)
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Use smaller area for ratio calculation
        min_area = min(bbox1_area, bbox2_area)
        return intersection_area / min_area if min_area > 0 else 0.0
    
    def get_bbox_margin_zone(self, bbox: List[float]) -> Optional[str]:
        """
        Determine if bbox center is in header or footer margin zone.
        
        Args:
            bbox: [x0, y0, x1, y1]
            
        Returns:
            'header', 'footer', or None (body area)
        """
        if not bbox or len(bbox) < 4 or not self.section_data:
            return None
        
        page_height = self.section_data.get('page_height_pt', 842)
        margin_top = self.section_data.get('margin_top_pt', 72)
        margin_bottom = self.section_data.get('margin_bottom_pt', 72)
        
        # Calculate Y center of bbox
        y_center = (bbox[1] + bbox[3]) / 2
        
        # Header zone: Y center < marginTop
        if y_center < margin_top:
            return 'header'
        
        # Footer zone: Y center > (pageHeight - marginBottom)
        if y_center > page_height - margin_bottom:
            return 'footer'
        
        return None
    
    def correct_header_footer_label(self, label: str, bbox: List[float]) -> str:
        """
        Correct page_header/page_footer labels if not in appropriate margin zone.
        
        Args:
            label: Original Docling label
            bbox: [x0, y0, x1, y1]
            
        Returns:
            Corrected label (may be changed to 'text')
        """
        if label not in ('page_header', 'page_footer'):
            return label
        
        zone = self.get_bbox_margin_zone(bbox)
        
        if label == 'page_header' and zone != 'header':
            return 'text'
        
        if label == 'page_footer' and zone != 'footer':
            return 'text'
        
        return label
    
    def fallback_label(self, item: Dict) -> str:
        """
        Generate fallback label from alignment element_type when no Docling match.
        
        Args:
            item: Aligned item with element_type, source, zone, etc.
            
        Returns:
            Appropriate label string
        """
        if not item:
            return 'text'
        
        # Header/footer source
        if item.get('source') == 'header_footer' and item.get('zone'):
            return 'page_header' if item['zone'] == 'header' else 'page_footer'
        
        element_type = str(item.get('element_type', '')).lower()
        
        if 'table' in element_type:
            return 'table'
        if 'list' in element_type:
            return 'list_item'
        if 'caption' in element_type:
            return 'caption'
        if 'title' in element_type:
            return 'title'
        if 'header' in element_type:
            return 'section_header'
        if 'footer' in element_type:
            return 'page_footer'
        if 'formula' in element_type:
            return 'formula'
        if 'code' in element_type:
            return 'code'
        if 'paragraph' in element_type:
            return 'paragraph'
        
        return 'text'
    
    def is_picture_area(self, item: Dict) -> bool:
        """
        Check if item contains image or shape content.
        
        Args:
            item: Aligned item to check
            
        Returns:
            True if item has image/shape content
        """
        if not item:
            return False
        return bool(
            item.get('is_image_part') or 
            item.get('has_pdf_image') or 
            item.get('has_shape_units')
        )

    def _is_text_only_item(self, item: Dict) -> bool:
        if not item:
            return False
        if item.get('source') != 'alignment':
            return False
        element_type = str(item.get('element_type', '')).lower()
        if 'table' in element_type or item.get('has_table_units'):
            return False
        return not (
            item.get('is_image_part') or
            item.get('has_pdf_image') or
            item.get('has_shape_units')
        )

    def _is_caption_candidate(self, text: Optional[str]) -> bool:
        if not text:
            return False
        return bool(self.CAPTION_TEXT_REGEX.match(text.strip()))
    
    @staticmethod
    def merge_bboxes(bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> Optional[List[float]]:
        """Merge two bboxes into one encompassing bbox."""
        if not bbox1:
            return bbox2
        if not bbox2:
            return bbox1
        return [
            min(bbox1[0], bbox2[0]),  # x0
            min(bbox1[1], bbox2[1]),  # y0
            max(bbox1[2], bbox2[2]),  # x1
            max(bbox1[3], bbox2[3])   # y1
        ]

    def _x_overlap_ratio(self, bbox1: Optional[List[float]], bbox2: Optional[List[float]]) -> float:
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        x0 = max(bbox1[0], bbox2[0])
        x1 = min(bbox1[2], bbox2[2])
        if x0 >= x1:
            return 0.0
        w1 = bbox1[2] - bbox1[0]
        w2 = bbox2[2] - bbox2[0]
        min_w = min(w1, w2)
        return (x1 - x0) / min_w if min_w > 0 else 0.0

    def _has_item_above(
        self,
        bbox: List[float],
        items: List[Dict],
        require_x_overlap: bool = True
    ) -> bool:
        for item in items:
            ibox = item.get('bbox')
            if not ibox or len(ibox) < 4:
                continue
            if ibox[3] <= bbox[1]:
                gap = bbox[1] - ibox[3]
                if gap <= self.CAPTION_VERTICAL_GAP_MAX and (
                    not require_x_overlap or
                    self._x_overlap_ratio(bbox, ibox) >= self.CAPTION_X_OVERLAP_MIN
                ):
                    return True
        return False

    def _has_item_below(
        self,
        bbox: List[float],
        items: List[Dict],
        require_x_overlap: bool = True
    ) -> bool:
        for item in items:
            ibox = item.get('bbox')
            if not ibox or len(ibox) < 4:
                continue
            if ibox[1] >= bbox[3]:
                gap = ibox[1] - bbox[3]
                if gap <= self.CAPTION_VERTICAL_GAP_MAX and (
                    not require_x_overlap or
                    self._x_overlap_ratio(bbox, ibox) >= self.CAPTION_X_OVERLAP_MIN
                ):
                    return True
        return False
    
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
                            'openxml_idx': parent_openxml_idx,
                            'has_pdf_image': has_image,
                            'has_shape_units': has_shape,
                            'has_table_units': True,
                            'is_picture_area': has_image or has_shape
                        })
            elif alignment.get('merged_bbox'):
                # Non-table elements
                matched_units = alignment.get('matched_pdf_units', [])
                has_shape = any(u.get('item_type') == 'shape' for u in matched_units)
                has_image = any(u.get('item_type') == 'image' for u in matched_units)
                has_table_units = any(u.get('item_type') in ('table', 'hline_table') for u in matched_units)
                is_picture_area = bool(
                    alignment.get('is_image_part') or has_shape or has_image
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
                    'is_picture_area': is_picture_area
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
                    'is_picture_area': False
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
                    all_text_only = all(self._is_text_only_item(m['item']) for m in matching_items)
                    
                    # Special case: Docling 'picture' with multiple shapes
                    should_merge_picture = is_picture_label and len(matching_items) > 1 and has_shape_units
                    
                    # Merge if: multiple items AND same element AND not mixed parts
                    # OR: picture label with shapes
                    should_merge = (len(matching_items) > 1 and all_same_element and not has_mixed_parts and not has_image_part) or should_merge_picture
                    if is_picture_label and (has_table_units or has_image_part or has_pdf_image):
                        # Keep picture parts split by cell/image to avoid oversized bboxes.
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
                        
                        avg_overlap /= len(matching_items)
                        
                        # Determine label
                        merged_label = doc_item.get('label')
                        if merged_label == 'picture':
                            if not has_pdf_image:
                                merged_label = 'text'
                            elif not any(
                                m['item'].get('is_picture_area') for m in matching_items
                            ) and not any(
                                m['item'].get('has_shape_units') for m in matching_items
                            ):
                                if all_text_only:
                                    merged_label = 'caption'
                        
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
                            'is_text_only_item': all_text_only
                        })
                    else:
                        # Don't merge - add each item separately
                        for m in matching_items:
                            item = m['item']
                            
                            final_label = doc_item.get('label')
                            if final_label == 'picture':
                                if not item.get('has_pdf_image'):
                                    final_label = 'text'
                                elif not item.get('is_picture_area') and not item.get('has_shape_units'):
                                    if self._is_text_only_item(item):
                                        final_label = 'caption'
                            
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
                                'is_text_only_item': self._is_text_only_item(item)
                            })
        
        # Add remaining unmatched aligned items (no Docling match)
        for idx, item in enumerate(all_aligned_items):
            if idx in used_indices:
                continue
            
            label = 'unknown'
            if item.get('source') == 'header_footer' and item.get('zone'):
                label = 'page_header' if item['zone'] == 'header' else 'page_footer'
            elif item.get('is_image_part') and item.get('has_pdf_image'):
                label = 'picture'
            elif item.get('is_image_part'):
                label = self.fallback_label(item)
            
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
                'is_text_only_item': self._is_text_only_item(item)
            })
        
        # Post-pass: force picture label for image/shape areas that overlap any picture prediction
        if has_docling:
            picture_preds = [
                d for d in docling_predictions
                if d.get('label') == 'picture' and d.get('bbox')
            ]
            if picture_preds:
                for result in fused_results:
                    if result.get('label') == 'picture' or not result.get('has_pdf_image'):
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
