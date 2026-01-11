
import fitz
import base64
from typing import List, Dict, Any, Tuple, Optional
from models import Dokumen
from utils.char_grouping import (
    collect_all_chars, find_overlapping_groups, 
    get_groups_in_y_range, check_boundary_crossing
)

class MergingExtractionService:
    def __init__(self):
        pass

    def extract_and_process(self, doc_id: int, page_num: int) -> Dict[str, Any]:
        """
        Main entry point: Extracts raw data and processes it into a unified, sorted list of items.
        Mirrors the flow of:
        1. Backend: extract_merging (pymupdf_routes.py)
        2. Frontend: processMergingResponse (pdf_extraction.js)
        """
        # 1. Extract Raw Data (Backend Logic)
        raw_data = self._extract_raw_data(doc_id, page_num)
        
        # 2. Process and Normalize Data (Frontend Logic Port)
        processed_result = self._process_merging_response(raw_data)
        
        return processed_result

    def _extract_raw_data(self, doc_id: int, page_num: int) -> Dict[str, Any]:
        """
        Extracts raw PDF data: char groups, tables, images, shapes.
        Logic ported from `extract_merging` in `pymupdf_routes.py`.
        """
        document = Dokumen.query.get(doc_id)
        if not document:
             raise ValueError(f"Document {doc_id} not found")

        with fitz.open(document.dokumen_pdf_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                raise ValueError(f"Page {page_num} out of range")
            
            page = pdf[page_num - 1]
            width = page.rect.width
            height = page.rect.height

            # Get rawdict for character grouping
            rawdict_data = page.get_text("rawdict")
            rawdict_data = self._sanitize(rawdict_data)
            
            # Step 1: Character grouping
            all_chars = collect_all_chars(rawdict_data)
            char_groups = find_overlapping_groups(all_chars)

            # Step 1.5: Detect page images and shapes
            page_images = page.get_image_info()
            shapes_list = []
            page_images_list = []

            for img in page_images:
                img_bbox = list(img['bbox'])
                img_width = img.get('width', 0)
                img_height = img.get('height', 0)

                # Find overlapping groups ( >= 50% Y overlap)
                overlapping_groups = []
                for group in char_groups:
                    if self._group_overlaps_image(group.get('merged_bbox'), img_bbox):
                        overlapping_groups.append(group)
                
                if overlapping_groups:
                    # SHAPE (Image + Text)
                    # Sort groups by reading order
                    overlapping_groups.sort(key=lambda g: (g['merged_bbox'][1], g['merged_bbox'][0]))
                    
                    # Merge bboxes
                    all_bboxes = [img_bbox] + [g['merged_bbox'] for g in overlapping_groups]
                    merged_bbox = self._merge_bboxes(all_bboxes)
                    
                    merged_text = ' '.join([g.get('text', '') for g in overlapping_groups]).strip()
                    
                    shapes_list.append({
                        'type': 'shape',
                        'bbox': merged_bbox,
                        'text': merged_text,
                        'image_bbox': img_bbox,
                        'image_xref': img.get('xref'),
                        'groups_count': len(overlapping_groups),
                        'claimed_groups_texts': [g.get('text', '')[:30] for g in overlapping_groups]
                    })
                    
                    # Mark groups as claimed
                    for g in overlapping_groups:
                        g['claimed_by_shape'] = True
                else:
                    # Pure Image
                    if img_width >= 50 and img_height >= 50:
                        page_images_list.append({
                            'bbox': img_bbox,
                            'xref': img.get('xref'),
                            'width': img_width,
                            'height': img_height,
                            'name': img.get('name', '')
                        })

            # Step 2: Find Tables (Basic for now, can implement the complex logic later if needed)
            basic_tables_finder = page.find_tables()
            basic_tables = basic_tables_finder.tables if hasattr(basic_tables_finder, 'tables') else list(basic_tables_finder)
            
            basic_table_list = []
            for t_idx, table in enumerate(basic_tables):
                table_bbox = list(table.bbox)
                
                # Check overlap with shapes
                overlaps_shape = False
                for shape in shapes_list:
                    if self._simple_bbox_overlap(table_bbox, shape['bbox']):
                        overlaps_shape = True
                        break
                
                if overlaps_shape:
                    continue

                basic_table_list.append({
                    'table_index': t_idx,
                    'bbox': table_bbox,
                    'row_count': table.row_count,
                    'col_count': table.col_count,
                    'cells': [list(c) for c in table.cells] if table.cells else []
                })
                
                # Mark groups as claimed by table
                for group in char_groups:
                    if not group.get('claimed_by_shape') and self._bbox_overlaps(group['merged_bbox'], table_bbox):
                        group['claimed_by_table'] = True

            # Filter unclaimed groups
            unclaimed_groups = [g for g in char_groups if not g.get('claimed_by_shape') and not g.get('claimed_by_table')]

            return {
                'width': width,
                'height': height,
                'char_groups': unclaimed_groups,
                'tables': basic_table_list,
                'shapes': shapes_list,
                'page_images': page_images_list
            }

    def _process_merging_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalization and Sorting.
        Logic ported from `pdf_extraction.js` (processMergingResponse + sortByReadingOrder).
        """
        items = []

        # 1. Normalize items into unified structure
        # Group
        for group in data.get('char_groups', []):
            items.append({
                'type': 'group',
                'data': {
                    'text': group.get('text'),
                    'mergedBbox': group.get('merged_bbox'),
                    'isSingle': group.get('is_single'),
                    'blockIdx': group.get('block_idx')
                },
                'bbox': group.get('merged_bbox')
            })
        
        # Table
        for table in data.get('tables', []):
            items.append({
                'type': 'table',
                'data': table,
                'bbox': table.get('bbox')
            })

        # Shape
        for shape in data.get('shapes', []):
            items.append({
                'type': 'shape',
                'data': shape,
                'bbox': shape.get('bbox')
            })
            
        # Image
        for img in data.get('page_images', []):
            items.append({
                'type': 'image',
                'data': img,
                'bbox': img.get('bbox')
            })

        # 2. Sort by Reading Order
        items = self._sort_by_reading_order(items)

        return {
            'items': items,
            'width': data['width'],
            'height': data['height'],
            'stats': {
                'total': len(items),
                'groups': len([i for i in items if i['type'] == 'group']),
                'tables': len([i for i in items if i['type'] == 'table']),
                'shapes': len([i for i in items if i['type'] == 'shape']),
                'images': len([i for i in items if i['type'] == 'image'])
            }
        }

    def _sort_by_reading_order(self, items: List[Dict]) -> List[Dict]:
        """
        Sorts items by Y, then X.
        Logic ported from `pdf_extraction.js` -> `sortByReadingOrder`
        """
        def compare_items(a, b):
            if not a.get('bbox') or not b.get('bbox'):
                return 0
            
            yA0, _, _, yA1 = a['bbox']
            yB0, _, _, yB1 = b['bbox']
            
            heightA = yA1 - yA0
            heightB = yB1 - yB0
            
            overlap_start = max(yA0, yB0)
            overlap_end = min(yA1, yB1)
            overlap_amount = max(0, overlap_end - overlap_start)
            
            smaller_height = min(heightA, heightB)
            overlap_ratio = overlap_amount / smaller_height if smaller_height > 0 else 0
            
            OVERLAP_THRESHOLD = 0.30
            is_same_line = overlap_ratio >= OVERLAP_THRESHOLD
            
            if is_same_line:
                # Same line: sort by X
                return a['bbox'][0] - b['bbox'][0]
            else:
                # Different lines: sort by Y
                return yA0 - yB0
        
        from functools import cmp_to_key
        return sorted(items, key=cmp_to_key(compare_items))

    # --- Helpers ---

    def _sanitize(self, obj):
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        elif isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(i) for i in obj]
        return obj

    def _merge_bboxes(self, bboxes):
        if not bboxes:
            return None
        x0 = min(b[0] for b in bboxes)
        y0 = min(b[1] for b in bboxes)
        x1 = max(b[2] for b in bboxes)
        y1 = max(b[3] for b in bboxes)
        return [x0, y0, x1, y1]

    def _group_overlaps_image(self, group_bbox, img_bbox, min_y_overlap_ratio=0.5):
        if not group_bbox or not img_bbox:
            return False
        
        # X overlap
        if group_bbox[2] < img_bbox[0] or group_bbox[0] > img_bbox[2]:
            return False
        
        # Y overlap
        group_height = group_bbox[3] - group_bbox[1]
        if group_height <= 0:
            return False
        
        y_overlap_start = max(group_bbox[1], img_bbox[1])
        y_overlap_end = min(group_bbox[3], img_bbox[3])
        y_overlap_amount = max(0, y_overlap_end - y_overlap_start)
        
        return (y_overlap_amount / group_height) >= min_y_overlap_ratio

    def _simple_bbox_overlap(self, bbox1, bbox2):
        if not bbox1 or not bbox2:
            return False
        return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                    bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])

    def _bbox_overlaps(self, bbox1, bbox2, y_threshold=0.7):
        """Checks if bbox1 is 'inside' bbox2 with Y threshold"""
        if not bbox1 or not bbox2:
            return False
        
        if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]:
            return False
            
        y_overlap_start = max(bbox1[1], bbox2[1])
        y_overlap_end = min(bbox1[3], bbox2[3])
        y_overlap = max(0, y_overlap_end - y_overlap_start)
        
        bbox1_height = bbox1[3] - bbox1[1]
        if bbox1_height <= 0:
            return False
            
        return (y_overlap / bbox1_height) >= y_threshold
