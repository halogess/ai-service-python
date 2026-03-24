"""
PDF Extraction Service
Handles PDF processing using PyMuPDF (fitz) - extracts text, images, and char groups.
Ports logic from pymupdf_routes.py for robust extraction.
"""

import fitz
import os
import logging
import base64
from typing import Optional, List, Dict, Any, Tuple
from utils.char_grouping import (
    collect_all_chars, 
    find_overlapping_groups,
    get_groups_in_y_range,
    check_boundary_crossing
)

logger = logging.getLogger(__name__)

# =============================================================================
# Helper functions for segment and table detection
# Ported from pymupdf_routes.py
# =============================================================================

from services.pdf_table_geometry import (
    collect_horizontal_segments,
    collect_vertical_segments,
    merge_by_y_and_x,
    get_segments_at_y,
    detect_column_boundaries_from_segments,
    guess_table_clips_from_hlines,
    detect_cells_from_segments,
)


class PDFExtractor:
    """Service class for extracting content from PDF files"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = None
        
    def __enter__(self):
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def open(self):
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        logger.info(f"Opened PDF: {self.pdf_path} ({self.doc.page_count} pages)")
        
    def close(self):
        if self.doc:
            self.doc.close()
            self.doc = None
            
    @property
    def page_count(self) -> int:
        return self.doc.page_count if self.doc else 0
    
    def get_page(self, page_num: int):
        if not self.doc: raise RuntimeError("PDF not opened")
        if page_num < 0 or page_num >= self.doc.page_count:
            raise ValueError(f"Invalid page number: {page_num}")
        return self.doc[page_num]
    
    def extract_merging_data(self, page_num: int) -> dict:
        """
        Comprehensive extraction for merging process (Text + Tables + Images + Shapes).
        Replicates extract_merging logic from legacy routes.
        """
        page = self.get_page(page_num)
        
        # 1. Get rawdict and char grouping
        rawdict_data = page.get_text("rawdict")
        
        # Sanitize
        def sanitize(obj):
            if isinstance(obj, bytes):
                return obj.decode('utf-8', errors='replace')
            elif isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(i) for i in obj]
            return obj
        rawdict_data = sanitize(rawdict_data)
        
        all_chars = collect_all_chars(rawdict_data)
        char_groups = find_overlapping_groups(all_chars)
        
        # 2. Detect Images and Shapes
        page_images = page.get_image_info()
        page_images_list = []
        shapes_list = []
        
        def merge_bboxes(bboxes):
            if not bboxes: return None
            x0 = min(b[0] for b in bboxes)
            y0 = min(b[1] for b in bboxes)
            x1 = max(b[2] for b in bboxes)
            y1 = max(b[3] for b in bboxes)
            return [x0, y0, x1, y1]
        
        def group_overlaps_image(group_bbox, img_bbox):
            if not group_bbox or not img_bbox: return False
            if group_bbox[2] < img_bbox[0] or group_bbox[0] > img_bbox[2]: return False
            g_h = group_bbox[3] - group_bbox[1]
            if g_h <= 0: return False
            y_start = max(group_bbox[1], img_bbox[1])
            y_end = min(group_bbox[3], img_bbox[3])
            return (max(0, y_end - y_start) / g_h) >= 0.5

        def simple_bbox_overlap(bbox1, bbox2):
            if not bbox1 or not bbox2: return False
            return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                        bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
        
        def bbox_overlaps(bbox1, bbox2, y_threshold=0.7):
            """Check if bbox1 overlaps bbox2 with a minimum Y overlap ratio."""
            if not bbox1 or not bbox2:
                return False
            
            # Require any X overlap
            if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]:
                return False
            
            y_overlap_start = max(bbox1[1], bbox2[1])
            y_overlap_end = min(bbox1[3], bbox2[3])
            y_overlap = max(0, y_overlap_end - y_overlap_start)
            
            bbox1_height = bbox1[3] - bbox1[1]
            if bbox1_height <= 0:
                return False
            
            return (y_overlap / bbox1_height) >= y_threshold

        for img in page_images:
            img_bbox = img.get('bbox')
            if not img_bbox: continue
            
            # Find all char_groups that overlap with this image
            overlapping_groups = [g for g in char_groups if g.get('merged_bbox') and group_overlaps_image(g.get('merged_bbox'), img_bbox)]
            
            if overlapping_groups:
                # Shape
                # Match legacy reading order: if Y overlaps, sort by X; else by Y
                from functools import cmp_to_key

                def compare_groups(g1, g2):
                    b1 = g1.get('merged_bbox', [0, 0, 0, 0])
                    b2 = g2.get('merged_bbox', [0, 0, 0, 0])
                    y_overlap = not (b1[3] < b2[1] or b2[3] < b1[1])
                    if y_overlap:
                        return -1 if b1[0] < b2[0] else (1 if b1[0] > b2[0] else 0)
                    return -1 if b1[1] < b2[1] else (1 if b1[1] > b2[1] else 0)

                overlapping_groups.sort(key=cmp_to_key(compare_groups))
                merged_text = ' '.join([g.get('text', '') for g in overlapping_groups]).strip()
                group_bboxes = [g['merged_bbox'] for g in overlapping_groups]
                merged_bbox = merge_bboxes([list(img_bbox)] + group_bboxes)
                
                content = [{
                    'type': 'text',
                    'text': merged_text,
                    'bbox': merge_bboxes(group_bboxes),
                    'groups_count': len(overlapping_groups)
                }]
                
                shapes_list.append({
                    'type': 'shape',
                    'bbox': merged_bbox,
                    'text': merged_text,
                    'image_bbox': list(img_bbox),
                    'content': content
                })
                
                for g in overlapping_groups:
                    g['claimed_by_shape'] = True
            else:
                # Standalone Image
                if img.get('width', 0) >= 50 and img.get('height', 0) >= 50:
                    page_images_list.append({
                        'bbox': list(img_bbox),
                        'xref': img.get('xref'),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'name': img.get('name', '')
                    })

        # 3. Basic Tables (fitz)
        basic_tables_finder = page.find_tables()
        basic_tables = getattr(basic_tables_finder, 'tables', []) or list(basic_tables_finder)
        
        basic_table_list = []
        for t_idx, table in enumerate(basic_tables):
            table_bbox = list(table.bbox) if table.bbox else None
            # Skip if overlaps shape
            if table_bbox and any(simple_bbox_overlap(table_bbox, s['bbox']) for s in shapes_list):
               continue
               
            rows_list = []
            for r_idx, row in enumerate(table.rows):
                row_cells = []
                for c_idx, cell in enumerate(row.cells):
                    cell_content = []
                    if cell:
                        cell_bbox = list(cell)
                        # Find overlapping char groups (>=50% Y overlap)
                        cell_groups = [
                            g for g in char_groups
                            if not g.get('claimed_by_shape')
                            and g.get('merged_bbox')
                            and bbox_overlaps(g.get('merged_bbox'), cell_bbox, y_threshold=0.5)
                        ]
                        if cell_groups:
                             for g in cell_groups: g['claimed_by_table'] = True
                             merged_text = ' '.join([g.get('text', '') for g in cell_groups])
                             cell_content.append({'type': 'text', 'text': merged_text, 'bbox': merge_bboxes([g['merged_bbox'] for g in cell_groups])})

                        # Add images overlapping this cell (>=50% Y overlap)
                        for img in page_images_list:
                            img_bbox = img.get('bbox')
                            if img_bbox and bbox_overlaps(img_bbox, cell_bbox, y_threshold=0.5):
                                cell_content.append({
                                    'type': 'image',
                                    'bbox': img_bbox,
                                    'xref': img.get('xref'),
                                    'width': img.get('width'),
                                    'height': img.get('height'),
                                    'name': img.get('name', '')
                                })

                        row_cells.append({'row': r_idx, 'col': c_idx, 'bbox': cell_bbox, 'content': cell_content})
                        
                rows_list.append({'cells': row_cells})
                
            basic_table_list.append({
                'table_index': t_idx,
                'bbox': table_bbox,
                'rows': rows_list
            })

        # 4. H-Line Tables
        hline_clips, _, _ = guess_table_clips_from_hlines(page)
        hline_table_list = []
        
        for clip in hline_clips:
            clip_bbox = [clip.x0, clip.y0, clip.x1, clip.y1]
            if any(bbox_overlaps(clip_bbox, bt['bbox']) for bt in basic_table_list): continue
            if any(bbox_overlaps(clip_bbox, im['bbox'], y_threshold=0.0) for im in page_images_list): continue
            
            # Check for unclaimed groups (>=30% Y overlap)
            clip_groups = [
                g for g in char_groups
                if not g.get('claimed_by_shape')
                and not g.get('claimed_by_table')
                and g.get('merged_bbox')
                and bbox_overlaps(g.get('merged_bbox'), clip_bbox, y_threshold=0.3)
            ]
            
            if not clip_groups: continue
            
            for g in clip_groups: g['claimed_by_table'] = True
            
            # Detect cells
            h_segs = collect_horizontal_segments(page)
            v_segs = collect_vertical_segments(page)
            
            cells, _ = detect_cells_from_segments(h_segs, v_segs, clip_bbox, clip_groups)
            
            # Map content to cells
            rows_dict = {}
            for cell in cells:
                r, c = cell['row'], cell['col']
                if r not in rows_dict: rows_dict[r] = {'cells': []}
                
                cell_bbox = cell['bbox']
                cell_content_groups = [
                    g for g in clip_groups
                    if bbox_overlaps(g['merged_bbox'], cell_bbox, y_threshold=0.5)
                ]
                cell_content = []
                if cell_content_groups:
                    combined_text = ' '.join([g.get('text', '') for g in cell_content_groups])
                    cell_content.append({'type': 'text', 'text': combined_text})
                
                rows_dict[r]['cells'].append({
                    'row': r, 'col': c, 'bbox': cell_bbox, 'content': cell_content
                })
            
            hline_table_list.append({
                'bbox': clip_bbox,
                'rows': [rows_dict[r] for r in sorted(rows_dict.keys())]
            })

        # 5. Finalize unclaimed groups
        unclaimed_groups = [g for g in char_groups if not g.get('claimed_by_shape') and not g.get('claimed_by_table')]
        
        return {
            'success': True,
            'width': page.rect.width,
            'height': page.rect.height,
            'char_groups': unclaimed_groups,
            'shapes': shapes_list,
            'page_images': page_images_list,
            'basic_tables': basic_table_list,
            'hline_tables': hline_table_list
        }

def extract_pdf_merging_data(pdf_path: str, page_num: int) -> dict:
    with PDFExtractor(pdf_path) as extractor:
        return extractor.extract_merging_data(page_num)
