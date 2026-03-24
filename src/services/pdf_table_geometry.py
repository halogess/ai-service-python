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


def collect_horizontal_segments(page: fitz.Page) -> List[Tuple[float, float, float]]:
    """Collect horizontal line segments from page drawings"""
    segs = []
    y_eps = 1.5
    for d in page.get_drawings():
        color = d.get("color")
        fill = d.get("fill")
        
        # Skip white lines
        is_white = False
        if color:
            if len(color) >= 3 and all(c > 0.95 for c in color[:3]):
                is_white = True
        if not color and fill:
            if len(fill) >= 3 and all(c > 0.95 for c in fill[:3]):
                is_white = True
        
        if is_white:
            continue
        
        for it in d["items"]:
            kind = it[0]
            if kind == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.y - p2.y) <= y_eps:
                    x0, x1 = sorted([p1.x, p2.x])
                    segs.append((p1.y, x0, x1))
            elif kind == "re":
                r = it[1]
                segs.append((r.y0, r.x0, r.x1))
                segs.append((r.y1, r.x0, r.x1))
    return segs

def collect_vertical_segments(page: fitz.Page) -> List[Tuple[float, float, float]]:
    """Collect vertical line segments for cell detection"""
    segs = []
    x_eps = 1.5
    for d in page.get_drawings():
        color = d.get("color")
        fill = d.get("fill")
        
        is_white = False
        if color:
            if len(color) >= 3 and all(c > 0.95 for c in color[:3]):
                is_white = True
        if not color and fill:
            if len(fill) >= 3 and all(c > 0.95 for c in fill[:3]):
                is_white = True
        
        if is_white:
            continue
        
        for it in d["items"]:
            kind = it[0]
            if kind == "l":
                p1, p2 = it[1], it[2]
                if abs(p1.x - p2.x) <= x_eps:  # Vertical line
                    y0, y1 = sorted([p1.y, p2.y])
                    segs.append((p1.x, y0, y1))
            elif kind == "re":
                r = it[1]
                segs.append((r.x0, r.y0, r.y1))
                segs.append((r.x1, r.y0, r.y1))
    return segs

def merge_by_y_and_x(segs: List[Tuple[float, float, float]]) -> List[Dict]:
    """Merge horizontal segments by Y position."""
    y_merge = 2.0
    segs = sorted(segs, key=lambda s: s[0])
    levels = []
    for y, x0, x1 in segs:
        seg_str = f"[{round(x0,1)}-{round(x1,1)}]"
        if not levels or abs(y - levels[-1]["y"]) > y_merge:
            levels.append({
                "y": y, 
                "x0": x0, 
                "x1": x1, 
                "raw_segs": [seg_str],
                "raw_count": 1
            })
        else:
            levels[-1]["x0"] = min(levels[-1]["x0"], x0)
            levels[-1]["x1"] = max(levels[-1]["x1"], x1)
            levels[-1]["raw_segs"].append(seg_str)
            levels[-1]["raw_count"] += 1
    return levels

def get_segments_at_y(segs: List[Tuple[float, float, float]], y: float, tolerance: float = 3.0) -> List[Tuple[float, float]]:
    """Get all raw segments at a given Y level (within tolerance)"""
    return [(s[1], s[2]) for s in segs if abs(s[0] - y) <= tolerance]

def detect_column_boundaries_from_segments(segments_at_y: List[Tuple[float, float]]) -> List[float]:
    """Detect column boundaries from gaps between column segments."""
    if len(segments_at_y) < 2:
        return []
    
    # Filter out small segments (< 3pt)
    column_segs = [(x0, x1) for x0, x1 in segments_at_y if (x1 - x0) >= 3]
    
    if len(column_segs) < 2:
        return []
    
    # Sort by x0
    sorted_segs = sorted(column_segs, key=lambda s: s[0])
    
    boundaries = []
    for i in range(len(sorted_segs) - 1):
        right_edge = sorted_segs[i][1]
        left_edge = sorted_segs[i + 1][0]
        
        if left_edge > right_edge:
            boundaries.append((right_edge + left_edge) / 2)
    
    return boundaries

def guess_table_clips_from_hlines(page: fitz.Page, min_rules: int = 2) -> Tuple[List[fitz.Rect], List[Dict], List[Dict]]:
    """
    Detect table clips from horizontal lines using character group analysis.
    Returns (clips, grouping_log, clip_validation_log)
    """
    segs = collect_horizontal_segments(page)
    if not segs:
        return [], [], []
    levels = merge_by_y_and_x(segs)
    if not levels:
        return [], [], []
    
    # Get rawdict and build character groups for the page
    rawdict_data = page.get_text("rawdict")
    all_chars = collect_all_chars(rawdict_data)
    all_groups = find_overlapping_groups(all_chars)
    
    groups = []
    cur_group = [levels[0]]
    cur_x0 = levels[0]["x0"]
    cur_x1 = levels[0]["x1"]
    column_boundaries = []
    grouping_log = []
    
    for i, lv in enumerate(levels[1:], 1):
        prev_y = levels[i-1]["y"]
        curr_y = lv["y"]
        y_gap = curr_y - prev_y
        
        log_entry = {'i': i, 'prev_y': round(prev_y, 1), 'curr_y': round(curr_y, 1), 'y_gap': round(y_gap, 1)}
        
        if y_gap < 3:
            cur_group.append(lv)
            continue
        
        row_groups = get_groups_in_y_range(all_groups, prev_y, curr_y)
        
        is_separator = False
        segments_at_prev_y = get_segments_at_y(segs, prev_y)
        segment_boundaries = detect_column_boundaries_from_segments(segments_at_prev_y)
        
        if len(segment_boundaries) > 0:
            if not column_boundaries:
                column_boundaries = segment_boundaries
            else:
                if len(row_groups) > 0:
                    is_crossing, crossing_info = check_boundary_crossing(row_groups, column_boundaries)
                    if is_crossing:
                        is_separator = True
        
        if is_separator:
            groups.append(cur_group)
            cur_group = []
            cur_x0 = None
            cur_x1 = None
            column_boundaries = []
        elif not cur_group:
            if len(segment_boundaries) >= 1:
                cur_group = [lv]
                cur_x0 = lv["x0"]
                cur_x1 = lv["x1"]
                column_boundaries = segment_boundaries
        else:
            x0_diff = abs(lv["x0"] - cur_x0) if cur_x0 else 0
            x1_diff = abs(lv["x1"] - cur_x1) if cur_x1 else 0
            
            if x0_diff > 30 or x1_diff > 30:
                groups.append(cur_group)
                cur_group = [lv]
                cur_x0 = lv["x0"]
                cur_x1 = lv["x1"]
                column_boundaries = []
            else:
                cur_group.append(lv)
                cur_x0 = min(cur_x0, lv["x0"]) if cur_x0 else lv["x0"]
                cur_x1 = max(cur_x1, lv["x1"]) if cur_x1 else lv["x1"]
    
    groups.append(cur_group)
    
    clips = []
    for g in groups:
        if len(g) >= min_rules:
            x0 = min(v["x0"] for v in g)
            x1 = max(v["x1"] for v in g)
            y0 = g[0]["y"]
            y1 = g[-1]["y"]
            clip = fitz.Rect(x0, y0, x1, y1) & page.rect
            clips.append(clip)
    
    def has_horizontal_outlier(clip, all_groups, tolerance=10):
        clip_x0, clip_y0, clip_x1, clip_y1 = clip.x0, clip.y0, clip.x1, clip.y1
        for g in all_groups:
            text = g.get('text', '').strip()
            if not text:
                continue
            g_x0, g_y0, g_x1, g_y1 = g['merged_bbox']
            if g_y1 < clip_y0 - tolerance or g_y0 > clip_y1 + tolerance:
                continue
            if g_x0 < clip_x0 - tolerance or g_x1 > clip_x1 + tolerance:
                return True, {'outlier_x': [round(g_x0, 1), round(g_x1, 1)]}
        return False, {}
    
    validated_clips = []
    clip_validation_log = []
    for clip in clips:
        has_outlier, outlier_info = has_horizontal_outlier(clip, all_groups)
        if has_outlier:
            clip_validation_log.append({'status': 'REJECTED'})
        else:
            validated_clips.append(clip)
            clip_validation_log.append({'status': 'VALID'})
            
    return validated_clips, grouping_log, clip_validation_log

def detect_cells_from_segments(h_segs, v_segs, clip_bbox, char_groups=None):
    """Detect cells from horizontal segments within a clip area."""
    x0_clip, y0_clip, x1_clip, y1_clip = clip_bbox
    tolerance = 3.0
    merge_tolerance = 10.0
    min_segment_width = 3.0
    
    def merge_close_positions(positions, merge_tol):
        if not positions:
            return []
        sorted_pos = sorted(positions)
        merged = [sorted_pos[0]]
        for pos in sorted_pos[1:]:
            if pos - merged[-1] > merge_tol:
                merged.append(pos)
        return merged
    
    # Get unique Y positions
    y_positions = set()
    for y, sx0, sx1 in h_segs:
        if y0_clip - tolerance <= y <= y1_clip + tolerance:
            y_positions.add(round(y, 1))
    y_positions = sorted(y_positions)
    y_positions = merge_close_positions(y_positions, merge_tolerance)
    
    cells = []
    first_row_boundaries = None
    
    def detect_column_boundaries_from_text(groups, y_top, y_bottom):
        if not groups: return []
        row_groups = []
        for g in groups:
            g_bbox = g.get('merged_bbox')
            if not g_bbox: continue
            g_y0, g_y1 = g_bbox[1], g_bbox[3]
            g_height = g_y1 - g_y0
            if g_height <= 0: continue
            overlap_start = max(g_y0, y_top)
            overlap_end = min(g_y1, y_bottom)
            if (overlap_end - overlap_start) / g_height >= 0.5:
                row_groups.append(g)
        
        if len(row_groups) < 2: return []
        row_groups.sort(key=lambda g: g['merged_bbox'][0])
        boundaries = []
        for j in range(len(row_groups) - 1):
            right = row_groups[j]['merged_bbox'][2]
            left = row_groups[j+1]['merged_bbox'][0]
            if left - right >= 5:
                boundaries.append((right + left) / 2)
        return boundaries

    for row_idx in range(len(y_positions) - 1):
        y_top = y_positions[row_idx]
        y_bottom = y_positions[row_idx + 1]
        
        segments_at_row = get_segments_at_y(h_segs, y_top, merge_tolerance)
        cell_segments = [(x0, x1) for x0, x1 in segments_at_row if (x1 - x0) >= min_segment_width]
        cell_segments = sorted(cell_segments, key=lambda s: s[0])
        
        merged_segments = []
        for seg in cell_segments:
            if not merged_segments:
                merged_segments.append(list(seg))
            else:
                prev = merged_segments[-1]
                if seg[0] < prev[1]:
                    prev[1] = max(prev[1], seg[1])
                else:
                    merged_segments.append(list(seg))
        
        if len(merged_segments) > 1 and first_row_boundaries is None:
            first_row_boundaries = []
            for i in range(len(merged_segments) - 1):
                r = merged_segments[i][1]
                l = merged_segments[i+1][0]
                first_row_boundaries.append((r+l)/2)
        
        if len(merged_segments) <= 1 and char_groups:
            text_boundaries = detect_column_boundaries_from_text(char_groups, y_top, y_bottom)
            if text_boundaries:
                all_x = [x0_clip] + sorted(text_boundaries) + [x1_clip]
                merged_segments = [[all_x[i], all_x[i+1]] for i in range(len(all_x)-1)]
        
        if len(merged_segments) <= 1 and first_row_boundaries:
            all_x = [x0_clip] + first_row_boundaries + [x1_clip]
            merged_segments = [[all_x[i], all_x[i+1]] for i in range(len(all_x)-1)]
            
        for col_idx, (seg_x0, seg_x1) in enumerate(merged_segments):
            cells.append({
                'bbox': [seg_x0, y_top, seg_x1, y_bottom],
                'row': row_idx,
                'col': col_idx
            })
            
    return cells, {}
