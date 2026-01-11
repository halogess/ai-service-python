"""
PyMuPDF Extraction Routes
Routes for extracting PDF content using various PyMuPDF methods.
"""
from flask import Blueprint, render_template, jsonify
import fitz
import base64

from models import db, TestingDokumen
from utils.char_grouping import (
    collect_all_chars, find_overlapping_groups, 
    get_groups_in_y_range, check_boundary_crossing
)

pymupdf_bp = Blueprint('pymupdf', __name__)


@pymupdf_bp.route('/pymupdf-documents')
def pymupdf_documents():
    """List all testing documents for PyMuPDF extraction"""
    documents = TestingDokumen.query.all()
    return render_template('pymupdf_documents.html', documents=documents)


@pymupdf_bp.route('/pymupdf-extract/<int:doc_id>')
def pymupdf_extract(doc_id):
    """Render extraction page for a specific document"""
    document = TestingDokumen.query.get_or_404(doc_id)
    return render_template('pymupdf_extract.html', doc=document)


def sanitize_for_json(obj):
    """Recursively convert bytes to base64 strings for JSON serialization"""
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode('utf-8')
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, tuple):
        return [sanitize_for_json(item) for item in obj]
    return obj


# =============================================================================
# Module-level helper functions for table clip detection
# Used by both Auto-Clip and Merging endpoints
# =============================================================================

def collect_horizontal_segments(page):
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


def collect_vertical_segments(page):
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


def merge_by_y_and_x(segs):
    """Merge horizontal segments by Y position. Tracks component segments for debugging."""
    y_merge = 2.0
    segs = sorted(segs, key=lambda s: s[0])
    levels = []
    for y, x0, x1 in segs:
        seg_str = f"[{round(x0,1)}-{round(x1,1)}]"  # Format: [x0-x1]
        if not levels or abs(y - levels[-1]["y"]) > y_merge:
            levels.append({
                "y": y, 
                "x0": x0, 
                "x1": x1, 
                "raw_segs": [seg_str],  # Track component segments
                "raw_count": 1
            })
        else:
            levels[-1]["x0"] = min(levels[-1]["x0"], x0)
            levels[-1]["x1"] = max(levels[-1]["x1"], x1)
            levels[-1]["raw_segs"].append(seg_str)
            levels[-1]["raw_count"] += 1
    return levels


def get_segments_at_y(segs, y, tolerance=3.0):
    """Get all raw segments at a given Y level (within tolerance)"""
    return [(s[1], s[2]) for s in segs if abs(s[0] - y) <= tolerance]


def detect_column_boundaries_from_segments(segments_at_y):
    """
    Detect column boundaries from gaps between column segments.
    - Ignore small segments (< 3pt) - these are noise/divider lines
    - Keep segments >= 3pt - these are actual columns
    - Find gaps between these column segments
    """
    if len(segments_at_y) < 2:
        return []
    
    # Filter out small segments (< 3pt), keep only actual column segments
    column_segs = [(x0, x1) for x0, x1 in segments_at_y if (x1 - x0) >= 3]
    
    if len(column_segs) < 2:
        return []
    
    # Sort by x0
    sorted_segs = sorted(column_segs, key=lambda s: s[0])
    
    boundaries = []
    for i in range(len(sorted_segs) - 1):
        right_edge = sorted_segs[i][1]  # x1 of current
        left_edge = sorted_segs[i + 1][0]  # x0 of next
        
        # Any gap between column segments is a boundary
        if left_edge > right_edge:
            boundaries.append((right_edge + left_edge) / 2)
    
    return boundaries


def guess_table_clips_from_hlines(page, min_rules=2):
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
    
    # Group lines into tables based on character group column consistency
    groups = []
    cur_group = [levels[0]]
    
    # Establish initial column bounds from first line
    cur_x0 = levels[0]["x0"]
    cur_x1 = levels[0]["x1"]
    
    # Track column boundaries (X positions between columns)
    column_boundaries = []
    grouping_log = []
    
    for i, lv in enumerate(levels[1:], 1):
        prev_y = levels[i-1]["y"]
        curr_y = lv["y"]
        y_gap = curr_y - prev_y
        
        log_entry = {'i': i, 'prev_y': round(prev_y, 1), 'curr_y': round(curr_y, 1), 'y_gap': round(y_gap, 1)}
        
        # Skip if lines are too close (same row border)
        if y_gap < 3:
            cur_group.append(lv)
            log_entry['action'] = 'merge (y_gap < 3)'
            grouping_log.append(log_entry)
            continue
        
        # Get character groups in this Y range
        row_groups = get_groups_in_y_range(all_groups, prev_y, curr_y)
        log_entry['char_groups_count'] = len(row_groups)
        
        # DEBUG: show each group's X range and text (same as Auto-Clip)
        if len(row_groups) > 0:
            groups_detail = []
            for g in sorted(row_groups, key=lambda x: x['merged_bbox'][0]):
                groups_detail.append({
                    'x': [round(g['merged_bbox'][0], 1), round(g['merged_bbox'][2], 1)],
                    'text': g.get('text', '')[:20]
                })
            log_entry['groups'] = groups_detail
        
        is_separator = False
        
        # Detect column boundaries from RAW SEGMENTS at this Y level
        segments_at_prev_y = get_segments_at_y(segs, prev_y)
        segment_boundaries = detect_column_boundaries_from_segments(segments_at_prev_y)
        log_entry['segments_at_y'] = len(segments_at_prev_y)
        log_entry['segment_boundaries'] = [round(b, 1) for b in segment_boundaries]
        
        if len(segment_boundaries) > 0:
            if not column_boundaries:
                column_boundaries = segment_boundaries
                log_entry['new_boundaries'] = len(column_boundaries)
                log_entry['boundary_positions'] = [round(b, 1) for b in column_boundaries]
                log_entry['boundary_source'] = 'segments'
            else:
                if len(row_groups) > 0:
                    is_crossing, crossing_info = check_boundary_crossing(row_groups, column_boundaries)
                    log_entry['boundaries_count'] = len(column_boundaries)
                    log_entry['boundary_positions'] = [round(b, 1) for b in column_boundaries]
                    
                    if is_crossing:
                        is_separator = True
                        log_entry['crossing'] = crossing_info

        
        if is_separator:
            log_entry['action'] = 'SPLIT (boundary crossing)'
            groups.append(cur_group)
            cur_group = []
            cur_x0 = None
            cur_x1 = None
            column_boundaries = []
        elif not cur_group:
            if len(segment_boundaries) >= 1:
                log_entry['action'] = 'START new table'
                cur_group = [lv]
                cur_x0 = lv["x0"]
                cur_x1 = lv["x1"]
                column_boundaries = segment_boundaries
            else:
                log_entry['action'] = 'skip (no segment boundaries)'
        else:
            x0_diff = abs(lv["x0"] - cur_x0) if cur_x0 else 0
            x1_diff = abs(lv["x1"] - cur_x1) if cur_x1 else 0
            
            if x0_diff > 30 or x1_diff > 30:
                log_entry['action'] = 'SPLIT (X alignment)'
                groups.append(cur_group)
                cur_group = [lv]
                cur_x0 = lv["x0"]
                cur_x1 = lv["x1"]
                column_boundaries = []
            else:
                log_entry['action'] = 'merge'
                cur_group.append(lv)
                cur_x0 = min(cur_x0, lv["x0"]) if cur_x0 else lv["x0"]
                cur_x1 = max(cur_x1, lv["x1"]) if cur_x1 else lv["x1"]
        
        grouping_log.append(log_entry)
    
    groups.append(cur_group)
    
    # Build clips from groups (no padding)
    clips = []
    for g in groups:
        if len(g) >= min_rules:
            x0 = min(v["x0"] for v in g)
            x1 = max(v["x1"] for v in g)
            y0 = g[0]["y"]
            y1 = g[-1]["y"]
            clip = fitz.Rect(x0, y0, x1, y1) & page.rect
            clips.append(clip)
    
    # Filter out clips with horizontal outliers
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
                return True, {'outlier_x': [round(g_x0, 1), round(g_x1, 1)], 
                              'clip_x': [round(clip_x0, 1), round(clip_x1, 1)],
                              'text': text[:30]}
        
        return False, {}
    
    # Validate clips
    validated_clips = []
    clip_validation_log = []
    for clip in clips:
        has_outlier, outlier_info = has_horizontal_outlier(clip, all_groups)
        if has_outlier:
            clip_validation_log.append({
                'clip': [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                'status': 'REJECTED',
                'reason': 'horizontal_outlier',
                'outlier': outlier_info
            })
        else:
            validated_clips.append(clip)
            clip_validation_log.append({
                'clip': [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                'status': 'VALID'
            })
    
    return validated_clips, grouping_log, clip_validation_log


def detect_cells_from_segments(h_segs, v_segs, clip_bbox, char_groups=None):
    """
    Detect cells from horizontal segments within a clip area.
    NEW APPROACH: Use H-segments directly as cell X ranges.
    - Large segments (≥3pt width) = actual cell content areas
    - Small segments (<3pt width) = dividers/separators (ignored)
    
    Returns list of cell dicts with bbox [x0, y0, x1, y1].
    """
    x0_clip, y0_clip, x1_clip, y1_clip = clip_bbox
    tolerance = 3.0
    merge_tolerance = 10.0  # Merge Y positions within 10pt
    min_segment_width = 3.0  # Minimum width to be considered a cell (not a divider)
    
    def merge_close_positions(positions, merge_tol):
        """Merge positions that are within merge_tol of each other."""
        if not positions:
            return []
        sorted_pos = sorted(positions)
        merged = [sorted_pos[0]]
        for pos in sorted_pos[1:]:
            if pos - merged[-1] > merge_tol:
                merged.append(pos)
        return merged
    
    # Get unique Y positions from horizontal segments within clip
    y_positions = set()
    for y, sx0, sx1 in h_segs:
        if y0_clip - tolerance <= y <= y1_clip + tolerance:
            y_positions.add(round(y, 1))
    y_positions = sorted(y_positions)
    # Merge close Y positions (handles double horizontal lines from rectangles)
    y_positions = merge_close_positions(y_positions, merge_tolerance)
    
    # Helper function to detect column boundaries from char groups
    def detect_column_boundaries_from_text(groups, y_top, y_bottom):
        """Detect column boundaries from gaps between character groups in a row."""
        if not groups:
            return []
        
        # Find groups in this row (50% Y overlap)
        row_groups = []
        for g in groups:
            g_bbox = g.get('merged_bbox')
            if not g_bbox:
                continue
            g_y0, g_y1 = g_bbox[1], g_bbox[3]
            g_height = g_y1 - g_y0
            if g_height <= 0:
                continue
            overlap_start = max(g_y0, y_top)
            overlap_end = min(g_y1, y_bottom)
            overlap = max(0, overlap_end - overlap_start)
            if overlap / g_height >= 0.5:
                row_groups.append(g)
        
        if len(row_groups) < 2:
            return []
        
        # Sort by X position
        row_groups.sort(key=lambda g: g['merged_bbox'][0])
        
        # Find gaps between adjacent groups (at least 5pt gap)
        boundaries = []
        for j in range(len(row_groups) - 1):
            right_edge = row_groups[j]['merged_bbox'][2]
            left_edge = row_groups[j + 1]['merged_bbox'][0]
            if left_edge - right_edge >= 5:
                boundaries.append((right_edge + left_edge) / 2)
        
        return boundaries
    
    # For each pair of Y positions (row), find cells from H-segments
    cells = []
    row_debug = []
    
    # Collect all column boundaries from first row with h-segment gaps (for propagation)
    first_row_boundaries = None
    
    for row_idx in range(len(y_positions) - 1):
        y_top = y_positions[row_idx]
        y_bottom = y_positions[row_idx + 1]
        y_mid = (y_top + y_bottom) / 2
        
        # Get H-segments at the TOP boundary of this row (within tolerance)
        segments_at_row = get_segments_at_y(h_segs, y_top, merge_tolerance)
        
        # Filter: only keep large segments (≥3pt) - these are actual cells
        cell_segments = [(x0, x1) for x0, x1 in segments_at_row if (x1 - x0) >= min_segment_width]
        
        # Sort by x0
        cell_segments = sorted(cell_segments, key=lambda s: s[0])
        
        # Merge ONLY truly overlapping segments (not adjacent ones)
        # Adjacent segments with gaps are separate cells
        merged_segments = []
        for seg in cell_segments:
            if not merged_segments:
                merged_segments.append(list(seg))
            else:
                prev = merged_segments[-1]
                # Only merge if segment OVERLAPS with previous (starts before previous ends)
                if seg[0] < prev[1]:  # Overlapping
                    prev[1] = max(prev[1], seg[1])  # Extend
                else:
                    # Gap exists - this is a separate cell
                    merged_segments.append(list(seg))
        
        # If we have cells from h-segments, save boundaries for propagation
        if len(merged_segments) > 1 and first_row_boundaries is None:
            # Extract boundaries from segment gaps
            first_row_boundaries = []
            for i in range(len(merged_segments) - 1):
                right_edge = merged_segments[i][1]
                left_edge = merged_segments[i + 1][0]
                first_row_boundaries.append((right_edge + left_edge) / 2)
        
        x_source = 'h_segments_direct'
        
        # FALLBACK 1: If no cells from h-segments, try text-based detection
        if len(merged_segments) <= 1 and char_groups:
            text_boundaries = detect_column_boundaries_from_text(char_groups, y_top, y_bottom)
            if text_boundaries:
                # Rebuild segments from text boundaries
                all_x = [x0_clip] + sorted(text_boundaries) + [x1_clip]
                merged_segments = [[all_x[i], all_x[i+1]] for i in range(len(all_x) - 1)]
                x_source = 'text_based'
        
        # FALLBACK 2: If still no cells, use propagated boundaries from first row
        if len(merged_segments) <= 1 and first_row_boundaries:
            all_x = [x0_clip] + first_row_boundaries + [x1_clip]
            merged_segments = [[all_x[i], all_x[i+1]] for i in range(len(all_x) - 1)]
            x_source = 'propagated'
        
        # Create cells from merged segments
        for col_idx, (seg_x0, seg_x1) in enumerate(merged_segments):
            cell = {
                'bbox': [seg_x0, y_top, seg_x1, y_bottom],
                'row': row_idx,
                'col': col_idx
            }
            cells.append(cell)
        
        row_debug.append({
            'row': row_idx,
            'y_range': [y_top, y_bottom],
            'raw_segments': len(segments_at_row),
            'cell_segments': len(cell_segments),
            'merged_segments': len(merged_segments),
            'cells_in_row': len(merged_segments),
            'x_source': x_source,
            'segments_detail': [f"[{round(s[0],1)}-{round(s[1],1)}]" for s in merged_segments[:8]]
        })
    
    return cells, {
        'method': 'h_segments_direct',
        'y_lines': len(y_positions),
        'y_positions': y_positions,
        'total_cells': len(cells),
        'row_debug': row_debug
    }



@pymupdf_bp.route('/pymupdf-api/text-dict/<int:doc_id>/<int:page_num>')
def extract_text_dict(doc_id, page_num):
    """Extract page content using page.get_text("dict")"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            result = page.get_text("dict")
            # Sanitize bytes data for JSON serialization
            result = sanitize_for_json(result)
            
        return jsonify({
            'success': True,
            'method': 'get_text("dict")',
            'page': page_num,
            'data': result
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/text-rawdict/<int:doc_id>/<int:page_num>')
def extract_text_rawdict(doc_id, page_num):
    """Extract page content using page.get_text("rawdict")"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            result = page.get_text("rawdict")
            # Sanitize bytes data for JSON serialization
            result = sanitize_for_json(result)
            
        return jsonify({
            'success': True,
            'method': 'get_text("rawdict")',
            'page': page_num,
            'data': result
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/images/<int:doc_id>/<int:page_num>')
def extract_images(doc_id, page_num):
    """Extract images using page.get_images(full=True)"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            images = page.get_images(full=True)
            
            # Format image data for JSON response
            image_list = []
            for img in images:
                image_info = {
                    'xref': img[0],
                    'smask': img[1],
                    'width': img[2],
                    'height': img[3],
                    'bpc': img[4],
                    'colorspace': img[5],
                    'alt_colorspace': img[6],
                    'name': img[7],
                    'filter': img[8] if len(img) > 8 else None,
                    'referencer': img[9] if len(img) > 9 else None
                }
                
                # Try to extract image preview (base64)
                try:
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    if base_image:
                        image_info['ext'] = base_image.get('ext', '')
                        # Include small preview if image is not too large
                        if base_image.get('width', 0) <= 500 and base_image.get('height', 0) <= 500:
                            image_info['preview_base64'] = base64.b64encode(base_image['image']).decode('utf-8')
                except:
                    pass
                
                image_list.append(image_info)
            
        return jsonify({
            'success': True,
            'method': 'get_images(full=True)',
            'page': page_num,
            'count': len(image_list),
            'data': image_list
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/image-info/<int:doc_id>/<int:page_num>')
def extract_image_info(doc_id, page_num):
    """Extract images with bounding boxes using page.get_image_info()"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            image_info_list = page.get_image_info()
            
            # Format for JSON response
            images = []
            for img in image_info_list:
                images.append({
                    'xref': img.get('xref'),
                    'bbox': list(img.get('bbox', [])),
                    'width': img.get('width'),
                    'height': img.get('height'),
                    'name': img.get('name', ''),
                    'cs-name': img.get('cs-name', ''),
                    'size': img.get('size', 0)
                })
        
        return jsonify({
            'success': True,
            'method': 'get_image_info()',
            'page': page_num,
            'count': len(images),
            'data': images
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/drawings/<int:doc_id>/<int:page_num>')
def extract_drawings(doc_id, page_num):
    """Extract drawings using page.get_drawings()"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            drawings = page.get_drawings()
            
            # Convert drawing objects to serializable dict
            drawing_list = []
            for drawing in drawings:
                draw_dict = {
                    'rect': list(drawing.get('rect', [])) if drawing.get('rect') else None,
                    'items': [],
                    'color': list(drawing.get('color', [])) if drawing.get('color') else None,
                    'fill': list(drawing.get('fill', [])) if drawing.get('fill') else None,
                    'width': drawing.get('width'),
                    'stroke_opacity': drawing.get('stroke_opacity'),
                    'fill_opacity': drawing.get('fill_opacity'),
                    'even_odd': drawing.get('even_odd'),
                    'closePath': drawing.get('closePath'),
                    'dashes': drawing.get('dashes'),
                    'lineCap': drawing.get('lineCap'),
                    'lineJoin': drawing.get('lineJoin'),
                }
                
                # Process items (path operations)
                items = drawing.get('items', [])
                for item in items:
                    item_type = item[0] if len(item) > 0 else None
                    item_data = {'type': item_type}
                    
                    if item_type == 'l':  # Line
                        item_data['from'] = list(item[1]) if len(item) > 1 else None
                        item_data['to'] = list(item[2]) if len(item) > 2 else None
                    elif item_type == 're':  # Rectangle
                        item_data['rect'] = list(item[1]) if len(item) > 1 else None
                    elif item_type == 'qu':  # Quad
                        item_data['quad'] = list(item[1]) if len(item) > 1 else None
                    elif item_type == 'c':  # Curve
                        item_data['p1'] = list(item[1]) if len(item) > 1 else None
                        item_data['p2'] = list(item[2]) if len(item) > 2 else None
                        item_data['p3'] = list(item[3]) if len(item) > 3 else None
                        item_data['p4'] = list(item[4]) if len(item) > 4 else None
                    
                    draw_dict['items'].append(item_data)
                
                drawing_list.append(draw_dict)
            
        return jsonify({
            'success': True,
            'method': 'get_drawings()',
            'page': page_num,
            'count': len(drawing_list),
            'data': drawing_list
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/find-tables/<int:doc_id>/<int:page_num>')
def extract_tables(doc_id, page_num):
    """Find tables on the page using customizable horizontal and vertical strategies."""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        # Get strategy parameters from query string (default: 'lines')
        from flask import request
        h_strategy = request.args.get('horizontal_strategy', 'lines')
        v_strategy = request.args.get('vertical_strategy', 'lines')
        
        # Validate strategies
        valid_strategies = ['lines', 'lines_strict', 'text']
        if h_strategy not in valid_strategies:
            h_strategy = 'lines'
        if v_strategy not in valid_strategies:
            v_strategy = 'lines'
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            
            def serialize_tables(tables, strategy_label):
                table_list = []
                for t_idx, table in enumerate(tables):
                    table_dict = {
                        'table_index': t_idx,
                        'strategy': strategy_label,
                        'bbox': list(table.bbox) if table.bbox else None,
                        'row_count': table.row_count,
                        'col_count': table.col_count,
                        'header': {
                            'bbox': list(table.header.bbox) if hasattr(table.header, 'bbox') and table.header.bbox else None,
                            'names': list(table.header.names) if hasattr(table.header, 'names') and table.header.names else None,
                            'cells': [list(c) if c else None for c in table.header.cells] if hasattr(table.header, 'cells') and table.header.cells else None
                        } if hasattr(table, 'header') and table.header else {'bbox': None, 'names': None, 'cells': None},
                        'cells': [list(c) if c else None for c in table.cells] if table.cells else None
                    }
                    
                    rows_list = []
                    for r_idx, row in enumerate(table.rows):
                        row_cells = []
                        if hasattr(row, 'cells') and row.cells:
                            for c_idx, cell in enumerate(row.cells):
                                cell_text = ""
                                if cell:
                                    cell_text = page.get_text("text", clip=cell).strip()
                                row_cells.append({
                                    'row': r_idx,
                                    'col': c_idx,
                                    'bbox': list(cell) if cell else None,
                                    'text': cell_text
                                })
                                
                        rows_list.append({
                            'bbox': list(row.bbox) if hasattr(row, 'bbox') and row.bbox else None,
                            'cells': row_cells
                        })
                    table_dict['rows'] = rows_list
                    
                    try:
                        table_dict['extract'] = table.extract()
                    except:
                        table_dict['extract'] = None
                    
                    table_list.append(table_dict)
                return table_list

            # Strategy mode only - use H/V strategy parameters
            strategy_label = f"h:{h_strategy}, v:{v_strategy}"
            tables = page.find_tables(
                horizontal_strategy=h_strategy,
                vertical_strategy=v_strategy
            )
            data_tables = serialize_tables(tables, strategy_label)
        
        return jsonify({
            'success': True,
            'method': 'find_tables()',
            'page': page_num,
            'horizontal_strategy': h_strategy,
            'vertical_strategy': v_strategy,
            'data': data_tables,
            'count': len(data_tables)
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/find-tables-autoclip/<int:doc_id>/<int:page_num>')
def extract_tables_autoclip(doc_id, page_num):
    """Find tables using auto-clip from horizontal lines."""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        from flask import request
        
        # Read configurable parameters from query string
        min_rules = int(request.args.get('min_rules', 2))
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            
            # Helper functions for auto-clip detection
            def collect_horizontal_segments(page):
                segs = []
                y_eps = 1.5
                for d in page.get_drawings():
                    # Check color - skip if white (stroke or fill close to white)
                    color = d.get("color")  # stroke color
                    fill = d.get("fill")    # fill color
                    
                    # Skip if stroke color is white (1,1,1) or near white
                    is_white = False
                    if color:
                        # color is tuple like (r, g, b) where values 0-1
                        if len(color) >= 3 and all(c > 0.95 for c in color[:3]):
                            is_white = True
                    if not color and fill:
                        # Check fill color if no stroke
                        if len(fill) >= 3 and all(c > 0.95 for c in fill[:3]):
                            is_white = True
                    
                    if is_white:
                        continue  # Skip white lines
                    
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
            
            def collect_vertical_segments(page):
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
                                segs.append((p1.x, y0, y1))  # (x, y0, y1)
                        elif kind == "re":
                            r = it[1]
                            segs.append((r.x0, r.y0, r.y1))
                            segs.append((r.x1, r.y0, r.y1))
                return segs

            def merge_by_y_and_x(segs):
                y_merge = 2.0
                segs = sorted(segs, key=lambda s: s[0])
                levels = []
                for y, x0, x1 in segs:
                    if not levels or abs(y - levels[-1]["y"]) > y_merge:
                        levels.append({"y": y, "x0": x0, "x1": x1})
                    else:
                        levels[-1]["x0"] = min(levels[-1]["x0"], x0)
                        levels[-1]["x1"] = max(levels[-1]["x1"], x1)
                return levels
            
            def get_segments_at_y(segs, y, tolerance=3.0):
                """Get all raw segments at a given Y level (within tolerance)"""
                return [(s[1], s[2]) for s in segs if abs(s[0] - y) <= tolerance]
            
            def detect_column_boundaries_from_segments(segments_at_y):
                """
                Detect column boundaries from gaps between column segments.
                - Ignore small segments (< 3pt) - these are noise/divider lines
                - Keep segments >= 3pt - these are actual columns
                - Find gaps between these column segments
                """
                if len(segments_at_y) < 2:
                    return []
                
                # Filter out small segments (< 3pt), keep only actual column segments
                column_segs = [(x0, x1) for x0, x1 in segments_at_y if (x1 - x0) >= 3]
                
                if len(column_segs) < 2:
                    return []
                
                # Sort by x0
                sorted_segs = sorted(column_segs, key=lambda s: s[0])
                
                boundaries = []
                for i in range(len(sorted_segs) - 1):
                    right_edge = sorted_segs[i][1]  # x1 of current
                    left_edge = sorted_segs[i + 1][0]  # x0 of next
                    
                    # Any gap between column segments is a boundary
                    if left_edge > right_edge:
                        boundaries.append((right_edge + left_edge) / 2)
                
                return boundaries

            def detect_cells_from_segments(h_segs, v_segs, clip_bbox, char_groups=None):
                """
                Detect cells from horizontal and vertical segments within a clip area.
                If no vertical segments exist, fallback to column boundaries from H-segment gaps.
                If that fails, fallback to text-based column detection from char groups.
                Returns list of cell bboxes [x0, y0, x1, y1].
                """
                x0_clip, y0_clip, x1_clip, y1_clip = clip_bbox
                tolerance = 3.0
                merge_tolerance = 10.0  # Merge positions within 10pt (handles double-line borders)
                
                def merge_close_positions(positions, merge_tol):
                    """Merge positions that are within merge_tol of each other."""
                    if not positions:
                        return []
                    sorted_pos = sorted(positions)
                    merged = [sorted_pos[0]]
                    for pos in sorted_pos[1:]:
                        if pos - merged[-1] > merge_tol:
                            merged.append(pos)
                    return merged
                
                def detect_column_boundaries_from_text(groups, y_positions):
                    """
                    Detect column boundaries from gaps between character groups.
                    Groups text by row (Y level), find X gaps within each row, then
                    find consistent gaps across rows.
                    """
                    if not groups or len(y_positions) < 2:
                        return []
                    
                    # For each row (between adjacent Y positions), find X positions of groups
                    all_row_gaps = []
                    
                    for i in range(len(y_positions) - 1):
                        row_y0 = y_positions[i]
                        row_y1 = y_positions[i + 1]
                        
                        # Find groups in this row
                        row_groups = []
                        for g in groups:
                            g_bbox = g.get('merged_bbox')
                            if not g_bbox:
                                continue
                            g_y0, g_y1 = g_bbox[1], g_bbox[3]
                            # Check if group overlaps with this row (50% of group height)
                            g_height = g_y1 - g_y0
                            if g_height <= 0:
                                continue
                            overlap_start = max(g_y0, row_y0)
                            overlap_end = min(g_y1, row_y1)
                            overlap = max(0, overlap_end - overlap_start)
                            if overlap / g_height >= 0.5:
                                row_groups.append(g)
                        
                        if len(row_groups) < 2:
                            continue
                        
                        # Sort groups by X position
                        row_groups.sort(key=lambda g: g['merged_bbox'][0])
                        
                        # Find gaps between adjacent groups
                        for j in range(len(row_groups) - 1):
                            right_edge = row_groups[j]['merged_bbox'][2]  # x1 of current
                            left_edge = row_groups[j + 1]['merged_bbox'][0]  # x0 of next
                            
                            # Gap must be significant (at least 5pt)
                            if left_edge - right_edge >= 5:
                                gap_center = (right_edge + left_edge) / 2
                                all_row_gaps.append(round(gap_center, 1))
                    
                    if not all_row_gaps:
                        return []
                    
                    # Find consistent gap positions (appearing in multiple rows)
                    # Merge close gaps and count occurrences
                    gap_counts = {}
                    for gap in all_row_gaps:
                        # Find if there's a similar gap already
                        found = False
                        for existing in gap_counts:
                            if abs(existing - gap) <= 15:  # Tolerance for same column boundary
                                gap_counts[existing] += 1
                                found = True
                                break
                        if not found:
                            gap_counts[gap] = 1
                    
                    # Return gaps that appear in at least 1 row (can be relaxed if needed)
                    boundaries = [g for g, count in gap_counts.items() if count >= 1]
                    return sorted(boundaries)
                
                # Get unique Y positions from horizontal segments within clip
                y_positions = set()
                for y, sx0, sx1 in h_segs:
                    if y0_clip - tolerance <= y <= y1_clip + tolerance:
                        y_positions.add(round(y, 1))
                y_positions = sorted(y_positions)
                # Merge close Y positions
                y_positions = merge_close_positions(y_positions, merge_tolerance)
                
                # Get unique X positions from vertical segments within clip
                # Must check BOTH X and Y overlap to prevent segments from other tables leaking in
                x_positions = set()
                for seg in v_segs:
                    # Safety check: ensure segment has 3 values (x, y0, y1)
                    if len(seg) != 3:
                        continue
                    x, sy0, sy1 = seg
                    # Check X is within clip
                    if x0_clip - tolerance <= x <= x1_clip + tolerance:
                        # Also check Y overlaps with clip
                        if sy1 >= y0_clip - tolerance and sy0 <= y1_clip + tolerance:
                            x_positions.add(round(x, 1))
                x_positions = sorted(x_positions)
                # Merge close X positions (handles double vertical lines)
                x_positions = merge_close_positions(x_positions, merge_tolerance)
                
                # FALLBACK 1: If no vertical segments, use column boundaries from H-segment gaps
                x_source = 'v_segments'
                if len(x_positions) < 2:
                    # Get horizontal segments at each Y level within clip
                    # and find column boundaries from gaps between them
                    all_boundaries = set()
                    for y in y_positions:
                        segments_at_y = get_segments_at_y(h_segs, y, tolerance)
                        boundaries = detect_column_boundaries_from_segments(segments_at_y)
                        for b in boundaries:
                            all_boundaries.add(round(b, 1))
                    
                    if all_boundaries:
                        # Use clip edges + boundaries as X positions
                        x_positions = [round(x0_clip, 1)] + sorted(all_boundaries) + [round(x1_clip, 1)]
                        x_source = 'h_segment_gaps'
                
                # FALLBACK 2: If still no boundaries, use text-based column detection
                if len(x_positions) < 2 and char_groups:
                    text_boundaries = detect_column_boundaries_from_text(char_groups, y_positions)
                    if text_boundaries:
                        x_positions = [round(x0_clip, 1)] + text_boundaries + [round(x1_clip, 1)]
                        x_source = 'text_based'
                
                # FALLBACK 3: Last resort - use clip edges only (1 column)
                if len(x_positions) < 2:
                    x_positions = [round(x0_clip, 1), round(x1_clip, 1)]
                    x_source = 'clip_edges'
                
                # Create cells from grid intersections
                cells = []
                for i in range(len(y_positions) - 1):
                    for j in range(len(x_positions) - 1):
                        cell = {
                            'bbox': [x_positions[j], y_positions[i], x_positions[j+1], y_positions[i+1]],
                            'row': i,
                            'col': j
                        }
                        cells.append(cell)
                
                return cells, {'y_lines': len(y_positions), 'x_lines': len(x_positions), 'x_source': x_source}


            def guess_table_clips_from_hlines(page, min_rules):
                segs = collect_horizontal_segments(page)
                if not segs:
                    return [], []
                levels = merge_by_y_and_x(segs)
                if not levels:
                    return [], []
                
                # Get rawdict and build character groups ONCE for the page
                rawdict_data = page.get_text("rawdict")
                all_chars = collect_all_chars(rawdict_data)
                all_groups = find_overlapping_groups(all_chars)
                
                # Group lines into tables based on character group column consistency
                groups = []
                cur_group = [levels[0]]
                
                # Establish initial column bounds from first line
                cur_x0 = levels[0]["x0"]
                cur_x1 = levels[0]["x1"]
                
                # Track column boundaries (X positions between columns)
                column_boundaries = []  # Established from first multi-column row
                grouping_log = []  # Debug log for grouping decisions
                
                for i, lv in enumerate(levels[1:], 1):
                    prev_y = levels[i-1]["y"]
                    curr_y = lv["y"]
                    y_gap = curr_y - prev_y
                    
                    log_entry = {'i': i, 'prev_y': round(prev_y, 1), 'curr_y': round(curr_y, 1), 'y_gap': round(y_gap, 1)}
                    
                    # Skip if lines are too close (same row border)
                    if y_gap < 3:
                        cur_group.append(lv)
                        log_entry['action'] = 'merge (y_gap < 3)'
                        grouping_log.append(log_entry)
                        continue
                    
                    # Get character groups in this Y range
                    row_groups = get_groups_in_y_range(all_groups, prev_y, curr_y)
                    log_entry['char_groups_count'] = len(row_groups)
                    
                    # DEBUG: show each group's X range and text
                    if len(row_groups) > 0:
                        groups_detail = []
                        for g in sorted(row_groups, key=lambda x: x['merged_bbox'][0]):
                            groups_detail.append({
                                'x': [round(g['merged_bbox'][0], 1), round(g['merged_bbox'][2], 1)],
                                'text': g.get('text', '')[:20]
                            })
                        log_entry['groups'] = groups_detail
                    
                    is_separator = False
                    
                    # Detect column boundaries from RAW SEGMENTS at this Y level
                    segments_at_prev_y = get_segments_at_y(segs, prev_y)
                    segment_boundaries = detect_column_boundaries_from_segments(segments_at_prev_y)
                    log_entry['segments_at_y'] = len(segments_at_prev_y)
                    log_entry['segment_boundaries'] = [round(b, 1) for b in segment_boundaries]
                    
                    if len(segment_boundaries) > 0:
                        # Use segment-based boundaries (more accurate)
                        if not column_boundaries:
                            # First row - establish boundaries from segments
                            column_boundaries = segment_boundaries
                            log_entry['new_boundaries'] = len(column_boundaries)
                            log_entry['boundary_positions'] = [round(b, 1) for b in column_boundaries]
                            log_entry['boundary_source'] = 'segments'
                        else:
                            # Check if any char group CROSSES a column boundary
                            if len(row_groups) > 0:
                                is_crossing, crossing_info = check_boundary_crossing(row_groups, column_boundaries)
                                log_entry['boundaries_count'] = len(column_boundaries)
                                log_entry['boundary_positions'] = [round(b, 1) for b in column_boundaries]
                                
                                if is_crossing:
                                    is_separator = True
                                    log_entry['crossing'] = crossing_info
                    # No segment boundaries = no column structure (single column or no table)
                    
                    if is_separator:
                        log_entry['action'] = 'SPLIT (boundary crossing)'
                        # End current table group
                        groups.append(cur_group)
                        # Reset - don't start new group yet, wait for real table row
                        cur_group = []
                        cur_x0 = None
                        cur_x1 = None
                        column_boundaries = []
                    elif not cur_group:
                        # We're looking for start of new table after a split
                        # Only start new group if this row has segment-based boundaries
                        if len(segment_boundaries) >= 1:  # Has column structure from segments
                            log_entry['action'] = 'START new table'
                            cur_group = [lv]
                            cur_x0 = lv["x0"]
                            cur_x1 = lv["x1"]
                            column_boundaries = segment_boundaries
                            log_entry['new_boundaries'] = len(segment_boundaries)
                            log_entry['boundary_source'] = 'segments'
                        else:
                            log_entry['action'] = 'skip (no segment boundaries)'
                    else:
                        # Normal case - check X alignment
                        x0_diff = abs(lv["x0"] - cur_x0) if cur_x0 else 0
                        x1_diff = abs(lv["x1"] - cur_x1) if cur_x1 else 0
                        
                        if x0_diff > 30 or x1_diff > 30:
                            log_entry['action'] = 'SPLIT (X alignment)'
                            groups.append(cur_group)
                            cur_group = [lv]
                            cur_x0 = lv["x0"]
                            cur_x1 = lv["x1"]
                            column_boundaries = []
                        else:
                            log_entry['action'] = 'merge'
                            cur_group.append(lv)
                            cur_x0 = min(cur_x0, lv["x0"]) if cur_x0 else lv["x0"]
                            cur_x1 = max(cur_x1, lv["x1"]) if cur_x1 else lv["x1"]
                    
                    grouping_log.append(log_entry)
                
                groups.append(cur_group)
                
                # Build clips from groups (no padding)
                clips = []
                for g in groups:
                    if len(g) >= min_rules:
                        x0 = min(v["x0"] for v in g)
                        x1 = max(v["x1"] for v in g)
                        y0 = g[0]["y"]
                        y1 = g[-1]["y"]
                        clip = fitz.Rect(x0, y0, x1, y1) & page.rect
                        clips.append(clip)
                
                # Filter out clips with horizontal outliers (groups outside X bounds)
                # This removes formulas where text like "Recall =" extends beyond fraction bar
                def has_horizontal_outlier(clip, all_groups, tolerance=10):
                    """Check if any char group in Y range extends outside X bounds"""
                    clip_x0, clip_y0, clip_x1, clip_y1 = clip.x0, clip.y0, clip.x1, clip.y1
                    
                    for g in all_groups:
                        # Skip whitespace-only groups (false positives)
                        text = g.get('text', '').strip()
                        if not text:
                            continue
                        
                        g_x0, g_y0, g_x1, g_y1 = g['merged_bbox']
                        
                        # Check if group is in Y range (with some tolerance)
                        if g_y1 < clip_y0 - tolerance or g_y0 > clip_y1 + tolerance:
                            continue  # Group not in Y range
                        
                        # Check if group extends outside X bounds
                        if g_x0 < clip_x0 - tolerance or g_x1 > clip_x1 + tolerance:
                            return True, {'outlier_x': [round(g_x0, 1), round(g_x1, 1)], 
                                          'clip_x': [round(clip_x0, 1), round(clip_x1, 1)],
                                          'text': text[:30]}
                    
                    return False, {}
                
                # Validate clips
                validated_clips = []
                clip_validation_log = []
                for clip in clips:
                    has_outlier, outlier_info = has_horizontal_outlier(clip, all_groups)
                    if has_outlier:
                        clip_validation_log.append({
                            'clip': [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                            'status': 'REJECTED',
                            'reason': 'horizontal_outlier',
                            'outlier': outlier_info
                        })
                    else:
                        validated_clips.append(clip)
                        clip_validation_log.append({
                            'clip': [round(clip.x0, 1), round(clip.y0, 1), round(clip.x1, 1), round(clip.y1, 1)],
                            'status': 'VALID'
                        })
                
                return validated_clips, grouping_log, clip_validation_log
            
            # Detect table areas from horizontal lines - with debug info
            segs = collect_horizontal_segments(page)
            v_segs = collect_vertical_segments(page)  # For cell detection
            levels = merge_by_y_and_x(segs) if segs else []
            clips, grouping_log, clip_validation_log = guess_table_clips_from_hlines(page, min_rules)
            all_tables = []
            clip_logs = []  # Detailed logs for each clip
            all_cells = []  # All detected cells
            
            if clips:
                for clip_idx, clip in enumerate(clips):
                    clip_log = {
                        'clip_idx': clip_idx,
                        'clip_bbox': [clip.x0, clip.y0, clip.x1, clip.y1],
                        'clip_size': [clip.width, clip.height]
                    }
                    
                    # Detect cells from H+V segments
                    clip_cells, cell_debug = detect_cells_from_segments(
                        segs, v_segs, 
                        [clip.x0, clip.y0, clip.x1, clip.y1]
                    )
                    clip_log['cells_detected'] = len(clip_cells)
                    clip_log['cell_debug'] = cell_debug
                    
                    # Add cells with clip_idx reference
                    for cell in clip_cells:
                        cell['clip_idx'] = clip_idx
                        all_cells.append(cell)
                    
                    # Get text content in clip area to see if there's text
                    clip_text = page.get_text("text", clip=clip).strip()
                    clip_log['text_length'] = len(clip_text)
                    clip_log['text_preview'] = clip_text[:100] if clip_text else "(empty)"
                    
                    # Skip find_tables - just use custom cell detection from segments
                    # find_tables() doesn't work well for tables without vertical lines
                    
                    clip_logs.append(clip_log)
                
                # No find_tables - data_tables will be empty, cells come from detect_cells_from_segments
                data_tables = []
            else:
                # No clips found
                data_tables = []
            
            # Convert clips to serializable format
            clips_data = []
            for clip in clips:
                clips_data.append([clip.x0, clip.y0, clip.x1, clip.y1])
            
            return jsonify({
                'success': True,
                'method': 'find_tables() + auto_clip',
                'page': page_num,
                'debug': {
                    'segments_count': len(segs) if segs else 0,
                    'v_segments_count': len(v_segs) if v_segs else 0,
                    'raw_segments': [{'y': round(s[0], 1), 'x0': round(s[1], 1), 'x1': round(s[2], 1)} for s in (segs[:50] if segs else [])],  # First 50 raw segments
                    'levels_count': len(levels) if levels else 0,
                    'levels': levels[:20] if levels else [],  # First 20 levels for debug
                    'grouping_log': grouping_log[:15] if grouping_log else [],  # First 15 grouping decisions
                    'clip_validation': clip_validation_log,  # Valid/rejected clips with reasons
                    'clips_found': len(clips) if clips else 0,
                    'min_rules': min_rules,
                    'clip_logs': clip_logs
                },
                'clips': clips_data,
                'clips_found': len(clips) if clips else 0,
                'cells': all_cells,  # Detected cells
                'cells_count': len(all_cells),
                'data': data_tables,
                'count': len(data_tables)
            })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@pymupdf_bp.route('/pymupdf-api/merging/<int:doc_id>/<int:page_num>')
def extract_merging(doc_id, page_num):
    """Extract rawdict for character grouping and find_tables for table detection.
    Frontend will combine these: group characters by overlap, then replace 
    groups that overlap with tables with table structure.
    """
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            
            # Get rawdict for character grouping
            rawdict_data = page.get_text("rawdict")
            
            # Sanitize bytes in rawdict
            def sanitize(obj):
                if isinstance(obj, bytes):
                    return obj.decode('utf-8', errors='replace')
                elif isinstance(obj, dict):
                    return {k: sanitize(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize(i) for i in obj]
                return obj
            
            rawdict_data = sanitize(rawdict_data)
            
            # Step 1: Character grouping from rawdict
            all_chars = collect_all_chars(rawdict_data)
            char_groups = find_overlapping_groups(all_chars)
            
            # Debug log to track all groups and their status
            debug_groups_log = []
            for idx, group in enumerate(char_groups):
                debug_groups_log.append({
                    'index': idx,
                    'text': group.get('text', '')[:50],
                    'bbox': group.get('merged_bbox'),
                    'claimed_by': None,
                    'reason': None
                })
            
            # Step 1.5: Detect page images and shapes
            page_images = page.get_image_info()
            page_images_list = []
            shapes_list = []
            
            # Helper to merge multiple bboxes into one (needed for shape detection)
            def merge_bboxes(bboxes):
                """Merge multiple bboxes into a single encompassing bbox"""
                if not bboxes:
                    return None
                x0 = min(b[0] for b in bboxes)
                y0 = min(b[1] for b in bboxes)
                x1 = max(b[2] for b in bboxes)
                y1 = max(b[3] for b in bboxes)
                return [x0, y0, x1, y1]
            
            # Helper to check bbox overlap with minimum 50% Y overlap of group
            def group_overlaps_image(group_bbox, img_bbox, min_y_overlap_ratio=0.5):
                """Check if group overlaps with image with at least 50% Y overlap of group height"""
                if not group_bbox or not img_bbox:
                    return False
                
                # Check X overlap (any overlap is fine)
                x_overlap = not (group_bbox[2] < img_bbox[0] or group_bbox[0] > img_bbox[2])
                if not x_overlap:
                    return False
                
                # Check Y overlap with minimum ratio
                group_height = group_bbox[3] - group_bbox[1]
                if group_height <= 0:
                    return False
                
                y_overlap_start = max(group_bbox[1], img_bbox[1])
                y_overlap_end = min(group_bbox[3], img_bbox[3])
                y_overlap_amount = max(0, y_overlap_end - y_overlap_start)
                
                y_overlap_ratio = y_overlap_amount / group_height
                return y_overlap_ratio >= min_y_overlap_ratio
            
            # Helper to check simple bbox overlap (any overlap) - used for table/shape checks
            def simple_bbox_overlap(bbox1, bbox2):
                if not bbox1 or not bbox2:
                    return False
                return not (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
                           bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])
            
            for img in page_images:
                img_bbox = img.get('bbox')
                if not img_bbox:
                    continue
                
                img_bbox_list = list(img_bbox)
                img_width = img.get('width', 0)
                img_height = img.get('height', 0)
                
                # Find all char_groups that overlap with this image (at least 50% Y overlap)
                overlapping_groups = []
                for group in char_groups:
                    group_bbox = group.get('merged_bbox')
                    if group_bbox and group_overlaps_image(group_bbox, img_bbox_list):
                        overlapping_groups.append(group)
                
                if overlapping_groups:
                    # This is a SHAPE - image + text groups combined
                    # Sort groups by reading order: Y first, but if Y overlaps (same line), use X
                    def reading_order_key(g):
                        bbox = g.get('merged_bbox', [0, 0, 0, 0])
                        return (bbox[1], bbox[0])  # (Y, X)
                    
                    # Custom sort: check Y overlap for reading order
                    def compare_groups(g1, g2):
                        b1 = g1.get('merged_bbox', [0, 0, 0, 0])
                        b2 = g2.get('merged_bbox', [0, 0, 0, 0])
                        # Check if Y ranges overlap
                        y_overlap = not (b1[3] < b2[1] or b2[3] < b1[1])
                        if y_overlap:
                            return b1[0] - b2[0]  # Same line - sort by X
                        return b1[1] - b2[1]  # Different lines - sort by Y
                    
                    from functools import cmp_to_key
                    overlapping_groups.sort(key=cmp_to_key(compare_groups))
                    
                    # Merge all bboxes (including image bbox)
                    all_bboxes = [img_bbox_list] + [g['merged_bbox'] for g in overlapping_groups]
                    merged_bbox = merge_bboxes(all_bboxes)
                    
                    # Merge all texts (now sorted by Y)
                    merged_text = ' '.join([g.get('text', '') for g in overlapping_groups]).strip()
                    
                    # Merge only group bboxes for content (not including image bbox)
                    group_bboxes = [g['merged_bbox'] for g in overlapping_groups]
                    merged_content_bbox = merge_bboxes(group_bboxes)
                    
                    # Debug: Calculate overlap ratios for each group
                    debug_overlapping = []
                    for og in overlapping_groups:
                        og_bbox = og.get('merged_bbox', [0,0,0,0])
                        og_height = og_bbox[3] - og_bbox[1]
                        y_ov_start = max(og_bbox[1], img_bbox_list[1])
                        y_ov_end = min(og_bbox[3], img_bbox_list[3])
                        y_ov_amt = max(0, y_ov_end - y_ov_start)
                        y_ratio = y_ov_amt / og_height if og_height > 0 else 0
                        debug_overlapping.append({
                            'text': og.get('text', '')[:30],
                            'group_y': [round(og_bbox[1], 1), round(og_bbox[3], 1)],
                            'img_y': [round(img_bbox_list[1], 1), round(img_bbox_list[3], 1)],
                            'y_overlap_ratio': round(y_ratio, 2)
                        })
                    
                    # Build single content item with merged text and bbox
                    content = [{
                        'type': 'text',
                        'text': merged_text,
                        'bbox': merged_content_bbox,
                        'groups_count': len(overlapping_groups),
                        'debug_groups': debug_overlapping  # Add debug info
                    }]
                    
                    shapes_list.append({
                        'type': 'shape',
                        'bbox': merged_bbox,
                        'text': merged_text,
                        'image_bbox': img_bbox_list,
                        'image_xref': img.get('xref'),
                        'groups_count': len(overlapping_groups),
                        'content': content,
                        'claimed_groups_texts': [g.get('text', '')[:30] for g in overlapping_groups]  # Debug: show which groups were claimed
                    })
                    
                    # Mark these groups as claimed by shape
                    for g in overlapping_groups:
                        g['claimed_by_shape'] = True
                        # Update debug log
                        for dl in debug_groups_log:
                            if dl['text'][:50] == g.get('text', '')[:50]:
                                dl['claimed_by'] = 'shape'
                                dl['reason'] = f"Y-overlap >= 50% with image bbox {img_bbox_list}"
                                break
                else:
                    # Pure image without text overlay
                    # Skip small images (less than 50x50 pixels)
                    if img_width < 50 or img_height < 50:
                        continue  # Skip small standalone images
                    
                    page_images_list.append({
                        'bbox': img_bbox_list,
                        'xref': img.get('xref'),
                        'width': img.get('width'),
                        'height': img.get('height'),
                        'name': img.get('name', '')
                    })
            
            # Step 2: find_tables() without parameters (basic PyMuPDF detection)
            basic_tables_finder = page.find_tables()
            basic_tables = basic_tables_finder.tables if hasattr(basic_tables_finder, 'tables') else list(basic_tables_finder)
            
            # Helper function to check bbox overlap with Y threshold
            def bbox_overlaps(bbox1, bbox2, y_threshold=0.7):
                """
                Check if bbox1 overlaps with bbox2.
                Uses Y overlap threshold: bbox1 is considered 'inside' bbox2 if
                at least y_threshold (70%) of bbox1's height is within bbox2's Y range.
                bbox format: [x0, y0, x1, y1]
                """
                if not bbox1 or not bbox2:
                    return False
                
                # First check basic X overlap (any overlap)
                if bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2]:
                    return False  # No X overlap at all
                
                # Calculate Y overlap amount
                y_overlap_start = max(bbox1[1], bbox2[1])
                y_overlap_end = min(bbox1[3], bbox2[3])
                y_overlap = max(0, y_overlap_end - y_overlap_start)
                
                # Calculate bbox1's height
                bbox1_height = bbox1[3] - bbox1[1]
                if bbox1_height <= 0:
                    return False
                
                # Check if Y overlap is at least threshold of bbox1's height
                y_overlap_ratio = y_overlap / bbox1_height
                return y_overlap_ratio >= y_threshold
            
            # Serialize basic tables (skip those overlapping with shapes)
            basic_table_list = []
            for t_idx, table in enumerate(basic_tables):
                table_bbox = list(table.bbox) if table.bbox else None
                
                # Check if this table overlaps with any shape
                overlaps_shape = False
                if table_bbox:
                    for shape in shapes_list:
                        shape_bbox = shape.get('bbox')
                        if shape_bbox and simple_bbox_overlap(table_bbox, shape_bbox):
                            overlaps_shape = True
                            break
                
                if overlaps_shape:
                    continue  # Skip - this table overlaps with a shape
                
                table_dict = {
                    'table_index': t_idx,
                    'bbox': table_bbox,
                    'row_count': table.row_count,
                    'col_count': table.col_count,
                    'cells': [list(c) if c else None for c in table.cells] if table.cells else None
                }
                
                rows_list = []
                for r_idx, row in enumerate(table.rows):
                    row_cells = []
                    if hasattr(row, 'cells') and row.cells:
                        for c_idx, cell in enumerate(row.cells):
                            cell_content = []  # Array of content items
                            
                            if cell:
                                cell_bbox = list(cell)
                                
                                # Find character groups that overlap with this cell (50% Y threshold)
                                # Skip groups already claimed by shapes
                                cell_char_groups = []
                                for group in char_groups:
                                    if group.get('claimed_by_shape'):
                                        continue  # Skip - already part of a shape
                                    group_bbox = group.get('merged_bbox')
                                    if group_bbox and bbox_overlaps(group_bbox, cell_bbox, y_threshold=0.5):
                                        cell_char_groups.append(group)
                                        # Mark group as claimed by basic_table
                                        group['claimed_by_table'] = True
                                        # Update debug log
                                        for dl in debug_groups_log:
                                            if dl['text'][:50] == group.get('text', '')[:50] and dl['claimed_by'] is None:
                                                dl['claimed_by'] = 'basic_table'
                                                dl['reason'] = f"50% Y-overlap with cell {cell_bbox}"
                                                break
                                
                                # Merge all character group bboxes and texts into one content item
                                if cell_char_groups:
                                    group_bboxes = [g['merged_bbox'] for g in cell_char_groups if g.get('merged_bbox')]
                                    merged_text_bbox = merge_bboxes(group_bboxes)
                                    merged_text = ' '.join([g.get('text', '') for g in cell_char_groups]).strip()
                                    
                                    if merged_text:
                                        cell_content.append({
                                            'type': 'text',
                                            'text': merged_text,
                                            'bbox': merged_text_bbox,
                                            'groups_count': len(cell_char_groups)
                                        })
                                
                                # Check for images overlapping with this cell (50% Y threshold)
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
                            
                            cell_data = {
                                'row': r_idx,
                                'col': c_idx,
                                'bbox': list(cell) if cell else None,
                                'content': cell_content
                            }
                            
                            row_cells.append(cell_data)
                    rows_list.append({
                        'bbox': list(row.bbox) if hasattr(row, 'bbox') and row.bbox else None,
                        'cells': row_cells
                    })
                table_dict['rows'] = rows_list
                
                try:
                    table_dict['extract'] = table.extract()
                except:
                    table_dict['extract'] = None
                
                basic_table_list.append(table_dict)
            
            # Step 3: Horizontal line table detection for unclaimed groups
            hline_clips, grouping_log, validation_log = guess_table_clips_from_hlines(page, min_rules=2)
            
            # Filter clips to only those containing unclaimed character groups
            # (groups not already inside any basic_table)
            hline_table_list = []
            
            for clip_idx, clip in enumerate(hline_clips):
                # clip is a fitz.Rect object, convert to list bbox
                clip_bbox = [clip.x0, clip.y0, clip.x1, clip.y1]
                
                # Check if this clip overlaps with any basic_table
                overlaps_basic_table = False
                for bt in basic_table_list:
                    if bt['bbox'] and bbox_overlaps(clip_bbox, bt['bbox']):
                        overlaps_basic_table = True
                        break
                
                if overlaps_basic_table:
                    continue  # Skip - already covered by find_tables()
                
                # Check if this clip overlaps with any image (any overlap = skip)
                overlaps_image = False
                for img in page_images_list:
                    img_bbox = img.get('bbox')
                    if img_bbox and bbox_overlaps(clip_bbox, img_bbox, y_threshold=0.0):
                        overlaps_image = True
                        break
                
                if overlaps_image:
                    continue  # Skip - this is an image area, not a table
                
                # Find character groups within this clip (use 30% threshold - just need to know if groups exist in clip)
                # Skip groups already claimed by shapes or basic_tables
                clip_groups = []
                for group in char_groups:
                    if group.get('claimed_by_shape') or group.get('claimed_by_table'):
                        continue  # Skip - already part of a shape or basic_table
                    group_bbox = group.get('merged_bbox')
                    if group_bbox and bbox_overlaps(group_bbox, clip_bbox, y_threshold=0.3):
                        clip_groups.append(group)
                        # Mark as claimed by hline_table and update debug log
                        group['claimed_by_table'] = True
                        for dl in debug_groups_log:
                            if dl['text'][:50] == group.get('text', '')[:50]:
                                dl['claimed_by'] = 'hline_table'
                                dl['reason'] = f"30% Y-overlap with hline clip {clip_bbox}"
                                break
                
                if not clip_groups:
                    continue  # No text content in this clip
                
                # Build table structure from clip
                # Create a simple table structure
                hline_table = {
                    'table_index': clip_idx,
                    'bbox': clip_bbox,
                    'row_count': 1,
                    'col_count': 1,
                    'groups': [],
                    'rows': [],
                    'cells': []
                }
                
                # Detect grid cells from segments within clip
                h_segs = collect_horizontal_segments(page)
                v_segs = collect_vertical_segments(page)
                
                # Filter segments to only those within clip bbox
                h_segs_in_clip = [(y, x0, x1) for (y, x0, x1) in h_segs 
                                  if clip_bbox[1] - 3 <= y <= clip_bbox[3] + 3]
                v_segs_in_clip = [(x, y0, y1) for (x, y0, y1) in v_segs 
                                  if clip_bbox[0] - 3 <= x <= clip_bbox[2] + 3 and
                                     y0 <= clip_bbox[3] + 3 and y1 >= clip_bbox[1] - 3]
                
                cells, cell_debug = detect_cells_from_segments(h_segs_in_clip, v_segs_in_clip, clip_bbox, clip_groups)
                
                # Organize cells by rows
                if cells:
                    hline_table['cells'] = cells
                    hline_table['cell_debug'] = cell_debug
                    
                    # Group cells by row
                    rows_dict = {}
                    for cell in cells:
                        row_idx = cell.get('row', 0)
                        if row_idx not in rows_dict:
                            rows_dict[row_idx] = []
                        rows_dict[row_idx].append(cell)
                    
                    # Build rows structure with cells and content
                    for row_idx in sorted(rows_dict.keys()):
                        row_cells = rows_dict[row_idx]
                        row_data = {
                            'row_index': row_idx,
                            'cells': []
                        }
                        
                        for cell in row_cells:
                            cell_bbox = cell.get('bbox', [])
                            cell_data = {
                                'col': cell.get('col', 0),
                                'bbox': cell_bbox,
                                'content': []
                            }
                            
                            # Find char groups within this cell (use 50% threshold for content assignment)
                            cell_char_groups = []
                            for g in clip_groups:
                                g_bbox = g.get('merged_bbox')
                                if g_bbox and cell_bbox and bbox_overlaps(g_bbox, cell_bbox, y_threshold=0.5):
                                    cell_char_groups.append(g)
                            
                            # Merge all character group bboxes and texts into one content item (like basic_tables)
                            if cell_char_groups:
                                group_bboxes = [g['merged_bbox'] for g in cell_char_groups if g.get('merged_bbox')]
                                merged_text_bbox = merge_bboxes(group_bboxes) if group_bboxes else None
                                merged_text = ' '.join([g.get('text', '') for g in cell_char_groups]).strip()
                                
                                if merged_text:
                                    cell_data['content'].append({
                                        'type': 'text',
                                        'text': merged_text,
                                        'bbox': merged_text_bbox,
                                        'groups_count': len(cell_char_groups)
                                    })
                            
                            row_data['cells'].append(cell_data)
                        
                        hline_table['rows'].append(row_data)
                    
                    hline_table['row_count'] = len(hline_table['rows'])
                    if hline_table['rows']:
                        hline_table['col_count'] = max(len(r['cells']) for r in hline_table['rows'])
                
                # Add groups with their merged text
                for g in clip_groups:
                    hline_table['groups'].append({
                        'text': g.get('text', ''),
                        'bbox': g.get('merged_bbox'),
                        'chars_count': len(g.get('chars', []))
                    })
                
                # Build content similar to basic_tables
                content_items = []
                if clip_groups:
                    group_bboxes = [g['merged_bbox'] for g in clip_groups if g.get('merged_bbox')]
                    merged_text_bbox = merge_bboxes(group_bboxes) if group_bboxes else None
                    merged_text = ' '.join([g.get('text', '') for g in clip_groups]).strip()
                    
                    if merged_text:
                        content_items.append({
                            'type': 'text',
                            'text': merged_text,
                            'bbox': merged_text_bbox,
                            'groups_count': len(clip_groups)
                        })
                
                # Check for images in this clip
                for img in page_images_list:
                    img_bbox = img.get('bbox')
                    if img_bbox and bbox_overlaps(clip_bbox, img_bbox):
                        content_items.append({
                            'type': 'image',
                            'bbox': img_bbox,
                            'xref': img.get('xref'),
                            'width': img.get('width'),
                            'height': img.get('height'),
                            'name': img.get('name', '')
                        })
                
                hline_table['content'] = content_items
                hline_table_list.append(hline_table)
        
        # Filter unclaimed groups (not claimed by shape or table)
        unclaimed_groups = []
        for group in char_groups:
            if not group.get('claimed_by_shape') and not group.get('claimed_by_table'):
                unclaimed_groups.append({
                    'text': group.get('text', ''),
                    'merged_bbox': group.get('merged_bbox'),
                    'is_single': group.get('is_single', False),
                    'block_idx': group.get('block_idx')
                })
        
        # Filter page_images to exclude those overlapping with any table
        # This ensures frontend receives only standalone images
        filtered_page_images = []
        for img in page_images_list:
            img_bbox = img.get('bbox')
            if not img_bbox:
                continue
            
            overlaps_table = False
            # Check against basic_tables
            for table in basic_table_list:
                table_bbox = table.get('bbox')
                if table_bbox and simple_bbox_overlap(img_bbox, table_bbox):
                    overlaps_table = True
                    break
            
            # Check against hline_tables
            if not overlaps_table:
                for hTable in hline_table_list:
                    hTable_bbox = hTable.get('bbox')
                    if hTable_bbox and simple_bbox_overlap(img_bbox, hTable_bbox):
                        overlaps_table = True
                        break
            
            if not overlaps_table:
                filtered_page_images.append(img)
        
        return jsonify({
            'success': True,
            'method': 'merging()',
            'page': page_num,
            'width': rawdict_data.get('width', 0),
            'height': rawdict_data.get('height', 0),
            # Step 1: Character grouping (rawdict)
            'rawdict': rawdict_data,
            'char_groups_count': len(char_groups),
            'char_groups': unclaimed_groups,  # Add unclaimed groups
            'char_groups_unclaimed_count': len(unclaimed_groups),
            # Step 1.5: Page images and shapes
            'page_images': filtered_page_images,
            'page_images_count': len(filtered_page_images),
            'shapes': shapes_list,
            'shapes_count': len(shapes_list),
            # Step 2: Basic find_tables() results
            'basic_tables': basic_table_list,
            'basic_tables_count': len(basic_table_list),
            # Step 3: Horizontal line table detection
            'hline_tables': hline_table_list,
            'hline_tables_count': len(hline_table_list),
            # Debug: All groups with claim status
            'debug_groups_log': debug_groups_log
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@pymupdf_bp.route('/pymupdf-api/page-count/<int:doc_id>')
def get_page_count(doc_id):
    """Get total page count for a document"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            page_count = pdf.page_count
            
        return jsonify({
            'success': True,
            'page_count': page_count
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@pymupdf_bp.route('/pymupdf-api/page-image/<int:doc_id>/<int:page_num>')
def get_page_image(doc_id, page_num):
    """Render PDF page as PNG image (base64)"""
    try:
        document = TestingDokumen.query.get_or_404(doc_id)
        
        with fitz.open(document.testing_dokumen_path) as pdf:
            if page_num < 1 or page_num > pdf.page_count:
                return jsonify({'error': 'Page out of range'}), 400
            
            page = pdf[page_num - 1]
            # Render at 1.5x zoom for better quality
            mat = fitz.Matrix(1.5, 1.5)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img_base64 = base64.b64encode(img_data).decode('utf-8')
            
        return jsonify({
            'success': True,
            'page': page_num,
            'width': pix.width,
            'height': pix.height,
            'image_base64': img_base64
        })
    except Exception as e:
        import traceback
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500



