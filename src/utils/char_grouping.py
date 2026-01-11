"""
PyMuPDF Character Grouping Module (Python port)
Detects overlapping characters based on X-coordinate and groups them together.
Port of char_grouping.js to Python.
"""

# Constants
X_TOLERANCE = 3  # X tolerance in PDF points
Y_OVERLAP_MIN_RATIO = 0.3  # Minimum 30% overlap of min height required
LINE_OVERLAP_THRESHOLD = 0.5  # Minimum 50% overlap to be considered same line (for reading order)


def collect_all_chars(rawdict_data):
    """
    Collect all characters from rawdict data into a flat list.
    
    Args:
        rawdict_data: PyMuPDF page.get_text("rawdict") output
        
    Returns:
        List of char dicts with bbox and indices
    """
    chars = []
    blocks = rawdict_data.get('blocks', [])
    
    for block_idx, block in enumerate(blocks):
        # Skip image blocks (type 1)
        if block.get('type', 0) == 1:
            continue
            
        lines = block.get('lines', [])
        for line_idx, line in enumerate(lines):
            spans = line.get('spans', [])
            for span_idx, span in enumerate(spans):
                span_chars = span.get('chars', [])
                for char_idx, char in enumerate(span_chars):
                    bbox = char.get('bbox')
                    char_text = char.get('c', '')
                    
                    # Skip whitespace-only characters
                    if not char_text or char_text.isspace():
                        continue
                    
                    if bbox and len(bbox) >= 4:
                        chars.append({
                            'c': char_text,
                            'bbox': list(bbox),
                            'block_idx': block_idx,
                            'line_idx': line_idx,
                            'span_idx': span_idx,
                            'char_idx': char_idx
                        })
    
    return chars


def x_overlap(bbox1, bbox2, tolerance=X_TOLERANCE):
    """
    Check if bbox2's x0 is within bbox1's x range (x0 to x1) with tolerance.
    
    Args:
        bbox1: [x0, y0, x1, y1] - the reference bbox
        bbox2: [x0, y0, x1, y1] - the bbox to check
        tolerance: tolerance in PDF points
        
    Returns:
        bool
    """
    return bbox2[0] >= (bbox1[0] - tolerance) and bbox2[0] <= (bbox1[2] + tolerance)


def y_overlap(bbox1, bbox2):
    """
    Check if bbox2 has at least 30% Y overlap with bbox1 (based on minimum height).
    
    Args:
        bbox1: [x0, y0, x1, y1]
        bbox2: [x0, y0, x1, y1]
        
    Returns:
        bool
    """
    height1 = bbox1[3] - bbox1[1]
    height2 = bbox2[3] - bbox2[1]
    min_height = min(height1, height2)
    
    if min_height <= 0:
        return False
    
    # Calculate Y overlap amount
    overlap_start = max(bbox1[1], bbox2[1])
    overlap_end = min(bbox1[3], bbox2[3])
    overlap_amount = max(0, overlap_end - overlap_start)
    
    # Check if overlap is at least 30% of min height
    overlap_ratio = overlap_amount / min_height
    return overlap_ratio >= Y_OVERLAP_MIN_RATIO


def calculate_y_overlap_ratio(g1, g2):
    """
    Calculate Y overlap ratio between two groups (for line detection).
    Uses the smaller group's height as the denominator.
    
    Args:
        g1: Group with 'merged_bbox' [x0, y0, x1, y1]
        g2: Group with 'merged_bbox' [x0, y0, x1, y1]
        
    Returns:
        Float: overlap ratio (0.0 to 1.0)
    """
    bbox1 = g1['merged_bbox']
    bbox2 = g2['merged_bbox']
    
    height1 = bbox1[3] - bbox1[1]
    height2 = bbox2[3] - bbox2[1]
    min_height = min(height1, height2)
    
    if min_height <= 0:
        return 0.0
    
    overlap_start = max(bbox1[1], bbox2[1])
    overlap_end = min(bbox1[3], bbox2[3])
    overlap_amount = max(0, overlap_end - overlap_start)
    
    return overlap_amount / min_height


def sort_groups_reading_order(groups, line_threshold=LINE_OVERLAP_THRESHOLD):
    """
    Sort groups in reading order: top-to-bottom by line, left-to-right within line.
    Groups with significant Y overlap are considered same line.
    
    This fixes the issue where tall elements (like fractions) would appear
    before shorter left-side content due to pure Y-based sorting.
    
    Args:
        groups: List of groups with 'merged_bbox'
        line_threshold: Minimum Y overlap ratio to be considered same line
        
    Returns:
        List of groups sorted in reading order
    """
    if not groups:
        return []
    
    if len(groups) == 1:
        return groups
    
    # Build lines by clustering groups with Y overlap
    lines = []
    remaining = list(groups)
    
    while remaining:
        # Start a new line with the topmost remaining group
        remaining.sort(key=lambda g: g['merged_bbox'][1])  # Sort by Y
        current_line = [remaining.pop(0)]
        
        # Find all groups that overlap with this line
        changed = True
        while changed:
            changed = False
            new_remaining = []
            for g in remaining:
                # Check if this group overlaps with any group in current line
                overlaps = False
                for line_g in current_line:
                    if calculate_y_overlap_ratio(g, line_g) >= line_threshold:
                        overlaps = True
                        break
                
                if overlaps:
                    current_line.append(g)
                    changed = True
                else:
                    new_remaining.append(g)
            remaining = new_remaining
        
        lines.append(current_line)
    
    # Sort lines by their top Y (min y0 in line)
    lines.sort(key=lambda line: min(g['merged_bbox'][1] for g in line))
    
    # Sort groups within each line by X (left to right)
    for line in lines:
        line.sort(key=lambda g: g['merged_bbox'][0])
    
    # Flatten back to list
    return [g for line in lines for g in line]


def is_overlapping(group_bbox, char_bbox, x_tol=X_TOLERANCE):
    """
    Check if two bboxes overlap based on X and Y criteria.
    
    Args:
        group_bbox: merged bbox of the group [x0, y0, x1, y1]
        char_bbox: bbox of character to check [x0, y0, x1, y1]
        x_tol: X tolerance in PDF points
        
    Returns:
        bool
    """
    return x_overlap(group_bbox, char_bbox, x_tol) and y_overlap(group_bbox, char_bbox)


def find_overlapping_groups(chars):
    """
    Find all overlapping character groups using DFS (Depth-First Search).
    
    Args:
        chars: Flat list of char dicts from collect_all_chars
        
    Returns:
        List of groups, each with 'chars', 'merged_bbox', 'text'
    """
    if not chars:
        return []
    
    # Sort by X first (left to right), then by Y (top to bottom)
    sorted_chars = sorted(chars, key=lambda c: (c['bbox'][0], c['bbox'][1]))
    
    groups = []
    processed = set()
    
    for i in range(len(sorted_chars)):
        if i in processed:
            continue
        
        # DFS with stack
        stack = [i]
        group = {
            'chars': [],
            'merged_bbox': None
        }
        
        while stack:
            current_idx = stack.pop()
            if current_idx in processed:
                continue
            
            current_char = sorted_chars[current_idx]
            processed.add(current_idx)
            group['chars'].append(current_char)
            
            # Track the block_idx of the group (first char sets it)
            if 'block_idx' not in group:
                group['block_idx'] = current_char.get('block_idx')
            
            # Update merged bbox
            if group['merged_bbox'] is None:
                group['merged_bbox'] = list(current_char['bbox'])
            else:
                group['merged_bbox'][0] = min(group['merged_bbox'][0], current_char['bbox'][0])
                group['merged_bbox'][1] = min(group['merged_bbox'][1], current_char['bbox'][1])
                group['merged_bbox'][2] = max(group['merged_bbox'][2], current_char['bbox'][2])
                group['merged_bbox'][3] = max(group['merged_bbox'][3], current_char['bbox'][3])
            
            # Find all neighbors that overlap with merged bbox
            # IMPORTANT: Only allow merging characters from the SAME block
            group_block_idx = group.get('block_idx')
            for j in range(len(sorted_chars)):
                if j in processed or j in stack:
                    continue
                
                other_char = sorted_chars[j]
                
                # Skip if different block_idx - prevents cross-block merging
                if other_char.get('block_idx') != group_block_idx:
                    continue
                
                # Check overlap with merged bbox
                if is_overlapping(group['merged_bbox'], other_char['bbox'], X_TOLERANCE):
                    stack.append(j)
        
        # Add group (both singles and multi-char groups)
        if group['chars']:
            # Sort chars by X position
            group['chars'].sort(key=lambda c: c['bbox'][0])
            group['text'] = ''.join(c['c'] for c in group['chars'])
            group['is_single'] = len(group['chars']) == 1
            groups.append(group)
    
    # Sort groups in reading order (line-aware: top-to-bottom, left-to-right within line)
    # This fixes ordering issues with tall elements like fractions
    groups = sort_groups_reading_order(groups)
    
    return groups


def get_groups_in_y_range(groups, y_top, y_bottom):
    """
    Get all character groups within a Y range.
    
    Args:
        groups: List of groups from find_overlapping_groups
        y_top: Top Y coordinate
        y_bottom: Bottom Y coordinate
        
    Returns:
        List of groups whose merged_bbox intersects with [y_top, y_bottom]
    """
    result = []
    for g in groups:
        bbox = g['merged_bbox']
        # Check if group's Y range overlaps with [y_top, y_bottom]
        if bbox[3] > y_top and bbox[1] < y_bottom:
            result.append(g)
    return result


def detect_column_gaps_from_groups(groups):
    """
    Detect column boundaries (gaps) from character groups.
    Groups are sorted by X, and gaps > threshold are column boundaries.
    
    Args:
        groups: List of groups sorted by Y
        
    Returns:
        List of X positions where column boundaries exist
    """
    if len(groups) < 2:
        return []
    
    # Sort by X position
    sorted_groups = sorted(groups, key=lambda g: g['merged_bbox'][0])
    
    boundaries = []
    gap_threshold = 15  # Minimum gap between columns
    
    for i in range(len(sorted_groups) - 1):
        right_edge = sorted_groups[i]['merged_bbox'][2]  # x1 of current group
        left_edge = sorted_groups[i + 1]['merged_bbox'][0]  # x0 of next group
        
        gap = left_edge - right_edge
        if gap > gap_threshold:
            # Gap center is the column boundary
            boundaries.append((right_edge + left_edge) / 2)
    
    return boundaries


def calculate_coverage(groups, row_width):
    """
    Calculate text coverage ratio for a row.
    
    Args:
        groups: List of character groups in the row
        row_width: Total width of the row (x1 - x0)
        
    Returns:
        Float: ratio of text width to row width (0.0 to 1.0+)
    """
    if not groups or row_width <= 0:
        return 0.0
    
    # Calculate total text width (sum of all group widths)
    total_text_width = sum(g['merged_bbox'][2] - g['merged_bbox'][0] for g in groups)
    
    return total_text_width / row_width


def count_large_gaps(groups, threshold=30):
    """
    Count number of gaps between groups that are larger than threshold.
    Larger gaps indicate column separators (tables have these, paragraphs don't).
    
    Args:
        groups: List of character groups in the row
        threshold: Minimum gap size to count as "large" (default 30pt)
        
    Returns:
        Int: number of large gaps
    """
    if len(groups) < 2:
        return 0
    
    sorted_groups = sorted(groups, key=lambda g: g['merged_bbox'][0])
    
    large_gap_count = 0
    for i in range(len(sorted_groups) - 1):
        right_edge = sorted_groups[i]['merged_bbox'][2]
        left_edge = sorted_groups[i + 1]['merged_bbox'][0]
        gap = left_edge - right_edge
        if gap > threshold:
            large_gap_count += 1
    
    return large_gap_count


def detect_column_boundaries(groups):
    """
    Detect column boundary X positions from character groups.
    Boundaries are the midpoints between consecutive groups.
    
    Args:
        groups: List of character groups (must have 2+ groups for boundaries)
        
    Returns:
        List of X positions where column boundaries exist
    """
    if len(groups) < 2:
        return []
    
    # Sort by X position
    sorted_groups = sorted(groups, key=lambda g: g['merged_bbox'][0])
    
    boundaries = []
    for i in range(len(sorted_groups) - 1):
        right_edge = sorted_groups[i]['merged_bbox'][2]  # x1 of current group
        left_edge = sorted_groups[i + 1]['merged_bbox'][0]  # x0 of next group
        
        # Only create boundary if there's a gap
        if left_edge > right_edge:
            # Midpoint of the gap is the boundary
            boundaries.append((right_edge + left_edge) / 2)
    
    return boundaries


def check_boundary_crossing(groups, boundaries):
    """
    Check if any character group crosses (spans across) a column boundary.
    
    Args:
        groups: List of character groups to check
        boundaries: List of X positions of column boundaries
        
    Returns:
        Tuple: (is_crossing: bool, crossing_info: dict or None)
    """
    if not groups or not boundaries:
        return False, None
    
    for g in groups:
        x0 = g['merged_bbox'][0]
        x1 = g['merged_bbox'][2]
        
        for boundary in boundaries:
            # Group crosses boundary if it starts before AND ends after
            if x0 < boundary and x1 > boundary:
                return True, {
                    'group_x': [round(x0, 1), round(x1, 1)],
                    'boundary': round(boundary, 1),
                    'text_preview': g.get('text', '')[:30]
                }
    
    return False, None
