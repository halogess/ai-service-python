"""
Text-in-Box Merging Utility

Takes character groups (from char_grouping.js) and combines text content
that falls within each group's merged bbox.
"""


def merge_text_in_boxes(rawdict, char_groups):
    """
    For each character group, find all text spans that fall within its bbox
    and merge them into a single text string.
    
    Args:
        rawdict: PyMuPDF page.get_text("rawdict") output
        char_groups: List of character groups with merged bboxes
                     [{id, text, bbox: [x0, y0, x1, y1], chars: [...]}]
    
    Returns:
        List of merged text boxes:
        [{
            group_id: str,
            bbox: [x0, y0, x1, y1],
            original_text: str,  # from char_grouping
            merged_text: str,    # all text found within bbox
            spans: [{text, origin, bbox}]  # individual spans found
        }]
    """
    if not rawdict or not char_groups:
        return []
    
    # Extract all text spans from rawdict
    all_spans = []
    for block in rawdict.get('blocks', []):
        if block.get('type') != 0:  # Only text blocks
            continue
        for line in block.get('lines', []):
            for span in line.get('spans', []):
                span_bbox = span.get('bbox', [])
                if len(span_bbox) >= 4:
                    all_spans.append({
                        'text': span.get('text', ''),
                        'bbox': span_bbox,
                        'origin': span.get('origin', []),
                        'font': span.get('font', ''),
                        'size': span.get('size', 0)
                    })
    
    merged_boxes = []
    
    for group in char_groups:
        group_bbox = group.get('bbox', [])
        if len(group_bbox) < 4:
            continue
        
        gx0, gy0, gx1, gy1 = group_bbox
        
        # Find all spans that overlap with this group's bbox
        matching_spans = []
        for span in all_spans:
            sx0, sy0, sx1, sy1 = span['bbox']
            
            # Check if span overlaps with group bbox
            if spans_overlap(gx0, gy0, gx1, gy1, sx0, sy0, sx1, sy1):
                matching_spans.append(span)
        
        # Sort spans by y position (top to bottom), then x position (left to right)
        matching_spans.sort(key=lambda s: (s['bbox'][1], s['bbox'][0]))
        
        # Merge text from all matching spans
        merged_text = ' '.join(s['text'] for s in matching_spans)
        
        merged_boxes.append({
            'group_id': group.get('id', ''),
            'bbox': group_bbox,
            'original_text': group.get('text', ''),
            'merged_text': merged_text.strip(),
            'span_count': len(matching_spans),
            'spans': [{
                'text': s['text'],
                'bbox': s['bbox']
            } for s in matching_spans]
        })
    
    return merged_boxes


def spans_overlap(gx0, gy0, gx1, gy1, sx0, sy0, sx1, sy1, threshold=0.5):
    """
    Check if a span bbox overlaps with a group bbox.
    Uses intersection over span area to determine overlap.
    """
    # Calculate intersection
    ix0 = max(gx0, sx0)
    iy0 = max(gy0, sy0)
    ix1 = min(gx1, sx1)
    iy1 = min(gy1, sy1)
    
    if ix0 >= ix1 or iy0 >= iy1:
        return False
    
    intersection_area = (ix1 - ix0) * (iy1 - iy0)
    span_area = (sx1 - sx0) * (sy1 - sy0)
    
    if span_area <= 0:
        return False
    
    # If at least threshold% of span is inside group bbox, consider it a match
    return (intersection_area / span_area) >= threshold


def merge_text_in_drawings(rawdict, drawings):
    """
    For each drawing rectangle, find all text spans that fall within it
    and merge them.
    
    Args:
        rawdict: PyMuPDF page.get_text("rawdict") output
        drawings: List of drawing rectangles [{rect: [x0, y0, x1, y1], ...}]
    
    Returns:
        List of merged text boxes for drawings
    """
    if not rawdict or not drawings:
        return []
    
    # Extract rectangles from drawings
    rect_boxes = []
    for i, drawing in enumerate(drawings):
        rect = drawing.get('rect', [])
        if len(rect) >= 4:
            rect_boxes.append({
                'id': f'drawing_{i}',
                'bbox': rect,
                'text': ''
            })
    
    return merge_text_in_boxes(rawdict, rect_boxes)


def merge_groups_by_container(char_groups, containers):
    """
    Merge character groups that fall within the same container (table cell/shape).
    
    Args:
        char_groups: List of character groups from char_grouping.js
                     [{id, text, bbox: [x0, y0, x1, y1]}]
        containers: List of container rectangles (cells/shapes)
                    [{id, rect: [x0, y0, x1, y1]}]
    
    Returns:
        List of merged containers with combined text:
        [{
            container_id: str,
            bbox: [x0, y0, x1, y1],
            merged_text: str,
            group_ids: [list of group ids inside],
            group_count: int
        }]
    """
    if not char_groups or not containers:
        return []
    
    merged_containers = []
    used_groups = set()  # Track which groups have been assigned
    
    for container in containers:
        container_rect = container.get('rect', [])
        if len(container_rect) < 4:
            continue
        
        cx0, cy0, cx1, cy1 = container_rect
        
        # Find all char groups whose center falls within this container
        matching_groups = []
        for group in char_groups:
            group_bbox = group.get('bbox', [])
            if len(group_bbox) < 4:
                continue
            
            gx0, gy0, gx1, gy1 = group_bbox
            # Calculate group center
            group_cx = (gx0 + gx1) / 2
            group_cy = (gy0 + gy1) / 2
            
            # Check if group center is inside container
            if cx0 <= group_cx <= cx1 and cy0 <= group_cy <= cy1:
                group_id = group.get('id', '')
                if group_id not in used_groups:
                    matching_groups.append(group)
                    used_groups.add(group_id)
        
        if not matching_groups:
            continue
        
        # Sort groups by y position (top to bottom), then x (left to right)
        matching_groups.sort(key=lambda g: (g['bbox'][1], g['bbox'][0]))
        
        # Merge text from all matching groups
        merged_text = ' '.join(g.get('text', '') for g in matching_groups)
        
        merged_containers.append({
            'container_id': container.get('id', ''),
            'bbox': container_rect,
            'merged_text': merged_text.strip(),
            'group_ids': [g.get('id', '') for g in matching_groups],
            'group_count': len(matching_groups)
        })
    
    return merged_containers
