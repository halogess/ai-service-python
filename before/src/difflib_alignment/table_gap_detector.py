"""Table gap detection (Phase 5) - Split table fix"""


def detect_and_remove_gap_cells(table_cells_buffer, log_file=None):
    """Remove cells after large vertical gap (split table fix)"""
    
    if not table_cells_buffer or len(table_cells_buffer) <= 1:
        return
    
    # Group by page
    from collections import defaultdict
    cells_by_page = defaultdict(list)
    for c in table_cells_buffer:
        cells_by_page[c['page']].append(c)
    
    for page_num, page_cells in cells_by_page.items():
        if len(page_cells) <= 1:
            continue
        
        # Sort by Y
        page_cells.sort(key=lambda c: c['bbox']['y0'])
        
        # Calculate dynamic gap threshold
        cell_heights = [c['bbox']['y1'] - c['bbox']['y0'] for c in page_cells]
        avg_cell_height = sum(cell_heights) / len(cell_heights) if cell_heights else 30.0
        gap_threshold = avg_cell_height * 5
        
        # Find largest internal gap
        max_gap = 0
        split_idx = -1
        for i in range(len(page_cells) - 1):
            curr_y1 = page_cells[i]['bbox']['y1']
            next_y0 = page_cells[i + 1]['bbox']['y0']
            gap = next_y0 - curr_y1
            if gap > max_gap:
                max_gap = gap
                split_idx = i + 1
        
        # If gap exceeds threshold, remove cells AFTER the gap
        if max_gap > gap_threshold and split_idx > 0:
            cells_to_remove = page_cells[split_idx:]
            for c in cells_to_remove:
                if c in table_cells_buffer:
                    table_cells_buffer.remove(c)
            if log_file:
                log_file.write(f"  [Phase 5 Post] Removed {len(cells_to_remove)} cells after {max_gap:.0f}px gap\n")


def validate_cell_vertical_position(candidate_bbox, table_cells_buffer, page_num, log_file=None):
    """Validate if candidate cell is within acceptable Y range of existing cluster"""
    
    if not candidate_bbox or not table_cells_buffer:
        return True
    
    # Get all cells on same page
    same_page_cells = [c for c in table_cells_buffer if c['page'] == page_num]
    if not same_page_cells:
        return True
    
    # Calculate Y bounds of existing cluster
    cluster_min_y = min(c['bbox']['y0'] for c in same_page_cells)
    cluster_max_y = max(c['bbox']['y1'] for c in same_page_cells)
    
    curr_y0 = candidate_bbox[1]
    curr_y1 = candidate_bbox[3]
    
    # Calculate dynamic gap threshold
    cell_heights = [c['bbox']['y1'] - c['bbox']['y0'] for c in same_page_cells]
    avg_cell_height = sum(cell_heights) / len(cell_heights) if cell_heights else 30.0
    gap_threshold = max(avg_cell_height * 5, 50.0)
    
    # Check if candidate is FAR BELOW the existing cluster
    gap_below = curr_y0 - cluster_max_y
    
    if gap_below > gap_threshold:
        if log_file:
            log_file.write(f"  [Phase 5] Rejecting cell: {gap_below:.0f}px below cluster (threshold={gap_threshold:.0f})\n")
        return False
    
    return True
