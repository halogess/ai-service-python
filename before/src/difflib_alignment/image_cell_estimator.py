"""Image-only cell position estimation based on table layout"""


def estimate_image_only_cell_positions(table_cells_buffer, num_cols_per_row, log_file=None):
    """Estimate bbox for image-only cells based on table grid layout
    
    Strategy:
    1. Group cells by row (based on cell_idx / num_cols)
    2. Calculate column X-ranges from aligned cells
    3. Calculate row Y-ranges from aligned cells  
    4. Assign estimated bbox to image-only cells based on their row/column position
    
    Args:
        table_cells_buffer: list of cell results from build_table_cells
        num_cols_per_row: number of columns per row in the table
        log_file: optional log file
    """
    if not table_cells_buffer or num_cols_per_row <= 0:
        return
    
    # Build row/column info from cell index
    cells_by_row = {}  # row_idx -> list of cells
    cells_by_col = {}  # col_idx -> list of cells
    
    for cell in table_cells_buffer:
        elem_id = cell.get('element_id', '')
        if '_cell_' not in elem_id:
            continue
        
        try:
            cell_idx = int(elem_id.split('_cell_')[1])
        except:
            continue
        
        row_idx = cell_idx // num_cols_per_row
        col_idx = cell_idx % num_cols_per_row
        
        if row_idx not in cells_by_row:
            cells_by_row[row_idx] = []
        cells_by_row[row_idx].append((col_idx, cell))
        
        if col_idx not in cells_by_col:
            cells_by_col[col_idx] = []
        cells_by_col[col_idx].append((row_idx, cell))
    
    # Calculate column X-ranges from aligned cells
    column_x_ranges = {}  # col_idx -> (x0_min, x1_max)
    for col_idx, row_cells in cells_by_col.items():
        valid_x0s = []
        valid_x1s = []
        for row_idx, cell in row_cells:
            bbox = cell.get('bbox', {})
            # Only use cells with valid bbox (not 0,0,0,0)
            if bbox.get('x0', 0) > 0 or bbox.get('x1', 0) > 0:
                valid_x0s.append(bbox.get('x0', 0))
                valid_x1s.append(bbox.get('x1', 0))
        
        if valid_x0s and valid_x1s:
            column_x_ranges[col_idx] = (min(valid_x0s), max(valid_x1s))
    
    # Calculate row Y-ranges from aligned cells
    row_y_ranges = {}  # row_idx -> (y0_min, y1_max)
    for row_idx, col_cells in cells_by_row.items():
        valid_y0s = []
        valid_y1s = []
        for col_idx, cell in col_cells:
            bbox = cell.get('bbox', {})
            if bbox.get('y0', 0) > 0 or bbox.get('y1', 0) > 0:
                valid_y0s.append(bbox.get('y0', 0))
                valid_y1s.append(bbox.get('y1', 0))
        
        if valid_y0s and valid_y1s:
            row_y_ranges[row_idx] = (min(valid_y0s), max(valid_y1s))
    
    if log_file:
        log_file.write(f"Column X-ranges: {column_x_ranges}\n")
        log_file.write(f"Row Y-ranges: {row_y_ranges}\n")
    
    # Now update image-only cells with estimated positions
    for cell in table_cells_buffer:
        # Only process image-only cells with invalid bbox
        if not cell.get('is_image_only_cell'):
            continue
        
        bbox = cell.get('bbox', {})
        if bbox.get('x0', 0) > 0 or bbox.get('y0', 0) > 0:
            continue  # Already has valid bbox
        
        elem_id = cell.get('element_id', '')
        if '_cell_' not in elem_id:
            continue
        
        try:
            cell_idx = int(elem_id.split('_cell_')[1])
        except:
            continue
        
        row_idx = cell_idx // num_cols_per_row
        col_idx = cell_idx % num_cols_per_row
        
        # Get estimated X from column
        x0_est, x1_est = 0, 0
        if col_idx in column_x_ranges:
            x0_est, x1_est = column_x_ranges[col_idx]
        elif column_x_ranges:
            # Fallback: use leftmost known column
            first_col = min(column_x_ranges.keys())
            x0_est, x1_est = column_x_ranges[first_col]
        
        # Get estimated Y from row
        y0_est, y1_est = 0, 0
        if row_idx in row_y_ranges:
            y0_est, y1_est = row_y_ranges[row_idx]
        elif row_y_ranges:
            # Fallback: estimate from previous row
            prev_rows = [r for r in row_y_ranges.keys() if r < row_idx]
            if prev_rows:
                prev_row = max(prev_rows)
                prev_y0, prev_y1 = row_y_ranges[prev_row]
                row_height = prev_y1 - prev_y0
                gap = row_height * 0.1  # 10% gap estimate
                y0_est = prev_y1 + gap
                y1_est = y0_est + row_height
            else:
                # Use first available row
                first_row = min(row_y_ranges.keys())
                y0_est, y1_est = row_y_ranges[first_row]
        
        if x0_est > 0 or y0_est > 0:
            cell['bbox'] = {'x0': x0_est, 'y0': y0_est, 'x1': x1_est, 'y1': y1_est}
            cell['bbox_estimated'] = True
            
            if log_file:
                log_file.write(f"Estimated {elem_id}: row={row_idx}, col={col_idx}, bbox={cell['bbox']}\n")


def get_table_columns_from_docx(elem_id, elements):
    """Get number of columns per row for a table element from DOCX structure
    
    Returns:
        int: number of columns per row, or 0 if unknown
    """
    for elem in elements:
        # Handle both dict and object access
        if isinstance(elem, dict):
            current_id = elem.get('delemen_id')
            json_tree = elem.get('delemen_json_tree', {})
        else:
            current_id = getattr(elem, 'delemen_id', None)
            json_tree = getattr(elem, 'delemen_json_tree', {})

        if current_id == elem_id:
            content = {}
            if isinstance(json_tree, dict):
                content = json_tree.get('content', {})
            
            if isinstance(content, dict) and 'rows' in content:
                rows = content.get('rows', [])
                if rows:
                    # Use first row's cell count
                    first_row = rows[0]
                    return len(first_row.get('cells', []))
    
    return 0
