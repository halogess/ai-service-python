"""Table column alignment utilities"""


def align_table_columns_x(table_cells, table_id):
    """Align X coordinates of cells in the same column"""
    if not table_cells:
        return
    
    # Group cells into columns based on x-center
    cols = []
    sorted_cells = sorted(table_cells, key=lambda c: (c['bbox']['x0'] + c['bbox']['x1'])/2)
    
    if sorted_cells:
        current_col = [sorted_cells[0]]
        last_cnt = (sorted_cells[0]['bbox']['x0'] + sorted_cells[0]['bbox']['x1'])/2
        
        for i in range(1, len(sorted_cells)):
            curr = sorted_cells[i]
            curr_cnt = (curr['bbox']['x0'] + curr['bbox']['x1'])/2
            
            if abs(curr_cnt - last_cnt) < 40:
                current_col.append(curr)
                last_cnt = (last_cnt * (len(current_col)-1) + curr_cnt) / len(current_col)
            else:
                cols.append(current_col)
                current_col = [curr]
                last_cnt = curr_cnt
        cols.append(current_col)
    
    # For each column, calculate unified width
    for col_idx, col in enumerate(cols):
        if not col:
            continue
        
        widths = [c['bbox']['x1'] - c['bbox']['x0'] for c in col]
        if not widths:
            continue
        
        sorted_widths = sorted(widths)
        median_width = sorted_widths[len(widths)//2]
        
        normal_cells = []
        outliers = []
        
        for c in col:
            w = c['bbox']['x1'] - c['bbox']['x0']
            if w > median_width * 1.5:
                outliers.append(c)
            else:
                normal_cells.append(c)
        
        if not normal_cells:
            normal_cells = col
        
        x0s = [c['bbox']['x0'] for c in normal_cells]
        x1s = [c['bbox']['x1'] for c in normal_cells]
        
        unified_x0 = min(x0s)
        unified_x1 = max(x1s)
        
        for c in normal_cells:
            c['bbox']['x0'] = unified_x0
            c['bbox']['x1'] = unified_x1
    
    # Global header fix
    all_x0 = [c['bbox']['x0'] for c in table_cells]
    all_x1 = [c['bbox']['x1'] for c in table_cells]
    
    if all_x0 and all_x1:
        table_min_x = min(all_x0)
        table_max_x = max(all_x1)
        
        for c in table_cells:
            text_lower = c['text'].lower()
            if text_lower.startswith('tabel') or text_lower.startswith('table'):
                c['bbox']['x0'] = table_min_x
                c['bbox']['x1'] = table_max_x
