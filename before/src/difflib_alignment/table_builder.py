"""Build table elements from grouped tokens"""

from .config import TABLE_MERGE_Y_MAX_GAP
from .table_cell_processor import process_table_cell
from .table_gap_detector import detect_and_remove_gap_cells
from .table_structure_cache import TableStructureCache


def build_table_container(cell_groups, main_page):
    """Build table container bbox dari cells"""
    cells_by_page = {}
    for cell_idx, words in cell_groups.items():
        if words:
            page = words[0]['page']
            if page not in cells_by_page:
                cells_by_page[page] = []
            cells_by_page[page].append(cell_idx)
    
    if not cells_by_page:
        return None
    
    main_page = max(cells_by_page.keys(), key=lambda p: len(cells_by_page[p]))
    main_page_cells = cells_by_page[main_page]
    
    cell_bboxes = []
    for cell_idx in main_page_cells:
        words = cell_groups[cell_idx]
        if words:
            x0 = min(w["bbox"]["x0"] for w in words)
            y0 = min(w["bbox"]["y0"] for w in words)
            x1 = max(w["bbox"]["x1"] for w in words)
            y1 = max(w["bbox"]["y1"] for w in words)
            cell_bboxes.append({'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    
    if not cell_bboxes:
        return None
    
    # Filter outliers
    sorted_bboxes = sorted(cell_bboxes, key=lambda b: (b['y0'] + b['y1']) / 2)
    
    if len(sorted_bboxes) > 1:
        clusters = []
        current_cluster = [sorted_bboxes[0]]
        
        for i in range(1, len(sorted_bboxes)):
            prev = sorted_bboxes[i-1]
            curr = sorted_bboxes[i]
            
            gap = curr['y0'] - prev['y1']
            
            if gap > TABLE_MERGE_Y_MAX_GAP:
                clusters.append(current_cluster)
                current_cluster = []
            
            current_cluster.append(curr)
        
        clusters.append(current_cluster)
        
        if len(clusters) > 1:
            main_cluster = max(clusters, key=len)
            cell_bboxes = main_cluster

    table_x0 = min(b['x0'] for b in cell_bboxes)
    table_y0 = min(b['y0'] for b in cell_bboxes)
    table_x1 = max(b['x1'] for b in cell_bboxes)
    table_y1 = max(b['y1'] for b in cell_bboxes)
    
    return {"x0": table_x0, "y0": table_y0, "x1": table_x1, "y1": table_y1}, main_page


def build_table_cells(elem_id, cell_groups, element_cell_texts, pdf_path, used_pdf_cells, 
                      log_file=None, image_only_cells_map=None, skip_gap_detection=False, 
                      table_cell_images=None, empty_cells_data=None):
    """Build table cell elements
    
    Args:
        empty_cells_data: list of (cell_idx, row_idx, col_idx) tuples for empty cells
    """
    table_cells_buffer = []
    pymupdf_table_cache = {}
    table_structure_cache = TableStructureCache(pdf_path)
    
    all_cell_indices = set(cell_groups.keys())
    if elem_id in element_cell_texts:
        all_cell_indices.update(element_cell_texts[elem_id].keys())
    
    # Add image-only cells
    if image_only_cells_map and elem_id in image_only_cells_map:
        for cell_idx in image_only_cells_map[elem_id]:
            all_cell_indices.add(cell_idx)
    
    # Add empty cells
    empty_cells_info = {}  # cell_idx -> (row_idx, col_idx)
    if empty_cells_data:
        for cell_idx, row_idx, col_idx in empty_cells_data:
            all_cell_indices.add(cell_idx)
            empty_cells_info[cell_idx] = (row_idx, col_idx)
    
    sorted_cell_indices = sorted([k for k in all_cell_indices if isinstance(k, int) and k != -1])

    # Determine main page from existing cells
    main_page = 0
    for cell_idx in sorted_cell_indices:
        words = cell_groups.get(cell_idx, [])
        if words:
            main_page = words[0]["page"]
            break

    for cell_idx in sorted_cell_indices:
        words = cell_groups.get(cell_idx, [])
        
        cell_content = []
        if elem_id in element_cell_texts and cell_idx in element_cell_texts[elem_id]:
            cell_content = element_cell_texts[elem_id][cell_idx]
        
        # Determine cell_text from content list for backward compatibility
        cell_text = ""
        if isinstance(cell_content, list):
            texts = [item['value'] for item in cell_content if item.get('type') == 'text']
            cell_text = " ".join(texts)
        elif isinstance(cell_content, str):
            cell_text = cell_content
            cell_content = [{'type': 'text', 'value': cell_text}] if cell_text else []
        
        # Check if this is image-only cell
        is_image_only = False
        if image_only_cells_map and elem_id in image_only_cells_map:
            if cell_idx in image_only_cells_map[elem_id]:
                is_image_only = True
        
        # Check if this is empty cell
        is_empty_cell = cell_idx in empty_cells_info
        
        page_num = 0
        if words:
            page_num = words[0]["page"]
        elif table_cells_buffer:
            page_num = table_cells_buffer[-1]['page']
        else:
            page_num = main_page
        
        if is_empty_cell and not words:
            # Handle empty cell with spatial alignment
            row_idx, col_idx = empty_cells_info[cell_idx]
            empty_bbox = table_structure_cache.get_empty_cell_bbox(page_num, row_idx, col_idx)
            
            if empty_bbox:
                empty_result = {
                    "text": "",
                    "matched_text": "",
                    "bbox": empty_bbox,
                    "bboxes": [],
                    "page": page_num,
                    "element_id": f"{elem_id}_cell_{cell_idx}",
                    "parent_element_id": elem_id,
                    "confidence": 0.7,  # Lower confidence for spatial matching
                    "before_align_bboxes": [],
                    "is_table_cell": True,
                    "is_empty_cell": True
                }
                table_cells_buffer.append(empty_result)
                
                if log_file:
                    log_file.write(f"Empty cell {elem_id}_cell_{cell_idx} aligned spatially to row={row_idx}, col={col_idx}\n")
            continue
        
        cell_result = process_table_cell(
            cell_idx, words, cell_content, page_num, elem_id,
            pdf_path, pymupdf_table_cache, used_pdf_cells, is_image_only, table_cell_images
        )
        
        if cell_result:
            table_cells_buffer.append(cell_result)
    
    # Apply gap detection only if not skipped
    if not skip_gap_detection:
        detect_and_remove_gap_cells(table_cells_buffer, log_file)
    
    return table_cells_buffer

