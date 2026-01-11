"""Build final aligned elements from grouped tokens"""

from .docx_extractor import extract_text_from_json_tree
from .shape_element_builder import build_shape_container, build_single_shape_element


def build_element_metadata(elements):
    """Build metadata untuk setiap element"""
    element_texts = {}
    element_is_table = {}
    element_is_shape_array = {}
    element_cell_texts = {}
    element_shape_data = {}
    element_empty_cells = {}
    element_table_structure = {}

    for elem in elements:
        # Handle both dict and object access
        if isinstance(elem, dict):
            json_tree = elem.get('delemen_json_tree', {})
            elem_id = elem.get('delemen_id')
        else:
            json_tree = getattr(elem, 'delemen_json_tree', {})
            elem_id = getattr(elem, 'delemen_id', None)
        
        elem_text = extract_text_from_json_tree(json_tree)
        element_texts[elem_id] = elem_text
        
        # Get cells with empty cells and table structure info
        cells_result = extract_text_from_json_tree(
            json_tree, 
            return_cells=True,
            return_empty_cells=True,
            return_table_structure=True
        )
        
        if isinstance(cells_result, tuple) and len(cells_result) == 4:
            cells, image_only_cells, empty_cells, table_structure = cells_result
        elif isinstance(cells_result, tuple):
            cells = cells_result[0]
            empty_cells = []
            table_structure = {}
        else:
            cells = cells_result if isinstance(cells_result, list) else []
            empty_cells = []
            table_structure = {}
        
        element_is_table[elem_id] = isinstance(cells, list) and len(cells) > 0
        
        if element_is_table[elem_id]:
            # cells is now list of (idx, content_list, row_idx, col_idx)
            # Store as dict: {idx: content, ...}
            element_cell_texts[elem_id] = {}
            for cell_tuple in cells:
                if len(cell_tuple) >= 2:
                    idx = cell_tuple[0]
                    content = cell_tuple[1]
                    element_cell_texts[elem_id][idx] = content
            
            # Store empty cells and table structure
            if empty_cells:
                element_empty_cells[elem_id] = empty_cells
            if table_structure:
                element_table_structure[elem_id] = table_structure
        
        shapes = extract_text_from_json_tree(json_tree, return_shapes=True)
        element_is_shape_array[elem_id] = isinstance(shapes, list) and len(shapes) > 0
        if element_is_shape_array[elem_id]:
            element_shape_data[elem_id] = {idx: (text, shape_item) for idx, text, shape_item in shapes}

    return (element_texts, element_is_table, element_is_shape_array, element_cell_texts, 
            element_shape_data, element_empty_cells, element_table_structure)


def build_shape_elements(elem_id, cell_groups, element_texts, element_shape_data):
    """Build elements untuk shape array"""
    final_aligned = []
    
    shape_order = []
    for shape_idx, words in cell_groups.items():
        if words:
            first_pdf_idx = min(w.get('pdf_index', float('inf')) for w in words)
            shape_order.append((first_pdf_idx, shape_idx))
    shape_order.sort()
    sorted_shape_indices = [shape_idx for _, shape_idx in shape_order]
    
    # Build container
    container = build_shape_container(elem_id, sorted_shape_indices, cell_groups, element_texts)
    if container:
        final_aligned.append(container)
    
    # Build individual shapes
    for shape_idx in sorted_shape_indices:
        words = cell_groups[shape_idx]
        shape_elem = build_single_shape_element(elem_id, shape_idx, words, element_shape_data)
        if shape_elem:
            final_aligned.append(shape_elem)
            
            # Add to parent's children list
            for parent in final_aligned:
                if parent.get('element_id') == elem_id and parent.get('is_shape_container'):
                    parent['children'].append(f"{elem_id}_shape_{shape_idx}")
                    break
    
    return final_aligned
