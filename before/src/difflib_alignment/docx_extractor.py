"""DOCX content extraction utilities"""

from .docx_table_extractor import extract_table_cells
from .docx_shape_extractor import extract_shapes
from .docx_text_extractor import extract_text_recursive


def extract_text_from_json_tree(json_tree, return_cells=False, return_shapes=False, 
                                  return_image_only_cells=False, return_empty_cells=False,
                                  return_table_structure=False):
    """Ekstrak teks dari dokumen_elemen_json_tree (OpenXML)"""
    if not json_tree:
        if return_image_only_cells or return_empty_cells:
            return ([], [], [], {'row_count': 0, 'col_count': 0})
        return "" if not (return_cells or return_shapes) else []

    # Unwrap content wrapper
    if isinstance(json_tree, dict) and "content" in json_tree and isinstance(json_tree["content"], dict):
        json_tree = json_tree["content"]
    
    # Cek shapes
    if return_shapes:
        shapes = extract_shapes(json_tree)
        if shapes:
            return shapes
    
    # Cek table
    if isinstance(json_tree, dict) and "rows" in json_tree:
        cells, image_only_cells, empty_cells, table_structure = extract_table_cells(json_tree)
        
        if return_image_only_cells or return_empty_cells or return_table_structure:
            return (cells, image_only_cells, empty_cells, table_structure)
        if return_cells:
            return cells
        # For plain text extraction, join cell text values
        return " ".join(content_item['value'] for cell_tuple in cells 
                        for content_item in cell_tuple[1]  # cell_tuple is (idx, content_list, row, col)
                        if content_item.get('type') == 'text')

    return extract_text_recursive(json_tree)
