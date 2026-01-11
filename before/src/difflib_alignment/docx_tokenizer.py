"""DOCX tokenization utilities"""

from .text_utils import tokenize
from .docx_extractor import extract_text_from_json_tree
from .formula_utils import has_formula_in_tree


def build_docx_tokens(elements):
    """Build global DOCX token stream"""
    docx_tokens = []
    docx_owner = []
    docx_cell_index = []
    docx_is_formula = []
    image_only_cells_map = {}
    empty_cells_map = {}
    table_structure_map = {}
    
    for elem in elements:
        # Handle both dict and object access
        if isinstance(elem, dict):
            json_tree = elem.get('delemen_json_tree', {})
            elem_id = elem.get('delemen_id')
        else:
            json_tree = getattr(elem, 'delemen_json_tree', {})
            elem_id = getattr(elem, 'delemen_id', None)

        elem_text = extract_text_from_json_tree(json_tree)
        
        cells_result = extract_text_from_json_tree(
            json_tree, 
            return_cells=True, 
            return_image_only_cells=True,
            return_empty_cells=True,
            return_table_structure=True
        )
        
        # New format: (cells, image_only_cells, empty_cells, table_structure)
        if isinstance(cells_result, tuple) and len(cells_result) == 4:
            cells, image_only_cells, empty_cells, table_structure = cells_result
        elif isinstance(cells_result, tuple) and len(cells_result) == 2:
            # Legacy fallback
            cells, image_only_cells = cells_result
            empty_cells = []
            table_structure = {}
        else:
            cells = cells_result if isinstance(cells_result, list) else []
            image_only_cells = set()
            empty_cells = []
            table_structure = {}
        
        is_table = isinstance(cells, list) and len(cells) > 0
        
        if image_only_cells:
            image_only_cells_map[elem_id] = image_only_cells
        
        if empty_cells:
            empty_cells_map[elem_id] = empty_cells
        
        if table_structure:
            table_structure_map[elem_id] = table_structure
        
        shapes = extract_text_from_json_tree(json_tree, return_shapes=True)
        is_shape_array = isinstance(shapes, list) and len(shapes) > 0
        
        has_formula = has_formula_in_tree(json_tree)
        
        if is_table:
            for cell_tuple in cells:
                # New format: (cell_idx, cell_content, row_idx, col_idx)
                if len(cell_tuple) == 4:
                    cell_idx, cell_content, row_idx, col_idx = cell_tuple
                else:
                    # Legacy format (cell_idx, cell_content)
                    cell_idx, cell_content = cell_tuple[:2]
                
                # cell_content is list of dicts (text/image items)
                # Shapes are already expanded to separate cells by docx_table_extractor
                if isinstance(cell_content, list):
                    cell_text = " ".join(item['value'] for item in cell_content if item.get('type') == 'text')
                    if cell_text.strip():
                        toks = tokenize(cell_text, is_formula=has_formula)
                        docx_tokens.extend(toks)
                        docx_owner.extend([elem_id] * len(toks))
                        docx_cell_index.extend([cell_idx] * len(toks))
                        docx_is_formula.extend([has_formula] * len(toks))
                else:
                    cell_text = cell_content  # Fallback
                    if cell_text and cell_text.strip():
                        toks = tokenize(cell_text, is_formula=has_formula)
                        docx_tokens.extend(toks)
                        docx_owner.extend([elem_id] * len(toks))
                        docx_cell_index.extend([cell_idx] * len(toks))
                        docx_is_formula.extend([has_formula] * len(toks))

        elif is_shape_array:
            for shape_idx, shape_text, _ in shapes:
                toks = tokenize(shape_text, is_formula=has_formula)
                docx_tokens.extend(toks)
                docx_owner.extend([elem_id] * len(toks))
                docx_cell_index.extend([shape_idx] * len(toks))
                docx_is_formula.extend([has_formula] * len(toks))
        else:
            toks = tokenize(elem_text, is_formula=has_formula)
            docx_tokens.extend(toks)
            docx_owner.extend([elem_id] * len(toks))
            docx_cell_index.extend([-1] * len(toks))
            docx_is_formula.extend([has_formula] * len(toks))

    return (docx_tokens, docx_owner, docx_cell_index, docx_is_formula, 
            image_only_cells_map, empty_cells_map, table_structure_map)

