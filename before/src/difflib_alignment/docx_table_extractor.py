"""Table extraction from DOCX JSON tree"""


def extract_table_cells(json_tree):
    """Extract table cells from JSON tree
    
    Returns:
        tuple: (cells, image_only_cells, empty_cells, table_structure)
            cells: list of (cell_index, cell_content_list, row_idx, col_idx) tuples
            image_only_cells: set of cell indices that contain only images
            empty_cells: list of (cell_index, row_idx, col_idx) tuples for empty cells
            table_structure: dict with row_count, col_count info
    """
    if not isinstance(json_tree, dict) or "rows" not in json_tree:
        return ([], set(), [], {'row_count': 0, 'col_count': 0})
    
    cells = []
    image_only_cells = set()
    empty_cells = []
    cell_index = 0
    
    rows = json_tree.get("rows", [])
    row_count = len(rows)
    col_count = 0
    
    for row_idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        
        row_cells = row.get("cells", [])
        col_count = max(col_count, len(row_cells))
        
        for col_idx, cell in enumerate(row_cells):
            content_list = []
            has_image = False
            is_empty = False
            
            if isinstance(cell, str):
                if cell.strip():
                    content_list = [{'type': 'text', 'value': cell}]
                else:
                    is_empty = True
                
            elif isinstance(cell, dict):
                content_list = _extract_cell_content(cell)
                has_image = _has_image(cell)
                if not content_list:
                    is_empty = True
                
            elif isinstance(cell, list):
                # Empty cell
                if not cell:
                    is_empty = True
                else:
                    # Check for shapes to enable granular splitting
                    has_shape = any(isinstance(item, dict) and item.get("type") == "shape" for item in cell)
                    
                    if has_shape:
                        # MONOLITH BEHAVIOR: Each shape becomes its OWN cell entry
                        for item in cell:
                            if isinstance(item, dict):
                                if item.get("type") == "shape":
                                    # Extract content from shape
                                    item_content = _extract_cell_content(item)
                                    if item_content:
                                        # Each shape is a SEPARATE cell with its own index
                                        cells.append((cell_index, item_content, row_idx, col_idx))
                                    cell_index += 1  # Increment for EACH shape
                                elif item.get("type") == "image":
                                    image_only_cells.add(cell_index)
                                    cell_index += 1
                                elif item.get("type") == "text":
                                    # Text item in mixed cell
                                    text_content = [item]
                                    cells.append((cell_index, text_content, row_idx, col_idx))
                                    cell_index += 1
                        # Skip the normal cell_index increment at end
                        continue
                    else:
                        # Regular cell (mixed text/images) -> Extract granular content
                        content_list = _extract_cell_content(cell)
                        has_image = _has_image(cell)
                    
                    if not content_list:
                        is_empty = True

            
            # Handle results
            if is_empty:
                empty_cells.append((cell_index, row_idx, col_idx))
            elif content_list:
                cells.append((cell_index, content_list, row_idx, col_idx))
                
                # Check if it has image but no text
                has_text = any(c['type'] == 'text' for c in content_list)
                if has_image and not has_text:
                    image_only_cells.add(cell_index)
            
            cell_index += 1
    
    table_structure = {
        'row_count': row_count,
        'col_count': col_count,
        'total_cells': cell_index
    }
    
    return (cells, image_only_cells, empty_cells, table_structure)


def _extract_cell_content(cell_node):
    """Recursively extract content items from cell node
    
    Returns:
        list of dict: [{'type': 'text', 'value': '...'}, {'type': 'image', ...}]
    """
    if not cell_node:
        return []
    
    items = []
    
    def rec(node):
        if isinstance(node, dict):
            node_type = node.get("type")
            
            # Text node
            if node_type == "text" and "value" in node:
                val = node["value"].strip()
                if val:
                    items.append({'type': 'text', 'value': val})
            elif node_type == "math" and "text" in node:
                val = node["text"].strip()
                if val:
                    items.append({'type': 'text', 'value': val})
            elif node_type == "image":
                items.append({'type': 'image'})
            
            # Recurse through children - BUT exclude metadata keys like 'id' and 'name'
            # These contain TextBox IDs that should NOT be part of content text
            for k, v in node.items():
                if k not in ("type", "value", "text", "rId", "id", "name"):
                    rec(v)
        elif isinstance(node, list):
            for item in node:
                rec(item)
        elif isinstance(node, str):
            val = node.strip()
            if val:
                items.append({'type': 'text', 'value': val})
    
    rec(cell_node)
    return items


def _has_image(cell_node):
    """Check if cell contains image"""
    if not cell_node:
        return False
    
    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "image":
                return True
            for v in node.values():
                if check(v):
                    return True
        elif isinstance(node, list):
            for item in node:
                if check(item):
                    return True
        return False
    
    return check(cell_node)
