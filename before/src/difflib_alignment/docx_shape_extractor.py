"""DOCX shape extraction utilities"""


def extract_shapes(json_tree):
    """Extract shapes dari JSON tree"""
    if not isinstance(json_tree, dict) or "content" not in json_tree:
        return []
    
    content = json_tree["content"]
    if not isinstance(content, list):
        return []
    
    has_shapes = any(isinstance(item, dict) and item.get('type') == 'shape' for item in content)
    if not has_shapes:
        return []
    
    shapes = []
    shape_index = 0
    
    for item in content:
        if not isinstance(item, dict) or item.get('type') != 'shape':
            continue
        
        if "content" in item and isinstance(item["content"], list):
            sub_idx = 0
            has_sub_splits = False
            for sub_item in item["content"]:
                if isinstance(sub_item, dict):
                    from .docx_extractor import extract_text_from_json_tree
                    sub_text = extract_text_from_json_tree(sub_item, return_cells=False, return_shapes=False)
                    if sub_text and sub_text.strip():
                        shapes.append((f"{shape_index}_{sub_idx}", sub_text, item))
                        sub_idx += 1
                        has_sub_splits = True
            
            if not has_sub_splits:
                from .docx_extractor import extract_text_from_json_tree
                shape_text = extract_text_from_json_tree(item, return_cells=False, return_shapes=False)
                if shape_text and shape_text.strip():
                    shapes.append((shape_index, shape_text, item))
        else:
            from .docx_extractor import extract_text_from_json_tree
            shape_text = extract_text_from_json_tree(item, return_cells=False, return_shapes=False)
            if shape_text and shape_text.strip():
                shapes.append((shape_index, shape_text, item))
        
        shape_index += 1
    
    return shapes
