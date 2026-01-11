"""Image alignment workflow"""

from .docx_extractor import extract_text_from_json_tree


def collect_image_items(elements):
    """Collect all image items dari elements"""
    image_items = []
    table_cell_images = []
    
    for elem in elements:
        json_tree = elem.delemen_json_tree
        if not json_tree or not isinstance(json_tree, dict):
            continue
        
        # Images in content array
        content = json_tree.get('content', [])
        if isinstance(content, list):
            image_index = 0
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get('type')
                    if item_type in ('image', 'drawing'):
                        rId = item.get('rId') or item.get('value')
                        if rId:
                            image_items.append((elem.delemen_id, rId, elem.delemen_sequence, image_index))
                            image_index += 1
                    elif item_type == 'shape' and 'content' in item:
                        for shape_item in item['content']:
                            if isinstance(shape_item, dict):
                                shape_type = shape_item.get('type')
                                if shape_type in ('image', 'drawing'):
                                    rId = shape_item.get('rId') or shape_item.get('value')
                                    if rId:
                                        image_items.append((elem.delemen_id, rId, elem.delemen_sequence, image_index))
                                        image_index += 1
        
        # Images in table cells
        if isinstance(json_tree.get('content'), dict) and 'rows' in json_tree.get('content'):
            rows = json_tree['content']['rows']
            cell_index = 0
            for row in rows:
                if isinstance(row, dict):
                    for cell in row.get('cells', []):
                        if isinstance(cell, list):
                            for item in cell:
                                if isinstance(item, dict):
                                    item_type = item.get('type')
                                    if item_type in ('image', 'drawing'):
                                        rId = item.get('rId') or item.get('value')
                                        if rId:
                                            table_cell_images.append((elem.delemen_id, cell_index, rId))
                        elif isinstance(cell, dict):
                            # Handle dict cell format
                            def extract_images_from_node(node):
                                if isinstance(node, dict):
                                    if node.get('type') in ('image', 'drawing'):
                                        rId = node.get('rId') or node.get('value')
                                        if rId:
                                            table_cell_images.append((elem.delemen_id, cell_index, rId))
                                    for v in node.values():
                                        extract_images_from_node(v)
                                elif isinstance(node, list):
                                    for item in node:
                                        extract_images_from_node(item)
                            extract_images_from_node(cell)
                        cell_index += 1
    
    return image_items, table_cell_images


def align_standalone_images(image_items, final_aligned, pdf_images, used_images, elements):
    """Align standalone images (non-table)"""
    aligned_count = 0
    
    for elem_id, rId, sequence, img_idx in image_items:
        already_aligned = any(
            a['element_id'] == elem_id and a.get('rId') == rId and a.get('image_index') == img_idx 
            for a in final_aligned
        )
        if already_aligned:
            continue
        
        # Find prev/next element pages
        elem_idx = next((i for i, e in enumerate(elements) if e.delemen_id == elem_id), None)
        target_page = None
        
        if elem_idx is not None:
            # Get next element page
            if elem_idx < len(elements) - 1:
                next_elem_id = elements[elem_idx + 1].delemen_id
                for aligned in final_aligned:
                    if aligned.get('element_id') == next_elem_id or str(aligned.get('element_id')).startswith(f"{next_elem_id}_"):
                        target_page = aligned['page']
                        break
            
            # Fallback to prev element
            if target_page is None and elem_idx > 0:
                prev_elem_id = elements[elem_idx - 1].delemen_id
                for aligned in final_aligned:
                    if aligned.get('element_id') == prev_elem_id or str(aligned.get('element_id')).startswith(f"{prev_elem_id}_"):
                        target_page = aligned['page']
                        break
        
        if target_page is None:
            target_page = 0
        
        # Find best matching image
        best_image = None
        best_score = 0
        
        for page_idx in [target_page] + [p for p in range(len(pdf_images)) if abs(p - target_page) <= 2 and p != target_page]:
            if page_idx not in pdf_images:
                continue
            
            for img in pdf_images[page_idx]:
                img_key = (page_idx, img['xref'])
                if img_key in used_images:
                    continue
                
                score = 100 if page_idx == target_page else max(0, 50 - abs(page_idx - target_page) * 10)
                
                if score > best_score:
                    best_score = score
                    best_image = (page_idx, img)
        
        if best_image:
            page_idx, img = best_image
            bbox = img['bbox']
            # Handle both list and dict bbox formats
            if isinstance(bbox, list):
                bbox_dict = {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]}
            else:
                bbox_dict = bbox
            
            used_images.add((page_idx, img['xref']))
            aligned_count += 1
            
            final_aligned.append({
                "text": "[IMAGE]",
                "matched_text": "[IMAGE]",
                "bbox": bbox_dict,
                "bboxes": [{"page": page_idx, "bbox": bbox_dict}],
                "page": page_idx,
                "element_id": elem_id,
                "rId": rId,
                "image_index": img_idx,
                "confidence": 0.8,
                "before_align_bboxes": [bbox_dict],
                "is_image": True,
            })
    
    return aligned_count
