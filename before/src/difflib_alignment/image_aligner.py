"""Image alignment utilities"""


def align_table_cell_images(final_aligned, table_cell_images, pdf_images, used_images, log_file=None):
    """Align images dalam table cells
    
    Strategy: Group cells by table element, then match images to cells based on Y-order.
    Images in the same column should be assigned in vertical order.
    """
    aligned_count = 0
    
    # Group cells with images by table element
    table_cells = {}  # {table_id: [(cell_idx, rId, cell_entry), ...]}
    
    for elem_id, cell_idx, rId in table_cell_images:
        cell_elem_id = f"{elem_id}_cell_{cell_idx}"
        
        # Skip if already aligned
        already_aligned = any(a.get('element_id') == cell_elem_id and a.get('has_image_bbox') for a in final_aligned)
        if already_aligned:
            continue
        
        cell_entry = next((a for a in final_aligned if a.get('element_id') == cell_elem_id), None)
        if not cell_entry:
            if log_file:
                log_file.write(f"WARNING: Cell {cell_elem_id} not found for image rId={rId}\n")
            continue
        
        if elem_id not in table_cells:
            table_cells[elem_id] = []
        table_cells[elem_id].append((cell_idx, rId, cell_entry))
    
    # Process each table
    for table_id, cells in table_cells.items():
        # Sort cells by cell_idx to maintain document order
        cells_sorted = sorted(cells, key=lambda x: x[0])
        
        # Find pages where this table's cells appear
        cell_pages = set(c[2]['page'] for c in cells_sorted)
        
        # Collect candidate images from these pages and nearby
        all_pages = set()
        for p in cell_pages:
            all_pages.update([p - 1, p, p + 1])
        
        candidate_images = []
        for page_idx in sorted(all_pages):
            if page_idx in pdf_images:
                for img in pdf_images[page_idx]:
                    img_key = (page_idx, img['xref'])
                    if img_key not in used_images:
                        candidate_images.append({
                            'page': page_idx,
                            'img': img,
                            'key': img_key,
                            'y0': img['bbox'][1],
                            'x_center': (img['bbox'][0] + img['bbox'][2]) / 2
                        })
        
        # Sort candidate images by page, then by Y position
        candidate_images.sort(key=lambda x: (x['page'], x['y0']))
        
        if log_file:
            log_file.write(f"\nTable {table_id}: {len(cells_sorted)} cells with images, {len(candidate_images)} candidate images\n")
        
        # Match cells to images in order
        img_idx = 0
        for cell_idx, rId, cell_entry in cells_sorted:
            if img_idx >= len(candidate_images):
                if log_file:
                    log_file.write(f"  WARNING: No image available for cell_{cell_idx} rId={rId}\n")
                continue
            
            # Find best matching image from remaining candidates
            best_img = None
            best_score = -1
            best_idx = -1
            
            cell_page = cell_entry['page']
            
            for j in range(img_idx, min(img_idx + 3, len(candidate_images))):  # Look at next 3 images max
                cand = candidate_images[j]
                score = 0
                
                # Prefer same page
                if cand['page'] == cell_page:
                    score += 100
                elif abs(cand['page'] - cell_page) == 1:
                    score += 50
                
                # Prefer similar X position (same column)
                cell_x = (cell_entry['bbox']['x0'] + cell_entry['bbox']['x1']) / 2
                x_diff = abs(cand['x_center'] - cell_x)
                if x_diff < 100:
                    score += 50 - x_diff / 2
                
                if score > best_score:
                    best_score = score
                    best_img = cand
                    best_idx = j
            
            if best_img:
                img = best_img['img']
                page_idx = best_img['page']
                img_bbox = img['bbox']
                
                used_images.add(best_img['key'])
                aligned_count += 1
                
                # Remove from candidates
                candidate_images.pop(best_idx)
                
                cell_entry['bboxes'].append({
                    'page': page_idx,
                    'bbox': {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
                })
                
                if cell_entry.get('is_image_only_cell') and cell_entry.get('confidence', 1.0) == 0.0:
                    cell_entry['bbox'] = {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
                    cell_entry['page'] = page_idx
                    cell_entry['confidence'] = 1.0
                else:
                    if page_idx == cell_entry['page']:
                        # Only expand bbox if on same page
                        cell_entry['bbox']['x0'] = min(cell_entry['bbox']['x0'], img_bbox[0])
                        cell_entry['bbox']['y0'] = min(cell_entry['bbox']['y0'], img_bbox[1])
                        cell_entry['bbox']['x1'] = max(cell_entry['bbox']['x1'], img_bbox[2])
                        cell_entry['bbox']['y1'] = max(cell_entry['bbox']['y1'], img_bbox[3])
                
                cell_entry['has_image_bbox'] = True
                
                if cell_entry.get('content_items'):
                    for i, item in enumerate(cell_entry['content_items']):
                        if item.get('type') == 'image' and cell_entry['content_bboxes'][i] is None:
                            cell_entry['content_bboxes'][i] = {
                                'page': page_idx,
                                'bbox': {'x0': img_bbox[0], 'y0': img_bbox[1], 'x1': img_bbox[2], 'y1': img_bbox[3]}
                            }
                            break
                
                if log_file:
                    log_file.write(f"  cell_{cell_idx} rId={rId} -> page {page_idx + 1} y0={img_bbox[1]:.1f}\n")
            else:
                if log_file:
                    log_file.write(f"  WARNING: No matching image for cell_{cell_idx} rId={rId}\n")
    
    return aligned_count

