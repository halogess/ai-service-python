import json
import os

def reassign_unaligned_tokens(doc_id):
    """
    Membaca alignment_difflib_xxx.json dan melakukan reassignment unaligned tokens
    ke elemen terdekat berdasarkan Y-position, lalu menyimpan ke alignment_final_xxx.json
    """
    try:
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, 'assets', str(doc_id))
        difflib_path = os.path.join(assets_dir, f'alignment_difflib_{doc_id}.json')
        final_path = os.path.join(assets_dir, f'alignment_final_{doc_id}.json')
        
        if not os.path.exists(difflib_path):
            return {"error": f"File {difflib_path} tidak ditemukan. Jalankan Align Difflib terlebih dahulu."}
        
        # Load alignment difflib
        with open(difflib_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check if data has success key
        if isinstance(data, dict) and 'success' in data:
            data = data.get('data', {})
        
        aligned_words = data.get('aligned_words', [])
        unaligned_tokens = data.get('unaligned_tokens', [])
        
        initial_unaligned_count = len(unaligned_tokens)
        reassigned_count = 0
        
        # Assign unaligned tokens to nearest element by Y position
        token_to_elem = {}  # Map token to element for text reconstruction
        
        for token in unaligned_tokens[:]:
            token_page = token['page']
            token_bbox = token['bbox']
            
            # Find elements on same page
            candidates = [w for w in aligned_words if w['page'] == token_page]
            if not candidates:
                continue
            
            # Find element with Y overlap (with tolerance) and closest X distance
            best_elem = None
            best_score = float('inf')
            
            for elem in candidates:
                elem_bbox = elem['bbox']
                
                # Calculate Y overlap with tolerance
                token_y_center = (token_bbox['y0'] + token_bbox['y1']) / 2
                elem_y_center = (elem_bbox['y0'] + elem_bbox['y1']) / 2
                
                # Check if Y centers are close (within element height)
                elem_height = elem_bbox['y1'] - elem_bbox['y0']
                y_distance = abs(token_y_center - elem_y_center)
                
                # Y overlap check: token Y range overlaps with element Y range OR close enough
                y_overlap = not (token_bbox['y1'] < elem_bbox['y0'] or token_bbox['y0'] > elem_bbox['y1'])
                y_close = y_distance < elem_height * 1.5  # Within 1.5x element height
                
                if y_overlap or y_close:
                    # Calculate X distance
                    if token_bbox['x1'] < elem_bbox['x0']:
                        x_distance = elem_bbox['x0'] - token_bbox['x1']
                    elif token_bbox['x0'] > elem_bbox['x1']:
                        x_distance = token_bbox['x0'] - elem_bbox['x1']
                    else:
                        x_distance = 0
                    
                    # Score: prioritize Y proximity, then X distance
                    score = y_distance * 10 + x_distance
                    
                    if score < best_score:
                        best_score = score
                        best_elem = elem
            
            if best_elem:
                # Add token to element's before_align_bboxes
                if 'before_align_bboxes' not in best_elem:
                    best_elem['before_align_bboxes'] = []
                best_elem['before_align_bboxes'].append(token['bbox'])
                
                # Store token info for text reconstruction
                elem_id = id(best_elem)
                if elem_id not in token_to_elem:
                    token_to_elem[elem_id] = []
                token_to_elem[elem_id].append({
                    'text': token['text'],
                    'x0': token['bbox']['x0'],
                    'y0': token['bbox']['y0']
                })
                
                # Expand element bbox to include token
                best_elem['bbox']['x0'] = min(best_elem['bbox']['x0'], token['bbox']['x0'])
                best_elem['bbox']['y0'] = min(best_elem['bbox']['y0'], token['bbox']['y0'])
                best_elem['bbox']['x1'] = max(best_elem['bbox']['x1'], token['bbox']['x1'])
                best_elem['bbox']['y1'] = max(best_elem['bbox']['y1'], token['bbox']['y1'])
                
                # Remove from unaligned
                unaligned_tokens.remove(token)
                reassigned_count += 1
        
        # Rebuild text for each element based on X position
        for elem in aligned_words:
            elem_id = id(elem)
            
            # Get all bboxes with their original text
            all_tokens = []
            
            # Add original tokens from before_align_bboxes
            if 'before_align_bboxes' in elem:
                # Try to extract original tokens from matched_text or text
                original_text = elem.get('matched_text', elem.get('text', ''))
                original_tokens = original_text.split()
                
                for i, bbox in enumerate(elem['before_align_bboxes']):
                    token_text = original_tokens[i] if i < len(original_tokens) else ''
                    all_tokens.append({
                        'text': token_text,
                        'x0': bbox['x0'],
                        'y0': bbox['y0']
                    })
            
            # Add reassigned tokens
            if elem_id in token_to_elem:
                for token_info in token_to_elem[elem_id]:
                    all_tokens.append(token_info)
            
            # Sort by X position and rebuild text
            if all_tokens:
                all_tokens.sort(key=lambda t: (t.get('y0', 0), t['x0']))
                elem['text'] = ' '.join([t['text'] for t in all_tokens if t['text']])
        
        # Save to alignment_final
        final_data = {
            "success": True,
            "data": {
                "aligned_words": aligned_words,
                "unaligned_elements": data.get('unaligned_elements', []),
                "unaligned_tokens": unaligned_tokens,
                "stats": {
                    "total_words": len(aligned_words),
                    "assigned_words": len(aligned_words),
                    "unaligned_count": len(data.get('unaligned_elements', [])),
                    "unaligned_tokens_count": len(unaligned_tokens),
                    "unaligned_tokens_reassigned": reassigned_count,
                    "coverage": 1.0,
                }
            }
        }
        
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "initial_unaligned": initial_unaligned_count,
            "reassigned": reassigned_count,
            "remaining_unaligned": len(unaligned_tokens),
            "output_file": final_path
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
