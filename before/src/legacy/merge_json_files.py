"""Merge alignment JSON dengan docling JSON"""

import json
import os


def merge_from_json_files(reassign2_file, docling_json_path, output_path):
    """
    Final merge: Rebuild text from reassign2 result.
    
    Args:
        reassign2_file: Path ke reassign2_merged_alignment_xxx.json
        docling_json_path: Path ke testing_prediction_xxx.json (docling)
        output_path: Path output merged JSON
    
    Returns:
        List of merged results
    """
    # Load reassign2 result
    with open(reassign2_file, 'r', encoding='utf-8') as f:
        reassign2_data = json.load(f)
    
    merged_results = reassign2_data.get('merged_results', [])
    unaligned_tokens = reassign2_data.get('unaligned_tokens', [])
    # Rebuild text for elements with reassigned tokens using REAL bboxes
    for merged in merged_results:
        if 'reassigned_tokens' in merged and merged['reassigned_tokens']:
            all_tokens = []
            
            # Use real bboxes from words[] array if available
            if 'words' in merged and merged['words']:
                for word in merged['words']:
                    bbox = word.get('bbox', word)
                    if isinstance(bbox, dict):
                        all_tokens.append({
                            'text': word.get('text', ''),
                            'x0': bbox.get('x0', 0),
                            'y0': bbox.get('y0', 0)
                        })
            else:
                # Fallback: split text with estimated positions from element bbox
                tokens = merged['text'].split()
                elem_width = merged['bbox']['x1'] - merged['bbox']['x0']
                token_width = elem_width / max(len(tokens), 1)
                for i, t in enumerate(tokens):
                    all_tokens.append({
                        'text': t,
                        'x0': merged['bbox']['x0'] + i * token_width,
                        'y0': merged['bbox']['y0']
                    })
            
            # Add reassigned tokens with their real positions
            for rt in merged['reassigned_tokens']:
                all_tokens.append({
                    'text': rt['text'],
                    'x0': rt['x0'],
                    'y0': rt['y0']
                })
            
            # Sort by Y first (line), then by X (reading order)
            all_tokens.sort(key=lambda t: (round(t['y0'] / 5) * 5, t['x0']))
            
            # Rebuild text in correct order
            merged['text'] = ' '.join([t['text'] for t in all_tokens if t['text']])

    
    # Get stats from reassign2
    stats = reassign2_data.get('stats', {})
    
    # Save final merged result (with text rebuild)
    output_data = {
        'merged_results': merged_results,
        'unaligned_tokens': unaligned_tokens,
        'stats': stats
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Final merged saved to: {output_path}")
    
    return merged_results


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union"""
    if isinstance(bbox1, dict):
        b1 = [bbox1['x0'], bbox1['y0'], bbox1['x1'], bbox1['y1']]
    else:
        b1 = bbox1
    
    if isinstance(bbox2, dict):
        b2 = [bbox2['x0'], bbox2['y0'], bbox2['x1'], bbox2['y1']]
    else:
        b2 = bbox2
    
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def merge_step1(alignment_json_path, docling_json_path, output_path):
    """Step 1: Merge elements with same Y + reassign unaligned tokens"""
    with open(alignment_json_path, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)
    
    aligned_words = alignment_data.get('data', {}).get('aligned_words', [])
    unaligned_tokens = alignment_data.get('data', {}).get('unaligned_tokens', [])
    
    by_page = {}
    for aligned in aligned_words:
        page = aligned['page']
        if page not in by_page:
            by_page[page] = []
        by_page[page].append(aligned)
    
    merged_results = []
    
    for page, elements in by_page.items():
        elements.sort(key=lambda e: (e['bbox']['y0'] + e['bbox']['y1']) / 2)
        
        i = 0
        while i < len(elements):
            current = elements[i]
            current_y = (current['bbox']['y0'] + current['bbox']['y1']) / 2
            
            group = [current]
            j = i + 1
            while j < len(elements):
                next_elem = elements[j]
                next_y = (next_elem['bbox']['y0'] + next_elem['bbox']['y1']) / 2
                if abs(next_y - current_y) < 5:
                    group.append(next_elem)
                    j += 1
                else:
                    break
            
            if len(group) == 1:
                elem = group[0]
            else:
                group.sort(key=lambda e: e['bbox']['x0'])
                elem = group[0].copy()
                elem['text'] = ' '.join([e['text'] for e in group])
                elem['bbox'] = {
                    'x0': min(e['bbox']['x0'] for e in group),
                    'y0': min(e['bbox']['y0'] for e in group),
                    'x1': max(e['bbox']['x1'] for e in group),
                    'y1': max(e['bbox']['y1'] for e in group)
                }
                # Merge words from all group elements with real bboxes
                merged_words = []
                for e in group:
                    if 'words' in e:
                        merged_words.extend(e['words'])
                if merged_words:
                    elem['words'] = merged_words
            
            # Build result with words array containing real bboxes
            result_elem = {
                'element_id': elem['element_id'],
                'text': elem['text'],
                'bbox': elem['bbox'],
                'page': page,
                'confidence': elem.get('confidence', 1.0)
            }
            
            # Include words with real bboxes for later text rebuild
            if 'words' in elem:
                result_elem['words'] = elem['words']
            
            merged_results.append(result_elem)
            i = j

    
    # REASSIGN UNALIGNED TOKENS - Now assign to merged_results instead of aligned_words
    reassigned = 0
    for token in unaligned_tokens[:]:
        token_page = token['page']
        token_bbox = token['bbox']
        
        # Get all candidates from MERGED_RESULTS (not aligned_words)
        candidates = []
        for elem in merged_results:
            if elem['page'] == token_page:
                candidates.append(elem)
        
        if not candidates:
            continue
        
        # Find best match: prioritize element with Y overlap and closest distance
        best = None
        best_score = 0
        
        for cand in candidates:
            y_overlap = not (token_bbox['y1'] < cand['bbox']['y0'] or token_bbox['y0'] > cand['bbox']['y1'])
            
            if y_overlap:
                y_overlap_size = min(token_bbox['y1'], cand['bbox']['y1']) - max(token_bbox['y0'], cand['bbox']['y0'])
                
                if y_overlap_size > best_score:
                    best_score = y_overlap_size
                    best = cand
        
        if best:
            if 'reassigned_tokens' not in best:
                best['reassigned_tokens'] = []
            best['reassigned_tokens'].append({'text': token['text'], 'x0': token_bbox['x0'], 'y0': token_bbox['y0'], 'x1': token_bbox['x1'], 'y1': token_bbox['y1']})

            best['bbox']['x0'] = min(best['bbox']['x0'], token_bbox['x0'])
            best['bbox']['y0'] = min(best['bbox']['y0'], token_bbox['y0'])
            best['bbox']['x1'] = max(best['bbox']['x1'], token_bbox['x1'])
            best['bbox']['y1'] = max(best['bbox']['y1'], token_bbox['y1'])
            
            # Update parent bboxes (only on same page for split tables)
            parent_id = best.get('parent_element_id')
            token_page = token['page']
            while parent_id:
                # Find parent on same page (handle split tables)
                parent = next((e for e in aligned_words 
                             if (e['element_id'] == parent_id or e['element_id'] == f"{parent_id}_page_{token_page}") 
                             and e['page'] == token_page), None)
                if parent:
                    parent['bbox']['x0'] = min(parent['bbox']['x0'], token_bbox['x0'])
                    parent['bbox']['y0'] = min(parent['bbox']['y0'], token_bbox['y0'])
                    parent['bbox']['x1'] = max(parent['bbox']['x1'], token_bbox['x1'])
                    parent['bbox']['y1'] = max(parent['bbox']['y1'], token_bbox['y1'])
                    parent_id = parent.get('parent_element_id')
                else:
                    break
            
            unaligned_tokens.remove(token)
            reassigned += 1
    
    # Rebuild text with reassigned tokens using REAL bboxes
    for elem in merged_results:
        if 'reassigned_tokens' in elem and elem['reassigned_tokens']:
            all_tokens = []
            
            # Use real bboxes from words[] array if available
            if 'words' in elem and elem['words']:
                for word in elem['words']:
                    bbox = word.get('bbox', word)
                    if isinstance(bbox, dict):
                        all_tokens.append({
                            'text': word.get('text', ''),
                            'x0': bbox.get('x0', 0),
                            'y0': bbox.get('y0', 0)
                        })
            else:
                # Fallback: split text (no real position available)
                tokens = elem['text'].split()
                # Use element bbox to estimate positions
                elem_width = elem['bbox']['x1'] - elem['bbox']['x0']
                token_width = elem_width / max(len(tokens), 1)
                for i, t in enumerate(tokens):
                    all_tokens.append({
                        'text': t,
                        'x0': elem['bbox']['x0'] + i * token_width,
                        'y0': elem['bbox']['y0']
                    })
            
            # Add reassigned tokens with their real positions
            for rt in elem['reassigned_tokens']:
                all_tokens.append({
                    'text': rt['text'],
                    'x0': rt['x0'],
                    'y0': rt['y0']
                })
            
            # Sort by Y first (line), then by X (reading order)
            all_tokens.sort(key=lambda t: (round(t['y0'] / 5) * 5, t['x0']))
            
            # Rebuild text in correct order
            elem['text'] = ' '.join([t['text'] for t in all_tokens if t['text']])
            
            # Update words array to include reassigned tokens
            if 'words' not in elem:
                elem['words'] = []
            for rt in elem['reassigned_tokens']:
                elem['words'].append({
                    'text': rt['text'],
                    'bbox': {'x0': rt['x0'], 'y0': rt['y0'], 'x1': rt.get('x1', rt['x0']+20), 'y1': rt.get('y1', rt['y0']+10)},
                    'reassigned': True
                })

    
    output_data = {
        'merged_results': merged_results,
        'aligned_words': aligned_words,
        'unaligned_tokens': unaligned_tokens,
        'stats': {
            'total_merged': len(merged_results),
            'reassigned_tokens': reassigned,
            'unaligned_tokens': len(unaligned_tokens)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data['stats']


def merge_step2(reassign1_file, docling_file, output_path):
    """Step 2: Reassign unaligned tokens via docling overlap + assign labels
    
    Input: reassign1_merged_alignment_xxx.json + docling
    Output: reassign2_merged_alignment_xxx.json
    """
    # Load reassign1 result (contains merged_results and unaligned_tokens)
    with open(reassign1_file, 'r', encoding='utf-8') as f:
        reassign1_data = json.load(f)
    
    merged_results = reassign1_data.get('merged_results', [])
    unaligned_tokens = reassign1_data.get('unaligned_tokens', [])
    
    # Load docling
    with open(docling_file, 'r', encoding='utf-8') as f:
        docling_raw = json.load(f)
    
    # Parse docling
    docling_data = []
    page_heights = {}
    
    pdf_path = os.path.dirname(os.path.dirname(docling_file))
    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    if pdf_files:
        import fitz
        pdf_doc = fitz.open(os.path.join(pdf_path, pdf_files[0]))
        for i in range(len(pdf_doc)):
            page_heights[i] = pdf_doc[i].rect.height
        pdf_doc.close()
    
    if isinstance(docling_raw, dict) and ('texts' in docling_raw or 'tables' in docling_raw):
        for text in docling_raw.get('texts', []):
            if isinstance(text, dict) and 'prov' in text:
                prov_list = text['prov'] if isinstance(text['prov'], list) else [text['prov']]
                for prov in prov_list:
                    if isinstance(prov, dict) and 'bbox' in prov:
                        page_num = prov.get('page_no', 1) - 1
                        bbox = prov['bbox']
                        page_height = page_heights.get(page_num, 842)
                        y0 = page_height - bbox.get('b', 0)
                        y1 = page_height - bbox.get('t', 0)
                        docling_data.append({
                            'page': page_num,
                            'bbox_x0': bbox.get('l', 0),
                            'bbox_y0': min(y0, y1),
                            'bbox_x1': bbox.get('r', 0),
                            'bbox_y1': max(y0, y1),
                            'label': str(text.get('label', 'text')).split('.')[-1].lower()
                        })
        
        for table in docling_raw.get('tables', []):
            if isinstance(table, dict) and 'prov' in table:
                prov_list = table['prov'] if isinstance(table['prov'], list) else [table['prov']]
                for prov in prov_list:
                    if isinstance(prov, dict) and 'bbox' in prov:
                        page_num = prov.get('page_no', 1) - 1
                        bbox = prov['bbox']
                        page_height = page_heights.get(page_num, 842)
                        y0 = page_height - bbox.get('b', 0)
                        y1 = page_height - bbox.get('t', 0)
                        docling_data.append({
                            'page': page_num,
                            'bbox_x0': bbox.get('l', 0),
                            'bbox_y0': min(y0, y1),
                            'bbox_x1': bbox.get('r', 0),
                            'bbox_y1': max(y0, y1),
                            'label': 'table'
                        })
    
    docling_by_page = {}
    for item in docling_data:
        page = item.get('page', 0)
        if page not in docling_by_page:
            docling_by_page[page] = []
        docling_by_page[page].append(item)
    
    # First: Assign labels to merged elements
    match_count = 0
    for merged in merged_results:
        page = merged['page']
        elem_bbox = merged['bbox']
        best_label = None
        best_iou = 0.0
        
        if page in docling_by_page:
            for doc_item in docling_by_page[page]:
                doc_bbox = {
                    'x0': doc_item['bbox_x0'],
                    'y0': doc_item['bbox_y0'],
                    'x1': doc_item['bbox_x1'],
                    'y1': doc_item['bbox_y1']
                }
                iou = calculate_iou(elem_bbox, doc_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_label = doc_item.get('label')
        
        if best_label and best_iou > 0.01:
            merged['docling_label'] = best_label
            merged['iou'] = best_iou
            match_count += 1
    
    # Second: Reassign unaligned tokens
    reassigned_count = 0
    for token in unaligned_tokens[:]:
        token_page = token['page']
        token_bbox = token['bbox']
        
        if token_page not in docling_by_page:
            continue
        
        overlapping_docling = None
        for doc_item in docling_by_page[token_page]:
            doc_bbox = {
                'x0': doc_item['bbox_x0'],
                'y0': doc_item['bbox_y0'],
                'x1': doc_item['bbox_x1'],
                'y1': doc_item['bbox_y1']
            }
            
            if calculate_iou(token_bbox, doc_bbox) > 0:
                overlapping_docling = doc_bbox
                break
        
        if overlapping_docling:
            best_merged = None
            best_iou = 0
            
            for merged in merged_results:
                if merged['page'] == token_page:
                    iou = calculate_iou(merged['bbox'], overlapping_docling)
                    if iou > best_iou:
                        best_iou = iou
                        best_merged = merged
            
            if best_merged and best_iou > 0.01:
                if 'reassigned_tokens' not in best_merged:
                    best_merged['reassigned_tokens'] = []
                best_merged['reassigned_tokens'].append({
                    'text': token['text'],
                    'x0': token_bbox['x0'],
                    'y0': token_bbox['y0']
                })
                
                best_merged['bbox']['x0'] = min(best_merged['bbox']['x0'], token_bbox['x0'])
                best_merged['bbox']['y0'] = min(best_merged['bbox']['y0'], token_bbox['y0'])
                best_merged['bbox']['x1'] = max(best_merged['bbox']['x1'], token_bbox['x1'])
                best_merged['bbox']['y1'] = max(best_merged['bbox']['y1'], token_bbox['y1'])
                
                unaligned_tokens.remove(token)
                reassigned_count += 1
    
    output_data = {
        'merged_results': merged_results,
        'unaligned_tokens': unaligned_tokens,
        'stats': {
            'total_merged': len(merged_results),
            'matched_labels': match_count,
            'reassigned_tokens': reassigned_count,
            'remaining_unaligned': len(unaligned_tokens)
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data['stats']
