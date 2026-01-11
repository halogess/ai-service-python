"""Fallback alignment for unaligned shapes"""

import difflib
from .text_utils import tokenize


def align_unaligned_shapes(element_groups, element_is_shape_array, element_shape_data, 
                           pdf_tokens, pdf_bboxes, pdf_pages, used_pdf_indices):
    """Independent alignment untuk shapes yang tidak ter-align"""
    
    for elem_id, is_shape_array in element_is_shape_array.items():
        if not is_shape_array:
            continue
            
        shape_data = element_shape_data.get(elem_id, {})
        for shape_idx, (shape_text, shape_item) in shape_data.items():
            # Check if already aligned
            if element_groups[elem_id][shape_idx]:
                continue
            
            # Try independent alignment
            shape_tokens = tokenize(shape_text)
            if not shape_tokens:
                continue
            
            # Get target pages from other shapes
            target_pages = set()
            for other_idx, words in element_groups[elem_id].items():
                if words:
                    target_pages.add(words[0]['page'])
            
            # Collect candidate PDF tokens (unused)
            candidate_indices = [i for i in range(len(pdf_tokens)) if i not in used_pdf_indices]
            
            # Filter by page if we have hints
            if target_pages:
                expanded_pages = set()
                for p in target_pages:
                    expanded_pages.update([p-1, p, p+1])
                candidate_indices = [i for i in candidate_indices if pdf_pages[i] in expanded_pages]
            
            if not candidate_indices:
                continue

            # Find best sequence match
            matched_indices, score = find_best_sequence_match(shape_tokens, candidate_indices, pdf_tokens)
            
            if matched_indices and score > 0.6:
                # Add to element_groups
                for pdf_idx in matched_indices:
                    bbox = pdf_bboxes[pdf_idx]
                    element_groups[elem_id][shape_idx].append({
                        "text": pdf_tokens[pdf_idx],
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "page": pdf_pages[pdf_idx],
                        "is_formula": False,
                        "pdf_index": pdf_idx,
                        "docx_token_index": -1,
                    })
                    used_pdf_indices.add(pdf_idx)


def find_best_sequence_match(target_tokens, candidate_indices, pdf_tokens):
    """Find best matching sequence in candidates"""
    if not target_tokens or not candidate_indices:
        return None, 0
    
    # Group candidates into contiguous segments
    segments = []
    if not candidate_indices:
        return None, 0
        
    current_seg = [candidate_indices[0]]
    for i in range(1, len(candidate_indices)):
        if candidate_indices[i] == candidate_indices[i-1] + 1:
            current_seg.append(candidate_indices[i])
        else:
            if len(current_seg) >= len(target_tokens) * 0.5:
                segments.append(current_seg)
            current_seg = [candidate_indices[i]]
    segments.append(current_seg)
    
    best_match = None
    best_score = 0.0
    
    for seg in segments:
        seg_tokens = [pdf_tokens[i] for i in seg]
        sm_local = difflib.SequenceMatcher(None, target_tokens, seg_tokens, autojunk=False)
        match = sm_local.find_longest_match(0, len(target_tokens), 0, len(seg_tokens))
        
        if match.size > 0:
            score = match.size / len(target_tokens)
            if score > best_score and score > 0.6:
                best_score = score
                matched_indices = seg[match.b : match.b + match.size]
                best_match = matched_indices
    
    return best_match, best_score
