"""Fallback alignment for unaligned table cells (tables with out-of-order content)"""

import difflib
import fitz
from .text_utils import tokenize
from .table_structure_cache import get_table_structure_from_pdf


def align_unaligned_table_cells(element_groups, element_is_table, element_cell_texts,
                                 pdf_tokens, pdf_bboxes, pdf_pages, used_pdf_indices, 
                                 pdf_path=None, log_file=None):
    """Independent alignment untuk table cells yang tidak ter-align atau partial aligned
    
    This handles cases where:
    1. Table content appears in different order in DOCX vs PDF
    2. Bullet tokens in PDF disrupt alignment sequence
    3. Cells have partial coverage (some tokens aligned, some not)
    4. Entire tables have no aligned cells (e.g., continuation tables)
    
    NEW: Uses PyMuPDF cell bbox as primary constraint to prevent matching
    tokens from wrong areas (e.g., formula section below table).
    """
    
    # Pre-compute page for each element for fallback
    element_page_hints = {}
    sorted_elem_ids = sorted(element_groups.keys())
    
    last_known_page = 0
    for elem_id in sorted_elem_ids:
        elem_data = element_groups.get(elem_id, {})
        # Find any page from this element's cells
        for cell_idx, words in elem_data.items():
            if words:
                last_known_page = words[0]['page']
                break
        element_page_hints[elem_id] = last_known_page
    
    # Cache PyMuPDF table structures per page
    pymupdf_cache = {}
    
    def get_pymupdf_table_for_page(page_num):
        """Get cached PyMuPDF table structure for page"""
        if pdf_path is None:
            return None
        if page_num not in pymupdf_cache:
            pymupdf_cache[page_num] = get_table_structure_from_pdf(pdf_path, page_num)
        return pymupdf_cache[page_num]
    
    def get_cell_bbox_from_pymupdf(page_num, cell_idx, num_cols=9):
        """Get cell bbox from PyMuPDF table structure.
        
        Args:
            page_num: PDF page number (0-indexed)
            cell_idx: DOCX cell index
            num_cols: Estimated number of columns in table
            
        Returns:
            dict {'x0': ..., 'y0': ..., 'x1': ..., 'y1': ...} or None if not found
        """
        tables = get_pymupdf_table_for_page(page_num)
        if not tables:
            return None
        
        # Estimate row and column index from cell_idx
        row_idx = cell_idx // num_cols if num_cols > 0 else cell_idx
        col_idx = cell_idx % num_cols if num_cols > 0 else 0
        
        for table in tables:
            cells_by_pos = table.get('cells_by_pos', {})
            
            # Try exact cell match first
            exact_bbox = cells_by_pos.get((row_idx, col_idx))
            if exact_bbox:
                return {'x0': exact_bbox[0], 'y0': exact_bbox[1], 
                        'x1': exact_bbox[2], 'y1': exact_bbox[3]}
            
            # If exact cell not found, try to get row bbox (all cells in row)
            row_cells = [(r, c, bbox) for (r, c), bbox in cells_by_pos.items() if r == row_idx]
            
            if row_cells:
                # Get bbox covering entire row
                x0 = min(bbox[0] for _, _, bbox in row_cells)
                y0 = min(bbox[1] for _, _, bbox in row_cells)
                x1 = max(bbox[2] for _, _, bbox in row_cells)
                y1 = max(bbox[3] for _, _, bbox in row_cells)
                return {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
            
            # If row not found, try to estimate from table bbox
            table_bbox = table.get('bbox')
            if table_bbox:
                row_count = table.get('row_count', 10)
                col_count = table.get('col_count', num_cols)
                
                table_width = table_bbox[2] - table_bbox[0]
                table_height = table_bbox[3] - table_bbox[1]
                
                row_height = table_height / row_count if row_count > 0 else 30
                col_width = table_width / col_count if col_count > 0 else 50
                
                # Estimate cell position
                est_x0 = table_bbox[0] + (col_idx * col_width)
                est_y0 = table_bbox[1] + (row_idx * row_height)
                est_x1 = est_x0 + col_width
                est_y1 = est_y0 + row_height
                
                # Clamp to table bounds and add margin
                if est_y0 < table_bbox[3]:
                    return {
                        'x0': max(table_bbox[0], est_x0 - 5),
                        'y0': max(table_bbox[1], est_y0 - 5),
                        'x1': min(table_bbox[2], est_x1 + 5),
                        'y1': min(table_bbox[3], est_y1 + 5)
                    }
        
        return None
    
    for elem_id, is_table in element_is_table.items():
        if not is_table:
            continue
        
        cell_texts = element_cell_texts.get(elem_id, {})
        
        # Get page hint for this table from element ordering
        table_page_hint = element_page_hints.get(elem_id, None)
        
        # Check each cell
        for cell_idx, cell_value in cell_texts.items():
            # Handle structured content (list of items) vs legacy string
            if isinstance(cell_value, list):
                cell_text = " ".join(item['value'] for item in cell_value if item.get('type') == 'text')
            else:
                cell_text = str(cell_value)

            # Skip empty cells
            if not cell_text or not cell_text.strip():
                continue
            
            cell_tokens = tokenize(cell_text)
            if not cell_tokens:
                continue
            
            existing_words = element_groups[elem_id].get(cell_idx, [])
            
            # Calculate current coverage
            aligned_token_count = len(existing_words)
            expected_token_count = len(cell_tokens)
            coverage = aligned_token_count / expected_token_count if expected_token_count > 0 else 0
            
            # If already well-aligned (>80% coverage), skip
            if coverage > 0.8:
                continue
            
            # Get spatial hints from existing aligned tokens
            cell_bbox_hint = None
            target_page = None
            if existing_words:
                target_page = existing_words[0]['page']
                # Use existing aligned tokens to estimate cell bbox
                x0 = min(w['bbox']['x0'] for w in existing_words)
                y0 = min(w['bbox']['y0'] for w in existing_words)
                x1 = max(w['bbox']['x1'] for w in existing_words)
                y1 = max(w['bbox']['y1'] for w in existing_words)
                cell_bbox_hint = {'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1}
            
            # Get target pages from other cells if no hint
            if target_page is None:
                target_pages = set()
                for other_idx, words in element_groups[elem_id].items():
                    if words:
                        target_pages.add(words[0]['page'])
                if target_pages:
                    target_page = min(target_pages)  # Start with earliest page
            
            # Use table page hint from element ordering if still no hint
            if target_page is None and table_page_hint is not None:
                target_page = table_page_hint
                if log_file:
                    log_file.write(f"Using element order hint page {target_page+1} for {elem_id}_{cell_idx}\\n")
            
            # Collect candidate PDF tokens (unused)
            candidate_indices = [i for i in range(len(pdf_tokens)) if i not in used_pdf_indices]
            
            # Filter by page - expand search to nearby pages
            if target_page is not None:
                expanded_pages = set([max(0, target_page-1), target_page, target_page+1])
                candidate_indices = [i for i in candidate_indices if pdf_pages[i] in expanded_pages]
            
            # *** PyMuPDF cell bbox constraint with cascading fallback ***
            # Priority: 1) X+Y from PyMuPDF, 2) Y-only from PyMuPDF, 3) existing bbox hint
            pymupdf_bbox = None
            if target_page is not None and isinstance(cell_idx, int):
                pymupdf_bbox = get_cell_bbox_from_pymupdf(target_page, cell_idx)
            
            constraint_applied = False
            
            if pymupdf_bbox and candidate_indices:
                margin = 15  # Allow 15px margin
                
                # Try 1: X+Y constraint (strictest)
                xy_filtered = [i for i in candidate_indices 
                               if (pymupdf_bbox['x0'] - margin <= pdf_bboxes[i][0] <= pymupdf_bbox['x1'] + margin
                                   and pymupdf_bbox['y0'] - margin <= pdf_bboxes[i][1] <= pymupdf_bbox['y1'] + margin)]
                
                if xy_filtered:
                    candidate_indices = xy_filtered
                    constraint_applied = True
                    if log_file:
                        log_file.write(f"PyMuPDF X+Y: {elem_id}_{cell_idx} -> bbox [{pymupdf_bbox['x0']:.0f},{pymupdf_bbox['y0']:.0f},{pymupdf_bbox['x1']:.0f},{pymupdf_bbox['y1']:.0f}], {len(xy_filtered)} candidates\\n")
                else:
                    # Try 2: Y-only constraint (fallback - for tables with no column borders)
                    y_filtered = [i for i in candidate_indices 
                                  if pymupdf_bbox['y0'] - margin <= pdf_bboxes[i][1] <= pymupdf_bbox['y1'] + margin]
                    
                    if y_filtered:
                        candidate_indices = y_filtered
                        constraint_applied = True
                        if log_file:
                            log_file.write(f"PyMuPDF Y-only: {elem_id}_{cell_idx} -> Y [{pymupdf_bbox['y0']:.0f},{pymupdf_bbox['y1']:.0f}], {len(y_filtered)} candidates\\n")
            
            # Try 3: FALLBACK - Use existing aligned tokens bbox hint
            if not constraint_applied and cell_bbox_hint and candidate_indices:
                # Relax Y range for wrapped cells
                y_extension = (len(cell_text) / 50) * 15
                min_y = cell_bbox_hint['y0'] - 10
                max_y = cell_bbox_hint['y1'] + y_extension + 20
                
                y_filtered = [i for i in candidate_indices 
                              if min_y <= pdf_bboxes[i][1] <= max_y]
                if y_filtered:
                    candidate_indices = y_filtered
            
            if not candidate_indices:
                continue

            # Find best sequence match - use lower threshold for partial recovery
            matched_indices, score = find_best_sequence_match(cell_tokens, candidate_indices, pdf_tokens, threshold=0.3)
            
            # If sequence matching failed, try individual token matching (for scattered content)
            if not matched_indices or score < 0.3:
                matched_indices = match_individual_tokens(cell_tokens, candidate_indices, pdf_tokens, pdf_bboxes)
                score = len(matched_indices) / len(cell_tokens) if cell_tokens else 0
            
            if matched_indices and score > 0.2:  # Lower threshold for recovery
                if log_file:
                    log_file.write(f"Fallback Align Cell {elem_id}_{cell_idx}: +{len(matched_indices)} tokens (score {score:.2f}, was {coverage:.0%})\\n")
                
                # Add new matched tokens to element_groups
                for pdf_idx in matched_indices:
                    # Skip if already used (might have been added elsewhere)
                    if pdf_idx in used_pdf_indices:
                        continue
                    
                    bbox = pdf_bboxes[pdf_idx]
                    element_groups[elem_id][cell_idx].append({
                        "text": pdf_tokens[pdf_idx],
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "page": pdf_pages[pdf_idx],
                        "is_formula": False,
                        "pdf_index": pdf_idx,
                        "docx_token_index": -1,  # Fallback alignment
                    })
                    used_pdf_indices.add(pdf_idx)


def match_individual_tokens(target_tokens, candidate_indices, pdf_tokens, pdf_bboxes):
    """Match individual tokens when sequence matching fails (for scattered content with bullets)
    
    This greedily matches each target token to the best candidate, prioritizing
    tokens that appear in reading order (top to bottom, left to right).
    """
    if not target_tokens or not candidate_indices:
        return []
    
    # Build lookup for candidate tokens
    candidate_lookup = {}  # normalized_text -> list of indices
    for idx in candidate_indices:
        text = pdf_tokens[idx].lower().strip()
        if text not in candidate_lookup:
            candidate_lookup[text] = []
        candidate_lookup[text].append(idx)
    
    matched = []
    used_candidates = set()
    
    for target_token in target_tokens:
        target_lower = target_token.lower().strip()
        
        # Exact match
        if target_lower in candidate_lookup:
            candidates = [c for c in candidate_lookup[target_lower] if c not in used_candidates]
            if candidates:
                # Pick the one closest to reading order (smallest Y, then smallest X)
                best_candidate = min(candidates, key=lambda c: (pdf_bboxes[c][1], pdf_bboxes[c][0]))
                matched.append(best_candidate)
                used_candidates.add(best_candidate)
                continue
        
        # Partial match (for truncated tokens)
        for candidate_text, indices in candidate_lookup.items():
            if target_lower in candidate_text or candidate_text in target_lower:
                candidates = [c for c in indices if c not in used_candidates]
                if candidates:
                    best_candidate = min(candidates, key=lambda c: (pdf_bboxes[c][1], pdf_bboxes[c][0]))
                    matched.append(best_candidate)
                    used_candidates.add(best_candidate)
                    break
    
    return matched


def find_best_sequence_match(target_tokens, candidate_indices, pdf_tokens, threshold=0.5):
    """Find best matching sequence in candidates
    
    Args:
        target_tokens: List of tokens to match
        candidate_indices: List of PDF token indices to search in
        pdf_tokens: Full list of PDF tokens
        threshold: Minimum score threshold (default 0.5)
    """
    if not target_tokens or not candidate_indices:
        return None, 0
    
    # Group candidates into contiguous segments
    segments = []
    if not candidate_indices:
        return None, 0
    
    # Sort candidates for proper segment grouping
    sorted_candidates = sorted(candidate_indices)
    
    current_seg = [sorted_candidates[0]]
    for i in range(1, len(sorted_candidates)):
        # Allow small gaps (1-2 tokens) to bridge over bullets
        if sorted_candidates[i] <= sorted_candidates[i-1] + 3:
            current_seg.append(sorted_candidates[i])
        else:
            # Accept smaller segments for better coverage
            if len(current_seg) >= min(2, len(target_tokens) * 0.2):
                segments.append(current_seg)
            current_seg = [sorted_candidates[i]]
    
    # Always add the last segment
    if current_seg and len(current_seg) >= min(2, len(target_tokens) * 0.2):
        segments.append(current_seg)
    
    best_match = None
    best_score = 0.0
    
    for seg in segments:
        seg_tokens = [pdf_tokens[i] for i in seg]
        sm_local = difflib.SequenceMatcher(None, target_tokens, seg_tokens, autojunk=False)
        
        # Try to get all matching blocks
        all_matched = []
        for block in sm_local.get_matching_blocks():
            if block.size > 0:
                for k in range(block.size):
                    all_matched.append(seg[block.b + k])
        
        if all_matched:
            score = len(all_matched) / len(target_tokens)
            if score > best_score and score > threshold:
                best_score = score
                best_match = all_matched
    
    return best_match, best_score

