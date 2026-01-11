"""Table cell matching utilities"""

import re


def match_docx_cell_to_pdf_cell(docx_text, pdf_table_cells, pdf_cell_texts, exclude_indices=None):
    """Match DOCX cell text ke PDF cell terbaik"""
    if not docx_text or not pdf_cell_texts:
        return None
    
    if exclude_indices is None:
        exclude_indices = set()
    
    def normalize(s):
        s = s.strip().lower()
        s = re.sub(r'\s+', ' ', s)
        s = re.sub(r'[^\w\s]', '', s)
        return s
    
    docx_norm = normalize(docx_text)
    best_idx = None
    best_score = 0
    
    for i, pdf_text in enumerate(pdf_cell_texts):
        if i in exclude_indices:
            continue
            
        if not pdf_text:
            continue

        # Height heuristic: reject giant merged cells
        cell_bbox = pdf_table_cells[i]
        cell_h = cell_bbox[3] - cell_bbox[1]
        
        est_lines = max(1, len(pdf_text) / 50)
        est_h = est_lines * 30
        
        if cell_h > 200 and cell_h > (est_h * 5):
            continue

        pdf_norm = normalize(pdf_text)
        
        # Exact match
        if docx_norm == pdf_norm:
            return i
        
        # Substring match
        if docx_norm in pdf_norm or pdf_norm in docx_norm:
            score = min(len(docx_norm), len(pdf_norm)) / max(len(docx_norm), len(pdf_norm), 1)
            if score > best_score:
                best_score = score
                best_idx = i
                continue
        
        # Token overlap
        docx_tokens = set(docx_norm.split())
        pdf_tokens = set(pdf_norm.split())
        
        if docx_tokens and pdf_tokens:
            overlap = len(docx_tokens & pdf_tokens)
            total = len(docx_tokens | pdf_tokens)
            token_score = overlap / total if total > 0 else 0
            
            if token_score > 0.6 and token_score > best_score:
                best_score = token_score
                best_idx = i
    
    if best_score > 0.4:
        return best_idx
    
    return None
