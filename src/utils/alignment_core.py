"""
Core alignment logic using difflib
Migrated from before/alignment_core.py
"""

import difflib
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


def perform_global_alignment(docx_tokens: List[str], pdf_tokens: List[str]) -> Tuple[List[Optional[int]], Dict[int, List[int]]]:
    """
    Perform global difflib alignment between DOCX and PDF tokens.
    
    Args:
        docx_tokens: List of tokens from DOCX
        pdf_tokens: List of tokens from PDF
        
    Returns:
        Tuple of:
        - docx_to_pdf: List mapping each DOCX token index to PDF token index (or None)
        - docx_to_pdf_multi: Dict mapping DOCX token index to multiple PDF indices (for split tokens)
    """
    sm = difflib.SequenceMatcher(a=docx_tokens, b=pdf_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    docx_to_pdf = [None] * len(docx_tokens)
    docx_to_pdf_multi = {}
    last_pdf_idx = -1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # Direct match - map each DOCX token to corresponding PDF token
            for k in range(min(i2 - i1, j2 - j1)):
                pdf_idx = j1 + k
                if pdf_idx > last_pdf_idx:
                    docx_to_pdf[i1 + k] = pdf_idx
                    last_pdf_idx = pdf_idx
                    
        elif tag == "replace" and i2 - i1 == 1 and j2 - j1 > 1:
            # One DOCX token matches multiple PDF tokens (e.g., hyphenated words)
            docx_token = docx_tokens[i1]
            pdf_segment = pdf_tokens[j1:j2]
            
            combined = "".join(pdf_segment)
            if combined == docx_token or combined.replace("-", "") == docx_token:
                pdf_indices = list(range(j1, j2))
                if all(idx > last_pdf_idx for idx in pdf_indices):
                    docx_to_pdf[i1] = j1
                    docx_to_pdf_multi[i1] = pdf_indices
                    last_pdf_idx = j2 - 1

    return docx_to_pdf, docx_to_pdf_multi


def group_aligned_tokens(
    docx_owner: List[int],
    docx_cell_index: List[Optional[int]],
    docx_is_formula: List[bool],
    docx_to_pdf: List[Optional[int]],
    docx_to_pdf_multi: Dict[int, List[int]],
    pdf_tokens: List[str],
    pdf_bboxes: List[List[float]],
    pdf_pages: List[int]
) -> Dict[int, Dict[Optional[int], List[dict]]]:
    """
    Group aligned tokens by element ID.
    
    Args:
        docx_owner: List of element IDs that own each DOCX token
        docx_cell_index: List of cell indices for table cells (None for non-table)
        docx_is_formula: List of boolean flags for formula tokens
        docx_to_pdf: Mapping from DOCX token index to PDF token index
        docx_to_pdf_multi: Mapping for multi-token matches
        pdf_tokens: List of PDF tokens
        pdf_bboxes: List of PDF bounding boxes [x0, y0, x1, y1]
        pdf_pages: List of page numbers for each PDF token
        
    Returns:
        Dict[element_id][cell_index] = List of matched token info
    """
    element_groups = defaultdict(lambda: defaultdict(list))

    for i, elem_id in enumerate(docx_owner):
        j = docx_to_pdf[i]
        if j is None:
            continue

        cell_idx = docx_cell_index[i]
        is_formula = docx_is_formula[i]
        
        if i in docx_to_pdf_multi:
            # Multiple PDF tokens for one DOCX token
            for pdf_idx in docx_to_pdf_multi[i]:
                bbox = pdf_bboxes[pdf_idx]
                element_groups[elem_id][cell_idx].append({
                    "text": pdf_tokens[pdf_idx],
                    "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                    "page": pdf_pages[pdf_idx],
                    "is_formula": is_formula,
                    "pdf_index": pdf_idx,
                    "docx_token_index": i,
                })
        else:
            # Single PDF token match
            bbox = pdf_bboxes[j]
            element_groups[elem_id][cell_idx].append({
                "text": pdf_tokens[j],
                "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                "page": pdf_pages[j],
                "is_formula": is_formula,
                "pdf_index": j,
                "docx_token_index": i,
            })

    return dict(element_groups)


def merge_bboxes(bboxes: List[dict]) -> dict:
    """
    Merge multiple bounding boxes into one encompassing bbox.
    
    Args:
        bboxes: List of bbox dicts with x0, y0, x1, y1 keys
        
    Returns:
        Merged bbox dict
    """
    if not bboxes:
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}
    
    return {
        "x0": min(b["x0"] for b in bboxes),
        "y0": min(b["y0"] for b in bboxes),
        "x1": max(b["x1"] for b in bboxes),
        "y1": max(b["y1"] for b in bboxes),
    }


def calculate_alignment_score(docx_tokens: List[str], pdf_tokens: List[str]) -> float:
    """
    Calculate alignment score between two token lists.
    
    Args:
        docx_tokens: DOCX tokens
        pdf_tokens: PDF tokens
        
    Returns:
        Score between 0.0 and 1.0
    """
    if not docx_tokens or not pdf_tokens:
        return 0.0
    
    sm = difflib.SequenceMatcher(a=docx_tokens, b=pdf_tokens, autojunk=False)
    return sm.ratio()
