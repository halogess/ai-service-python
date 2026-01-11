"""Core alignment logic using difflib"""

import difflib
from collections import defaultdict


def perform_global_alignment(docx_tokens, pdf_tokens):
    """Perform global difflib alignment"""
    sm = difflib.SequenceMatcher(a=docx_tokens, b=pdf_tokens, autojunk=False)
    opcodes = sm.get_opcodes()

    docx_to_pdf = [None] * len(docx_tokens)
    docx_to_pdf_multi = {}
    last_pdf_idx = -1

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(min(i2 - i1, j2 - j1)):
                pdf_idx = j1 + k
                if pdf_idx > last_pdf_idx:
                    docx_to_pdf[i1 + k] = pdf_idx
                    last_pdf_idx = pdf_idx
        elif tag == "replace" and i2 - i1 == 1 and j2 - j1 > 1:
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


def group_aligned_tokens(docx_owner, docx_cell_index, docx_is_formula, docx_to_pdf, docx_to_pdf_multi, 
                         pdf_tokens, pdf_bboxes, pdf_pages):
    """Group aligned tokens by element"""
    element_groups = defaultdict(lambda: defaultdict(list))

    for i, elem_id in enumerate(docx_owner):
        j = docx_to_pdf[i]
        if j is None:
            continue

        cell_idx = docx_cell_index[i]
        is_formula = docx_is_formula[i]
        
        if i in docx_to_pdf_multi:
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
            bbox = pdf_bboxes[j]
            element_groups[elem_id][cell_idx].append({
                "text": pdf_tokens[j],
                "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                "page": pdf_pages[j],
                "is_formula": is_formula,
                "pdf_index": j,
                "docx_token_index": i,
            })

    return element_groups
