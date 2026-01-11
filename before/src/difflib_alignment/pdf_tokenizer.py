"""PDF tokenization utilities"""

import fitz
from .pdf_extractor import iter_pdf_tokens_with_bboxes


def build_pdf_tokens(pdf_path):
    """Build global PDF token stream"""
    pdf_tokens = []
    pdf_bboxes = []
    pdf_pages = []

    with fitz.open(pdf_path) as pdf:
        for page_index in range(pdf.page_count):
            page = pdf[page_index]
            for tok, bbox, _ in iter_pdf_tokens_with_bboxes(page, page_index):
                pdf_tokens.append(tok)
                pdf_bboxes.append(bbox)
                pdf_pages.append(page_index)

    return pdf_tokens, pdf_bboxes, pdf_pages
