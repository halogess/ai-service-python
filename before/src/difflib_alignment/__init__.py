"""difflib_alignment - Modular alignment system for OpenXML and PDF"""

from .main import align_document
from .pdf_extractor import iter_pdf_tokens_with_bboxes

__all__ = ['align_document', 'iter_pdf_tokens_with_bboxes']
