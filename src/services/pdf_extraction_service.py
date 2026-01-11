"""
PDF Extraction Service
Handles PDF processing using PyMuPDF (fitz) - extracts text, images, and char groups
"""

import fitz
import os
import logging
from typing import Optional
from utils.char_grouping import collect_all_chars, find_overlapping_groups

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Service class for extracting content from PDF files"""
    
    def __init__(self, pdf_path: str):
        """
        Initialize PDF extractor.
        
        Args:
            pdf_path: Full path to the PDF file
        """
        self.pdf_path = pdf_path
        self.doc = None
        
    def __enter__(self):
        """Context manager entry - open the PDF."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close the PDF."""
        self.close()
        
    def open(self):
        """Open the PDF document."""
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF file not found: {self.pdf_path}")
        self.doc = fitz.open(self.pdf_path)
        logger.info(f"Opened PDF: {self.pdf_path} ({self.doc.page_count} pages)")
        
    def close(self):
        """Close the PDF document."""
        if self.doc:
            self.doc.close()
            self.doc = None
            
    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        return self.doc.page_count if self.doc else 0
    
    def get_page(self, page_num: int):
        """
        Get a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            PyMuPDF page object
        """
        if not self.doc:
            raise RuntimeError("PDF not opened")
        if page_num < 0 or page_num >= self.doc.page_count:
            raise ValueError(f"Invalid page number: {page_num}")
        return self.doc[page_num]
    
    def extract_char_groups(self, page_num: int) -> dict:
        """
        Extract character groups from a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            dict with:
            - 'page': Page number
            - 'groups': List of character groups
            - 'page_width': Page width
            - 'page_height': Page height
        """
        page = self.get_page(page_num)
        rawdict = page.get_text("rawdict")
        
        # Use char_grouping functions
        chars = collect_all_chars(rawdict)
        groups = find_overlapping_groups(chars)
        
        return {
            "page": page_num,
            "groups": groups,
            "page_width": rawdict.get("width", 0),
            "page_height": rawdict.get("height", 0),
        }

    
    def extract_all_char_groups(self) -> list:
        """
        Extract character groups from all pages.
        
        Returns:
            List of page results, each containing:
            - 'page': Page number
            - 'groups': List of character groups
            - 'page_width': Page width  
            - 'page_height': Page height
        """
        results = []
        for page_num in range(self.page_count):
            result = self.extract_char_groups(page_num)
            results.append(result)
            logger.debug(f"Page {page_num + 1}: {len(result['groups'])} groups")
        logger.info(f"Extracted char groups from {len(results)} pages")
        return results
    
    def extract_text(self, page_num: int, method: str = "text") -> str:
        """
        Extract text from a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            method: Extraction method ("text", "blocks", "dict", "rawdict")
            
        Returns:
            Extracted text or data
        """
        page = self.get_page(page_num)
        return page.get_text(method)
    
    def extract_images(self, page_num: int) -> list:
        """
        Extract images from a specific page.
        
        Args:
            page_num: Page number (0-indexed)
            
        Returns:
            List of image info dicts
        """
        page = self.get_page(page_num)
        return page.get_image_info()
    
    def render_page_to_image(self, page_num: int, output_path: str, dpi: int = 150) -> str:
        """
        Render a page to an image file.
        
        Args:
            page_num: Page number (0-indexed)
            output_path: Output image path
            dpi: Resolution in DPI
            
        Returns:
            Output path
        """
        page = self.get_page(page_num)
        zoom = dpi / 72  # 72 is default DPI
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pix.save(output_path)
        logger.debug(f"Rendered page {page_num + 1} to {output_path}")
        return output_path


def extract_pdf_char_groups(pdf_path: str) -> list:
    """
    Convenience function to extract all char groups from a PDF.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of page results with char groups
    """
    with PDFExtractor(pdf_path) as extractor:
        return extractor.extract_all_char_groups()
