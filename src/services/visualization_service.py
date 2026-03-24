"""
Visualization Service
Renders PDF pages with alignment and fusion bboxes overlayed.
Ported from classification.html and canvas_renderer.js
"""

import os
import fitz  # PyMuPDF
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import io


# Label colors for Docling/Classification (matches LABEL_COLORS from classification.html)
LABEL_COLORS = {
    'title': '#e74c3c',
    'text': '#3498db',
    'paragraph': '#3498db',
    'list_item': '#2ecc71',
    'table': '#f39c12',
    'picture': '#9b59b6',
    'caption': '#1abc9c',
    'section_header': '#e74c3c',
    'page_header': '#e67e22',
    'page_footer': '#c0392b',
    'footnote': '#8b4513',
    'formula': '#008080',
    'code': '#00CED1',
    'unknown': '#95a5a6'
}

# Docling-specific colors (slightly different shade, cyan theme)
DOCLING_LABEL_COLORS = {
    'title': '#006064',
    'text': '#0097a7',
    'paragraph': '#0097a7',
    'list_item': '#00838f',
    'table': '#00acc1',
    'picture': '#26c6da',
    'caption': '#4dd0e1',
    'section_header': '#006064',
    'page_header': '#00838f',
    'page_footer': '#006064',
    'footnote': '#004d40',
    'formula': '#00695c',
    'code': '#00796b',
    'unknown': '#607d8b'
}

# Alignment colors
ALIGNMENT_COLORS = {
    'default': '#4caf50',  # Green
    'table': '#e74c3c',    # Red
    'cell': '#c0392b',     # Darker red
    'image': '#27ae60',    # Bright green
    'text_part': '#3498db', # Blue
    'unaligned': '#e74c3c', # Red
    'duplicate_mapping': '#00ff00' # Green for duplicate OpenXML mapping
}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def hex_to_rgba(hex_color: str, alpha: int = 255) -> Tuple[int, int, int, int]:
    """Convert hex color to RGBA tuple."""
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, alpha)


from services.visualization_draw_mixin import VisualizationDrawMixin


class VisualizationService(VisualizationDrawMixin):


    """
    Service for generating visualization images of alignment and Docling fusion results.
    """

    def __init__(self, output_dir: str = 'visualization_output'):
        """
        Initialize visualization service.
        
        Args:
            output_dir: Directory to save visualization images
        """
        self.output_dir = output_dir
        self.scale = 1.5  # PDF to image scale factor (matches frontend)
        
        # Try to load a font, fallback to default if not available
        self.font = None
        self.font_small = None
        try:
            # Try common font paths
            font_paths = [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
            ]
            for path in font_paths:
                if os.path.exists(path):
                    self.font = ImageFont.truetype(path, 12)
                    self.font_small = ImageFont.truetype(path, 10)
                    break
        except:
            pass

    def _ensure_output_dir(self, subdir: str = None) -> str:
        """Ensure output directory exists and return path."""
        path = self.output_dir
        if subdir:
            path = os.path.join(path, subdir)
        os.makedirs(path, exist_ok=True)
        return path

    def render_pdf_page(self, pdf_path: str, page_num: int) -> Image.Image:
        """
        Render a PDF page to an image.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 0-based page number
            
        Returns:
            PIL Image of the rendered page
        """
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        
        # Render at scale factor (1.5 = 150% = ~108 DPI)
        mat = fitz.Matrix(self.scale, self.scale)
        pix = page.get_pixmap(matrix=mat)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        
        doc.close()
        return img

    def visualize_page(
        self,
        pdf_path: str,
        page_num: int,
        alignments: List[Dict] = None,
        fused_results: List[Dict] = None,
        header_footer_units: List[Dict] = None,
        unaligned_pdf_units: List[Dict] = None,
        duplicate_mapping_units: List[Dict] = None,
        save_separate: bool = True,
        doc_id: int = None,
        output_dir_override: str = None
    ) -> Dict[str, str]:
        """
        Generate visualization images for a page.
        
        Args:
            pdf_path: Path to PDF file
            page_num: 0-based page number
            alignments: Alignment results (not drawn as PDF units)
            fused_results: Docling fusion results
            header_footer_units: Header/footer PDF units (not drawn as PDF units)
            unaligned_pdf_units: PDF units that remain unaligned after final pass
            duplicate_mapping_units: PDF units aligned to OpenXML elements that also appear on other pages
            save_separate: Save alignment and fusion as separate images
            doc_id: Document ID for output folder naming
            output_dir_override: If provided, save directly to this directory (ignoring self.output_dir and doc_id nesting)
            
        Returns:
            Dict with paths to saved images
        """
        if output_dir_override:
            output_path = output_dir_override
            os.makedirs(output_path, exist_ok=True)
        else:
            folder_name = f"doc_{doc_id}" if doc_id else "visualization"
            output_path = self._ensure_output_dir(folder_name)
        
        saved_paths = {}
        
        # Render base PDF page
        base_image = self.render_pdf_page(pdf_path, page_num)
        
        # Draw fusion results (if available) - This is the ONLY one we want to save now
        if fused_results:
            fusion_image = self.draw_fusion_results(base_image.copy(), fused_results)
            if duplicate_mapping_units:
                fusion_image = self.draw_duplicate_mapping_units(
                    fusion_image,
                    duplicate_units=duplicate_mapping_units
                )
            if unaligned_pdf_units:
                fusion_image = self.draw_unaligned_pdf_units(
                    fusion_image,
                    unaligned_pdf_units=unaligned_pdf_units
                )
            fusion_path = os.path.join(output_path, f"page_{page_num + 1}_fused.png")
            fusion_image.save(fusion_path)
            saved_paths['fused'] = fusion_path
        
        return saved_paths

    def visualize_document(
        self,
        pdf_path: str,
        alignments_by_page: Dict[int, List[Dict]] = None,
        fusion_by_page: Dict[int, List[Dict]] = None,
        doc_id: int = None,
        max_pages: int = None
    ) -> Dict[str, Dict[str, str]]:
        """
        Generate visualization images for an entire document.
        
        Args:
            pdf_path: Path to PDF file
            alignments_by_page: Dict of page_num -> alignments
            fusion_by_page: Dict of page_num -> fusion results
            doc_id: Document ID for output folder
            max_pages: Maximum number of pages to process
            
        Returns:
            Dict of page_num -> saved image paths
        """
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        
        if max_pages:
            total_pages = min(total_pages, max_pages)
        
        all_paths = {}
        
        for page_num in range(total_pages):
            alignments = alignments_by_page.get(page_num, []) if alignments_by_page else None
            fused = fusion_by_page.get(page_num, []) if fusion_by_page else None
            
            paths = self.visualize_page(
                pdf_path, page_num, alignments, fused, doc_id=doc_id
            )
            all_paths[page_num] = paths
        
        return all_paths
