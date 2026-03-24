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


class VisualizationDrawMixin:


    def draw_bbox(
        self, 
        draw: ImageDraw.ImageDraw, 
        bbox: List[float], 
        color: str, 
        label: str = None,
        fill_alpha: int = 50,
        line_width: int = 2,
        dashed: bool = False
    ):
        """
        Draw a bounding box on the image.
        
        Args:
            draw: PIL ImageDraw object
            bbox: [x0, y0, x1, y1] in PDF points
            color: Hex color code
            label: Optional label text
            fill_alpha: Fill opacity (0-255)
            line_width: Border line width
            dashed: Whether to draw dashed lines (not fully supported in PIL)
        """
        if not bbox or len(bbox) < 4:
            return
        
        # Scale bbox to image coordinates
        x0 = int(bbox[0] * self.scale)
        y0 = int(bbox[1] * self.scale)
        x1 = int(bbox[2] * self.scale)
        y1 = int(bbox[3] * self.scale)
        
        # Draw filled rectangle with transparency
        rgb = hex_to_rgb(color)
        fill_color = (*rgb, fill_alpha)
        
        # Create a temporary image for the transparent fill
        # Note: PIL doesn't support alpha blending directly, so we use composite
        overlay = Image.new('RGBA', draw.im.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([x0, y0, x1, y1], fill=fill_color)
        
        # Composite isn't directly available on ImageDraw, so we skip fill for now
        # and just draw the outline
        draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
        
        # Draw label if provided
        if label:
            # Draw white background for label
            font = self.font_small or ImageFont.load_default()
            
            # Get text size (use font.getbbox for newer Pillow)
            try:
                text_bbox = font.getbbox(label)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width = len(label) * 6
                text_height = 10
            
            # Draw label background
            label_y = max(0, y0 - text_height - 4)
            draw.rectangle(
                [x0, label_y, x0 + text_width + 4, label_y + text_height + 2],
                fill='white'
            )
            
            # Draw label text
            draw.text((x0 + 2, label_y), label, fill=color, font=font)

    def _get_pdf_unit_key(self, unit: Dict) -> Optional[Tuple[str, Any]]:
        if unit.get('pdf_unit_id') is not None:
            return ('pdf_unit_id', unit['pdf_unit_id'])
        if unit.get('unit_id') is not None:
            return ('unit_id', unit['unit_id'])
        if unit.get('item_idx') is not None:
            return ('item_idx', unit['item_idx'])
        bbox = unit.get('bbox')
        if bbox and len(bbox) >= 4:
            return ('bbox', tuple(bbox))
        return None

    def _dedupe_pdf_units(self, units: Optional[List[Dict]]) -> List[Dict]:
        deduped = []
        seen = set()
        for unit in units or []:
            if not unit or not unit.get('bbox'):
                continue
            key = self._get_pdf_unit_key(unit)
            if key is None or key in seen:
                continue
            seen.add(key)
            deduped.append(unit)
        return deduped

    def _collect_unaligned_pdf_units(
        self,
        unaligned_pdf_units: Optional[List[Dict]]
    ) -> List[Dict]:
        return self._dedupe_pdf_units(unaligned_pdf_units or [])

    def draw_unaligned_pdf_units(
        self,
        image: Image.Image,
        unaligned_pdf_units: Optional[List[Dict]] = None
    ) -> Image.Image:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        draw = ImageDraw.Draw(image)
        unaligned_units = self._collect_unaligned_pdf_units(unaligned_pdf_units)

        for unit in unaligned_units:
            bbox = unit.get('bbox')
            if bbox:
                self.draw_bbox(draw, bbox, ALIGNMENT_COLORS['unaligned'], None, fill_alpha=0, line_width=3)

        return image

    def draw_duplicate_mapping_units(
        self,
        image: Image.Image,
        duplicate_units: Optional[List[Dict]] = None
    ) -> Image.Image:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')

        draw = ImageDraw.Draw(image)
        dup_units = self._dedupe_pdf_units(duplicate_units or [])

        for unit in dup_units:
            if unit.get('valid_cross_page_continuation'):
                continue
            bbox = unit.get('bbox')
            if bbox:
                self.draw_bbox(draw, bbox, ALIGNMENT_COLORS['duplicate_mapping'], None, fill_alpha=0, line_width=2)

        return image

    def draw_alignments(
        self,
        image: Image.Image,
        alignments: List[Dict],
        show_all: bool = True
    ) -> Image.Image:
        """
        Draw alignment bounding boxes on the image.
        
        Args:
            image: PIL Image to draw on
            alignments: List of alignment results
            show_all: Whether to show all alignments
            
        Returns:
            Image with alignments drawn
        """
        # Convert to RGBA for transparency support
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        draw = ImageDraw.Draw(image)
        
        for align in alignments:
            bbox = align.get('merged_bbox')
            if not bbox:
                continue
            
            # Determine color based on type
            if align.get('is_table'):
                color = ALIGNMENT_COLORS['table']
            elif align.get('is_image_part'):
                color = ALIGNMENT_COLORS['image']
            elif align.get('is_text_part'):
                color = ALIGNMENT_COLORS['text_part']
            else:
                color = ALIGNMENT_COLORS['default']
            
            # Create label
            elem_seq = align.get('element_sequence', '?')
            elem_type = align.get('element_type', '')
            label = f"#{elem_seq} {elem_type}"
            
            if align.get('is_text_part'):
                label += " [TXT]"
            elif align.get('is_image_part'):
                label += " [IMG]"
            
            self.draw_bbox(draw, bbox, color, label, fill_alpha=40, line_width=3)
            
            # Draw individual matched PDF units
            for unit in align.get('matched_pdf_units', []):
                unit_bbox = unit.get('bbox')
                if unit_bbox:
                    self.draw_bbox(draw, unit_bbox, color, None, fill_alpha=20, line_width=1)
        
        return image

    def draw_fusion_results(
        self,
        image: Image.Image,
        fused_results: List[Dict],
        use_docling_colors: bool = True,
        show_all: bool = True
    ) -> Image.Image:
        """
        Draw Docling fusion results on the image.
        
        Args:
            image: PIL Image to draw on
            fused_results: List of fused Docling results
            use_docling_colors: Use Docling color scheme (cyan) vs LayoutLM (mixed)
            show_all: Whether to show all results
            
        Returns:
            Image with fusion results drawn
        """
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        draw = ImageDraw.Draw(image)
        
        colors = DOCLING_LABEL_COLORS if use_docling_colors else LABEL_COLORS
        
        for idx, result in enumerate(fused_results):
            bbox = result.get('bbox')
            if not bbox:
                continue
            
            label = result.get('label', 'unknown')
            color = colors.get(label, colors['unknown'])
            
            # Create label text
            elem_seq = result.get('element_sequence')
            openxml_idx = result.get('openxml_idx')
            overlap = result.get('overlap', 0)
            display_seq = elem_seq
            if display_seq is None and openxml_idx is not None:
                if isinstance(openxml_idx, (list, tuple)):
                    display_seq = ','.join(str(idx) for idx in openxml_idx)
                else:
                    display_seq = openxml_idx
            
            if display_seq is not None:
                label_text = f"[{label}] #{display_seq}"
            else:
                label_text = f"[{label}]"

            if openxml_idx is not None and elem_seq is not None and label != 'table':
                if isinstance(openxml_idx, (list, tuple)):
                    show_ox = elem_seq not in openxml_idx
                    ox_text = ','.join(str(idx) for idx in openxml_idx)
                else:
                    show_ox = openxml_idx != elem_seq
                    ox_text = str(openxml_idx)
                if show_ox:
                    label_text += f" ox{ox_text}"
            
            if overlap > 0:
                label_text += f" {overlap:.0%}"
            
            self.draw_bbox(draw, bbox, color, label_text, fill_alpha=50, line_width=2)
        
        return image
