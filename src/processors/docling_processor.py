"""
Docling Processor
Handles document analysis using Docling library
Refactored from src/docling_processor.py
"""

import logging
import os
from typing import List, Dict, Optional
import fitz
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def process_pdf_with_docling(pdf_path: str, output_dir: Optional[str] = None) -> dict:
    """
    Process PDF with Docling for document analysis.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Optional output directory for results
    
    Returns:
        dict: Docling analysis results with bounding boxes
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableStructureOptions
    
    logger.info(f"Processing PDF with Docling: {pdf_path}")
    
    # Configure pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.do_ocr = False
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
        mode="accurate"
    )
    pipeline_options.generate_page_images = True
    pipeline_options.generate_picture_images = True
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    result = converter.convert(pdf_path)
    doc = result.document
    
    # Export to markdown
    markdown_content = doc.export_to_markdown()
    
    # Export to dict for JSON
    doc_dict = doc.export_to_dict()
    
    # Open PDF to get page dimensions
    pdf_doc = fitz.open(pdf_path)
    scale = 300/72  # 300 DPI
    
    # Extract bounding boxes per page
    pages_with_bbox = _extract_pages_with_bbox(doc, pdf_doc, scale)
    
    # Extract text blocks for matching
    text_blocks = _extract_text_blocks(doc)
    
    pdf_doc.close()
    logger.info(f"Docling processing completed: {len(pages_with_bbox)} pages, {len(text_blocks)} text blocks")
    
    return {
        "markdown": markdown_content,
        "document": doc_dict,
        "pages": doc_dict.get("pages", []),
        "pages_with_bbox": pages_with_bbox,
        "text_blocks": text_blocks
    }


def _extract_pages_with_bbox(doc, pdf_doc, scale: float) -> List[dict]:
    """
    Extract bounding boxes for all elements per page.
    
    Args:
        doc: Docling document
        pdf_doc: PyMuPDF document
        scale: Coordinate scale factor
        
    Returns:
        List of page data dicts
    """
    pages_with_bbox = []
    
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        page_height = page.rect.height
        
        page_data = {
            'page_number': page_num + 1,
            'size': {'width': page.rect.width, 'height': page_height},
            'elements': []
        }
        
        # Extract from texts (paragraphs, titles, list items, etc)
        for text in doc.texts:
            _add_element_if_on_page(page_data, text, page_num, page_height, scale)
        
        # Extract from pictures
        for picture in doc.pictures:
            _add_element_if_on_page(page_data, picture, page_num, page_height, scale, elem_type='picture')
        
        # Extract from tables
        for table in doc.tables:
            _add_element_if_on_page(page_data, table, page_num, page_height, scale, elem_type='table')
        
        # Extract from captions
        if hasattr(doc, 'captions'):
            for caption in doc.captions:
                _add_element_if_on_page(page_data, caption, page_num, page_height, scale, elem_type='caption')
        
        # Detect overlapping elements
        _detect_overlaps(page_data)
        
        pages_with_bbox.append(page_data)
        
        # Log element type distribution
        type_counts = {}
        for elem in page_data['elements']:
            elem_type = elem['type']
            type_counts[elem_type] = type_counts.get(elem_type, 0) + 1
        logger.debug(f"Page {page_data['page_number']}: {len(page_data['elements'])} elements - {type_counts}")
    
    return pages_with_bbox


def _add_element_if_on_page(
    page_data: dict, 
    element, 
    page_num: int, 
    page_height: float, 
    scale: float,
    elem_type: Optional[str] = None
):
    """Add element to page_data if it's on the specified page."""
    if not hasattr(element, 'prov') or not element.prov:
        return
        
    prov = element.prov[0] if isinstance(element.prov, list) else element.prov
    if not hasattr(prov, 'page_no') or prov.page_no != page_num + 1:
        return
        
    if not hasattr(prov, 'bbox') or not prov.bbox:
        return
    
    bbox = prov.bbox
    y_top = (page_height - bbox.t) * scale
    y_bottom = (page_height - bbox.b) * scale
    
    if elem_type is None:
        elem_type = str(element.label).split('.')[-1].lower() if hasattr(element, 'label') else 'text'
    
    text = element.text if hasattr(element, 'text') else ''
    
    page_data['elements'].append({
        'type': elem_type,
        'text': text,
        'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
    })


def _detect_overlaps(page_data: dict):
    """Detect overlapping elements in page_data."""
    elements = page_data['elements']
    
    for i, elem in enumerate(elements):
        elem['overlaps'] = []
        bbox1 = elem['bbox']
        
        for j, other in enumerate(elements):
            if i == j:
                continue
            
            bbox2 = other['bbox']
            
            # Check if bboxes overlap
            x_overlap = not (bbox1[2] <= bbox2[0] or bbox2[2] <= bbox1[0])
            y_overlap = not (bbox1[3] <= bbox2[1] or bbox2[3] <= bbox1[1])
            
            if x_overlap and y_overlap:
                # Calculate overlap area
                x_left = max(bbox1[0], bbox2[0])
                y_top = max(bbox1[1], bbox2[1])
                x_right = min(bbox1[2], bbox2[2])
                y_bottom = min(bbox1[3], bbox2[3])
                overlap_area = (x_right - x_left) * (y_bottom - y_top)
                
                elem['overlaps'].append({
                    'index': j,
                    'type': other['type'],
                    'overlap_area': round(overlap_area, 2)
                })


def _extract_text_blocks(doc) -> List[dict]:
    """
    Extract text blocks with line-level detail for matching.
    
    Args:
        doc: Docling document
        
    Returns:
        List of text block dicts
    """
    text_blocks = []
    sequence = 0
    
    logger.debug(f"doc.texts has {len(doc.texts)} elements")
    
    for element in doc.texts:
        if not hasattr(element, 'text') or not element.text or not element.text.strip():
            continue
            
        label = str(element.label).split('.')[-1].lower() if hasattr(element, 'label') else 'text'
        
        # Get bbox info
        bbox_info = None
        page_no = None
        if hasattr(element, 'prov') and element.prov:
            prov = element.prov[0] if isinstance(element.prov, list) else element.prov
            if hasattr(prov, 'page_no'):
                page_no = prov.page_no
            if hasattr(prov, 'bbox') and prov.bbox:
                bbox = prov.bbox
                bbox_info = {
                    'l': bbox.l,
                    't': bbox.t,
                    'r': bbox.r,
                    'b': bbox.b
                }
        
        # Normalize text
        text_normalized = element.text.replace('\t', ' ')
        text_db_format = element.text.replace('\t', ':t')
        lines = element.text.split('\n')
        lines_normalized = [line.replace('\t', ' ') for line in lines]
        lines_db_format = [line.replace('\t', ':t') for line in lines]
        
        text_blocks.append({
            'sequence': sequence,
            'type': label,
            'page': page_no,
            'bbox': bbox_info,
            'full_text': element.text,
            'full_text_normalized': text_normalized,
            'full_text_db_format': text_db_format,
            'lines': lines,
            'lines_normalized': lines_normalized,
            'lines_db_format': lines_db_format,
            'line_count': len(lines),
            'words': text_normalized.split(),
            'word_count': len(text_normalized.split())
        })
        sequence += 1
    
    logger.info(f"Extracted {len(text_blocks)} text blocks for matching")
    
    return text_blocks


def draw_bboxes_on_images(
    image_paths: List[str], 
    pages_with_bbox: List[dict], 
    output_dir: str
):
    """
    Draw bounding boxes on images.
    
    Args:
        image_paths: List of image file paths
        pages_with_bbox: List of page data with bboxes
        output_dir: Output directory for annotated images
    """
    os.makedirs(output_dir, exist_ok=True)
    
    colors = {
        'title': 'red',
        'section_header': 'orange',
        'heading': 'red',
        'text': 'blue',
        'paragraph': 'blue',
        'list_item': 'green',
        'table': 'purple',
        'figure': 'cyan',
        'picture': 'cyan',
        'caption': 'magenta',
        'page_header': 'yellow',
        'page_footer': 'pink',
        'footnote': 'brown',
        'formula': 'teal',
        'code': 'lime',
        'reference': 'violet',
    }
    
    for idx, (image_path, page_data) in enumerate(zip(image_paths, pages_with_bbox)):
        image = Image.open(image_path).convert('RGB')
        draw = ImageDraw.Draw(image)
        
        elements = page_data.get('elements', [])
        
        logger.debug(f"Page {idx+1}: Drawing {len(elements)} bboxes")
        
        drawn_count = 0
        for element in elements:
            bbox = element.get('bbox', [])
            elem_type = element.get('type', 'text').lower()
            
            if not bbox or len(bbox) != 4:
                continue
            
            # Bbox already in image coordinates [x0, y0, x1, y1]
            x0, y0, x1, y1 = bbox
            
            color = colors.get(elem_type, 'gray')
            draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
            draw.text((x0, max(y0 - 20, 0)), elem_type, fill=color)
            drawn_count += 1
        
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        image.save(output_path)
        logger.debug(f"Saved bbox visualization: {output_path} ({drawn_count} boxes drawn)")
