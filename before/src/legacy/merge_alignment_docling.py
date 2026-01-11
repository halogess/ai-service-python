"""Merge bbox alignment dari difflib_alignment dengan label dari docling"""

import fitz
from docling.document_converter import DocumentConverter


def merge_alignment_with_docling(pdf_path, elements, output_db=None):
    """
    Merge alignment bbox dengan docling labels.
    
    Args:
        pdf_path: Path ke PDF
        elements: List of DokumenElemen objects
        output_db: Optional database session untuk save
    
    Returns:
        List of merged results dengan format:
        {
            'element_id': int,
            'text': str,
            'bbox': {x0, y0, x1, y1},
            'page': int,
            'docling_label': str,
            'confidence': float
        }
    """
    # 1. Get alignment dari difflib
    from difflib_alignment import align_document
    alignment_result = align_document(pdf_path, elements)
    aligned_words = alignment_result['aligned_words']
    
    # 2. Get labels dari docling
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    
    # 3. Extract docling annotations per page
    pdf_doc = fitz.open(pdf_path)
    docling_annotations = {}  # page -> list of {bbox, label}
    
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        page_height = page.rect.height
        scale = 300/72
        annotations = []
        
        # Texts
        for text in doc.texts:
            if hasattr(text, 'prov') and text.prov:
                prov = text.prov[0] if isinstance(text.prov, list) else text.prov
                if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        y_top = (page_height - bbox.t) * scale
                        y_bottom = (page_height - bbox.b) * scale
                        annotations.append({
                            'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
                            'label': str(text.label).split('.')[-1].lower() if hasattr(text, 'label') else 'text'
                        })
        
        # Tables
        for table in doc.tables:
            if hasattr(table, 'prov') and table.prov:
                prov = table.prov[0] if isinstance(table.prov, list) else table.prov
                if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        y_top = (page_height - bbox.t) * scale
                        y_bottom = (page_height - bbox.b) * scale
                        annotations.append({
                            'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
                            'label': 'table'
                        })
        
        # Formulas
        if hasattr(doc, 'formulas'):
            for formula in doc.formulas:
                if hasattr(formula, 'prov') and formula.prov:
                    prov = formula.prov[0] if isinstance(formula.prov, list) else formula.prov
                    if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                        if hasattr(prov, 'bbox') and prov.bbox:
                            bbox = prov.bbox
                            y_top = (page_height - bbox.t) * scale
                            y_bottom = (page_height - bbox.b) * scale
                            annotations.append({
                                'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
                                'label': 'formula'
                            })
        
        docling_annotations[page_num] = annotations
    
    pdf_doc.close()
    
    # 4. Match alignment bbox dengan docling labels
    merged_results = []
    
    for aligned in aligned_words:
        elem_bbox = aligned['bbox']
        page = aligned['page']
        
        # Find best matching docling annotation
        best_label = None
        best_iou = 0.0
        
        if page in docling_annotations:
            for ann in docling_annotations[page]:
                iou = calculate_iou(elem_bbox, ann['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_label = ann['label']
        
        merged_results.append({
            'element_id': aligned['element_id'],
            'text': aligned['text'],
            'bbox': elem_bbox,
            'page': page,
            'docling_label': best_label if best_iou > 0.1 else None,
            'confidence': aligned.get('confidence', 1.0),
            'iou': best_iou
        })
    
    # 5. Save to database if provided
    if output_db:
        from models import DokumenElemenVisual
        for result in merged_results:
            dev = DokumenElemenVisual(
                dokumen_id=elements[0].dokumen_id if elements else None,
                dev_bbox_x0=result['bbox']['x0'],
                dev_bbox_y0=result['bbox']['y0'],
                dev_bbox_x1=result['bbox']['x1'],
                dev_bbox_y1=result['bbox']['y1'],
                dev_page=result['page'],
                dev_label=result['docling_label'],
                dev_text=result['text'],
                dokumen_elemen_id=result['element_id'] if isinstance(result['element_id'], int) else None
            )
            output_db.add(dev)
        output_db.commit()
    
    return merged_results


def calculate_iou(bbox1, bbox2):
    """Calculate Intersection over Union"""
    if isinstance(bbox1, dict):
        b1 = [bbox1['x0'], bbox1['y0'], bbox1['x1'], bbox1['y1']]
    else:
        b1 = bbox1
    
    if isinstance(bbox2, dict):
        b2 = [bbox2['x0'], bbox2['y0'], bbox2['x1'], bbox2['y1']]
    else:
        b2 = bbox2
    
    x_left = max(b1[0], b2[0])
    y_top = max(b1[1], b2[1])
    x_right = min(b1[2], b2[2])
    y_bottom = min(b1[3], b2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def label_to_int(label):
    """Convert docling label to integer"""
    label_map = {
        'title': 1,
        'paragraph': 2,
        'list_item': 3,
        'caption': 4,
        'table': 5,
        'picture': 6,
        'text': 7,
        'formula': 8,
        'footnote': 9,
        'code': 10,
        'page_header': 11,
        'page_footer': 12,
        'section_header': 13
    }
    return label_map.get(label, 0)
