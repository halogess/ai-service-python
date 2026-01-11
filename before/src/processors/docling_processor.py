from docling.document_converter import DocumentConverter
import fitz
import base64
import os
from PIL import Image, ImageDraw, ImageFont
import io
import time
from models import db, TestingHistory, TestingPrediction

COLOR_MAP = {
    'title': 'blue',
    'paragraph': 'green',
    'list_item': 'purple',
    'caption': 'orange',
    'table': 'red',
    'picture': 'cyan',
    'text': 'magenta',
    'formula': 'teal',
    'footnote': 'brown',
    'code': '#00CED1',
    'page_header': '#FF6B00',
    'page_footer': '#E91E63',
    'section_header': '#8B008B'
}

def process_pdf_with_docling(pdf_path, output_folder, doc_id=None, save_to_db=False):
    try:
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc = result.document
    except Exception as e:
        raise Exception(f"Docling conversion failed: {str(e)}")
    
    try:
        pdf_doc = fitz.open(pdf_path)
    except Exception as e:
        raise Exception(f"Failed to open PDF: {str(e)}")
    
    all_pages = []
    
    history_id = None
    if save_to_db and doc_id:
        start_time = time.time()
        history = TestingHistory(
            testing_dokumen_id=doc_id,
            testing_history_description='Docling Analysis'
        )
        db.session.add(history)
        db.session.flush()
        history_id = history.testing_history_id
    
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        page_height = page.rect.height
        scale = 300/72
        
        # Get annotations from all elements for this page
        annotations = []
        picture_bboxes = []
        
        # Extract images first to know picture regions
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            try:
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    rect = img_rects[0]
                    picture_bbox = [rect.x0 * scale, rect.y0 * scale, rect.x1 * scale, rect.y1 * scale]
                    picture_bboxes.append(picture_bbox)
                    annotations.append({
                        'label': 'picture',
                        'bbox': picture_bbox,
                        'text': f'[Image {img_index + 1}]'
                    })
            except:
                pass
        
        # Extract from texts (paragraphs, titles, list items, etc)
        for text in doc.texts:
            if hasattr(text, 'prov') and text.prov:
                prov = text.prov[0] if isinstance(text.prov, list) else text.prov
                if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        y_top = (page_height - bbox.t) * scale
                        y_bottom = (page_height - bbox.b) * scale
                        text_bbox = [bbox.l * scale, y_top, bbox.r * scale, y_bottom]
                        
                        # Skip if text overlaps with picture (>50% overlap)
                        if not is_inside_picture(text_bbox, picture_bboxes):
                            label = str(text.label).split('.')[-1].lower() if hasattr(text, 'label') else 'text'
                            text_content = text.text if hasattr(text, 'text') else ''
                            print(f"[DEBUG] Page {page_num+1} | Label: {label} | Text: {text_content[:50]}...")
                            annotations.append({
                                'label': label,
                                'bbox': text_bbox,
                                'text': text_content
                            })
        
        # Extract from tables
        for table in doc.tables:
            if hasattr(table, 'prov') and table.prov:
                prov = table.prov[0] if isinstance(table.prov, list) else table.prov
                if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        y_top = (page_height - bbox.t) * scale
                        y_bottom = (page_height - bbox.b) * scale
                        annotations.append({
                            'label': 'table',
                            'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
                            'text': ''
                        })
        
        # Extract from formulas
        if hasattr(doc, 'formulas'):
            for formula in doc.formulas:
                if hasattr(formula, 'prov') and formula.prov:
                    prov = formula.prov[0] if isinstance(formula.prov, list) else formula.prov
                    if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                        if hasattr(prov, 'bbox') and prov.bbox:
                            bbox = prov.bbox
                            y_top = (page_height - bbox.t) * scale
                            y_bottom = (page_height - bbox.b) * scale
                            formula_text = formula.text if hasattr(formula, 'text') else ''
                            annotations.append({
                                'label': 'formula',
                                'bbox': [bbox.l * scale, y_top, bbox.r * scale, y_bottom],
                                'text': formula_text
                            })
        
        # Render page with PyMuPDF
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale))
        img_bytes = pix.tobytes("png")
        
        # Draw bounding boxes on image
        img = Image.open(io.BytesIO(img_bytes))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        for ann in annotations:
            bbox = ann['bbox']
            color = COLOR_MAP.get(ann['label'], 'red')
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=3)
            draw.text((bbox[0], bbox[1] - 30), ann['label'], fill=color, font=font)
        
        # Save annotated image
        img_path = os.path.join(output_folder, f'page_{page_num + 1}.png')
        img.save(img_path)
        
        # Convert to base64
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        all_pages.append({
            'page_num': page_num + 1,
            'annotations': annotations,
            'image': img_str
        })
        
        # Save to database
        if save_to_db and history_id:
            for ann in annotations:
                bbox = ann['bbox']
                prediction = TestingPrediction(
                    testing_history_id=history_id,
                    testing_prediction_page=page_num + 1,
                    testing_prediction_bbox_x0=bbox[0],
                    testing_prediction_bbox_y0=bbox[1],
                    testing_prediction_bbox_x1=bbox[2],
                    testing_prediction_bbox_y1=bbox[3],
                    testing_prediction_label=ann['label'],
                    testing_prediction_word=ann.get('text', ''),
                    testing_prediction_confidence=1.0
                )
                db.session.add(prediction)
    
    if save_to_db and history_id:
        total_time = time.time() - start_time
        history.testing_history_processing_time = round(total_time, 2)
        db.session.commit()
    
    pdf_doc.close()
    return all_pages

def is_inside_picture(text_bbox, picture_bboxes, threshold=0.5):
    """Check if text bbox overlaps with any picture bbox by more than threshold"""
    tx0, ty0, tx1, ty1 = text_bbox
    text_area = (tx1 - tx0) * (ty1 - ty0)
    
    if text_area <= 0:
        return False
    
    for pic_bbox in picture_bboxes:
        px0, py0, px1, py1 = pic_bbox
        
        # Calculate intersection
        ix0 = max(tx0, px0)
        iy0 = max(ty0, py0)
        ix1 = min(tx1, px1)
        iy1 = min(ty1, py1)
        
        if ix0 < ix1 and iy0 < iy1:
            intersection_area = (ix1 - ix0) * (iy1 - iy0)
            overlap_ratio = intersection_area / text_area
            
            if overlap_ratio > threshold:
                return True
    
    return False
