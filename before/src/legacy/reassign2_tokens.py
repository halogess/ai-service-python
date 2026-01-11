"""
Reassignment 2: Untuk unaligned token dari reassign1, 
cek overlap dengan docling bbox, lalu merge dengan elemen OpenXML yang overlap
"""

import json
import os
import fitz
from docling.document_converter import DocumentConverter


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


def extract_docling_bboxes(pdf_path):
    """Extract bboxes from docling"""
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    doc = result.document
    
    pdf_doc = fitz.open(pdf_path)
    docling_bboxes = {}  # page -> list of {bbox, label}
    
    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        page_height = page.rect.height
        scale = 300/72
        bboxes = []
        
        # Texts
        for text in doc.texts:
            if hasattr(text, 'prov') and text.prov:
                prov = text.prov[0] if isinstance(text.prov, list) else text.prov
                if hasattr(prov, 'page_no') and prov.page_no == page_num + 1:
                    if hasattr(prov, 'bbox') and prov.bbox:
                        bbox = prov.bbox
                        y_top = (page_height - bbox.t) * scale
                        y_bottom = (page_height - bbox.b) * scale
                        bboxes.append({
                            'bbox': {
                                'x0': bbox.l * scale,
                                'y0': y_top,
                                'x1': bbox.r * scale,
                                'y1': y_bottom
                            },
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
                        bboxes.append({
                            'bbox': {
                                'x0': bbox.l * scale,
                                'y0': y_top,
                                'x1': bbox.r * scale,
                                'y1': y_bottom
                            },
                            'label': 'table'
                        })
        
        docling_bboxes[page_num] = bboxes
    
    pdf_doc.close()
    return docling_bboxes


def reassign2_unaligned_tokens(doc_id, pdf_path):
    """
    Reassignment 2: Cek unaligned tokens dari reassign1,
    overlap dengan docling bbox, merge dengan OpenXML element
    """
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        assets_dir = os.path.join(base_dir, 'assets', str(doc_id))
        
        reassign1_path = os.path.join(assets_dir, f'reassign1_merged_alignment_{doc_id}.json')
        alignment_difflib_path = os.path.join(assets_dir, f'alignment_difflib_{doc_id}.json')
        reassign2_path = os.path.join(assets_dir, f'reassign2_merged_alignment_{doc_id}.json')
        
        if not os.path.exists(reassign1_path):
            return {"error": "Reassign1 file not found. Run Reassign 1 first."}
        
        if not os.path.exists(alignment_difflib_path):
            return {"error": "Alignment difflib file not found."}
        
        # Load reassign1 results
        with open(reassign1_path, 'r', encoding='utf-8') as f:
            reassign1_data = json.load(f)
        
        merged_results = reassign1_data.get('merged_results', [])
        
        # Load alignment difflib to get unaligned tokens
        with open(alignment_difflib_path, 'r', encoding='utf-8') as f:
            difflib_data = json.load(f)
        
        if 'success' in difflib_data:
            difflib_data = difflib_data.get('data', {})
        
        unaligned_tokens = difflib_data.get('unaligned_tokens', [])
        
        # Extract docling bboxes
        docling_bboxes = extract_docling_bboxes(pdf_path)
        
        reassigned_count = 0
        
        # Process each unaligned token
        for token in unaligned_tokens[:]:
            token_page = token['page']
            token_bbox = token['bbox']
            
            # Find docling bbox that overlaps with token
            best_docling_bbox = None
            best_docling_iou = 0.0
            
            if token_page in docling_bboxes:
                for docling_item in docling_bboxes[token_page]:
                    iou = calculate_iou(token_bbox, docling_item['bbox'])
                    if iou > best_docling_iou:
                        best_docling_iou = iou
                        best_docling_bbox = docling_item
            
            # If found docling bbox overlap
            if best_docling_bbox and best_docling_iou > 0.1:
                # Find OpenXML element that overlaps with this token
                best_element = None
                best_elem_iou = 0.0
                
                for elem in merged_results:
                    if elem['page'] == token_page:
                        elem_iou = calculate_iou(token_bbox, elem['bbox'])
                        if elem_iou > best_elem_iou:
                            best_elem_iou = elem_iou
                            best_element = elem
                
                # Merge token with element
                if best_element and best_elem_iou > 0.05:
                    # Expand element bbox
                    best_element['bbox']['x0'] = min(best_element['bbox']['x0'], token_bbox['x0'])
                    best_element['bbox']['y0'] = min(best_element['bbox']['y0'], token_bbox['y0'])
                    best_element['bbox']['x1'] = max(best_element['bbox']['x1'], token_bbox['x1'])
                    best_element['bbox']['y1'] = max(best_element['bbox']['y1'], token_bbox['y1'])
                    
                    # Add token text to element
                    best_element['text'] += ' ' + token['text']
                    
                    # Mark as reassigned
                    if 'reassign2_tokens' not in best_element:
                        best_element['reassign2_tokens'] = []
                    best_element['reassign2_tokens'].append(token['text'])
                    
                    reassigned_count += 1
        
        # Save results
        output_data = {
            "merged_results": merged_results,
            "stats": {
                "total_elements": len(merged_results),
                "reassign2_count": reassigned_count,
                "remaining_unaligned": len(unaligned_tokens) - reassigned_count
            }
        }
        
        with open(reassign2_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        return {
            "success": True,
            "reassigned": reassigned_count,
            "remaining_unaligned": len(unaligned_tokens) - reassigned_count,
            "output_file": reassign2_path
        }
        
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
