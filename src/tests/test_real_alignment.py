"""
Test alignment dengan task real (dokumen_id=332)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['DB_HOST'] = 'localhost'

from database import SessionLocal
from models import Antrian, Dokumen
from services.alignment_service import AlignmentService
from services.pdf_extraction_service import PDFExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

db = SessionLocal()

# Get task 520 (dokumen_id=332)
task = db.query(Antrian).get(520)
if not task:
    print("[ERROR] Task 520 not found")
    exit(1)

print(f"Task ID: {task.antrian_id}")
print(f"Dokumen ID: {task.dokumen_id}")

# Get dokumen
doc = db.query(Dokumen).get(task.dokumen_id)
if not doc:
    print(f"[ERROR] Dokumen {task.dokumen_id} not found")
    exit(1)

print(f"Dokumen: {doc.dokumen_filename}")
print(f"PDF Path: {doc.dokumen_pdf_path}")

# Get full path
STORAGE_BASE = os.getenv('VOLUME_BASE_PATH', '/app/storage')
pdf_path = os.path.join(STORAGE_BASE, doc.dokumen_pdf_path)
print(f"Full path: {pdf_path}")

# Check if PDF exists
if not os.path.exists(pdf_path):
    print(f"[ERROR] PDF not found: {pdf_path}")
    exit(1)

print(f"[OK] PDF exists: {pdf_path}")

# Extract first page
print("\n[1] Extracting PDF...")
try:
    with PDFExtractor(pdf_path) as extractor:
        page = extractor.get_page(0)
        page_data = extractor.extract_merging_data(0)
        
        items = []
        for g in page_data.get('char_groups', []):
            items.append({
                'type': 'group',
                'bbox': g.get('merged_bbox'),
                'data': {'text': g.get('text', '')}
            })
        
        items.sort(key=lambda x: (x['bbox'][1] if x.get('bbox') else 0, x['bbox'][0] if x.get('bbox') else 0))
        
        extraction_results = [{
            'page': 1,
            'page_width': page.rect.width,
            'page_height': page.rect.height,
            'items': items
        }]
        
        print(f"[OK] Extracted {len(items)} items from page 1")
        if items:
            print(f"  First item: '{items[0]['data'].get('text', '')[:50]}'")
except Exception as e:
    print(f"[ERROR] Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Run alignment
print("\n[2] Running alignment...")
try:
    alignment_service = AlignmentService()
    alignment_results = alignment_service.align_document(
        extraction_results, 
        task.dokumen_id
    )
    
    print(f"[OK] Alignment completed")
    
    for result in alignment_results:
        page = result.get('page')
        success = result.get('success')
        alignments = result.get('alignments', [])
        unaligned = result.get('unaligned_pdf_units', [])
        
        print(f"\n  Page {page}:")
        print(f"    Success: {success}")
        print(f"    Alignments: {len(alignments)}")
        print(f"    Unaligned: {len(unaligned)}")
        
        if alignments:
            print(f"\n    First 3 alignments:")
            for i, align in enumerate(alignments[:3]):
                print(f"      {i+1}. Element {align.get('element_id')} Type={align.get('element_type')}")
                print(f"         Merged bbox: {align.get('merged_bbox')}")
                print(f"         Matched units: {len(align.get('matched_pdf_units', []))}")
        else:
            print(f"    [!] NO ALIGNMENTS!")
            print(f"    [!] Text tidak match antara PDF dan DokumenElemen")
            
            # Show sample PDF text
            print(f"\n    Sample PDF text (first 5 items):")
            for i, item in enumerate(extraction_results[0]['items'][:5]):
                text = item['data'].get('text', '')[:50]
                print(f"      {i+1}. '{text}'")
            
            # Show sample DokumenElemen text
            from models import DokumenElemen, DokumenPart, DokumenSection
            elements = db.query(DokumenElemen).join(
                DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id
            ).join(
                DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id
            ).filter(
                DokumenSection.dsec_ref_tipe == 'dokumen',
                DokumenSection.dsec_ref_id == task.dokumen_id,
                DokumenPart.dpart_type == 'body'
            ).order_by(DokumenElemen.delemen_sequence).limit(5).all()
            
            print(f"\n    Sample DokumenElemen text (first 5):")
            for i, elem in enumerate(elements):
                # Extract text from json_tree
                import json
                try:
                    json_tree = json.loads(elem.delemen_json_tree) if elem.delemen_json_tree else {}
                    # Simple text extraction
                    def extract_text(node):
                        if isinstance(node, dict):
                            if 'value' in node: return str(node['value'])
                            if 'text' in node: return str(node['text'])
                            return ' '.join(extract_text(v) for v in node.values() if v)
                        elif isinstance(node, list):
                            return ' '.join(extract_text(item) for item in node)
                        return str(node) if node else ''
                    
                    text = extract_text(json_tree)[:50]
                except:
                    text = '[parse error]'
                
                print(f"      {i+1}. ID={elem.delemen_id} Type={elem.delemen_type} Text='{text}'")
            
            # Check normalized text
            print(f"\n    Checking normalized text:")
            pdf_text = ' '.join(item['data'].get('text', '') for item in extraction_results[0]['items'][:10])
            print(f"      PDF (first 10 items): '{pdf_text[:100]}'")
            
            if elements:
                elem = elements[0]
                json_tree = json.loads(elem.delemen_json_tree) if elem.delemen_json_tree else {}
                elem_text = alignment_service._extract_text_from_json_tree(json_tree)
                elem_norm = alignment_service._normalize_text(elem_text)
                pdf_norm = alignment_service._normalize_text(pdf_text)
                print(f"      Element text: '{elem_text[:100]}'")
                print(f"      Element normalized: '{elem_norm[:100]}'")
                print(f"      PDF normalized: '{pdf_norm[:100]}'")
                
except Exception as e:
    print(f"[ERROR] Alignment failed: {e}")
    import traceback
    traceback.print_exc()
    db.close()
    exit(1)

db.close()
print("\n[OK] Test completed")
