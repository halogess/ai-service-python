"""
Debug script to compare what's being fed into alignment:
1. PDF extraction text
2. DokumenElemen text from database
3. Normalized versions of both
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ['DB_HOST'] = 'localhost'

from database import SessionLocal
from models import Antrian, Dokumen, DokumenElemen, DokumenPart, DokumenSection
from services.alignment_service import AlignmentService
from services.pdf_extraction_service import PDFExtractor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

STORAGE_BASE = os.getenv('VOLUME_BASE_PATH', '/app/storage')

def debug_alignment_data(doc_id: int = 332):
    """Debug alignment data for a specific document"""
    
    db = SessionLocal()
    service = AlignmentService()
    
    try:
        # 1. Get Document
        doc = db.query(Dokumen).get(doc_id)
        if not doc:
            print(f"[ERROR] Document {doc_id} not found")
            return
        
        print(f"=" * 80)
        print(f"DEBUGGING ALIGNMENT FOR DOCUMENT: {doc.dokumen_filename} (ID: {doc_id})")
        print(f"=" * 80)
        
        # 2. Get DokumenElemen from database
        elements = db.query(DokumenElemen).join(
            DokumenPart, DokumenElemen.dpart_id == DokumenPart.dpart_id
        ).join(
            DokumenSection, DokumenPart.dsec_id == DokumenSection.dsec_id
        ).filter(
            DokumenSection.dsec_ref_tipe == 'dokumen',
            DokumenSection.dsec_ref_id == doc_id,
            DokumenPart.dpart_type == 'body'
        ).order_by(DokumenElemen.delemen_sequence).limit(20).all()
        
        print(f"\n[1] DOKUMEN ELEMEN FROM DATABASE ({len(elements)} elements)")
        print("-" * 80)
        
        for i, elem in enumerate(elements[:10]):
            print(f"\n  [{i+1}] Element ID: {elem.delemen_id}, Type: {elem.delemen_type}, Seq: {elem.delemen_sequence}")
            
            # Parse JSON tree
            try:
                if isinstance(elem.delemen_json_tree, str):
                    json_tree = json.loads(elem.delemen_json_tree)
                else:
                    json_tree = elem.delemen_json_tree or {}
            except Exception as e:
                print(f"      [ERROR] JSON parse failed: {e}")
                json_tree = {}
            
            # Extract text using our function
            text_raw = service._extract_text_from_json_tree(json_tree)
            text_norm = service._normalize_text(text_raw)
            
            print(f"      Raw text:        '{text_raw[:80]}...'")
            print(f"      Normalized text: '{text_norm[:80]}...'")
            
            # Show JSON tree structure (first level keys only)
            if isinstance(json_tree, dict):
                print(f"      JSON keys: {list(json_tree.keys())[:10]}")
        
        # 3. Get PDF extraction data
        pdf_path = os.path.join(STORAGE_BASE, doc.dokumen_pdf_path)
        if not os.path.exists(pdf_path):
            print(f"\n[ERROR] PDF not found: {pdf_path}")
            return
        
        print(f"\n[2] PDF EXTRACTION DATA")
        print("-" * 80)
        
        with PDFExtractor(pdf_path) as extractor:
            page = extractor.get_page(0)
            page_data = extractor.extract_merging_data(0)
            
            # Transform to items
            items = []
            for g in page_data.get('char_groups', []):
                items.append({
                    'type': 'group',
                    'bbox': g.get('merged_bbox'),
                    'data': {'text': g.get('text', '')}
                })
            
            for s in page_data.get('shapes', []):
                items.append({
                    'type': 'shape',
                    'bbox': s.get('bbox'),
                    'data': {'text': s.get('text', ''), 'image_bbox': s.get('image_bbox')}
                })
            
            items.sort(key=lambda x: (x['bbox'][1] if x.get('bbox') else 0, x['bbox'][0] if x.get('bbox') else 0))
            
            print(f"  Extracted {len(items)} items from page 1")
            
            # Flatten using service
            pdf_units = service._flatten_extraction_items(items)
            
            print(f"  Flattened to {len(pdf_units)} PDF units")
            
            for i, unit in enumerate(pdf_units[:10]):
                text_raw = unit.get('text', '')
                text_norm = unit.get('text_normalized', '')
                print(f"\n  [{i+1}] PDF Unit {unit['unit_id']}")
                print(f"      Raw text:        '{text_raw[:60]}'")
                print(f"      Normalized text: '{text_norm[:60]}'")
        
        # 4. Compare first elements
        print(f"\n[3] DIRECT COMPARISON (First 5 elements)")
        print("-" * 80)
        
        openxml_units, _ = service._build_openxml_units(elements[:20])
        
        print(f"\n  OpenXML units built: {len(openxml_units)}")
        for i, ou in enumerate(openxml_units[:5]):
            print(f"  [{i+1}] OX Unit: elem_id={ou['elem_id']}, type={ou['elem_type']}")
            print(f"       text: '{ou['text'][:60]}'")
            print(f"       norm: '{ou['text_normalized'][:60]}'")
        
        print(f"\n[4] TEXT MATCHING TEST")
        print("-" * 80)
        
        # Simple character comparison
        pdf_concat = ''.join(u['text_normalized'] for u in pdf_units[:10])
        ox_concat = ''.join(u['text_normalized'] for u in openxml_units[:10])
        
        print(f"  PDF (first 10 units) normalized: '{pdf_concat[:100]}'")
        print(f"  OX  (first 10 units) normalized: '{ox_concat[:100]}'")
        
        # Check overlap
        import difflib
        sm = difflib.SequenceMatcher(None, pdf_concat[:200], ox_concat[:200])
        ratio = sm.ratio()
        print(f"\n  Similarity ratio (first 200 chars): {ratio:.2%}")
        
        if ratio < 0.5:
            print("\n  [!] LOW SIMILARITY - Text does not match!")
            print("      This is why alignment fails.")
            print("\n  Checking character-by-character differences:")
            
            # Find first difference
            for i, (p, o) in enumerate(zip(pdf_concat[:50], ox_concat[:50])):
                if p != o:
                    print(f"      Position {i}: PDF='{p}' (U+{ord(p):04X}) vs OX='{o}' (U+{ord(o):04X})")
                    break
        else:
            print("\n  [OK] Good similarity - alignment should work")
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    doc_id = int(sys.argv[1]) if len(sys.argv) > 1 else 332
    debug_alignment_data(doc_id)
