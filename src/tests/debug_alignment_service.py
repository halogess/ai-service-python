"""
Debug alignment service dengan bab2.pdf
Cek kenapa alignment tidak menghasilkan hasil
"""

import os
import sys
import json
import fitz

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService
from database import SessionLocal
from models import Dokumen, DokumenSection, DokumenPart, DokumenElemen
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_alignment():
    """Debug alignment dengan data real"""
    
    pdf_path = 'bab2.pdf'
    if not os.path.exists(pdf_path):
        print(f"[X] File tidak ditemukan: {pdf_path}")
        return
    
    print(f"[OK] Found: {pdf_path}\n")
    
    # Extract first page
    print("="*80)
    print("STEP 1: EXTRACTION")
    print("="*80)
    
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
            print(f"[*] First 3 items:")
            for i, item in enumerate(items[:3]):
                text = item['data'].get('text', '')[:50]
                print(f"    {i+1}. '{text}' bbox={item['bbox']}")
    
    # Check database
    print("\n" + "="*80)
    print("STEP 2: DATABASE CHECK")
    print("="*80)
    
    db = SessionLocal()
    try:
        # Check if any dokumen exists
        doc_count = db.query(Dokumen).count()
        print(f"[*] Total documents in database: {doc_count}")
        
        if doc_count == 0:
            print("[!] NO DOCUMENTS IN DATABASE!")
            print("[!] Alignment needs dokumen_id with DokumenElemen data")
            return
        
        # Get first document
        doc = db.query(Dokumen).first()
        doc_id = doc.dokumen_id
        print(f"[OK] Using document ID: {doc_id}")
        print(f"    Name: {doc.dokumen_nama}")
        
        # Check sections
        sections = db.query(DokumenSection).filter(
            DokumenSection.dsec_ref_tipe == 'dokumen',
            DokumenSection.dsec_ref_id == doc_id
        ).all()
        print(f"[*] Sections: {len(sections)}")
        
        if not sections:
            print("[!] NO SECTIONS!")
            return
        
        # Check parts
        section_ids = [s.dsec_id for s in sections]
        parts = db.query(DokumenPart).filter(DokumenPart.dsec_id.in_(section_ids)).all()
        print(f"[*] Parts: {len(parts)}")
        
        body_parts = [p for p in parts if p.dpart_type == 'body']
        print(f"[*] Body parts: {len(body_parts)}")
        
        if not body_parts:
            print("[!] NO BODY PARTS!")
            return
        
        # Check elements
        part_ids = [p.dpart_id for p in body_parts]
        elements = db.query(DokumenElemen).filter(DokumenElemen.dpart_id.in_(part_ids)).all()
        print(f"[*] Elements: {len(elements)}")
        
        if not elements:
            print("[!] NO DOKUMEN ELEMEN!")
            print("[!] Alignment cannot work without DokumenElemen data")
            return
        
        print(f"[OK] Found {len(elements)} elements to align")
        print(f"[*] First 3 elements:")
        for i, elem in enumerate(elements[:3]):
            text = elem.delemen_text[:50] if elem.delemen_text else '[no text]'
            print(f"    {i+1}. ID={elem.delemen_id} Type={elem.delemen_type} Text='{text}'")
        
        # Try alignment
        print("\n" + "="*80)
        print("STEP 3: ALIGNMENT")
        print("="*80)
        
        alignment_service = AlignmentService()
        
        print(f"[*] Running alignment for doc_id={doc_id}...")
        alignment_results = alignment_service.align_document(extraction_results, doc_id)
        
        print(f"[OK] Alignment completed")
        print(f"[*] Results for {len(alignment_results)} pages:")
        
        for result in alignment_results:
            page = result.get('page')
            success = result.get('success')
            alignments = result.get('alignments', [])
            unaligned = result.get('unaligned_pdf_units', [])
            
            print(f"\n    Page {page}:")
            print(f"      Success: {success}")
            print(f"      Alignments: {len(alignments)}")
            print(f"      Unaligned: {len(unaligned)}")
            
            if alignments:
                print(f"      First alignment:")
                first = alignments[0]
                print(f"        element_id: {first.get('element_id')}")
                print(f"        merged_bbox: {first.get('merged_bbox')}")
                print(f"        matched_units: {len(first.get('matched_pdf_units', []))}")
            else:
                print(f"      [!] NO ALIGNMENTS FOUND!")
                print(f"      [!] This means text in PDF doesn't match DokumenElemen text")
        
        # Save results
        output_file = 'debug_alignment_result.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(alignment_results, f, indent=2, ensure_ascii=False)
        print(f"\n[OK] Saved results to: {output_file}")
        
    except Exception as e:
        print(f"[X] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()

if __name__ == "__main__":
    debug_alignment()
