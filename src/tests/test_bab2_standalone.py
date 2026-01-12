"""
Test alignment dengan bab2.pdf TANPA database.
Menggunakan mock OpenXML units untuk test alignment logic.
"""

import os
import sys
import json
from unittest.mock import MagicMock

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Mock database before import
sys.modules['database'] = MagicMock()
mock_models = MagicMock()
sys.modules['models'] = mock_models

from services.pdf_extraction_service import PDFExtractor
from services.alignment_service import AlignmentService

def test_bab2_alignment():
    """Test bab2.pdf extraction and alignment without database"""
    
    pdf_path = os.path.join(os.path.dirname(__file__), '..', '..', 'bab2.pdf')
    if not os.path.exists(pdf_path):
        print(f"[X] bab2.pdf tidak ditemukan: {pdf_path}")
        return
    
    print(f"[OK] Found: {pdf_path}\n")
    
    # Step 1: Extract PDF
    print("=" * 60)
    print("STEP 1: PDF EXTRACTION")
    print("=" * 60)
    
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
        
        for img in page_data.get('page_images', []):
            items.append({
                'type': 'image',
                'bbox': img.get('bbox'),
                'data': {}
            })
        
        items.sort(key=lambda x: (x['bbox'][1] if x.get('bbox') else 0, x['bbox'][0] if x.get('bbox') else 0))
        
        print(f"[OK] Extracted {len(items)} items")
        print(f"\n[*] First 5 items:")
        for i, item in enumerate(items[:5]):
            text = item['data'].get('text', '')[:40] if item['data'].get('text') else f"[{item['type']}]"
            print(f"    {i+1}. Type={item['type']}, Text='{text}'")
        
        page_width = page.rect.width
        page_height = page.rect.height
    
    # Step 2: Test Flattening
    print("\n" + "=" * 60)
    print("STEP 2: FLATTEN EXTRACTION ITEMS")
    print("=" * 60)
    
    service = AlignmentService()
    pdf_units = service._flatten_extraction_items(items)
    
    print(f"[OK] Created {len(pdf_units)} PDF units")
    print(f"\n[*] First 5 PDF units:")
    for i, unit in enumerate(pdf_units[:5]):
        text = unit.get('text', '')[:40]
        norm = unit.get('text_normalized', '')[:40]
        print(f"    {i+1}. text='{text}' norm='{norm}'")
    
    # Step 3: Create Mock OpenXML Units (simulate DokumenElemen)
    print("\n" + "=" * 60)
    print("STEP 3: CREATE MOCK OPENXML UNITS")
    print("=" * 60)
    
    # Take sample text from PDF and use same text as mock OpenXML
    mock_openxml_units = []
    for i, unit in enumerate(pdf_units[:10]):  # First 10 units
        mock_openxml_units.append({
            'unit_id': f'ox_{i}',
            'elem_id': i + 1,
            'elem_seq': i + 1,
            'elem_type': 'paragraph',
            'text': unit['text'],
            'text_normalized': unit['text_normalized'],
            'is_cell': False,
            'has_shape': unit.get('item_type') == 'shape'
        })
    
    print(f"[OK] Created {len(mock_openxml_units)} mock OpenXML units")
    
    # Step 4: Run Alignment
    print("\n" + "=" * 60)
    print("STEP 4: RUN ALIGNMENT")
    print("=" * 60)
    
    result = service._perform_two_pass_alignment(pdf_units, mock_openxml_units, min_openxml_idx=0)
    
    alignments = result.get('final_alignments', [])
    unaligned_pdf = result.get('unaligned_pdf_indices', [])
    max_idx = result.get('max_openxml_idx', 0)
    
    print(f"[OK] Alignment completed!")
    print(f"    Alignments: {len(alignments)}")
    print(f"    Unaligned PDF units: {len(unaligned_pdf)}")
    print(f"    Max OpenXML idx: {max_idx}")
    
    if alignments:
        print(f"\n[*] First 3 alignments:")
        for i, align in enumerate(alignments[:3]):
            elem_id = align.get('element_id')
            matched = align.get('matched_pdf_units', [])
            bbox = align.get('merged_bbox')
            print(f"    {i+1}. Element {elem_id}: {len(matched)} matched units, bbox={bbox}")
    else:
        print("\n[!] NO ALIGNMENTS - checking why...")
        print(f"    PDF units count: {len(pdf_units)}")
        print(f"    OpenXML units count: {len(mock_openxml_units)}")
        
        if pdf_units and mock_openxml_units:
            print(f"\n    PDF[0] text_normalized: '{pdf_units[0]['text_normalized'][:50]}'")
            print(f"    OX[0] text_normalized: '{mock_openxml_units[0]['text_normalized'][:50]}'")
    
    # Step 5: Save Results
    output_dir = 'test_output_bab2'
    os.makedirs(output_dir, exist_ok=True)
    
    result_file = os.path.join(output_dir, 'alignment_test_result.json')
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'pdf_units_count': len(pdf_units),
            'openxml_units_count': len(mock_openxml_units),
            'alignments_count': len(alignments),
            'unaligned_count': len(unaligned_pdf),
            'alignments': alignments[:5],  # First 5 only
            'pdf_units_sample': pdf_units[:5],
            'openxml_units_sample': mock_openxml_units[:5]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Results saved to: {result_file}")
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
    
    if len(alignments) > 0:
        print("✓ ALIGNMENT LOGIC WORKS!")
    else:
        print("✗ ALIGNMENT FAILED - check debug info above")

if __name__ == "__main__":
    test_bab2_alignment()
