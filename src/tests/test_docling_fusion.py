"""
Test for DoclingFusionService
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.docling_fusion_service import DoclingFusionService

def test_fusion_service():
    """Test the fusion service with mock data"""
    print("=" * 60)
    print("TESTING DOCLING FUSION SERVICE")
    print("=" * 60)
    
    # Create service with section data
    section_data = {
        'page_height_pt': 842,
        'margin_top_pt': 72,
        'margin_bottom_pt': 72
    }
    service = DoclingFusionService(section_data)
    
    # Test 1: Overlap calculation
    print("\n[1] Testing overlap calculation...")
    bbox1 = [100, 100, 200, 200]  # 100x100 = 10000 area
    bbox2 = [120, 120, 220, 220]  # 80x80 = 6400 intersection, 64% of smaller box
    overlap = service.calculate_overlap(bbox1, bbox2)
    print(f"    Overlap between {bbox1} and {bbox2}: {overlap:.2%}")
    assert overlap > 0.3, f"Expected >30% overlap, got {overlap:.2%}"
    print("    ✓ Overlap calculation works")
    
    # Test 2: Margin zone detection
    print("\n[2] Testing margin zone detection...")
    header_bbox = [100, 20, 200, 50]  # Y center = 35, in header zone
    body_bbox = [100, 200, 200, 300]  # Y center = 250, in body
    footer_bbox = [100, 790, 200, 830]  # Y center = 810, in footer zone
    
    assert service.get_bbox_margin_zone(header_bbox) == 'header'
    assert service.get_bbox_margin_zone(body_bbox) is None
    assert service.get_bbox_margin_zone(footer_bbox) == 'footer'
    print("    ✓ Margin zone detection works")
    
    # Test 3: Label correction
    print("\n[3] Testing label correction...")
    # page_header in body should become 'text'
    corrected = service.correct_header_footer_label('page_header', body_bbox)
    assert corrected == 'text', f"Expected 'text', got '{corrected}'"
    # page_header in header zone should stay
    corrected = service.correct_header_footer_label('page_header', header_bbox)
    assert corrected == 'page_header'
    print("    ✓ Label correction works")
    
    # Test 4: Full fusion
    print("\n[4] Testing full fusion...")
    alignments = [
        {
            'element_id': 1,
            'element_sequence': 1,
            'element_type': 'paragraph',
            'merged_bbox': [100, 200, 400, 250],
            'matched_pdf_units': [
                {'item_type': 'group', 'text': 'Hello', 'bbox': [100, 200, 150, 250]},
                {'item_type': 'group', 'text': 'World', 'bbox': [160, 200, 210, 250]}
            ]
        },
        {
            'element_id': 2,
            'element_sequence': 2,
            'element_type': 'table',
            'is_table': True,
            'merged_bbox': [100, 300, 400, 500],
            'cells': [
                {'merged_bbox': [100, 300, 200, 350], 'text': 'Cell 1'}
            ],
            'matched_pdf_units': []
        }
    ]
    
    docling_predictions = [
        {'label': 'paragraph', 'bbox': [90, 190, 410, 260], 'confidence': 0.95},
        {'label': 'table', 'bbox': [95, 295, 410, 510], 'confidence': 0.88}
    ]
    
    header_footer_units = [
        {'bbox': [100, 20, 200, 50], 'text': 'Page 1', 'zone': 'header'}
    ]
    
    fused = service.fuse_alignments_with_docling(alignments, header_footer_units, docling_predictions)
    
    print(f"    Fused results: {len(fused)} items")
    for i, f in enumerate(fused):
        print(f"      [{i+1}] label={f.get('label')}, overlap={f.get('overlap', 0):.2%}, elem_id={f.get('element_id')}")
    
    # Should have: 1 paragraph + 1 table cell + 1 header
    assert len(fused) >= 2, f"Expected at least 2 fused results, got {len(fused)}"
    print("    ✓ Full fusion works")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == '__main__':
    test_fusion_service()
