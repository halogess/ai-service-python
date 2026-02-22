"""
Test for DoclingFusionService.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.docling_fusion_service import DoclingFusionService


def test_fusion_service():
    """Test fusion behavior with mock data."""
    print("=" * 60)
    print("TESTING DOCLING FUSION SERVICE")
    print("=" * 60)

    section_data = {
        'page_height_pt': 842,
        'margin_top_pt': 72,
        'margin_bottom_pt': 72
    }
    service = DoclingFusionService(section_data)

    print("\n[1] Testing overlap calculation...")
    bbox1 = [100, 100, 200, 200]
    bbox2 = [120, 120, 220, 220]
    overlap = service.calculate_overlap(bbox1, bbox2)
    print(f"    Overlap between {bbox1} and {bbox2}: {overlap:.2%}")
    assert overlap > 0.3, f"Expected >30% overlap, got {overlap:.2%}"
    print("    [OK] Overlap calculation works")

    print("\n[2] Testing margin zone detection...")
    header_bbox = [100, 20, 200, 50]
    body_bbox = [100, 200, 200, 300]
    footer_bbox = [100, 790, 200, 830]
    assert service.get_bbox_margin_zone(header_bbox) == 'header'
    assert service.get_bbox_margin_zone(body_bbox) is None
    assert service.get_bbox_margin_zone(footer_bbox) == 'footer'
    print("    [OK] Margin zone detection works")

    print("\n[3] Testing label correction...")
    corrected = service.correct_header_footer_label('page_header', body_bbox)
    assert corrected == 'text', f"Expected 'text', got '{corrected}'"
    corrected = service.correct_header_footer_label('page_header', header_bbox)
    assert corrected == 'page_header'
    print("    [OK] Label correction works")

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
    for i, item in enumerate(fused):
        print(
            f"      [{i+1}] label={item.get('label')}, "
            f"overlap={item.get('overlap', 0):.2%}, elem_id={item.get('element_id')}"
        )
    assert len(fused) >= 2, f"Expected at least 2 fused results, got {len(fused)}"
    print("    [OK] Full fusion works")

    print("\n[5] Testing table-to-code relabel for monospace non-table...")
    code_alignments = [
        {
            'element_id': 10,
            'element_sequence': 10,
            'element_type': 'paragraph',
            'merged_bbox': [100, 350, 520, 470],
            'matched_pdf_units': [
                {
                    'item_type': 'group',
                    'text': 'for i in range(10): print(i)',
                    'bbox': [100, 350, 520, 370]
                }
            ],
            'is_code_font': True,
            'is_code_style': True,
            'is_code_like_openxml': True,
            'font_families': ['courier new'],
            'style_ids': ['sttssegmenprogramcontent'],
            'is_text_part': False,
            'is_image_part': False
        }
    ]
    code_docling_predictions = [
        {'label': 'table', 'bbox': [95, 345, 525, 475], 'confidence': 0.9}
    ]
    code_fused = service.fuse_alignments_with_docling(code_alignments, [], code_docling_predictions)
    assert code_fused, "Expected fused output for code relabel test"
    assert code_fused[0].get('label') == 'code', f"Expected 'code', got {code_fused[0].get('label')}"
    print("    [OK] Table-to-code relabel works for monospace non-table")

    print("\n[6] Testing real table stays table...")
    real_table_alignments = [
        {
            'element_id': 11,
            'element_sequence': 11,
            'element_type': 'table',
            'is_table': True,
            'cells': [
                {
                    'row': 0,
                    'col': 0,
                    'text': 'col_a',
                    'merged_bbox': [100, 500, 220, 530],
                    'matched_pdf_units': [
                        {'item_type': 'table', 'text': 'col_a', 'bbox': [100, 500, 220, 530]}
                    ]
                }
            ]
        }
    ]
    real_table_preds = [{'label': 'table', 'bbox': [95, 495, 225, 535], 'confidence': 0.95}]
    table_fused = service.fuse_alignments_with_docling(real_table_alignments, [], real_table_preds)
    assert table_fused, "Expected fused output for real table test"
    assert table_fused[0].get('label') == 'table', f"Expected 'table', got {table_fused[0].get('label')}"
    print("    [OK] Real table remains table")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_fusion_service()
