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

    print("\n[5] Testing table prediction maps list element to list_item...")
    list_alignments = [
        {
            'element_id': 10,
            'element_sequence': 10,
            'element_type': 'list-item-1-1',
            'merged_bbox': [100, 350, 520, 390],
            'matched_pdf_units': [
                {'item_type': 'group', 'text': '1. langkah pertama', 'bbox': [100, 350, 520, 390]}
            ],
            'is_text_part': False,
            'is_image_part': False
        }
    ]
    list_docling_predictions = [
        {'label': 'table', 'bbox': [95, 345, 525, 395], 'confidence': 0.9}
    ]
    list_fused = service.fuse_alignments_with_docling(list_alignments, [], list_docling_predictions)
    assert list_fused, "Expected fused output for list relabel test"
    assert list_fused[0].get('label') == 'list_item', f"Expected 'list_item', got {list_fused[0].get('label')}"
    print("    [OK] Table prediction correctly relabeled to list_item")

    print("\n[6] Testing table prediction maps non-table to paragraph...")
    paragraph_alignments = [
        {
            'element_id': 11,
            'element_sequence': 11,
            'element_type': 'paragraph',
            'merged_bbox': [100, 400, 520, 460],
            'matched_pdf_units': [
                {'item_type': 'group', 'text': 'Ini paragraf biasa.', 'bbox': [100, 400, 520, 460]}
            ],
            'is_text_part': False,
            'is_image_part': False
        }
    ]
    paragraph_docling_predictions = [
        {'label': 'table', 'bbox': [95, 395, 525, 465], 'confidence': 0.9}
    ]
    paragraph_fused = service.fuse_alignments_with_docling(paragraph_alignments, [], paragraph_docling_predictions)
    assert paragraph_fused, "Expected fused output for paragraph relabel test"
    assert paragraph_fused[0].get('label') == 'paragraph', f"Expected 'paragraph', got {paragraph_fused[0].get('label')}"
    print("    [OK] Table prediction correctly relabeled to paragraph")

    print("\n[7] Testing table prediction maps image element to picture...")
    image_alignments = [
        {
            'element_id': 12,
            'element_sequence': 12,
            'element_type': 'figure',
            'merged_bbox': [100, 470, 320, 620],
            'matched_pdf_units': [
                {'item_type': 'image', 'text': '', 'bbox': [100, 470, 320, 620]}
            ],
            'is_text_part': False,
            'is_image_part': True
        }
    ]
    image_docling_predictions = [
        {'label': 'table', 'bbox': [95, 465, 325, 625], 'confidence': 0.95}
    ]
    image_fused = service.fuse_alignments_with_docling(image_alignments, [], image_docling_predictions)
    assert image_fused, "Expected fused output for image relabel test"
    assert image_fused[0].get('label') == 'picture', f"Expected 'picture', got {image_fused[0].get('label')}"
    print("    [OK] Table prediction correctly relabeled to picture")

    print("\n[8] Testing real table stays table...")
    real_table_alignments = [
        {
            'element_id': 13,
            'element_sequence': 13,
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

    print("\n[9] Testing chart shape keeps picture label without pdf image...")
    chart_shape_alignments = [
        {
            'element_id': 14,
            'element_sequence': 14,
            'element_type': 'paragraph',
            'merged_bbox': [120, 540, 420, 700],
            'matched_pdf_units': [
                {'item_type': 'shape', 'text': '[IMG]', 'bbox': [120, 540, 420, 700]}
            ],
            'is_text_part': False,
            'is_image_part': False,
            'is_openxml_chart': True
        }
    ]
    chart_shape_preds = [{'label': 'picture', 'bbox': [115, 535, 425, 705], 'confidence': 0.95}]
    chart_shape_fused = service.fuse_alignments_with_docling(chart_shape_alignments, [], chart_shape_preds)
    assert chart_shape_fused, "Expected fused output for chart shape picture test"
    assert chart_shape_fused[0].get('label') == 'picture', (
        f"Expected 'picture', got {chart_shape_fused[0].get('label')}"
    )
    print("    [OK] Chart shape stays picture without has_pdf_image")

    print("\n[10] Testing non-chart shape still downgrades to text without pdf image...")
    non_chart_shape_alignments = [
        {
            'element_id': 15,
            'element_sequence': 15,
            'element_type': 'paragraph',
            'merged_bbox': [120, 710, 420, 780],
            'matched_pdf_units': [
                {'item_type': 'shape', 'text': '[IMG]', 'bbox': [120, 710, 420, 780]}
            ],
            'is_text_part': False,
            'is_image_part': False,
            'is_openxml_chart': False
        }
    ]
    non_chart_shape_preds = [{'label': 'picture', 'bbox': [115, 705, 425, 785], 'confidence': 0.95}]
    non_chart_shape_fused = service.fuse_alignments_with_docling(
        non_chart_shape_alignments,
        [],
        non_chart_shape_preds
    )
    assert non_chart_shape_fused, "Expected fused output for non-chart shape picture test"
    assert non_chart_shape_fused[0].get('label') == 'text', (
        f"Expected 'text', got {non_chart_shape_fused[0].get('label')}"
    )
    print("    [OK] Non-chart shape remains text without has_pdf_image")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == '__main__':
    test_fusion_service()
