"""
Test structural labeling rule for figure panel markers like (a), (b), (c).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.merging_extraction_service import MergingExtractionService


def test_panel_marker_after_picture_becomes_caption_gambar():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'picture',
            'text': '',
            'bbox': [60, 100, 520, 320],
            'element_id': None,
            'element_type': 'figure'
        },
        {
            'label': 'section_header',
            'docling_label': 'section_header',
            'text': '(a)',
            'bbox': [80, 325, 120, 345],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(db=None, fused_results=fused_results, structural_state=state)

    assert fused_results[0].get('dev_label_struktural') == 'gambar'
    assert fused_results[1].get('dev_label_struktural') == 'caption_gambar'


def test_panel_marker_before_picture_becomes_caption_gambar():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'text',
            'docling_label': 'title',
            'text': '(b)',
            'bbox': [80, 200, 120, 220],
            'element_id': None,
            'element_type': 'paragraph'
        },
        {
            'label': 'picture',
            'text': '',
            'bbox': [60, 230, 520, 480],
            'element_id': None,
            'element_type': 'figure'
        }
    ]

    service._apply_structural_labels(db=None, fused_results=fused_results, structural_state=state)

    assert fused_results[0].get('dev_label_struktural') == 'caption_gambar'
    assert fused_results[1].get('dev_label_struktural') == 'gambar'


def test_panel_marker_without_adjacent_picture_is_not_caption_gambar():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'text',
            'text': '(c)',
            'bbox': [80, 200, 120, 220],
            'element_id': None,
            'element_type': 'paragraph'
        },
        {
            'label': 'paragraph',
            'text': 'Penjelasan biasa',
            'bbox': [60, 230, 520, 280],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(db=None, fused_results=fused_results, structural_state=state)

    assert fused_results[0].get('dev_label_struktural') != 'caption_gambar'


if __name__ == '__main__':
    test_panel_marker_after_picture_becomes_caption_gambar()
    test_panel_marker_before_picture_becomes_caption_gambar()
    test_panel_marker_without_adjacent_picture_is_not_caption_gambar()
    print("test_structural_figure_panel_caption: OK")
