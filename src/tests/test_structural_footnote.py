"""
Test structural labeling rule for footnote outputs from Docling.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.merging_extraction_service import MergingExtractionService


def test_visual_label_footnote_becomes_structural_footnote():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'footnote',
            'text': '1 Ini adalah catatan kaki.',
            'bbox': [60, 700, 520, 740],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(db=None, fused_results=fused_results, structural_state=state)

    assert fused_results[0].get('dev_label_struktural') == 'footnote'


def test_docling_label_footnote_fills_structural_label():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'text',
            'docling_label': 'footnote',
            'text': '2 Catatan kaki dari docling.',
            'bbox': [60, 745, 520, 780],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(db=None, fused_results=fused_results, structural_state=state)

    assert fused_results[0].get('dev_label_struktural') == 'footnote'


if __name__ == '__main__':
    test_visual_label_footnote_becomes_structural_footnote()
    test_docling_label_footnote_fills_structural_label()
    print("test_structural_footnote: OK")
