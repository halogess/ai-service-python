"""
Test structural labeling rule for code titles.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.merging_extraction_service import MergingExtractionService


def test_section_header_before_code_is_judul_kode():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'section_header',
            'text': 'Segmen Program 5.11 Script CaptureAndRecognizeCoroutine',
            'bbox': [60, 100, 520, 130],
            'element_id': None,
            'element_type': 'paragraph'
        },
        {
            'label': 'code',
            'text': '01: isProcessing = true;',
            'bbox': [60, 140, 520, 160],
            'element_id': None,
            'element_type': 'paragraph'
        },
        {
            'label': 'code',
            'text': '02: UpdateStatus("Mengambil gambar...");',
            'bbox': [60, 165, 520, 185],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(
        db=None,
        fused_results=fused_results,
        structural_state=state
    )

    assert fused_results[0].get('dev_label_struktural') == 'judul_kode'
    assert fused_results[1].get('dev_label_struktural') == 'kode'
    assert fused_results[2].get('dev_label_struktural') == 'kode'


def test_regular_section_header_not_forced_to_judul_kode():
    service = MergingExtractionService()
    state = service._new_structural_label_state()

    fused_results = [
        {
            'label': 'section_header',
            'text': 'Pendahuluan',
            'bbox': [60, 100, 300, 130],
            'element_id': None,
            'element_type': 'paragraph'
        },
        {
            'label': 'text',
            'text': 'Ini adalah paragraf biasa, bukan blok kode.',
            'bbox': [60, 140, 520, 180],
            'element_id': None,
            'element_type': 'paragraph'
        }
    ]

    service._apply_structural_labels(
        db=None,
        fused_results=fused_results,
        structural_state=state
    )

    assert fused_results[0].get('dev_label_struktural') != 'judul_kode'
    assert fused_results[0].get('dev_label_struktural') == 'section_header'


if __name__ == '__main__':
    test_section_header_before_code_is_judul_kode()
    test_regular_section_header_not_forced_to_judul_kode()
    print("test_structural_code_title: OK")
