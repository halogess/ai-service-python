"""
Smoke regression for reference routing:
- antrian_tipe -> (ref_tipe, ref_id)
- visual persistence uses dev_ref_tipe/dev_ref_id
- process_document routes correctly for dokumen and bab
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Bab, Dokumen
from services.antrian_service import AntrianService
from services.merging_extraction_service import MergingExtractionService


class DummyPage:
    rect = SimpleNamespace(width=595, height=842)


class DummyExtractor:
    def __init__(self, _pdf_path):
        self.page_count = 1

    def open(self):
        return None

    def close(self):
        return None

    def get_page(self, _idx):
        return DummyPage()

    def extract_merging_data(self, _idx):
        return {
            'char_groups': [],
            'basic_tables': [],
            'hline_tables': [],
            'shapes': [],
            'page_images': []
        }


def _make_query(model):
    q = MagicMock()
    if model is Dokumen:
        q.get.return_value = SimpleNamespace(dokumen_pdf_path='dummy/doc.pdf')
    elif model is Bab:
        q.get.return_value = SimpleNamespace(bab_pdf_path='dummy/bab.pdf')
    else:
        q.get.return_value = None
    return q


def test_reference_routing():
    service = AntrianService(MagicMock())
    assert service.get_task_reference(
        SimpleNamespace(antrian_tipe='dokumen', dokumen_id=101, bab_id=None, antrian_id=1)
    ) == ('dokumen', 101)
    assert service.get_task_reference(
        SimpleNamespace(antrian_tipe='buku', dokumen_id=None, bab_id=202, antrian_id=2)
    ) == ('bab', 202)


def test_visual_record_ref_fields():
    service = MergingExtractionService()
    db = MagicMock()
    db.query.return_value.filter.return_value.delete.return_value = None

    inserted = []
    db.add.side_effect = lambda obj: inserted.append(obj)

    service._replace_visual_records(
        db,
        'bab',
        202,
        1,
        [{'bbox': [1, 2, 3, 4], 'element_id': 9, 'label': 'text', 'text': 'x'}],
        structural_state=service._new_structural_label_state()
    )

    assert inserted
    assert inserted[0].dev_ref_tipe == 'bab'
    assert inserted[0].dev_ref_id == 202


def test_process_document_bab_and_dokumen_routing_with_legacy_alias():
    mock_session = MagicMock()
    mock_session.query.side_effect = _make_query

    with patch('services.merging_extraction_service.SessionLocal', return_value=mock_session), \
         patch('services.merging_extraction_service.PDFExtractor', DummyExtractor):
        service = MergingExtractionService()
        service.docling_service.classify_pdf = MagicMock(return_value={'success': True, 'predictions_by_page': {'1': []}})
        service.docling_service.classify_document = MagicMock(return_value={'success': True, 'predictions_by_page': {'1': []}})
        service.alignment_service.align = MagicMock(return_value={
            'success': True,
            'final_alignments': [],
            'header_footer_units': [],
            'page_debug': {},
            'unaligned_pdf_units': [],
            'max_openxml_idx': 0
        })
        service._save_alignment_results = MagicMock(return_value=[])
        service._replace_visual_records = MagicMock()

        # canonical bab path
        assert service.process_document(202, ref_tipe='bab', generate_visualizations=False, save_to_db=True)
        service.docling_service.classify_pdf.assert_called()
        assert service.alignment_service.align.call_args.kwargs.get('ref_tipe') == 'bab'
        call_args = service._replace_visual_records.call_args.args
        assert call_args[1] == 'bab'
        assert call_args[2] == 202

        # legacy alias buku -> canonical bab
        service.docling_service.classify_pdf.reset_mock()
        service._replace_visual_records.reset_mock()
        assert service.process_document(202, ref_tipe='buku', generate_visualizations=False, save_to_db=True)
        service.docling_service.classify_pdf.assert_called()
        assert service.alignment_service.align.call_args.kwargs.get('ref_tipe') == 'bab'
        call_args = service._replace_visual_records.call_args.args
        assert call_args[1] == 'bab'
        assert call_args[2] == 202

        # dokumen path
        service.docling_service.classify_pdf.reset_mock()
        service.docling_service.classify_document.reset_mock()
        service._replace_visual_records.reset_mock()
        assert service.process_document(101, ref_tipe='dokumen', generate_visualizations=False, save_to_db=True)
        service.docling_service.classify_document.assert_called()
        assert service.alignment_service.align.call_args.kwargs.get('ref_tipe') == 'dokumen'
        call_args = service._replace_visual_records.call_args.args
        assert call_args[1] == 'dokumen'
        assert call_args[2] == 101


if __name__ == '__main__':
    test_reference_routing()
    test_visual_record_ref_fields()
    test_process_document_bab_and_dokumen_routing_with_legacy_alias()
    print('test_ref_routing_regression: OK')
