import copy
import os
import sys
import unittest
from pathlib import Path


os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.merging_extraction_service import MergingExtractionService
from services.alignment_service import AlignmentService
from services.docling_fusion_service import DoclingFusionService


class CollapseTableVisualResultsTests(unittest.TestCase):
    def setUp(self):
        self.service = MergingExtractionService.__new__(MergingExtractionService)

    def collapse(self, fused_results):
        payload = copy.deepcopy(fused_results)
        return self.service._collapse_table_visual_results_for_page(payload)

    def test_same_page_table_rows_collapse_to_one_visual(self):
        fused_results = [
            {
                "bbox": [10, 10, 40, 20],
                "label": "table",
                "docling_label": "table",
                "text": "Header",
                "source": "cell",
                "element_id": 101,
                "element_type": "table",
                "element_sequence": 10,
                "openxml_idx": 10,
                "has_table_units": True,
                "merged_count": 1,
                "overlap": 0.42,
                "alignment_confidence": 0.61,
                "block_order": 3,
            },
            {
                "bbox": [10, 20, 55, 50],
                "label": "table",
                "docling_label": "table",
                "text": "Body",
                "source": "cell",
                "element_id": 101,
                "element_type": "table",
                "element_sequence": 10,
                "openxml_idx": 10,
                "has_table_units": True,
                "merged_count": 2,
                "overlap": 0.77,
                "alignment_confidence": 0.88,
                "block_order": 4,
            },
            {
                "bbox": [60, 60, 90, 80],
                "label": "text",
                "docling_label": "text",
                "text": "Outside",
                "source": "alignment",
                "element_id": 202,
                "element_type": "paragraph",
                "has_table_units": False,
            },
        ]

        collapsed = self.collapse(fused_results)

        self.assertEqual(2, len(collapsed))
        table_result = collapsed[0]
        self.assertEqual("table", table_result["label"])
        self.assertEqual("table", table_result["docling_label"])
        self.assertEqual("table_page_merge", table_result["source"])
        self.assertEqual([10.0, 10.0, 55.0, 50.0], table_result["bbox"])
        self.assertEqual("Header\nBody", table_result["text"])
        self.assertEqual(3, table_result["merged_count"])
        self.assertEqual(0.77, table_result["overlap"])
        self.assertEqual(0.88, table_result["alignment_confidence"])
        self.assertTrue(table_result["has_table_units"])
        self.assertEqual("tabel", table_result["dev_label_struktural"])
        self.assertEqual("Outside", collapsed[1]["text"])

    def test_caption_with_same_element_id_is_not_collapsed_into_table(self):
        fused_results = [
            {
                "bbox": [10, 10, 40, 20],
                "label": "table",
                "docling_label": "table",
                "text": "Row 1",
                "source": "cell",
                "element_id": 303,
                "element_type": "table",
                "has_table_units": True,
            },
            {
                "bbox": [10, 20, 40, 30],
                "label": "caption",
                "docling_label": "caption",
                "text": "Tabel 1.1 Caption",
                "source": "alignment",
                "element_id": 303,
                "element_type": "caption",
                "has_table_units": False,
            },
        ]

        collapsed = self.collapse(fused_results)

        self.assertEqual(2, len(collapsed))
        self.assertEqual("table", collapsed[0]["label"])
        self.assertEqual("caption", collapsed[1]["label"])
        self.assertEqual("Tabel 1.1 Caption", collapsed[1]["text"])

    def test_orphan_table_like_rows_without_element_id_are_not_collapsed(self):
        fused_results = [
            {
                "bbox": [5, 5, 25, 15],
                "label": "table",
                "docling_label": "table",
                "text": "Loose A",
                "source": "cell",
                "element_id": None,
                "element_type": "table",
                "has_table_units": True,
            },
            {
                "bbox": [5, 15, 25, 25],
                "label": "table",
                "docling_label": "table",
                "text": "Loose B",
                "source": "cell",
                "element_id": None,
                "element_type": "table",
                "has_table_units": True,
            },
        ]

        collapsed = self.collapse(fused_results)

        self.assertEqual(2, len(collapsed))
        self.assertEqual("Loose A", collapsed[0]["text"])
        self.assertEqual("Loose B", collapsed[1]["text"])

    def test_collapse_is_page_scoped_when_run_independently_per_page(self):
        page_one = [
            {
                "bbox": [0, 0, 10, 10],
                "label": "table",
                "docling_label": "table",
                "text": "P1",
                "source": "cell",
                "element_id": 404,
                "element_type": "table",
                "has_table_units": True,
            },
            {
                "bbox": [0, 10, 10, 20],
                "label": "table",
                "docling_label": "table",
                "text": "P1B",
                "source": "cell",
                "element_id": 404,
                "element_type": "table",
                "has_table_units": True,
            },
        ]
        page_two = [
            {
                "bbox": [0, 0, 12, 12],
                "label": "table",
                "docling_label": "table",
                "text": "P2",
                "source": "cell",
                "element_id": 404,
                "element_type": "table",
                "has_table_units": True,
            },
            {
                "bbox": [0, 12, 12, 24],
                "label": "table",
                "docling_label": "table",
                "text": "P2B",
                "source": "cell",
                "element_id": 404,
                "element_type": "table",
                "has_table_units": True,
            },
        ]

        collapsed_page_one = self.collapse(page_one)
        collapsed_page_two = self.collapse(page_two)

        self.assertEqual(1, len(collapsed_page_one))
        self.assertEqual(1, len(collapsed_page_two))
        self.assertEqual([0.0, 0.0, 10.0, 20.0], collapsed_page_one[0]["bbox"])
        self.assertEqual([0.0, 0.0, 12.0, 24.0], collapsed_page_two[0]["bbox"])


class CaptionRegressionGuardTests(unittest.TestCase):
    def test_caption_continuation_rejects_long_section_header(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        service.alignment_service = AlignmentService()

        self.assertFalse(
            service._is_caption_continuation_candidate(
                {
                    "label": "section_header",
                    "text": "Hasil Penguraian Data Kemudahan Aplikasi (a) Navigasi (b) Kontrol",
                }
            )
        )

    def test_synthetic_caption_bbox_is_not_widened_against_alignment_bbox(self):
        service = MergingExtractionService.__new__(MergingExtractionService)

        class _AlignmentStub:
            @staticmethod
            def _merge_bboxes(bboxes):
                xs0 = [bbox[0] for bbox in bboxes]
                ys0 = [bbox[1] for bbox in bboxes]
                xs1 = [bbox[2] for bbox in bboxes]
                ys1 = [bbox[3] for bbox in bboxes]
                return [min(xs0), min(ys0), max(xs1), max(ys1)]

        service.alignment_service = _AlignmentStub()
        fused_results = [
            {
                "source": "alignment",
                "element_id": 10,
                "label": "caption",
                "bbox": [250.0, 730.0, 372.0, 744.0],
                "repair_reason": "caption_fragment_inherit",
            }
        ]
        alignments = [
            {
                "element_id": 10,
                "merged_bbox": [250.0, 716.0, 372.0, 744.0],
            }
        ]

        service._sync_fused_bboxes_with_alignments(fused_results, alignments)

        self.assertEqual([250.0, 730.0, 372.0, 744.0], fused_results[0]["bbox"])


class AlignmentCaptionFragmentTests(unittest.TestCase):
    def setUp(self):
        self.service = AlignmentService()

    def test_select_caption_fragment_source_prefers_caption_like_neighbor(self):
        prev_alignment = {
            "element_text": "Gambar 3.2",
            "matched_pdf_units": [{"item_type": "shape", "bbox": [0, 0, 10, 10]}],
            "is_chart_caption_text": False,
        }
        next_alignment = {
            "element_text": "Paragraf biasa",
            "matched_pdf_units": [{"item_type": "shape", "bbox": [0, 0, 10, 10]}],
        }

        selected = self.service._select_caption_fragment_source_alignment(
            "Business Process Model",
            prev_alignment,
            next_alignment,
        )

        self.assertIs(selected, prev_alignment)

    def test_redundant_caption_fragment_alignment_is_skipped(self):
        alignments = [
            {
                "element_id": 100,
                "element_sequence": 10,
                "element_type": "paragraph",
                "element_text": "Gambar 3.2 Business Process Model",
                "matched_pdf_units": [{"item_type": "shape", "bbox": [10, 10, 50, 20]}],
                "merged_bbox": [10, 10, 50, 20],
                "is_chart_caption_text": True,
            }
        ]
        openxml_units = [
            {
                "elem_id": 101,
                "elem_seq": 11,
                "elem_type": "paragraph",
                "text": "Business Process Model",
            }
        ]

        rescued, debug = self.service._rescue_fragment_paragraph_alignments(
            openxml_units,
            alignments,
            page_sequence_range=(1, 20),
        )

        self.assertEqual(1, len(rescued))
        self.assertEqual([], debug)


class TextLikeParagraphLabelTests(unittest.TestCase):
    def test_visual_label_canonicalizes_paragraph_to_text(self):
        service = MergingExtractionService.__new__(MergingExtractionService)

        self.assertEqual(
            "text",
            service._get_visual_label({"label": "paragraph"}),
        )
        self.assertEqual(
            "text",
            service._get_visual_label({"docling_label": "paragraph"}),
        )

    def test_structural_labels_treat_paragraph_like_text(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "paragraph",
                "docling_label": "paragraph",
                "text": "Ini paragraf biasa yang berasal dari label paragraph.",
                "element_type": "paragraph",
                "element_id": 123,
            }
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("paragraf", fused_results[0]["dev_label_struktural"])

    def test_docling_fusion_normalizes_paragraph_outputs_to_text(self):
        service = DoclingFusionService()

        self.assertEqual("text", service.fallback_label({"element_type": "paragraph"}))
        self.assertEqual("text", service._resolve_table_prediction_label([]))
        self.assertEqual(
            "text",
            service._resolve_item_docling_label(
                {"label": "paragraph"},
                {
                    "bbox": [0, 0, 10, 10],
                    "element_type": "paragraph",
                },
            ),
        )


class CodeTitlePromotionTests(unittest.TestCase):
    def test_caption_lines_above_code_are_promoted_to_judul_kode(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "caption",
                "text": "Program 2.1.",
                "element_type": "paragraph",
                "element_id": 1,
            },
            {
                "label": "caption",
                "text": "Segmen Progam 2.1 Contoh Sederhana Penggunaan REST API",
                "element_type": "paragraph",
                "element_id": 2,
            },
            {
                "label": "code",
                "text": "1. GET /products",
                "element_type": "table",
                "element_id": 3,
            },
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("judul_kode", fused_results[0]["dev_label_struktural"])
        self.assertEqual("judul_kode", fused_results[1]["dev_label_struktural"])
        self.assertEqual("kode", fused_results[2]["dev_label_struktural"])

    def test_non_code_caption_above_code_is_not_promoted(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "caption",
                "text": "Gambar 2.1 Arsitektur Sistem",
                "bbox": [10, 10, 100, 20],
                "element_type": "paragraph",
                "element_id": 10,
            },
            {
                "label": "code",
                "text": "const value = 42;",
                "element_type": "table",
                "element_id": 11,
            },
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("caption", fused_results[0]["dev_label_struktural"])
        self.assertEqual("kode", fused_results[1]["dev_label_struktural"])

    def test_mid_sentence_algoritma_prose_is_not_promoted_to_judul_kode(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "text",
                "text": (
                    "Stemming merupakan suatu proses untuk menemukan kata dasar, "
                    "dan pada bagian ini dibahas algoritma partisi beserta imbuhan "
                    "(prefixes), sisipan (infixes), dan akhiran (suffixes)."
                ),
                "element_type": "paragraph",
                "element_id": 20,
            },
            {
                "label": "text",
                "text": (
                    "Pada tahun 1999, Abu-Salem dan Al-Omari menerapkan metodologi "
                    "stemming dengan algoritma tertentu untuk sistem pencarian informasi."
                ),
                "element_type": "paragraph",
                "element_id": 21,
            },
            {
                "label": "text",
                "text": (
                    "Pada tahun 2010, peneliti lain menggunakan algoritma Brute Force "
                    "untuk membendung kata Punjabi."
                ),
                "element_type": "paragraph",
                "element_id": 22,
            },
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("paragraf", fused_results[0]["dev_label_struktural"])
        self.assertEqual("paragraf", fused_results[1]["dev_label_struktural"])
        self.assertEqual("paragraf", fused_results[2]["dev_label_struktural"])

    def test_algorithm_heading_at_start_still_promotes_to_judul_kode(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "text",
                "text": "Algoritma Dijkstra",
                "element_type": "paragraph",
                "element_id": 30,
            },
            {
                "label": "code",
                "text": "dist[source] = 0;",
                "element_type": "table",
                "element_id": 31,
            },
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("judul_kode", fused_results[0]["dev_label_struktural"])
        self.assertEqual("kode", fused_results[1]["dev_label_struktural"])


class HeadingLikeListPromotionTests(unittest.TestCase):
    def test_text_heading_with_bullet_is_treated_as_list(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "text",
                "text": "•\t BAB III\t:\tPENGEMBANGAN PROTOTYPING",
                "element_type": "h3",
                "element_id": 20,
            }
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("list_level_1", fused_results[0]["dev_label_struktural"])

    def test_text_paragraph_with_bullet_stays_non_list(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        fused_results = [
            {
                "label": "text",
                "text": "• contoh paragraf biasa",
                "element_type": "paragraph",
                "element_id": 21,
            }
        ]

        service._apply_structural_labels(None, fused_results)

        self.assertEqual("paragraf", fused_results[0]["dev_label_struktural"])

if __name__ == "__main__":
    unittest.main()
