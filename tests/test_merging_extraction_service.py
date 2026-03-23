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


if __name__ == "__main__":
    unittest.main()
