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
SCRIPTS_DIR = ROOT_DIR / "scripts"
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_visual_alignment_lib.helpers import (
    is_valid_same_page_chart_caption_pair,
    is_valid_same_page_table_claim_set,
)


class EvaluateVisualAlignmentHelperTests(unittest.TestCase):
    def test_chart_caption_pair_accepts_picture_followed_by_caption(self):
        rows = [
            {"label": "picture", "bbox": [10, 10, 80, 60]},
            {"label": "caption", "bbox": [12, 64, 78, 78], "text": "Gambar 2.1 Hasil"},
        ]

        self.assertTrue(is_valid_same_page_chart_caption_pair(rows))

    def test_chart_caption_pair_rejects_large_gap(self):
        rows = [
            {"label": "picture", "bbox": [10, 10, 80, 60]},
            {"label": "caption", "bbox": [12, 180, 78, 194], "text": "Gambar 2.1 Hasil"},
        ]

        self.assertFalse(is_valid_same_page_chart_caption_pair(rows))

    def test_table_claim_set_requires_all_rows_to_be_table_like(self):
        valid_rows = [
            {"label": "table", "bbox": [0, 0, 10, 10]},
            {"label": "text", "bbox": [0, 10, 10, 20], "has_table_units": True},
        ]
        invalid_rows = [
            {"label": "table", "bbox": [0, 0, 10, 10]},
            {"label": "text", "bbox": [0, 10, 10, 20], "has_table_units": False},
        ]

        self.assertTrue(is_valid_same_page_table_claim_set(valid_rows))
        self.assertFalse(is_valid_same_page_table_claim_set(invalid_rows))


if __name__ == "__main__":
    unittest.main()
