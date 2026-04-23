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


class _AlignmentStub:
    @staticmethod
    def _extract_text_from_json_tree(tree):
        if isinstance(tree, dict):
            if isinstance(tree.get("content"), list):
                values = []
                for item in tree["content"]:
                    if isinstance(item, dict) and isinstance(item.get("content"), list):
                        for child in item["content"]:
                            if isinstance(child, dict) and child.get("type") == "text":
                                values.append(str(child.get("value") or child.get("text") or ""))
                    elif isinstance(item, dict) and item.get("type") == "text":
                        values.append(str(item.get("value") or item.get("text") or ""))
                return " ".join(value for value in values if value)
        return ""

    @staticmethod
    def _normalize_text(value):
        return " ".join(str(value or "").lower().split())


class _QueryStub:
    def __init__(self, all_result=None):
        self.filters = []
        self.all_result = list(all_result or [])

    def filter(self, *criteria):
        self.filters.extend(criteria)
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def all(self):
        return list(self.all_result)


class _DbStub:
    def __init__(self, query_results=None):
        self.query_results = list(query_results or [])
        self.queries = []

    def query(self, *_args, **_kwargs):
        config = self.query_results.pop(0) if self.query_results else {}
        query = _QueryStub(**config)
        self.queries.append(query)
        return query


class NoteRefScopeTests(unittest.TestCase):
    def setUp(self):
        self.service = MergingExtractionService.__new__(MergingExtractionService)
        self.service.alignment_service = _AlignmentStub()
        self.service._append_footnote_log = lambda *_args, **_kwargs: None

    def _get_filter_sql(self, query_stub, column_name):
        for criterion in query_stub.filters:
            sql = str(criterion)
            if column_name in sql:
                return sql
        self.fail(f"{column_name} filter not found")

    def test_load_note_targets_supports_bab(self):
        db = _DbStub(
            query_results=[
                {
                    "all_result": [
                        type(
                            "Row",
                            (),
                            {
                                "dnote_id": 10,
                                "dnote_kind": "footnote",
                                "dnote_type": "normal",
                                "dnote_json_tree": '{"content":[{"type":"paragraph","dfp_id":99,"content":[{"type":"text","value":"Catatan bab","dftx_id":12}]}]}',
                            },
                        )()
                    ]
                }
            ]
        )

        targets = self.service._load_note_targets_for_ref(db, "bab", 77)

        self.assertEqual(1, len(targets))
        self.assertEqual(10, targets[0]["element_id"])
        self.assertEqual("note", targets[0]["target_kind"])
        self.assertEqual("Catatan bab", targets[0]["text"])
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_tipe"))
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_id"))

    def test_load_note_targets_supports_aturan(self):
        db = _DbStub(
            query_results=[
                {
                    "all_result": [
                        type(
                            "Row",
                            (),
                            {
                                "dnote_id": 11,
                                "dnote_kind": "footnote",
                                "dnote_type": "normal",
                                "dnote_json_tree": '{"content":[{"type":"paragraph","dfp_id":199,"content":[{"type":"text","value":"Catatan aturan","dftx_id":21}]}]}',
                            },
                        )()
                    ]
                }
            ]
        )

        targets = self.service._load_note_targets_for_ref(db, "aturan", 88)

        self.assertEqual(1, len(targets))
        self.assertEqual("Catatan aturan", targets[0]["text"])
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_tipe"))
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_id"))

    def test_assign_docling_footnotes_queries_ref_scoped_notes(self):
        db = _DbStub(query_results=[{"all_result": []}])

        filtered_predictions, entries = self.service._assign_docling_footnotes(
            db,
            "bab",
            123,
            4,
            [{"label": "footnote", "bbox": [0, 0, 10, 10], "text": "Catatan"}],
            [{"docling_idx": 0, "docling_pred": {"label": "footnote", "bbox": [0, 0, 10, 10], "text": "Catatan"}, "bbox": [0, 0, 10, 10], "text": "Catatan"}],
        )

        self.assertEqual(
            [{"label": "footnote", "bbox": [0, 0, 10, 10], "text": "Catatan"}],
            filtered_predictions,
        )
        self.assertEqual([], entries)
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_tipe"))
        self.assertIn("=", self._get_filter_sql(db.queries[0], "dnote_ref_id"))


if __name__ == "__main__":
    unittest.main()
