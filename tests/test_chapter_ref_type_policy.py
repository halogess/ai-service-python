import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from services.antrian_service import AntrianService
from services.alignment.openxml_sections_mixin import AlignmentOpenXmlSectionsMixin
from services.merging_extraction_service import MergingExtractionService


class _QueryStub:
    def __init__(self, first_result=None, all_result=None):
        self.filters = []
        self.first_result = first_result
        self.all_result = list(all_result or [])
        self.deleted = False

    def join(self, *_args, **_kwargs):
        return self

    def filter(self, *criteria):
        self.filters.extend(criteria)
        return self

    def order_by(self, *_args, **_kwargs):
        return self

    def first(self):
        return self.first_result

    def all(self):
        return list(self.all_result)

    def delete(self, synchronize_session=False):
        self.deleted = not synchronize_session or True
        return 0


class _DbStub:
    def __init__(self, query_results=None):
        self.query_results = list(query_results or [])
        self.queries = []
        self.flushed = False

    def query(self, *_args, **_kwargs):
        config = self.query_results.pop(0) if self.query_results else {}
        query = _QueryStub(**config)
        self.queries.append(query)
        return query

    def flush(self):
        self.flushed = True


class ChapterRefTypeBoundaryTests(unittest.TestCase):
    def test_queue_task_type_buku_maps_to_canonical_bab_reference(self):
        service = AntrianService(db=None)
        task = SimpleNamespace(
            antrian_id=10,
            antrian_tipe="buku",
            bab_id=123,
            dokumen_id=None,
            aturan_id=None,
        )

        self.assertEqual(("bab", 123), service.get_task_reference(task))

    def test_merging_extraction_canonicalizes_buku_to_bab(self):
        self.assertEqual("bab", MergingExtractionService._canonical_ref_tipe("buku"))


class ChapterRefTypeReadPolicyTests(unittest.TestCase):
    def test_openxml_reader_uses_bab_only_for_canonical_chapter_flow(self):
        self.assertEqual(("bab",), AlignmentOpenXmlSectionsMixin._resolve_ref_tipe_for_read("bab"))

    def test_openxml_reader_still_canonicalizes_legacy_buku_input_to_bab(self):
        self.assertEqual(("bab",), AlignmentOpenXmlSectionsMixin._resolve_ref_tipe_for_read("buku"))


class ChapterRefTypeQueryPolicyTests(unittest.TestCase):
    def _get_ref_tipe_filter_sql(self, query_stub):
        for criterion in query_stub.filters:
            sql = str(criterion)
            if "dev_ref_tipe" in sql:
                return sql
        self.fail("dev_ref_tipe filter not found")

    def test_header_footer_section_lookup_uses_equality_for_bab(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        db = _DbStub(query_results=[{"first_result": (7,)}])

        page = service._resolve_section_start_page(db, "bab", 12, 34)

        self.assertEqual(7, page)
        ref_tipe_filter_sql = self._get_ref_tipe_filter_sql(db.queries[0])
        self.assertIn("=", ref_tipe_filter_sql)
        self.assertNotIn(" IN ", ref_tipe_filter_sql.upper())

    def test_existing_claim_lookup_uses_equality_for_bab(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        db = _DbStub(query_results=[{"all_result": []}])

        claims = service._collect_existing_claims_by_element(
            db,
            "bab",
            12,
            3,
            {101},
        )

        self.assertEqual({}, claims)
        ref_tipe_filter_sql = self._get_ref_tipe_filter_sql(db.queries[0])
        self.assertIn("=", ref_tipe_filter_sql)
        self.assertNotIn(" IN ", ref_tipe_filter_sql.upper())

    def test_visual_record_delete_uses_equality_for_bab(self):
        service = MergingExtractionService.__new__(MergingExtractionService)
        db = _DbStub(query_results=[{}])

        fused_results = service._replace_visual_records(
            db,
            "buku",
            12,
            3,
            [],
            apply_duplicate_claim_guard=False,
        )

        self.assertEqual([], fused_results)
        self.assertTrue(db.flushed)
        ref_tipe_filter_sql = self._get_ref_tipe_filter_sql(db.queries[0])
        self.assertIn("=", ref_tipe_filter_sql)
        self.assertNotIn(" IN ", ref_tipe_filter_sql.upper())


if __name__ == "__main__":
    unittest.main()
