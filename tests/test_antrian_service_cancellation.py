import os
import sys
import unittest
from datetime import datetime
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


class _QueryStub:
    def __init__(self, first_result=None):
        self.first_result = first_result
        self.filters = []

    def filter(self, *criteria):
        self.filters.extend(criteria)
        return self

    def first(self):
        return self.first_result


class _DbStub:
    def __init__(self, query_results=None):
        self.query_results = list(query_results or [])
        self.queries = []
        self.commit_count = 0

    def query(self, *_args, **_kwargs):
        config = self.query_results.pop(0) if self.query_results else {}
        query = _QueryStub(**config)
        self.queries.append(query)
        return query

    def commit(self):
        self.commit_count += 1


class AntrianServiceCancellationTests(unittest.TestCase):
    def test_is_task_cancelled_returns_true_for_cancelled_dokumen(self):
        db = _DbStub(
            query_results=[
                {"first_result": SimpleNamespace(dokumen_status="dibatalkan")}
            ]
        )
        service = AntrianService(db)
        task = SimpleNamespace(
            antrian_tipe="dokumen",
            dokumen_id=7,
            buku_id=None,
            bab_id=None,
        )

        self.assertTrue(service.is_task_cancelled(task))

    def test_is_task_cancelled_returns_true_for_cancelled_buku_via_bab(self):
        db = _DbStub(
            query_results=[
                {"first_result": SimpleNamespace(buku_id=9)},
                {"first_result": SimpleNamespace(buku_status="dibatalkan")},
            ]
        )
        service = AntrianService(db)
        task = SimpleNamespace(
            antrian_tipe="buku",
            dokumen_id=None,
            buku_id=None,
            bab_id=11,
        )

        self.assertTrue(service.is_task_cancelled(task))

    def test_mark_task_cancelled_clears_only_active_stages_and_commits_once(self):
        db = _DbStub()
        service = AntrianService(db)
        task = SimpleNamespace(
            antrian_id=13,
            antrian_extraction_status="completed",
            antrian_labeling_status="processing",
            antrian_validation_status="in_queue",
            antrian_error_message=None,
            antrian_updated_at=None,
        )

        service.mark_task_cancelled(task)

        self.assertEqual("completed", task.antrian_extraction_status)
        self.assertIsNone(task.antrian_labeling_status)
        self.assertIsNone(task.antrian_validation_status)
        self.assertEqual("Dibatalkan oleh pengguna.", task.antrian_error_message)
        self.assertIsInstance(task.antrian_updated_at, datetime)
        self.assertEqual(1, db.commit_count)

    def test_mark_task_cancelled_is_noop_when_no_active_stage_exists(self):
        db = _DbStub()
        service = AntrianService(db)
        task = SimpleNamespace(
            antrian_id=14,
            antrian_extraction_status="completed",
            antrian_labeling_status="failed",
            antrian_validation_status=None,
            antrian_error_message=None,
            antrian_updated_at=None,
        )

        service.mark_task_cancelled(task)

        self.assertEqual("completed", task.antrian_extraction_status)
        self.assertEqual("failed", task.antrian_labeling_status)
        self.assertIsNone(task.antrian_validation_status)
        self.assertIsNone(task.antrian_error_message)
        self.assertIsNone(task.antrian_updated_at)
        self.assertEqual(0, db.commit_count)


if __name__ == "__main__":
    unittest.main()
