import importlib
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("LOG_DIR", str(ROOT_DIR / "logs"))


class _WorkerDbStub:
    def __init__(self):
        self.rollback_called = False
        self.closed = False

    def rollback(self):
        self.rollback_called = True

    def close(self):
        self.closed = True

    def query(self, *_args, **_kwargs):
        raise AssertionError("Unexpected database query in aturan cancellation test path")


def _load_visual_worker(cancel_states, process_result=True, convert_exception=None):
    task = SimpleNamespace(
        antrian_id=51,
        antrian_tipe="aturan",
        aturan_id=8,
        dokumen_id=None,
        buku_id=None,
        bab_id=None,
    )
    db = _WorkerDbStub()
    state = {"calls": []}
    cancel_values = list(cancel_states)

    fake_database = ModuleType("database")
    fake_database.SessionLocal = lambda: db
    fake_database.engine = object()

    fake_models = ModuleType("models")
    fake_models.Base = SimpleNamespace(
        metadata=SimpleNamespace(create_all=lambda *args, **kwargs: None)
    )
    fake_models.Bab = type("Bab", (), {})
    fake_models.Dokumen = type("Dokumen", (), {})
    fake_models.Aturan = type("Aturan", (), {})

    class FakeAntrianService:
        def __init__(self, db_session):
            state["db_session"] = db_session

        def get_next_labeling_task(self):
            state["calls"].append(("get_next_labeling_task",))
            return task

        def is_task_cancelled(self, current_task):
            result = cancel_values.pop(0) if cancel_values else False
            state["calls"].append(("is_task_cancelled", current_task.antrian_id, result))
            return result

        def mark_task_cancelled(self, current_task, error_message="Dibatalkan oleh pengguna."):
            state["calls"].append(("mark_task_cancelled", current_task.antrian_id, error_message))

        def update_labeling_status(self, current_task, status, error_message=None):
            state["calls"].append(("update_labeling_status", current_task.antrian_id, status, error_message))

        def update_validation_status(self, current_task, status, error_message=None):
            state["calls"].append(("update_validation_status", current_task.antrian_id, status, error_message))

        def get_task_reference(self, current_task):
            state["calls"].append(("get_task_reference", current_task.antrian_id))
            return ("aturan", current_task.aturan_id)

        def get_full_pdf_path(self, current_task):
            state["calls"].append(("get_full_pdf_path", current_task.antrian_id))
            return "/tmp/storage/aturan/8/pdf/template.pdf"

        def get_output_directory(self, current_task):
            state["calls"].append(("get_output_directory", current_task.antrian_id))
            return "/tmp/storage/aturan/8/extraction/template"

    fake_antrian_module = ModuleType("services.antrian_service")
    fake_antrian_module.AntrianService = FakeAntrianService
    fake_antrian_module.STORAGE_BASE = "/tmp/storage"

    class FakeMergingExtractionService:
        def process_document(self, **kwargs):
            state["calls"].append(("process_document", kwargs))
            if isinstance(process_result, Exception):
                raise process_result
            return process_result

    fake_merging_module = ModuleType("services.merging_extraction_service")
    fake_merging_module.MergingExtractionService = FakeMergingExtractionService

    def fake_convert_pdf_to_images(pdf_path, output_dir=None):
        state["calls"].append(("convert_pdf_to_images", pdf_path, output_dir))
        if convert_exception is not None:
            raise convert_exception

    fake_pdf_module = ModuleType("services.pdf_image_service")
    fake_pdf_module.convert_pdf_to_images = fake_convert_pdf_to_images

    sys.modules.pop("workers.visual_worker", None)

    with patch.dict(
        sys.modules,
        {
            "database": fake_database,
            "models": fake_models,
            "services.antrian_service": fake_antrian_module,
            "services.merging_extraction_service": fake_merging_module,
            "services.pdf_image_service": fake_pdf_module,
        },
    ):
        module = importlib.import_module("workers.visual_worker")

    module.os.makedirs = lambda *args, **kwargs: None
    return module, state, db


class VisualWorkerCancellationTests(unittest.TestCase):
    def test_process_visual_task_skips_when_task_is_cancelled_before_start(self):
        module, state, db = _load_visual_worker(cancel_states=[True])

        result = module.process_visual_task()

        self.assertFalse(result)
        self.assertIn(("mark_task_cancelled", 51, "Dibatalkan oleh pengguna."), state["calls"])
        self.assertFalse(any(call[0] == "update_labeling_status" for call in state["calls"]))
        self.assertFalse(any(call[0] == "convert_pdf_to_images" for call in state["calls"]))
        self.assertFalse(any(call[0] == "process_document" for call in state["calls"]))
        self.assertFalse(any(call[0] == "update_validation_status" for call in state["calls"]))
        self.assertTrue(db.closed)

    def test_process_visual_task_does_not_handoff_validation_after_midflow_cancel(self):
        module, state, db = _load_visual_worker(cancel_states=[False, True])

        result = module.process_visual_task()

        self.assertFalse(result)
        self.assertIn(("update_labeling_status", 51, "processing", None), state["calls"])
        self.assertTrue(any(call[0] == "convert_pdf_to_images" for call in state["calls"]))
        self.assertTrue(any(call[0] == "process_document" for call in state["calls"]))
        self.assertIn(("mark_task_cancelled", 51, "Dibatalkan oleh pengguna."), state["calls"])
        self.assertFalse(any(call[0] == "update_validation_status" for call in state["calls"]))
        self.assertFalse(
            any(call[0] == "update_labeling_status" and call[2] == "completed" for call in state["calls"])
        )
        self.assertTrue(db.closed)

    def test_process_visual_task_marks_cancelled_in_exception_path(self):
        module, state, db = _load_visual_worker(
            cancel_states=[False, True],
            convert_exception=RuntimeError("boom"),
        )

        result = module.process_visual_task()

        self.assertFalse(result)
        self.assertTrue(db.rollback_called)
        self.assertIn(("mark_task_cancelled", 51, "Dibatalkan oleh pengguna."), state["calls"])
        self.assertFalse(
            any(call[0] == "update_labeling_status" and call[2] == "failed" for call in state["calls"])
        )
        self.assertFalse(any(call[0] == "update_validation_status" for call in state["calls"]))
        self.assertTrue(db.closed)


if __name__ == "__main__":
    unittest.main()
