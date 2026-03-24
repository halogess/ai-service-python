import importlib
import os
import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
SCRIPTS_DIR = ROOT_DIR / "scripts"
os.environ.setdefault("DB_HOST", "localhost")
os.environ.setdefault("DB_PORT", "3306")
os.environ.setdefault("DB_NAME", "test_db")
os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
for path in (SRC_DIR, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


MAX_PYTHON_FILE_LINES = 500
SCAN_ROOTS = ("src", "scripts", "tests")
SKIP_PARTS = {"__pycache__", "logs", "tmp", "venv", ".venv"}


class StructureGuardTests(unittest.TestCase):
    def test_python_files_stay_within_line_budget(self):
        offenders = []

        for root_name in SCAN_ROOTS:
            root = ROOT_DIR / root_name
            for path in root.rglob("*.py"):
                if any(part in SKIP_PARTS for part in path.parts):
                    continue
                line_count = sum(1 for _ in path.open("r", encoding="utf-8"))
                if line_count > MAX_PYTHON_FILE_LINES:
                    offenders.append((line_count, str(path.relative_to(ROOT_DIR)).replace("\\", "/")))

        self.assertFalse(
            offenders,
            "Python files above line budget: "
            + ", ".join(f"{path} ({line_count})" for line_count, path in sorted(offenders, reverse=True)),
        )

    def test_main_service_facades_import_cleanly(self):
        for module_name in (
            "services.merging_extraction_service",
            "services.alignment_service",
            "services.docling_fusion_service",
        ):
            with self.subTest(module=module_name):
                importlib.import_module(module_name)


if __name__ == "__main__":
    unittest.main()
