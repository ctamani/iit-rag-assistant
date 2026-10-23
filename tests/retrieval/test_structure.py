"""Regression checks for runtime isolation and module-based offline commands."""
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_runtime_loads_without_offline_modules_from_another_directory(tmp_path):
    code = """
import importlib.abc
import sys

class NoOfflineImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise AssertionError(f"Runtime imported offline code: {fullname}")
        return None

sys.meta_path.insert(0, NoOfflineImports())
from backend.app.retrieval.common import DATA
from backend.app.retrieval.retriever import Retriever
retriever = Retriever(model=object())
assert retriever.data == DATA
assert retriever.index.ntotal == len(retriever.chunks) == 2350
assert retriever.index.d == 768
assert DATA.is_absolute()
records = retriever.original_records(retriever.chunks[0])
assert [r["original_index"] for r in records] == retriever.chunks[0]["original_indices"]
assert not any(name.startswith("scripts.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_offline_entry_points_are_importable_as_modules():
    for module in ("preprocess", "build_index", "validate", "evaluate"):
        result = subprocess.run(
            [sys.executable, "-m", f"scripts.retrieval.{module}", "--help"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout
