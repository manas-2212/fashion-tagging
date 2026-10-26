"""Tests for src/reporting.py."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from src import reporting
from src.config import SEED
from src.reporting import (RESULT_COLUMNS, build_results_table, get_environment_info,
                           save_results_table)

ENV_KEYS = {"python", "platform", "cpu", "ram", "numpy", "pandas", "sklearn", "skimage", "torch",
            "torchvision", "cuda_available", "gpu_name", "torch_threads", "seed"}


def test_result_columns_are_the_specified_13():
    assert len(RESULT_COLUMNS) == 13 and RESULT_COLUMNS[0] == "model" and RESULT_COLUMNS[-1] == "notes"
    assert len(set(RESULT_COLUMNS)) == 13


def test_environment_info_has_all_keys_and_is_json_serialisable():
    info = get_environment_info()
    assert ENV_KEYS <= set(info)
    assert info["seed"] == SEED
    json.dumps(info)  # must not raise
    print(info)


def test_environment_info_survives_missing_libraries(monkeypatch):
    for name in ("torch", "torchvision", "skimage", "psutil"):
        monkeypatch.setitem(sys.modules, name, None)  # makes "import <name>" raise ImportError
    info = get_environment_info()
    assert info["torch"] == info["torchvision"] == info["skimage"] == "not installed"
    assert info["cuda_available"] == "not installed" and info["gpu_name"] is None
    assert info["torch_threads"] == "not installed"
    assert info["ram"] == "unknown"


def test_environment_info_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    info = get_environment_info()
    assert info["cuda_available"] is False and info["gpu_name"] is None


def test_partial_entries_give_tbd_never_numeric_defaults():
    df = build_results_table([{"model": "logreg", "representation": "pixels", "accuracy": 0.8},
                              {"model": "cnn", "macro_f1": None, "fit_seconds": float("nan")}])
    assert list(df.columns) == RESULT_COLUMNS and len(df) == 2
    assert df.loc[0, "accuracy"] == 0.8
    assert df.loc[0, "macro_f1"] == "TBD" and df.loc[1, "accuracy"] == "TBD"
    assert df.loc[1, "macro_f1"] == "TBD" and df.loc[1, "fit_seconds"] == "TBD"  # None / NaN -> TBD
    assert df.loc[1, "representation"] == "TBD"
    for col in RESULT_COLUMNS:  # nothing may be 0 or NaN unless it was a real measured value
        for v in df[col]:
            assert isinstance(v, str) or v == 0.8


def test_real_zero_is_kept_not_replaced():
    df = build_results_table([{"model": "x", "accuracy": 0}])
    assert df.loc[0, "accuracy"] == 0


def test_unknown_key_is_rejected():
    with pytest.raises(ValueError, match="unknown keys"):
        build_results_table([{"model": "x", "acuracy": 0.5}])


def test_empty_entries_give_empty_table_with_columns():
    df = build_results_table([])
    assert list(df.columns) == RESULT_COLUMNS and len(df) == 0


def test_save_results_table_writes_both_files_with_header_separator(tmp_path):
    df = build_results_table([{"model": "svm|rbf", "representation": "hog", "accuracy": 0.9},
                              {"model": "cnn"}])
    csv_path, md_path = tmp_path / "out" / "results.csv", tmp_path / "out" / "results.md"
    save_results_table(df, csv_path, md_path)
    assert csv_path.is_file() and md_path.is_file()
    back = pd.read_csv(csv_path, keep_default_na=False)
    assert list(back.columns) == RESULT_COLUMNS and len(back) == 2
    assert back.loc[1, "accuracy"] == "TBD"
    lines = md_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 + 2  # header + separator + 2 rows
    assert lines[0].startswith("| model |")
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines[1].strip("|").split("|")) == len(RESULT_COLUMNS)
    assert "svm\\|rbf" in lines[2]  # a pipe inside a cell is escaped
