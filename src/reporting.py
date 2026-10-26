"""Reporting helpers: environment info and the results table (CSV + Markdown)."""
import importlib
import math
import os
import platform
from pathlib import Path

import pandas as pd

from src.config import SEED

RESULT_COLUMNS = ["model", "representation", "training_config", "n_train_used", "accuracy", "macro_f1",
                  "infer_cpu_predict_only_ms_per_img", "infer_cpu_end_to_end_ms_per_img",
                  "infer_gpu_ms_per_img", "params", "params_note", "fit_seconds", "notes"]
# Per-class recall is a SEPARATE CSV (9 rows x 10 classes), not part of this table.

NOT_INSTALLED = "not installed"


def _import(name):
    """Import a module, or return None if it is missing or broken (never crash)."""
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def _version(name):
    mod = _import(name)
    if mod is None:
        return NOT_INSTALLED
    return str(getattr(mod, "__version__", "unknown"))


def get_environment_info():
    """Report only what this machine returns. Missing libraries give 'not installed', never a crash."""
    info = {"python": platform.python_version(), "platform": platform.platform(),
            "cpu": f"{platform.processor() or 'unknown'} ({os.cpu_count()} logical cores)"}
    psutil = _import("psutil")
    try:
        info["ram"] = f"{psutil.virtual_memory().total / 1024 ** 3:.1f} GB" if psutil else "unknown"
    except Exception:
        info["ram"] = "unknown"
    for name in ("numpy", "pandas", "sklearn", "skimage", "torch", "torchvision"):
        info[name] = _version(name)
    torch = _import("torch")  # all GPU/thread queries are guarded, so a CPU-only machine works
    if torch is None:
        info.update(cuda_available=NOT_INSTALLED, gpu_name=None, torch_threads=NOT_INSTALLED)
    else:
        try:
            cuda = bool(torch.cuda.is_available())
            info["cuda_available"] = cuda
            info["gpu_name"] = torch.cuda.get_device_name(0) if cuda else None
            info["torch_threads"] = int(torch.get_num_threads())
        except Exception:
            info.update(cuda_available="unknown", gpu_name=None, torch_threads="unknown")
    info["seed"] = SEED
    return info


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and value.strip() == ""


def build_results_table(entries):
    """One row per entry (a dict). Missing, None, NaN or empty values become the string 'TBD'."""
    rows = []
    for i, entry in enumerate(entries):
        unknown = set(entry) - set(RESULT_COLUMNS)
        if unknown:  # a typo in a key must not silently drop a measured value
            raise ValueError(f"entry {i} has unknown keys {sorted(unknown)}; allowed: {RESULT_COLUMNS}")
        rows.append({col: ("TBD" if _is_missing(entry.get(col)) else entry[col]) for col in RESULT_COLUMNS})
    return pd.DataFrame(rows, columns=RESULT_COLUMNS, dtype=object)


def _md_cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def save_results_table(df, csv_path, md_path):
    """Write the table as CSV and as a hand-written Markdown table (no tabulate dependency)."""
    csv_path, md_path = Path(csv_path), Path(md_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(_md_cell(c) for c in cols) + " |",
             "| " + " | ".join("---" for _ in cols) + " |"]  # header separator row
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_md_cell(v) for v in row.tolist()) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
