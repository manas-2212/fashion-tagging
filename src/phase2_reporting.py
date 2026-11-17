"""Phase 2 tables: CSV for analysis, Markdown for the README, LaTeX for the IEEE report (owner: Puneet).

Every number in the report is generated from a results file by this module; nothing is typed by hand.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiments import _json_default

_LATEX_ESCAPES = {"&": r"\&", "%": r"\%", "_": r"\_", "#": r"\#", "$": r"\$"}


def latex_escape(text) -> str:
    out = str(text)
    for k, v in _LATEX_ESCAPES.items():
        out = out.replace(k, v)
    return out


def format_cell(value, float_format: str = "{:.4f}") -> str:
    """Floats with a fixed format, integers with thousands separators, missing values as a dash."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    if isinstance(value, (float, np.floating)):
        return float_format.format(float(value))
    return str(value)


def mean_std(mean, std, digits: int = 4) -> str:
    """'0.9412 +- 0.0011' as a plain string (LaTeX conversion replaces +- by \\pm)."""
    return f"{mean:.{digits}f} +- {std:.{digits}f}"


def df_to_latex(df: pd.DataFrame, caption: str, label: str, float_format: str = "{:.4f}",
                column_format: str | None = None, wide: bool = False) -> str:
    """Minimal booktabs table (IEEEtran compatible). wide=True uses table* (spans both columns)."""
    env = "table*" if wide else "table"
    column_format = column_format or ("l" + "r" * (len(df.columns) - 1))
    lines = [f"\\begin{{{env}}}[t]", "\\centering", f"\\caption{{{latex_escape(caption)}}}", f"\\label{{{label}}}",
             "\\footnotesize", f"\\begin{{tabular}}{{{column_format}}}", "\\toprule",
             " & ".join(latex_escape(c) for c in df.columns) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = [latex_escape(format_cell(v, float_format)).replace("+-", "$\\pm$") for v in row]
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    return "\n".join(lines)


def df_to_markdown(df: pd.DataFrame, float_format: str = "{:.4f}") -> str:
    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(format_cell(v, float_format).replace("|", "/") for v in row) + " |"
            for row in df.itertuples(index=False)]
    return "\n".join([head, sep] + body) + "\n"


def save_table(df: pd.DataFrame, stem: str, results_dir, tables_dir=None, caption: str = "", label: str = "",
               float_format: str = "{:.4f}", wide: bool = False, latex_df: pd.DataFrame | None = None) -> dict:
    """Write <results_dir>/<stem>.csv and .md, and <tables_dir>/<stem>.tex when tables_dir is given.

    latex_df: optional slimmer/renamed version of the table for the report (the CSV keeps every column).
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = {"csv": results_dir / f"{stem}.csv", "md": results_dir / f"{stem}.md"}
    df.to_csv(paths["csv"], index=False)
    paths["md"].write_text(df_to_markdown(df, float_format), encoding="utf-8")
    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        paths["tex"] = tables_dir / f"{stem}.tex"
        paths["tex"].write_text(df_to_latex(latex_df if latex_df is not None else df, caption or stem,
                                            label or f"tab:{stem}", float_format, wide=wide), encoding="utf-8")
    return paths


def save_json(obj, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default), encoding="utf-8")
    return path


def flatten_records(records: list[dict]) -> pd.DataFrame:
    """One row per training run with scalar fields only (history and config dicts are summarised)."""
    rows = []
    for r in records:
        c = r["config"]
        rows.append({
            "name": r["name"], "seed": r["seed"], "arch": c["arch"],
            "arch_kwargs": json.dumps(c["arch_kwargs"], sort_keys=True), "optimizer": c["optimizer"], "lr": c["lr"],
            "weight_decay": c["weight_decay"], "schedule": c["schedule"], "label_smoothing": c["label_smoothing"],
            "aug": c["aug"], "max_epochs": c["max_epochs"], "patience": c["patience"], "n_params": r["n_params"],
            "val_accuracy": r["val_accuracy"], "val_macro_f1": r["val_macro_f1"], "val_loss": r["val_loss"],
            "train_accuracy_eval": r["train_accuracy_eval"], "generalisation_gap": r["generalisation_gap"],
            "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "stopped_early": r["stopped_early"],
            "diverged": r["diverged"], "train_seconds": r["train_seconds"], "device": r["device"], "amp": r["amp"],
        })
    return pd.DataFrame(rows)
