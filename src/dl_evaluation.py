"""Phase 2 evaluation: multi-seed statistics, ablation deltas, model selection, calibration, figures (owner: Rohan).

Phase 1 reported one training run per model, so its confidence intervals covered test-set sampling only.
Phase 2 trains every configuration with several seeds; this module turns the per-run records into
  * mean +- sample standard deviation over seeds (training randomness),
  * paired per-seed differences between an ablation and the base (same seed -> same initial weights and batches),
  * the frozen selection rule for the final deep model,
  * calibration (expected calibration error, reliability diagram) and in-group confusion,
  * the Phase 2 figures.
Nothing here trains or tunes a model. Phase 1 helpers (bootstrap_ci, mcnemar_test, ...) are re-used from
src/evaluation.py unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CLASS_NAMES, NUM_CLASSES
from src.dl_config import ECE_BINS, IN_GROUP

# Categorical colours in a fixed order (colour-blind-safe set), plus neutral inks.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GROUP_COLOURS = {"architecture": PALETTE[0], "regularisation": PALETTE[1],
                 "augmentation": PALETTE[2], "optimisation": PALETTE[6]}


def style_axes(ax, grid_axis: str = "y"):
    """Shared look: light grid on one axis, no top/right spines, muted tick labels."""
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    return ax


# ----------------------------------------------------------------------------- multi-seed statistics
def _std(values) -> float:
    """Sample standard deviation (ddof=1); 0.0 when there is a single value."""
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def summarise_runs(records: list[dict]) -> pd.DataFrame:
    """One row per configuration: mean and std over seeds of the validation metrics and training statistics."""
    if not records:
        raise ValueError("no run records")
    rows = []
    names = list(dict.fromkeys(r["name"] for r in records))     # keep first-seen order
    for name in names:
        runs = sorted((r for r in records if r["name"] == name), key=lambda r: r["seed"])
        seeds = [r["seed"] for r in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{name}: duplicate seeds {seeds}")
        col = lambda k: [r[k] for r in runs]
        rows.append({
            "name": name, "n_seeds": len(runs), "seeds": " ".join(str(s) for s in seeds),
            "val_macro_f1_mean": float(np.mean(col("val_macro_f1"))), "val_macro_f1_std": _std(col("val_macro_f1")),
            "val_accuracy_mean": float(np.mean(col("val_accuracy"))), "val_accuracy_std": _std(col("val_accuracy")),
            "val_loss_mean": float(np.mean(col("val_loss"))),
            "train_accuracy_mean": float(np.mean(col("train_accuracy_eval"))),
            "gap_mean": float(np.mean(col("generalisation_gap"))), "gap_std": _std(col("generalisation_gap")),
            "shirt_recall_mean": float(np.mean([r["val_per_class_recall"][6] for r in runs])),
            "n_params": int(runs[0]["n_params"]), "best_epoch_mean": float(np.mean(col("best_epoch"))),
            "epochs_run_mean": float(np.mean(col("epochs_run"))),
            "train_seconds_mean": float(np.mean(col("train_seconds"))),
            "any_diverged": bool(any(col("diverged"))),
        })
    return pd.DataFrame(rows)


def ablation_deltas(records: list[dict], base_name: str, metric: str = "val_macro_f1") -> pd.DataFrame:
    """Paired differences (variant - base) per seed, for every configuration that shares seeds with the base.

    delta_mean < 0 means removing/changing the factor HURT, i.e. the factor of the base helps.
    consistent_sign is True when every seed agrees on the direction. With 3 seeds this is descriptive evidence,
    not a significance test.
    """
    base = {r["seed"]: r[metric] for r in records if r["name"] == base_name}
    if not base:
        raise ValueError(f"no runs found for the base configuration {base_name!r}")
    rows = []
    for name in dict.fromkeys(r["name"] for r in records):
        if name == base_name:
            continue
        runs = {r["seed"]: r[metric] for r in records if r["name"] == name}
        shared = sorted(set(runs) & set(base))
        if not shared:
            continue
        d = np.array([runs[s] - base[s] for s in shared])
        rows.append({"name": name, "n_pairs": len(shared), "delta_mean": float(d.mean()), "delta_std": _std(d),
                     "delta_min": float(d.min()), "delta_max": float(d.max()),
                     "consistent_sign": bool((d > 0).all() or (d < 0).all()),
                     "base_mean": float(np.mean([base[s] for s in shared])),
                     "variant_mean": float(np.mean([runs[s] for s in shared]))})
    return pd.DataFrame(rows)


def select_final_config(summary: pd.DataFrame, required_seeds: int | None = None) -> str:
    """Frozen selection rule for the final deep model (validation only).

    Highest mean validation macro-F1 over seeds; ties -> higher mean accuracy -> fewer parameters -> name.
    Configurations that diverged, or that lack a full set of seeds, are not eligible.
    """
    ok = summary[~summary["any_diverged"]]
    if required_seeds is not None:
        ok = ok[ok["n_seeds"] >= required_seeds]
    if ok.empty:
        raise ValueError("no eligible configuration to select")
    order = ok.sort_values(["val_macro_f1_mean", "val_accuracy_mean", "n_params", "name"],
                           ascending=[False, False, True, True], kind="stable")
    return str(order.iloc[0]["name"])


def aggregate_seed_metrics(per_seed: dict) -> dict:
    """per_seed: {seed: {"accuracy": .., "macro_f1": ..}} -> mean/std/min/max over seeds."""
    acc = [m["accuracy"] for m in per_seed.values()]
    f1 = [m["macro_f1"] for m in per_seed.values()]
    return {"n_seeds": len(acc), "accuracy_mean": float(np.mean(acc)), "accuracy_std": _std(acc),
            "accuracy_min": float(np.min(acc)), "accuracy_max": float(np.max(acc)),
            "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": _std(f1)}


# ----------------------------------------------------------------------------- calibration
def expected_calibration_error(probs, y_true, n_bins: int = ECE_BINS) -> float:
    """ECE with equal-width confidence bins: sum_b (n_b / N) * |accuracy_b - mean confidence_b|.

    0 means that among predictions made with confidence c, a fraction c is correct.
    """
    probs, y_true = np.asarray(probs, dtype=float), np.asarray(y_true)
    if probs.ndim != 2 or len(probs) != len(y_true) or len(y_true) == 0:
        raise ValueError("probs must be (N,C) and y_true (N,) with N > 0")
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def reliability_curve(probs, y_true, n_bins: int = ECE_BINS) -> pd.DataFrame:
    """Per-bin count, mean confidence and accuracy (empty bins are dropped)."""
    probs, y_true = np.asarray(probs, dtype=float), np.asarray(y_true)
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    bins = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        if m.any():
            rows.append({"bin": b, "count": int(m.sum()), "confidence": float(conf[m].mean()),
                         "accuracy": float((pred[m] == y_true[m]).mean())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- confusion structure
def in_group_confusion_rate(cm, group=IN_GROUP) -> float:
    """Share of the group's images that are assigned to a DIFFERENT class of the same group.

    Default group: T-shirt/top, Pullover, Coat, Shirt (the classes Phase 1 found hardest to separate).
    """
    cm = np.asarray(cm)
    g = list(group)
    block = cm[np.ix_(g, g)]
    total = cm[g].sum()
    return float((block.sum() - np.trace(block)) / total) if total else float("nan")


def per_class_recall_table(recalls: dict) -> pd.DataFrame:
    """recalls: {model name: (10,) recall array} -> DataFrame indexed by class name."""
    return pd.DataFrame({k: np.asarray(v, dtype=float) for k, v in recalls.items()}, index=CLASS_NAMES)


# ----------------------------------------------------------------------------- figures
def plot_roadmap(summary: pd.DataFrame, names: list[str], labels: dict, metric: str = "val_macro_f1"):
    """Bar chart of the architecture roadmap: mean over seeds with +-1 std whiskers."""
    import matplotlib.pyplot as plt

    df = summary.set_index("name").loc[names]
    means, stds = df[f"{metric}_mean"].to_numpy(), df[f"{metric}_std"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds, width=0.55, color=PALETTE[0], ecolor=INK, capsize=4, error_kw={"linewidth": 1})
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.001, f"{m:.4f}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(n, n) for n in names], fontsize=9)
    lo = max(0.0, float((means - stds).min()) - 0.01)
    ax.set_ylim(lo, min(1.0, float((means + stds).max()) + 0.006))
    ax.set_ylabel("Validation macro-F1 (mean over seeds)")
    ax.set_title(f"Architecture roadmap: recipe change vs architecture change (n = {int(df['n_seeds'].min())} seeds)",
                 fontsize=11, color=INK)
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_ablation_deltas(deltas: pd.DataFrame, groups: dict, title_metric: str = "validation macro-F1"):
    """Horizontal bars: mean paired change vs the base for each ablation, whiskers = min..max over seeds."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    df = deltas[deltas["name"].isin(groups)].copy()
    df["group"] = df["name"].map(lambda n: groups[n][0])
    df["label"] = df["name"].map(lambda n: f"{groups[n][1]}: {groups[n][2]}")
    order = list(GROUP_COLOURS)
    df["g"] = df["group"].map(order.index)
    df = df.sort_values(["g", "delta_mean"], ascending=[False, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.5, 0.42 * len(df) + 1.6))
    y = np.arange(len(df))
    pts = 100 * df["delta_mean"].to_numpy()
    err = np.vstack([pts - 100 * df["delta_min"].to_numpy(), 100 * df["delta_max"].to_numpy() - pts])
    ax.barh(y, pts, xerr=err, height=0.6, color=[GROUP_COLOURS[g] for g in df["group"]], ecolor=INK, capsize=3,
            error_kw={"linewidth": 1})
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8.5)
    ax.set_xlabel(f"Change in {title_metric} vs base (percentage points; negative = the base factor helps)")
    ax.set_title("One-factor ablations of the Phase 2 base model (bars: mean over seeds, whiskers: min-max)",
                 fontsize=11, color=INK)
    ax.legend(handles=[Patch(color=GROUP_COLOURS[g], label=g) for g in order if g in set(df["group"])],
              fontsize=8.5, frameon=False, loc="best")
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def plot_generalisation_gap(summary: pd.DataFrame, names: list[str]):
    """Train accuracy (eval mode) next to validation accuracy for selected configurations."""
    import matplotlib.pyplot as plt

    df = summary.set_index("name").loc[names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(names) + 2), 4.2))
    ax.bar(x - 0.2, df["train_accuracy_mean"], width=0.38, color=PALETTE[0], label="train (eval mode)")
    ax.bar(x + 0.2, df["val_accuracy_mean"], width=0.38, color=PALETTE[1], label="validation")
    for xi, gap in zip(x, df["gap_mean"]):
        ax.text(xi, max(df["train_accuracy_mean"].iloc[xi], df["val_accuracy_mean"].iloc[xi]) + 0.002,
                f"gap {100 * gap:.1f} pp", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylim(max(0.0, float(df["val_accuracy_mean"].min()) - 0.03), 1.005)
    ax.set_ylabel("Accuracy (mean over seeds)")
    ax.set_title("Generalisation gap: what each regulariser does to over-fitting", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2)
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_learning_curves_compare(histories: dict, title: str = "Learning curves"):
    """histories: {label: history dict/DataFrame with epoch, train_loss, val_loss, val_acc}. Two panels."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for i, (label, h) in enumerate(histories.items()):
        h = pd.DataFrame(h)
        c = PALETTE[i % len(PALETTE)]
        axes[0].plot(h["epoch"], h["val_loss"], color=c, linewidth=2, label=f"{label}: validation")
        axes[0].plot(h["epoch"], h["train_loss"], color=c, linewidth=1.2, linestyle="--", label=f"{label}: train")
        axes[1].plot(h["epoch"], h["val_acc"], color=c, linewidth=2, label=label)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Loss (dashed = training, running average)", fontsize=10)
    axes[1].set_ylabel("Validation accuracy")
    axes[1].set_title("Validation accuracy", fontsize=10)
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        style_axes(ax)
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout()
    return fig


def plot_reliability(curves: dict, eces: dict, title: str = "Reliability diagram"):
    """curves: {label: reliability_curve DataFrame}. Points on the diagonal are perfectly calibrated."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1, linestyle=":", label="perfect calibration")
    for i, (label, df) in enumerate(curves.items()):
        ax.plot(df["confidence"], df["accuracy"], marker="o", markersize=4, linewidth=2,
                color=PALETTE[i % len(PALETTE)], label=f"{label} (ECE {eces[label]:.3f})")
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Accuracy in bin")
    ax.set_xlim(0.2, 1.0)
    ax.set_ylim(0.2, 1.0)
    ax.set_title(title, fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style_axes(ax, grid_axis="both")
    fig.tight_layout()
    return fig


def plot_per_class_recall(table: pd.DataFrame, title: str = "Per-class recall on the test set"):
    """Grouped bars: one group per class, one bar per model (at most 4 models for readability)."""
    import matplotlib.pyplot as plt

    cols = list(table.columns)[:4]
    x = np.arange(len(table))
    width = 0.8 / len(cols)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, c in enumerate(cols):
        ax.bar(x - 0.4 + width * (i + 0.5), table[c], width=width * 0.92, color=PALETTE[i], label=c)
    ax.set_xticks(x)
    ax.set_xticklabels(table.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(max(0.0, float(table[cols].min().min()) - 0.08), 1.005)
    ax.set_ylabel("Recall")
    ax.set_title(title, fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, ncol=len(cols), loc="upper center", bbox_to_anchor=(0.5, -0.24))
    style_axes(ax)
    fig.tight_layout()
    return fig


def plot_test_summary(rows: pd.DataFrame, title: str = "Test accuracy (mean over seeds)"):
    """Horizontal bars of test accuracy with +-1 std whiskers. rows: label, accuracy_mean, accuracy_std, kind."""
    import matplotlib.pyplot as plt

    kinds = list(dict.fromkeys(rows["kind"]))
    colour = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(kinds)}
    df = rows.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(df) + 1.5))
    y = np.arange(len(df))
    ax.barh(y, df["accuracy_mean"], xerr=df["accuracy_std"], height=0.6, color=[colour[k] for k in df["kind"]],
            ecolor=INK, capsize=3, error_kw={"linewidth": 1})
    for yi, m, s in zip(y, df["accuracy_mean"], df["accuracy_std"]):
        ax.text(m + s + 0.001, yi, f"{m:.4f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=9)
    lo = max(0.0, float(df["accuracy_mean"].min()) - 0.02)
    ax.set_xlim(lo, min(1.0, float((df["accuracy_mean"] + df["accuracy_std"]).max()) + 0.012))
    ax.set_xlabel("Test accuracy")
    ax.set_title(title, fontsize=11, color=INK)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=colour[k], label=k) for k in kinds], frameon=False, fontsize=8.5,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=len(kinds))
    style_axes(ax, grid_axis="x")
    fig.tight_layout()
    return fig


def plot_cascade(df: pd.DataFrame, best_tau: float):
    """Two stacked panels sharing the threshold axis: validation accuracy and the fraction of images routed."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(7, 5.2), sharex=True)
    x = np.arange(len(df))
    axes[0].plot(x, df["val_accuracy"], marker="o", color=PALETTE[0], linewidth=2)
    axes[0].set_ylabel("Validation accuracy")
    axes[1].bar(x, 100 * df["routed_fraction"], color=PALETTE[1], width=0.6)
    axes[1].set_ylabel("Images sent to the hybrid (%)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{t:g}" for t in df["tau"]])
    axes[1].set_xlabel("Confidence threshold tau (CNN decides when its confidence >= tau)")
    best_i = int(np.argmin(np.abs(df["tau"].to_numpy() - best_tau)))
    for ax in axes:
        ax.axvline(best_i, color=INK_MUTED, linestyle=":", linewidth=1)
        style_axes(ax)
    axes[0].set_title(f"Confidence-gated cascade: chosen tau = {best_tau:g}", fontsize=11, color=INK)
    fig.tight_layout()
    return fig
