"""Phase 2 context: data, device, folders and names shared by every stage (owner: Manas).

make_context() loads the training and validation data (never the test set), checks that the split is the Phase 1
split, and returns a Phase2Context that every stage function receives. smoke=True swaps in tiny subsets, a
2-epoch budget and separate output folders, so the whole pipeline can be checked in minutes.

load_test() is the only function in Phase 2 that reads the official test set. It is called by the test gate, and
by the post-gate analysis for the robustness and timing studies.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from src import ablation
from src.augment import background_value
from src.cnn import compute_mean_std, make_cnn_tensor
from src.config import DATA_DIR, RESULTS_DIR, ROOT
from src.data import (load_test_dataset, load_training_dataset, prepare_splits, stratified_subset,
                      verify_test_dataset, verify_training_dataset)
from src.dl_config import (HOG_PCA_FIT_N, HYBRID_TUNE_N, N_BOOT, P2_FIGURES_DIR, P2_RESULTS_DIR, P2_SEEDS,
                           ROBUSTNESS_N)
from src.features import cached_features
from src.phase2_experiments import RunStore

SMOKE = dict(n_train=2000, n_val=500, max_epochs=2, patience=2, seeds=(42, 43), tune_n=800, pca_fit_n=1000,
             test_n=500, robustness_n=200, n_boot=100, bench_n=256)
GATE_KINDS = ("p1_cnn", "final_dl", "h1_emb_svm", "h2_fusion_svm", "h3_cascade")
H0_NAME = "h0_hog_svm"
PRETTY = {"p1_cnn": "Phase 1 CNN (re-trained)", "final_dl": "Final deep model", H0_NAME: "H0: PCA-HOG + SVM",
          "h1_emb_svm": "H1: CNN embedding + SVM", "h2_fusion_svm": "H2: embedding + HOG fusion SVM",
          "h3_cascade": "H3: confidence-gated cascade"}


def gate_name(kind: str, seed: int) -> str:
    return f"{kind}@{int(seed)}"




# ----------------------------------------------------------------------------- context
@dataclass
class Phase2Context:
    """Everything a stage needs. Create it with make_context()."""
    results_dir: Path
    figures_dir: Path
    tables_dir: Path
    seeds: tuple
    device: torch.device
    smoke: bool
    configs: list
    data: dict
    store: RunStore
    settings: dict = field(default_factory=dict)

    @property
    def config_names(self) -> list:
        return [c.name for c in self.configs]

    def savefig(self, fig, name: str) -> Path:
        import matplotlib.pyplot as plt
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        path = self.figures_dir / name
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return path


def load_phase2_data(smoke: bool = False, device="cpu") -> dict:
    """Training and validation data only. Uses the SAME split as Phase 1 and proves it with the audit hash."""
    audit_path = RESULTS_DIR / "split_audit.json"
    old_hash = json.loads(audit_path.read_text())["idx_val_sha256"] if audit_path.is_file() else None
    X_full, y_full = load_training_dataset()
    verify_training_dataset(X_full, y_full)
    s = prepare_splits(X_full, y_full)
    new_hash = json.loads(audit_path.read_text())["idx_val_sha256"]
    if old_hash is not None and old_hash != new_hash:
        raise RuntimeError("the validation split differs from the Phase 1 split (audit hash mismatch)")
    X_tr, y_tr, X_val, y_val = s["X_train"], s["y_train"], s["X_val"], s["y_val"]
    if smoke:                                               # small stratified subsets: plumbing check only
        X_tr, y_tr, _ = stratified_subset(X_tr, y_tr, SMOKE["n_train"])
        X_val, y_val, _ = stratified_subset(X_val, y_val, SMOKE["n_val"])
    mean, std = compute_mean_std(X_tr)                      # TRAIN split only
    device = torch.device(device)
    return dict(X_train=X_tr, y_tr=y_tr, X_val=X_val, y_val=y_val, mean=mean, std=std,
                pad_value=background_value(mean, std), split_hash=new_hash,
                X_tr_t=make_cnn_tensor(X_tr, mean, std).to(device),
                X_val_t=make_cnn_tensor(X_val, mean, std).to(device))


def make_context(smoke: bool = False, seeds=None, max_epochs: int | None = None, config_names=None,
                 results_dir=None, device=None) -> Phase2Context:
    """Build the context. smoke=True uses tiny subsets, 2 epochs and separate *_smoke output folders."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if smoke:
        results_dir = Path(results_dir or (RESULTS_DIR / "phase2_smoke"))
        figures_dir, tables_dir = results_dir / "figures", results_dir / "tables"
        seeds = tuple(seeds or SMOKE["seeds"])
        max_epochs = max_epochs or SMOKE["max_epochs"]
        settings = dict(tune_n=SMOKE["tune_n"], pca_fit_n=SMOKE["pca_fit_n"], test_n=SMOKE["test_n"],
                        robustness_n=SMOKE["robustness_n"], n_boot=SMOKE["n_boot"], bench_n=SMOKE["bench_n"],
                        patience=SMOKE["patience"], cache_tag="p2smoke")
    else:
        results_dir = Path(results_dir or P2_RESULTS_DIR)
        figures_dir, tables_dir = P2_FIGURES_DIR, ROOT / "report" / "phase2_tables"
        seeds = tuple(seeds or P2_SEEDS)
        settings = dict(tune_n=HYBRID_TUNE_N, pca_fit_n=HOG_PCA_FIT_N, test_n=None, robustness_n=ROBUSTNESS_N,
                        n_boot=N_BOOT, bench_n=2000, patience=None, cache_tag="p2")
    settings["max_epochs"] = max_epochs
    configs = ablation.all_configs()
    if config_names:
        unknown = set(config_names) - {c.name for c in configs}
        if unknown:
            raise ValueError(f"unknown configuration names: {sorted(unknown)}")
        configs = [c for c in configs if c.name in set(config_names)]
    if max_epochs or settings["patience"]:
        configs = [ablation.with_budget(c, max_epochs, settings["patience"]) for c in configs]
    print(f"device = {device} | smoke = {smoke} | seeds = {seeds} | {len(configs)} configurations | "
          f"results -> {results_dir}")
    data = load_phase2_data(smoke=smoke, device=device)
    print(f"train {len(data['y_tr'])} | validation {len(data['y_val'])} | pixel mean {data['mean']:.4f} "
          f"std {data['std']:.4f} | split hash {data['split_hash'][:12]}")
    return Phase2Context(results_dir=results_dir, figures_dir=figures_dir, tables_dir=tables_dir, seeds=seeds,
                         device=device, smoke=smoke, configs=configs, data=data, store=RunStore(results_dir),
                         settings=settings)




# ----------------------------------------------------------------------------- small shared helpers
def get_final_name(ctx: Phase2Context) -> str:
    path = ctx.results_dir / "selection_preview.json"
    if not path.is_file():
        raise RuntimeError("run stage_summary first (selection_preview.json is missing)")
    return json.loads(path.read_text())["final_config"]




def hog_features(ctx: Phase2Context, X, split: str) -> np.ndarray:
    return cached_features(X, "hog", f"{ctx.settings['cache_tag']}_{split}", DATA_DIR / "cache")


def hybrid_dir(ctx: Phase2Context) -> Path:
    d = ctx.results_dir / "hybrid_models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_p1_name(ctx: Phase2Context) -> str:
    if ablation.P1_NAME not in ctx.config_names:
        raise RuntimeError(f"the reference configuration {ablation.P1_NAME} must be trained for the test gate")
    return ablation.P1_NAME


def load_test():
    X, y = load_test_dataset()
    verify_test_dataset(X, y)
    return X, y
