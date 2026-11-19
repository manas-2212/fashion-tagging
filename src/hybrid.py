"""Hybrid ML + DL models (owner: Anant).

Idea. Phase 1 showed that the CNN and the HOG models make partly DIFFERENT mistakes. HOG is a fixed description
of edge orientation; the CNN embedding is a learned description. The hybrid asks a testable question:
does adding the handcrafted description to the learned one help a classical classifier separate the hard classes?

Models (all classical heads are RBF-SVMs, the best classical model of Phase 1):
  H0  hog       : PCA(HOG)                      -> SVM   control: classical features only
  H1  emb       : CNN embedding                 -> SVM   control: learned features, classical classifier
  H2  fusion    : [CNN embedding, alpha*PCA(HOG)] -> SVM main hybrid: feature-level fusion
  H3  cascade   : CNN softmax decides when its confidence >= tau, otherwise H2 decides (confidence gate)

Why the controls matter: H2 is only "hybrid value" if it beats BOTH the CNN alone and H1. If H1 already matches
H2, the gain comes from the SVM head, not from the handcrafted features.

Leakage rules: the scaler, the PCA and the SVM are fitted on the 54k TRAIN split only; hyper-parameters and tau are
chosen on VALIDATION; the test set appears only inside the Phase 2 test gate.

Known limitation (documented in docs/phase2/hybrid_notes.md): the CNN has seen the training images, so its
training embeddings are cleaner than its embeddings of new images. The SVM is therefore fitted on slightly
"too easy" inputs. Validation is the judge of whether the hybrid still helps.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import NUM_CLASSES, SEED
from src.data import stratified_subset
from src.dl_config import HOG_PCA_FIT_N
from src.dl_models import embed_features

MODES = ("hog", "emb", "fusion")
_LABELS = list(range(NUM_CLASSES))


# ----------------------------------------------------------------------------- CNN embeddings
def extract_embeddings(model, X_t: torch.Tensor, device, batch_size: int = 512) -> np.ndarray:
    """Penultimate-layer features in eval mode: float32 (N, d). No gradients, no augmentation."""
    device = torch.device(device)
    model.to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X_t), batch_size):
            out.append(embed_features(model, X_t[i:i + batch_size].to(device)).float().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


# ----------------------------------------------------------------------------- feature builder
class HybridFeatures:
    """Builds the classifier input for one mode. fit() uses TRAIN data only.

    emb block : StandardScaler (zero mean, unit variance per dimension).
    hog block : PCA with whitening to `hog_dim` components (unit variance per component), multiplied by `alpha`.
                The PCA is unsupervised and is fitted on a stratified subset of the training HOG vectors
                (HOG_PCA_FIT_N rows) to bound memory; alpha sets how much the HOG block weighs in the RBF distance.
    """

    def __init__(self, mode: str, hog_dim: int = 64, alpha: float = 1.0, seed: int = SEED):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode, self.hog_dim, self.alpha, self.seed = mode, int(hog_dim), float(alpha), int(seed)
        self.scaler_, self.pca_ = None, None

    @property
    def uses_emb(self) -> bool:
        return self.mode in ("emb", "fusion")

    @property
    def uses_hog(self) -> bool:
        return self.mode in ("hog", "fusion")

    def fit(self, emb_tr=None, hog_tr=None, y_tr=None, pca=None):
        """`pca`: optional already-fitted PCA with >= hog_dim components (re-used across the tuning grid)."""
        if self.uses_emb:
            if emb_tr is None:
                raise ValueError(f"mode {self.mode!r} needs embeddings")
            self.scaler_ = StandardScaler().fit(emb_tr)
        if self.uses_hog:
            if pca is not None:
                if pca.n_components_ < self.hog_dim or not pca.whiten:
                    raise ValueError("supplied PCA must be whitened and have at least hog_dim components")
                self.pca_ = pca
            else:
                if hog_tr is None or y_tr is None:
                    raise ValueError(f"mode {self.mode!r} needs HOG features and labels to fit the PCA")
                self.pca_ = fit_hog_pca(hog_tr, y_tr, self.hog_dim, self.seed)
        return self

    def transform(self, emb=None, hog=None) -> np.ndarray:
        parts = []
        if self.uses_emb:
            parts.append(self.scaler_.transform(emb).astype(np.float32))
        if self.uses_hog:
            z = pca_transform(self.pca_, hog)[:, :self.hog_dim]
            parts.append((self.alpha * z).astype(np.float32))
        return np.concatenate(parts, axis=1)

    @property
    def output_dim(self) -> int:
        d = 0
        if self.uses_emb:
            d += int(self.scaler_.n_features_in_)
        if self.uses_hog:
            d += self.hog_dim
        return d


def fit_hog_pca(hog_tr: np.ndarray, y_tr: np.ndarray, n_components: int, seed: int = SEED,
                fit_n: int = HOG_PCA_FIT_N) -> PCA:
    """Whitened PCA fitted on a stratified subset of the TRAIN HOG matrix (unsupervised; labels only stratify)."""
    sub, _, _ = stratified_subset(hog_tr, y_tr, min(fit_n, len(y_tr)), seed=seed)
    n_components = min(int(n_components), sub.shape[0] - 1, sub.shape[1])
    return PCA(n_components=n_components, whiten=True, svd_solver="randomized", random_state=seed).fit(sub)


def pca_transform(pca: PCA, hog: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """PCA projection in chunks (the full HOG matrix is 54,000 x 6,084 floats; avoid a second copy of it)."""
    out = np.empty((len(hog), pca.n_components_), dtype=np.float32)
    for i in range(0, len(hog), chunk):
        out[i:i + chunk] = pca.transform(hog[i:i + chunk])
    return out


# ----------------------------------------------------------------------------- fit / tune
class HybridModel:
    """HybridFeatures + RBF-SVM. predict() takes the raw blocks (embeddings and/or HOG vectors)."""

    def __init__(self, mode: str, C: float = 10.0, hog_dim: int = 64, alpha: float = 1.0, seed: int = SEED):
        self.params = dict(mode=mode, C=float(C), hog_dim=int(hog_dim), alpha=float(alpha))
        self.features = HybridFeatures(mode, hog_dim=hog_dim, alpha=alpha, seed=seed)
        self.svm = SVC(kernel="rbf", gamma="scale", C=float(C), random_state=seed)
        self.fit_info_ = {}

    def fit(self, y_tr, emb_tr=None, hog_tr=None, pca=None):
        t0 = time.perf_counter()
        self.features.fit(emb_tr, hog_tr, y_tr, pca=pca)
        Z = self.features.transform(emb_tr, hog_tr)
        self.svm.fit(Z, y_tr)
        self.fit_info_ = dict(n_train_used=int(len(y_tr)), input_dim=int(Z.shape[1]),
                              n_support_vectors=int(self.svm.n_support_.sum()),
                              fit_seconds=time.perf_counter() - t0)
        return self

    def predict(self, emb=None, hog=None) -> np.ndarray:
        return self.svm.predict(self.features.transform(emb, hog)).astype(np.int64)


def _score(y_true, y_pred) -> dict:
    return {"val_accuracy": float(accuracy_score(y_true, y_pred)),
            "val_macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=_LABELS, zero_division=0))}


def tune_hybrid(mode: str, grid: dict, y_tr, y_val, emb_tr=None, emb_val=None, hog_tr=None, hog_val=None,
                subset_n: int | None = None, seed: int = SEED, pca=None):
    """Validation-based grid search. Returns (best_params, tuning_df).

    grid keys: C (all modes), hog_dim (hog, fusion), alpha (fusion). The SVM is fitted on a stratified subset of
    `subset_n` training rows (None = all) and scored on the full validation split. Selection: highest validation
    macro-F1, then highest accuracy, then the simplest setting (smaller hog_dim, smaller C).
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    n = len(y_tr)
    idx = np.arange(n) if not subset_n or subset_n >= n else stratified_subset(np.arange(n), y_tr, subset_n, seed)[2]
    sub = lambda a: None if a is None else a[idx]
    Cs = grid["C"]
    dims = grid["hog_dim"] if mode in ("hog", "fusion") else (0,)
    alphas = grid["alpha"] if mode == "fusion" else (1.0,)
    if mode in ("hog", "fusion") and pca is None:
        pca = fit_hog_pca(hog_tr, y_tr, max(dims), seed)   # fitted on TRAIN only, shared by the whole grid
    rows = []
    for hog_dim, alpha, C in itertools.product(dims, alphas, Cs):
        model = HybridModel(mode, C=C, hog_dim=hog_dim or 1, alpha=alpha, seed=seed)
        model.fit(y_tr[idx], sub(emb_tr), sub(hog_tr), pca=pca)
        rows.append({"mode": mode, "C": float(C), "hog_dim": int(hog_dim), "alpha": float(alpha),
                     "n_train_used": int(len(idx)), **_score(y_val, model.predict(emb_val, hog_val)),
                     "fit_seconds": model.fit_info_["fit_seconds"]})
    df = pd.DataFrame(rows)
    order = df.sort_values(["val_macro_f1", "val_accuracy", "hog_dim", "C"],
                           ascending=[False, False, True, True], kind="stable")
    best = order.iloc[0]
    params = {"mode": mode, "C": float(best["C"]), "hog_dim": int(best["hog_dim"]), "alpha": float(best["alpha"])}
    return params, df


def fit_hybrid(params: dict, y_tr, emb_tr=None, hog_tr=None, seed: int = SEED, pca=None) -> HybridModel:
    """Final fit on the full training split with frozen hyper-parameters."""
    model = HybridModel(params["mode"], C=params["C"], hog_dim=params["hog_dim"] or 1, alpha=params["alpha"],
                        seed=seed)
    return model.fit(y_tr, emb_tr, hog_tr, pca=pca)


# ----------------------------------------------------------------------------- confidence-gated cascade
def cascade_predict(cnn_probs: np.ndarray, expert_pred: np.ndarray, tau: float):
    """CNN prediction when max softmax probability >= tau, otherwise the expert (hybrid) prediction.

    Returns (y_pred int64 (N,), routed bool (N,)). tau = 0 routes nothing (pure CNN); tau > 1 routes everything.
    """
    cnn_probs = np.asarray(cnn_probs)
    expert_pred = np.asarray(expert_pred).astype(np.int64)
    if cnn_probs.ndim != 2 or len(cnn_probs) != len(expert_pred):
        raise ValueError("cnn_probs must be (N,10) and expert_pred (N,)")
    routed = cnn_probs.max(axis=1) < tau
    y_pred = np.where(routed, expert_pred, cnn_probs.argmax(axis=1)).astype(np.int64)
    return y_pred, routed


def tune_cascade(cnn_probs_val, expert_pred_val, y_val, taus):
    """Score every threshold on validation. Returns (best_tau, df[tau, routed_fraction, val_accuracy, val_macro_f1]).

    Selection: highest validation macro-F1, then accuracy, then the SMALLEST routed fraction (cheapest cascade).
    """
    rows = []
    for tau in taus:
        pred, routed = cascade_predict(cnn_probs_val, expert_pred_val, tau)
        rows.append({"tau": float(tau), "routed_fraction": float(routed.mean()), **_score(y_val, pred)})
    df = pd.DataFrame(rows)
    order = df.sort_values(["val_macro_f1", "val_accuracy", "routed_fraction"],
                           ascending=[False, False, True], kind="stable")
    return float(order.iloc[0]["tau"]), df


# ----------------------------------------------------------------------------- complementarity analysis
def complementarity(y_true, pred_a, pred_b) -> dict:
    """How different are the errors of two models on the same images?

    oracle_accuracy: accuracy of an imaginary selector that always picks the correct model when one exists.
    It is an UPPER BOUND on what any combination of the two predictions can reach, not an achievable number.
    """
    y_true, pred_a, pred_b = (np.asarray(v) for v in (y_true, pred_a, pred_b))
    a_ok, b_ok = pred_a == y_true, pred_b == y_true
    n = len(y_true)
    a_wrong = int((~a_ok).sum())
    return {"n": n, "both_correct": int((a_ok & b_ok).sum()), "only_a_correct": int((a_ok & ~b_ok).sum()),
            "only_b_correct": int((~a_ok & b_ok).sum()), "both_wrong": int((~a_ok & ~b_ok).sum()),
            "accuracy_a": float(a_ok.mean()), "accuracy_b": float(b_ok.mean()),
            "oracle_accuracy": float((a_ok | b_ok).mean()),
            "share_of_a_errors_fixed_by_b": float((~a_ok & b_ok).sum() / a_wrong) if a_wrong else 0.0}
