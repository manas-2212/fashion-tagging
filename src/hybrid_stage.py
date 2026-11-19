"""Stage 3: tune and fit the hybrid models, and build the test-time predictor (owner: Anant).

stage_hybrid uses the training and validation splits only. make_predict_all returns the function that the test
gate calls with the test images; it loads nothing by itself.
"""
from __future__ import annotations

import time

import pandas as pd

from src.cnn import make_cnn_tensor
from src.config import SEED
from src.dl_config import (CASCADE_TAUS, EMB_ONLY_GRID, HOG_ONLY_GRID, HYBRID_GRID)
from src.dl_evaluation import plot_cascade
from src.dl_train import load_checkpoint, predict_proba
from src.evaluation import evaluate_predictions
from src.hybrid import (cascade_predict, complementarity, extract_embeddings, fit_hog_pca, fit_hybrid, tune_cascade,
                        tune_hybrid)
from src.phase2_experiments import load_phase2_selection
from src.phase2_reporting import save_json, save_table
from src.phase2_context import (H0_NAME, Phase2Context, get_final_name, gate_name, hog_features, hybrid_dir, get_p1_name)


def stage_hybrid(ctx: Phase2Context) -> dict:
    """Tune (primary seed, validation) and fit (every seed, full train split) the hybrid models H0-H3."""
    import joblib

    final_name, d = get_final_name(ctx), ctx.data
    y_tr, y_val = d["y_tr"], d["y_val"]
    tune_n = min(ctx.settings["tune_n"], len(y_tr))
    t0 = time.perf_counter()
    hog_tr, hog_val = hog_features(ctx, d["X_train"], "train"), hog_features(ctx, d["X_val"], "val")
    print(f"HOG features ready: train {hog_tr.shape}, val {hog_val.shape} ({time.perf_counter() - t0:.0f}s)")
    max_dim = max(max(HYBRID_GRID["hog_dim"]), max(HOG_ONLY_GRID["hog_dim"]))
    pca = fit_hog_pca(hog_tr, y_tr, max_dim, SEED, fit_n=ctx.settings["pca_fit_n"])   # TRAIN only, shared
    cap = lambda grid: {**grid, "hog_dim": tuple(sorted({min(k, pca.n_components_) for k in grid["hog_dim"]}))}
    evr = float(pca.explained_variance_ratio_.sum())
    print(f"HOG PCA: {pca.n_components_} components keep {100 * evr:.1f}% of the HOG variance")

    tuning, rows, spec = [], [], {"final_config": final_name, "tune_n": int(tune_n), "pca_components": int(pca.n_components_),
                                  "pca_explained_variance": evr, "primary_seed": int(ctx.seeds[0])}
    # H0: handcrafted features only (does not depend on the CNN seed)
    p_h0, df = tune_hybrid("hog", cap(HOG_ONLY_GRID), y_tr, y_val, hog_tr=hog_tr, hog_val=hog_val,
                           subset_n=tune_n, pca=pca)
    tuning.append(df)
    h0 = fit_hybrid(p_h0, y_tr, hog_tr=hog_tr, pca=pca)
    joblib.dump(h0, hybrid_dir(ctx) / f"{H0_NAME}.joblib")
    h0_val = h0.predict(hog=hog_val)
    m = evaluate_predictions(y_val, h0_val, H0_NAME)
    rows.append({"model": H0_NAME, "seed": -1, "val_accuracy": m["accuracy"], "val_macro_f1": m["macro_f1"],
                 **{f"fit_{k}": v for k, v in h0.fit_info_.items()}})
    spec["h0"] = p_h0
    print(f"H0 {p_h0}: val acc {m['accuracy']:.4f} ({h0.fit_info_['fit_seconds']:.0f}s fit)")

    comp_rows, cascade_df = [], None
    for i, seed in enumerate(ctx.seeds):
        model = load_checkpoint(ctx.store.checkpoint_path(final_name, seed), ctx.device)
        emb_tr = extract_embeddings(model, d["X_tr_t"], ctx.device)
        emb_val = extract_embeddings(model, d["X_val_t"], ctx.device)
        cnn_probs = ctx.store.load_val_probs(final_name, seed)
        cnn_pred = cnn_probs.argmax(axis=1)
        if i == 0:                                          # hyper-parameters come from the primary seed only
            p_h1, df1 = tune_hybrid("emb", EMB_ONLY_GRID, y_tr, y_val, emb_tr=emb_tr, emb_val=emb_val,
                                    subset_n=tune_n)
            p_h2, df2 = tune_hybrid("fusion", cap(HYBRID_GRID), y_tr, y_val, emb_tr, emb_val, hog_tr, hog_val,
                                    subset_n=tune_n, pca=pca)
            tuning += [df1, df2]
            spec["h1"], spec["h2"] = p_h1, p_h2
        h1 = fit_hybrid(spec["h1"], y_tr, emb_tr=emb_tr)
        h2 = fit_hybrid(spec["h2"], y_tr, emb_tr=emb_tr, hog_tr=hog_tr, pca=pca)
        joblib.dump(h1, hybrid_dir(ctx) / f"{gate_name('h1_emb_svm', seed)}.joblib")
        joblib.dump(h2, hybrid_dir(ctx) / f"{gate_name('h2_fusion_svm', seed)}.joblib")
        h1_val, h2_val = h1.predict(emb=emb_val), h2.predict(emb=emb_val, hog=hog_val)
        if i == 0:
            tau, cascade_df = tune_cascade(cnn_probs, h2_val, y_val, CASCADE_TAUS)
            spec["tau"] = tau
        h3_val, routed = cascade_predict(cnn_probs, h2_val, spec["tau"])
        for name, pred, extra in (("final_dl", cnn_pred, {}), ("h1_emb_svm", h1_val, h1.fit_info_),
                                  ("h2_fusion_svm", h2_val, h2.fit_info_),
                                  ("h3_cascade", h3_val, {"routed_fraction": float(routed.mean())})):
            m = evaluate_predictions(y_val, pred, name)
            rows.append({"model": name, "seed": int(seed), "val_accuracy": m["accuracy"],
                         "val_macro_f1": m["macro_f1"], **{f"fit_{k}": v for k, v in extra.items()}})
        for a, b, pa, pb in (("final_dl", H0_NAME, cnn_pred, h0_val), ("final_dl", "h2_fusion_svm", cnn_pred, h2_val),
                             ("h1_emb_svm", "h2_fusion_svm", h1_val, h2_val)):
            comp_rows.append({"split": "validation", "seed": int(seed), "model_a": a, "model_b": b,
                              **complementarity(y_val, pa, pb)})
        print(f"seed {seed}: val acc CNN {(cnn_pred == y_val).mean():.4f} | H1 {(h1_val == y_val).mean():.4f} | "
              f"H2 {(h2_val == y_val).mean():.4f} | H3 {(h3_val == y_val).mean():.4f} "
              f"(tau {spec['tau']:g}, {100 * routed.mean():.1f}% routed)")

    save_table(pd.concat(tuning, ignore_index=True), "hybrid_tuning", ctx.results_dir)
    val_df = pd.DataFrame(rows)
    save_table(val_df, "hybrid_validation", ctx.results_dir)
    save_table(pd.DataFrame(comp_rows), "hybrid_complementarity_val", ctx.results_dir)
    save_table(cascade_df, "cascade_thresholds", ctx.results_dir)
    ctx.savefig(plot_cascade(cascade_df, spec["tau"]), "cascade_threshold.png")
    save_json(spec, ctx.results_dir / "hybrid_spec.json")
    print(f"hybrid spec: H1 {spec['h1']} | H2 {spec['h2']} | tau {spec['tau']:g}")
    return {"spec": spec, "validation": val_df}


def make_predict_all(ctx: Phase2Context):
    """Build the function the gate calls with the test images: {gate model name: {y_pred, probs}}."""
    import joblib

    sel = load_phase2_selection(ctx.results_dir)
    final_name, tau, p1_name = sel["final_config"], sel["hybrid"]["tau"], get_p1_name(ctx)
    mean, std = ctx.data["mean"], ctx.data["std"]

    def predict_all(X_test):
        X_t = make_cnn_tensor(X_test, mean, std)            # TRAIN mean/std
        hog = hog_features(ctx, X_test, "test")
        out = {}
        h0 = joblib.load(hybrid_dir(ctx) / f"{H0_NAME}.joblib")
        out[H0_NAME] = {"y_pred": h0.predict(hog=hog), "probs": None}
        for seed in sel["seeds"]:
            p1 = load_checkpoint(ctx.store.checkpoint_path(p1_name, seed), ctx.device)
            probs = predict_proba(p1, X_t, ctx.device)
            out[gate_name("p1_cnn", seed)] = {"y_pred": probs.argmax(axis=1), "probs": probs}
            model = load_checkpoint(ctx.store.checkpoint_path(final_name, seed), ctx.device)
            probs = predict_proba(model, X_t, ctx.device)
            out[gate_name("final_dl", seed)] = {"y_pred": probs.argmax(axis=1), "probs": probs}
            emb = extract_embeddings(model, X_t, ctx.device)
            h1 = joblib.load(hybrid_dir(ctx) / f"{gate_name('h1_emb_svm', seed)}.joblib")
            h2 = joblib.load(hybrid_dir(ctx) / f"{gate_name('h2_fusion_svm', seed)}.joblib")
            h2_pred = h2.predict(emb=emb, hog=hog)
            out[gate_name("h1_emb_svm", seed)] = {"y_pred": h1.predict(emb=emb), "probs": None}
            out[gate_name("h2_fusion_svm", seed)] = {"y_pred": h2_pred, "probs": None}
            out[gate_name("h3_cascade", seed)] = {"y_pred": cascade_predict(probs, h2_pred, tau)[0], "probs": None}
        return out

    return predict_all
