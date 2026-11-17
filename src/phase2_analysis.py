"""Stage 6: describe the test-gate outputs (owner: Rohan).

Multi-seed test metrics, bootstrap CIs, McNemar tests, per-class recall, confusion structure, calibration,
robustness and inference cost. Nothing is selected or tuned here, and the stage refuses to run before the gate.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from src.benchmarking import benchmark_cnn_inference
from src.cnn import make_cnn_tensor
from src.config import RESULTS_DIR
from src.data import stratified_subset
from src.dl_evaluation import (aggregate_seed_metrics, expected_calibration_error, in_group_confusion_rate,
                               per_class_recall_table, plot_per_class_recall, plot_reliability, plot_test_summary,
                               reliability_curve)
from src.dl_models import count_parameters
from src.dl_train import load_checkpoint, predict_proba
from src.evaluation import (bootstrap_ci, evaluate_predictions, mcnemar_test, plot_confusion_matrix,
                            top_confusions)
from src.features import extract_hog_features
from src.hybrid import (complementarity, extract_embeddings)
from src.phase2_experiments import (FLAG_NAME, load_gate_outputs, load_phase2_selection)
from src.phase2_reporting import mean_std, save_json, save_table
from src.reporting import get_environment_info
from src.robustness import evaluate_robustness, plot_robustness
from src.phase2_context import (GATE_KINDS, H0_NAME, PRETTY, Phase2Context, gate_name, hybrid_dir, load_test,
                                get_p1_name)


def _phase1_test_predictions(test_idx) -> dict:
    """Stored Phase 1 test predictions (same official test order), restricted to the evaluated indices."""
    path = RESULTS_DIR / "test_gate_outputs.npz"
    if not path.is_file():
        return {}
    z = np.load(path, allow_pickle=True)
    out = {}
    for key, label in (("cnn__pred", "Phase 1 CNN (original checkpoint)"),
                       ("svm__pixels__pred", "Phase 1 RBF-SVM on pixels")):
        if key in z.files:
            out[label] = np.asarray(z[key]).astype(np.int64)[test_idx]
    return out


def stage_analysis(ctx: Phase2Context) -> dict:
    """Describe the gate outputs. Nothing is selected or tuned here."""
    import joblib

    gate = load_gate_outputs(ctx.results_dir)               # raises if the gate has not run
    sel = load_phase2_selection(ctx.results_dir)
    y, res, seeds, primary = gate["y_test"], gate["results"], sel["seeds"], sel["primary_seed"]
    final_name, n_boot = sel["final_config"], ctx.settings["n_boot"]
    P = lambda kind, seed=primary: res[gate_name(kind, seed)]["y_pred"]

    # ---- per-seed and aggregated test metrics
    per_seed_rows, agg_rows, evals = [], [], {}
    for kind in GATE_KINDS:
        per_seed = {}
        for seed in seeds:
            e = evaluate_predictions(y, P(kind, seed), gate_name(kind, seed))
            evals[gate_name(kind, seed)] = e
            per_seed[seed] = e
            per_seed_rows.append({"model": kind, "seed": int(seed), "accuracy": e["accuracy"],
                                  "macro_f1": e["macro_f1"], "shirt_recall": float(e["per_class_recall"][6]),
                                  "in_group_confusion": in_group_confusion_rate(e["confusion"]),
                                  "n_errors": int((P(kind, seed) != y).sum())})
        a = aggregate_seed_metrics(per_seed)
        _, lo, hi = bootstrap_ci(y, P(kind), "accuracy", n_boot=n_boot)
        agg_rows.append({"model": kind, "label": PRETTY[kind], **a,
                         "primary_seed_accuracy": per_seed[primary]["accuracy"], "ci95_low": lo, "ci95_high": hi,
                         "shirt_recall_mean": float(np.mean([per_seed[s]["per_class_recall"][6] for s in seeds])),
                         "in_group_confusion_mean": float(np.mean(
                             [in_group_confusion_rate(per_seed[s]["confusion"]) for s in seeds]))})
    e0 = evaluate_predictions(y, res[H0_NAME]["y_pred"], H0_NAME)
    evals[H0_NAME] = e0
    _, lo, hi = bootstrap_ci(y, res[H0_NAME]["y_pred"], "accuracy", n_boot=n_boot)
    agg_rows.append({"model": H0_NAME, "label": PRETTY[H0_NAME], "n_seeds": 1, "accuracy_mean": e0["accuracy"],
                     "accuracy_std": 0.0, "accuracy_min": e0["accuracy"], "accuracy_max": e0["accuracy"],
                     "macro_f1_mean": e0["macro_f1"], "macro_f1_std": 0.0, "primary_seed_accuracy": e0["accuracy"],
                     "ci95_low": lo, "ci95_high": hi, "shirt_recall_mean": float(e0["per_class_recall"][6]),
                     "in_group_confusion_mean": in_group_confusion_rate(e0["confusion"])})
    metrics = pd.DataFrame(agg_rows)
    order = [H0_NAME, "p1_cnn", "final_dl", "h1_emb_svm", "h2_fusion_svm", "h3_cascade"]
    metrics = metrics.set_index("model").loc[order].reset_index()
    view = pd.DataFrame({
        "model": metrics["label"],
        "test accuracy": [mean_std(m, s) for m, s in zip(metrics["accuracy_mean"], metrics["accuracy_std"])],
        "test macro-F1": [mean_std(m, s) for m, s in zip(metrics["macro_f1_mean"], metrics["macro_f1_std"])],
        f"95% CI (seed {primary})": [f"[{a:.4f}, {b:.4f}]" for a, b in zip(metrics["ci95_low"], metrics["ci95_high"])],
        "Shirt recall": metrics["shirt_recall_mean"], "in-group confusion": metrics["in_group_confusion_mean"]})
    save_table(metrics, "test_metrics", ctx.results_dir, ctx.tables_dir, wide=True, latex_df=view,
               caption=f"Test results of the frozen Phase 2 models (mean +- std over {len(seeds)} training seeds; "
                       "the interval is a bootstrap over test images for one seed).", label="tab:p2_test")
    save_table(pd.DataFrame(per_seed_rows), "test_metrics_per_seed", ctx.results_dir)

    # ---- paired significance tests on the primary seed (Bonferroni over the planned comparisons)
    p1_stored = _phase1_test_predictions(gate["test_idx"])
    pairs = [("final_dl", P("final_dl"), "p1_cnn (re-trained)", P("p1_cnn")),
             ("h1_emb_svm", P("h1_emb_svm"), "final_dl", P("final_dl")),
             ("h2_fusion_svm", P("h2_fusion_svm"), "final_dl", P("final_dl")),
             ("h3_cascade", P("h3_cascade"), "final_dl", P("final_dl")),
             ("h2_fusion_svm", P("h2_fusion_svm"), "h1_emb_svm", P("h1_emb_svm"))]
    pairs += [("final_dl", P("final_dl"), label, pred) for label, pred in p1_stored.items()]
    mc_rows = []
    for name_a, pa, name_b, pb in pairs:
        t = mcnemar_test(y, pa, pb)
        mc_rows.append({"model_a": name_a, "model_b": name_b, "accuracy_a": float((pa == y).mean()),
                        "accuracy_b": float((pb == y).mean()), "a_right_b_wrong": t["n10"],
                        "a_wrong_b_right": t["n01"], "p_value": t["p_value"],
                        "p_bonferroni": min(1.0, t["p_value"] * len(pairs)), "method": t["method"]})
    mcnemar = pd.DataFrame(mc_rows)
    mcnemar["significant_0.05"] = mcnemar["p_bonferroni"] < 0.05
    save_table(mcnemar, "mcnemar_phase2", ctx.results_dir, ctx.tables_dir, wide=True, float_format="{:.4g}",
               latex_df=mcnemar[["model_a", "model_b", "a_right_b_wrong", "a_wrong_b_right", "p_bonferroni",
                                 "significant_0.05"]],
               caption=f"McNemar tests on the test set (seed {primary}), Bonferroni-corrected for {len(pairs)} "
                       "comparisons.", label="tab:p2_mcnemar")

    # ---- per-class recall, confusion matrices, top confusions
    recalls = {}
    if "Phase 1 CNN (original checkpoint)" in p1_stored:
        recalls["Phase 1 CNN"] = evaluate_predictions(y, p1_stored["Phase 1 CNN (original checkpoint)"], "p1")[
            "per_class_recall"]
    else:
        recalls["Phase 1 CNN (re-trained)"] = evals[gate_name("p1_cnn", primary)]["per_class_recall"]
    recalls["Final deep model"] = evals[gate_name("final_dl", primary)]["per_class_recall"]
    recalls["H2 fusion"] = evals[gate_name("h2_fusion_svm", primary)]["per_class_recall"]
    recalls["H0 HOG + SVM"] = e0["per_class_recall"]
    recall_table = per_class_recall_table(recalls)
    save_table(recall_table.reset_index().rename(columns={"index": "class"}), "per_class_recall_phase2",
               ctx.results_dir, ctx.tables_dir, caption=f"Per-class recall on the test set (seed {primary}).",
               label="tab:p2_recall", wide=True)
    ctx.savefig(plot_per_class_recall(recall_table), "per_class_recall_phase2.png")
    cm_final = evals[gate_name("final_dl", primary)]["confusion"]
    ctx.savefig(plot_confusion_matrix(cm_final, f"Final deep model ({final_name}, seed {primary}): test confusion"),
                "cm_final_dl.png")
    np.savez_compressed(ctx.results_dir / "confusion_matrices_phase2.npz",
                        **{k.replace("@", "__seed"): v["confusion"] for k, v in evals.items()})
    save_table(top_confusions(cm_final, 10), "top_confusions_final_dl", ctx.results_dir)

    # ---- hybrid complementarity on test (primary seed)
    comp = [{"split": "test", "seed": primary, "model_a": a, "model_b": b, **complementarity(y, pa, pb)}
            for a, b, pa, pb in (("final_dl", H0_NAME, P("final_dl"), res[H0_NAME]["y_pred"]),
                                 ("final_dl", "h2_fusion_svm", P("final_dl"), P("h2_fusion_svm")),
                                 ("h1_emb_svm", "h2_fusion_svm", P("h1_emb_svm"), P("h2_fusion_svm")))]
    save_table(pd.DataFrame(comp), "hybrid_complementarity_test", ctx.results_dir)

    # ---- calibration: test ECE per seed, and the label-smoothing ablation on validation
    cal_rows = []
    for kind in ("p1_cnn", "final_dl"):
        for seed in seeds:
            probs = res[gate_name(kind, seed)]["probs"]
            cal_rows.append({"model": kind, "split": "test", "seed": int(seed),
                             "ece": expected_calibration_error(probs, y),
                             "mean_confidence": float(probs.max(axis=1).mean()),
                             "accuracy": float((probs.argmax(axis=1) == y).mean())})
    curves, eces = {}, {}
    for name, label in (("base", "base (label smoothing 0.1)"), ("reg_no_ls", "no label smoothing"),
                        ("p1_cnn_p1_recipe", "Phase 1 CNN")):
        if name in ctx.config_names:
            probs = ctx.store.load_val_probs(name, primary)
            curves[label], eces[label] = reliability_curve(probs, ctx.data["y_val"]), \
                expected_calibration_error(probs, ctx.data["y_val"])
            cal_rows.append({"model": name, "split": "validation", "seed": int(primary), "ece": eces[label],
                             "mean_confidence": float(probs.max(axis=1).mean()),
                             "accuracy": float((probs.argmax(axis=1) == ctx.data["y_val"]).mean())})
    save_table(pd.DataFrame(cal_rows), "calibration", ctx.results_dir)
    if curves:
        ctx.savefig(plot_reliability(curves, eces, f"Reliability on validation (seed {primary})"), "reliability.png")

    # ---- robustness and inference cost need the test IMAGES again (allowed only after the gate)
    assert (ctx.results_dir / FLAG_NAME).exists()
    X_test_all, _ = load_test()
    X_eval = X_test_all[gate["test_idx"]]
    mean, std = ctx.data["mean"], ctx.data["std"]
    p1_model = load_checkpoint(ctx.store.checkpoint_path(get_p1_name(ctx), primary), ctx.device)
    final_model = load_checkpoint(ctx.store.checkpoint_path(final_name, primary), ctx.device)
    h0 = joblib.load(hybrid_dir(ctx) / f"{H0_NAME}.joblib")
    h2 = joblib.load(hybrid_dir(ctx) / f"{gate_name('h2_fusion_svm', primary)}.joblib")
    cnn_pred = lambda m: (lambda X: predict_proba(m, make_cnn_tensor(X, mean, std), ctx.device).argmax(axis=1))

    def h2_predict(X):
        emb = extract_embeddings(final_model, make_cnn_tensor(X, mean, std), ctx.device)
        return h2.predict(emb=emb, hog=extract_hog_features(X))

    Xr, yr, _ = stratified_subset(X_eval, y, min(ctx.settings["robustness_n"], len(y)))
    rob = evaluate_robustness({"Phase 1 CNN (re-trained)": cnn_pred(p1_model), "Final deep model": cnn_pred(final_model),
                               "H2 fusion": h2_predict, "H0 HOG + SVM": lambda X: h0.predict(hog=extract_hog_features(X))},
                              Xr, yr)
    save_table(rob, "robustness", ctx.results_dir)
    ctx.savefig(plot_robustness(rob), "robustness.png")

    n_b = min(ctx.settings["bench_n"], len(X_eval))
    X_b = X_eval[:n_b]
    X_bt = make_cnn_tensor(X_b, mean, std)
    bench_rows = []
    for label, m in (("Phase 1 CNN", p1_model), ("Final deep model", final_model)):
        cpu = benchmark_cnn_inference(m, X_bt, "cpu", repeats=3)
        row = {"model": label, "params": count_parameters(m), "cpu_ms_per_image": cpu["per_image_ms"],
               "gpu_ms_per_image": float("nan"), "includes_preprocessing": False, "n_images": n_b}
        if torch.cuda.is_available():
            row["gpu_ms_per_image"] = benchmark_cnn_inference(m, X_bt, "cuda", repeats=5)["per_image_ms"]
        bench_rows.append(row)
        m.to(ctx.device)
    times = []
    for _ in range(3):                                      # hybrid end-to-end from raw images (HOG + CNN + SVM)
        t0 = time.perf_counter()
        h2_predict(X_b[:min(500, n_b)])
        times.append(time.perf_counter() - t0)
    bench_rows.append({"model": "H2 fusion (end-to-end)", "params": count_parameters(final_model),
                       "cpu_ms_per_image": float(np.median(times)) / min(500, n_b) * 1000.0,
                       "gpu_ms_per_image": float("nan"), "includes_preprocessing": True, "n_images": min(500, n_b)})
    bench = pd.DataFrame(bench_rows)
    save_table(bench, "inference_phase2", ctx.results_dir, ctx.tables_dir,
               caption="Model size and inference time per image (median of repeated passes; the hybrid row is "
                       "end-to-end from raw images on the device used for the run, including HOG extraction).",
               label="tab:p2_inference")

    fig_rows = metrics[["label", "accuracy_mean", "accuracy_std"]].copy()
    fig_rows["kind"] = ["classical", "deep", "deep", "hybrid", "hybrid", "hybrid"]
    for label, pred in p1_stored.items():
        fig_rows = pd.concat([pd.DataFrame([{"label": label, "accuracy_mean": float((pred == y).mean()),
                                             "accuracy_std": 0.0, "kind": "Phase 1 (stored)"}]), fig_rows],
                             ignore_index=True)
    ctx.savefig(plot_test_summary(fig_rows, f"Test accuracy (mean over {len(seeds)} seeds, whiskers = 1 std)"),
                "test_summary.png")

    summary = {"final_config": final_name, "seeds": seeds, "n_test": int(len(y)), "smoke": bool(ctx.smoke),
               "test": {r["model"]: {k: r[k] for k in ("accuracy_mean", "accuracy_std", "macro_f1_mean",
                                                          "macro_f1_std", "ci95_low", "ci95_high")}
                        for r in metrics.to_dict(orient="records")},
               "hybrid": sel["hybrid"], "freeze_sha256_models": sel["gate_models"],
               "environment": get_environment_info()}
    save_json(summary, ctx.results_dir / "phase2_summary.json")
    save_json(get_environment_info(), ctx.results_dir / "environment_phase2.json")
    print(metrics[["label", "accuracy_mean", "accuracy_std", "macro_f1_mean", "ci95_low", "ci95_high"]]
          .to_string(index=False))
    return {"metrics": metrics, "mcnemar": mcnemar, "robustness": rob, "inference": bench}
