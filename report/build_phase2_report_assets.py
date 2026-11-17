"""Build every number and table used by the Phase 2 report from the results files (owner: Puneet).

Run from the repository root AFTER the full Phase 2 run:

    python report/build_phase2_report_assets.py

Writes
  report/phase2_tables/numbers.tex     one LaTeX macro per number quoted in the text (no number is typed by hand)
  report/phase2_tables/rt_*.tex        compact tables laid out for the IEEE two-column format
  results/phase2/report_numbers.json   the same numbers for the slides and the README
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, P1RES, OUT = ROOT / "results" / "phase2", ROOT / "results", ROOT / "report" / "phase2_tables"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = (42, 43, 44)

summ = pd.read_csv(RES / "ablation_summary.csv").set_index("name")
delt = pd.read_csv(RES / "ablation_deltas.csv").set_index("name")
arch = pd.read_csv(RES / "architecture_summary.csv").set_index("name")
runs = pd.read_csv(RES / "run_records.csv")
test = pd.read_csv(RES / "test_metrics.csv").set_index("model")
seedm = pd.read_csv(RES / "test_metrics_per_seed.csv")
mcn = pd.read_csv(RES / "mcnemar_phase2.csv")
hval = pd.read_csv(RES / "hybrid_validation.csv")
comp = pd.read_csv(RES / "hybrid_complementarity_test.csv")
cal = pd.read_csv(RES / "calibration.csv")
rob = pd.read_csv(RES / "robustness.csv")
inf = pd.read_csv(RES / "inference_phase2.csv").set_index("model")
recall = pd.read_csv(RES / "per_class_recall_phase2.csv").set_index("class")
casc = pd.read_csv(RES / "cascade_thresholds.csv")
spec = json.loads((RES / "hybrid_spec.json").read_text())
sel = json.loads((RES / "selected_configs_phase2.json").read_text())
flag = json.loads((RES / "test_gate_phase2_done.flag").read_text())
gate = np.load(RES / "test_gate_outputs_phase2.npz")
p1 = pd.read_csv(P1RES / "results_table.csv")
FINAL = sel["selected"]["final_config"]

N = {}          # name -> formatted string (LaTeX macro body)
RAW = {}        # name -> raw value (JSON)


def put(name, value, fmt="{:.4f}"):
    assert name.isalpha(), name
    assert name not in N, name
    RAW[name] = value.item() if hasattr(value, "item") else value
    N[name] = fmt.format(value) if not isinstance(value, str) else value


pp = lambda x: 100.0 * x                                     # fraction -> percentage points
ms = lambda m, s: f"{m:.4f} $\\pm$ {s:.4f}"

# ---- validation: roadmap and ablations
for key, name in (("ValPoneRecipeOne", "p1_cnn_p1_recipe"), ("ValPoneRecipeTwo", "p1_cnn_p2_recipe"),
                  ("ValPlain", "arch_plain"), ("ValBase", "base"), ("ValDeeper", "arch_deeper")):
    put(key, summ.loc[name, "val_macro_f1_mean"])
    put(key + "Std", summ.loc[name, "val_macro_f1_std"])
    put(key + "Gap", pp(summ.loc[name, "gap_mean"]), "{:.2f}")
put("ValNoAugGap", pp(summ.loc["aug_none", "gap_mean"]), "{:.2f}")
put("ValNoAugTrainAcc", summ.loc["aug_none", "train_accuracy_mean"])
put("ValPoneRecipeTwoTrainAcc", summ.loc["p1_cnn_p2_recipe", "train_accuracy_mean"])
rec = runs.pivot(index="seed", columns="name", values="val_macro_f1")
d_recipe = rec["p1_cnn_p2_recipe"] - rec["p1_cnn_p1_recipe"]
put("RecipeOnPoneDelta", pp(d_recipe.mean()), "{:+.2f}")
put("RecipeOnPoneAllNegative", "all three" if (d_recipe < 0).all() else "not all")
put("ArchGainSameRecipe", pp(summ.loc["arch_plain", "val_macro_f1_mean"] - summ.loc["p1_cnn_p2_recipe", "val_macro_f1_mean"]), "{:+.2f}")
put("BaseVsPoneGain", pp(summ.loc["base", "val_macro_f1_mean"] - summ.loc["p1_cnn_p1_recipe", "val_macro_f1_mean"]), "{:+.2f}")
for key, name in (("DeltaDeeper", "arch_deeper"), ("DeltaNarrow", "arch_narrow"), ("DeltaNoBN", "arch_no_bn"),
                  ("DeltaPlain", "arch_plain"), ("DeltaNoErase", "aug_no_erase"), ("DeltaNoFlip", "aug_no_flip"),
                  ("DeltaNoAug", "aug_none"), ("DeltaConstLR", "opt_constant_lr"), ("DeltaSGD", "opt_sgd"),
                  ("DeltaDropHalf", "reg_dropout05"), ("DeltaDropZero", "reg_dropout0"), ("DeltaNoLS", "reg_no_ls"),
                  ("DeltaNoWD", "reg_no_wd")):
    put(key, pp(delt.loc[name, "delta_mean"]), "{:+.2f}")
put("NoBNStd", pp(summ.loc["arch_no_bn", "val_macro_f1_std"]), "{:.2f}")
put("BaseStdPP", pp(summ.loc["base", "val_macro_f1_std"]), "{:.2f}")
put("ShortcutGain", -pp(delt.loc["arch_plain", "delta_mean"]), "{:+.2f}")
put("GainLS", -pp(delt.loc["reg_no_ls", "delta_mean"]), "{:.2f}")
put("NRuns", len(runs), "{:d}")
put("NConfigs", runs["name"].nunique(), "{:d}")
put("TrainMinutesTotal", runs["train_seconds"].sum() / 60.0, "{:.0f}")
put("TrainSecondsDeeper", summ.loc["arch_deeper", "train_seconds_mean"], "{:.0f}")
put("TrainSecondsPone", summ.loc["p1_cnn_p1_recipe", "train_seconds_mean"], "{:.0f}")
put("ParamsPone", int(arch.loc["p1_cnn_p1_recipe", "params_total"]), "{:,}")
put("ParamsBase", int(arch.loc["base", "params_total"]), "{:,}")
put("ParamsDeeper", int(arch.loc["arch_deeper", "params_total"]), "{:,}")
put("ParamsNarrow", int(arch.loc["arch_narrow", "params_total"]), "{:,}")
put("DenseSharePone", pp(arch.loc["p1_cnn_p1_recipe", "dense_share"]), "{:.1f}")
put("RFPone", int(arch.loc["p1_cnn_p1_recipe", "receptive_field"]), "{:d}")
put("RFBase", int(arch.loc["base", "receptive_field"]), "{:d}")
put("RFDeeper", int(arch.loc["arch_deeper", "receptive_field"]), "{:d}")
put("FinalName", FINAL.replace("_", r"\_"))

# ---- test
for key, name in (("TestPone", "p1_cnn"), ("TestFinal", "final_dl"), ("TestHone", "h1_emb_svm"),
                  ("TestHtwo", "h2_fusion_svm"), ("TestHthree", "h3_cascade"), ("TestHzero", "h0_hog_svm")):
    put(key, test.loc[name, "accuracy_mean"])
    put(key + "Std", test.loc[name, "accuracy_std"])
    put(key + "F", test.loc[name, "macro_f1_mean"])
    put(key + "Shirt", test.loc[name, "shirt_recall_mean"], "{:.3f}")
    put(key + "InGroup", pp(test.loc[name, "in_group_confusion_mean"]), "{:.1f}")
put("TestFinalCILow", test.loc["final_dl", "ci95_low"])
put("TestFinalCIHigh", test.loc["final_dl", "ci95_high"])
put("TestGainFinalVsPone", pp(test.loc["final_dl", "accuracy_mean"] - test.loc["p1_cnn", "accuracy_mean"]), "{:+.2f}")
err = seedm.pivot(index="seed", columns="model", values="n_errors")
put("ErrorsPoneMean", err["p1_cnn"].mean(), "{:.0f}")
put("ErrorsFinalMean", err["final_dl"].mean(), "{:.0f}")
put("ErrorReduction", 100.0 * (1 - err["final_dl"].mean() / err["p1_cnn"].mean()), "{:.0f}")
acc = seedm.pivot(index="seed", columns="model", values="accuracy")
d_h2, d_h1 = acc["h2_fusion_svm"] - acc["final_dl"], acc["h1_emb_svm"] - acc["final_dl"]
put("HtwoGain", pp(d_h2.mean()), "{:+.2f}")
put("HtwoGainMin", pp(d_h2.min()), "{:+.2f}")
put("HtwoGainMax", pp(d_h2.max()), "{:+.2f}")
put("HoneGain", pp(d_h1.mean()), "{:+.2f}")
vacc = hval[hval.seed > 0].pivot(index="seed", columns="model", values="val_accuracy")
put("HtwoGainVal", pp((vacc["h2_fusion_svm"] - vacc["final_dl"]).mean()), "{:+.2f}")
put("HzeroVal", hval.loc[hval.model == "h0_hog_svm", "val_accuracy"].iloc[0])
m = mcn.set_index(["model_a", "model_b"])
for key, pair in (("PHtwoVsFinal", ("h2_fusion_svm", "final_dl")), ("PHoneVsFinal", ("h1_emb_svm", "final_dl")),
                  ("PHtwoVsHone", ("h2_fusion_svm", "h1_emb_svm"))):
    put(key, m.loc[pair, "p_value"], "{:.2f}")
row = m.loc[("final_dl", "p1_cnn (re-trained)")]
put("McFinalOnly", int(row["a_right_b_wrong"]), "{:d}")
put("McPoneOnly", int(row["a_wrong_b_right"]), "{:d}")
row = m.loc[("h2_fusion_svm", "final_dl")]
put("McHtwoOnly", int(row["a_right_b_wrong"]), "{:d}")
put("McFinalOnlyVsHtwo", int(row["a_wrong_b_right"]), "{:d}")
put("NComparisons", len(mcn), "{:d}")
c = comp.set_index(["model_a", "model_b"]).loc[("final_dl", "h0_hog_svm")]
put("CompHogFixes", int(c["only_b_correct"]), "{:d}")
put("CompCnnErrors", int(c["only_b_correct"] + c["both_wrong"]), "{:d}")
put("CompHogFixShare", 100.0 * c["share_of_a_errors_fixed_by_b"], "{:.1f}")
put("CompOracle", c["oracle_accuracy"])
routed = [float((gate[f"final_dl@{s}::probs"].max(axis=1) < spec["tau"]).mean()) for s in SEEDS]
put("RoutedTest", 100.0 * float(np.mean(routed)), "{:.1f}")
put("RoutedVal", 100.0 * hval.loc[hval.model == "h3_cascade", "fit_routed_fraction"].mean(), "{:.1f}")
put("Tau", spec["tau"], "{:g}")
put("HtwoAlpha", spec["h2"]["alpha"], "{:g}")
put("HtwoDim", spec["h2"]["hog_dim"], "{:d}")
put("HtwoC", spec["h2"]["C"], "{:g}")
put("PcaVar", 100.0 * spec["pca_explained_variance"], "{:.1f}")
put("NGateModels", len(flag["models"]), "{:d}")
put("FreezeSha", flag["freeze_sha256"][:12])

# ---- Phase 1 reference numbers (stored results)
p1row = lambda model, rep: p1[(p1["model"] == model) & (p1["representation"] == rep)].iloc[0]
cnn1 = p1[p1["model"].str.lower().str.contains("cnn")].iloc[0]
put("PhaseOneCnnAcc", float(cnn1["accuracy"]))
put("PhaseOneHogSvmAcc", float(p1row("RBF-SVM", "HOG")["accuracy"]))
put("PhaseOneSvmAcc", float(p1row("RBF-SVM", "pixels")["accuracy"]))
put("PhaseOneShirt", recall.loc["Shirt", "Phase 1 CNN"], "{:.3f}")

# ---- calibration
ct = cal[cal.split == "test"].groupby("model")[["ece", "mean_confidence", "accuracy"]].mean()
put("EceFinalTest", ct.loc["final_dl", "ece"], "{:.3f}")
put("EcePoneTest", ct.loc["p1_cnn", "ece"], "{:.3f}")
put("ConfFinalTest", ct.loc["final_dl", "mean_confidence"], "{:.3f}")
cv = cal[cal.split == "validation"].set_index("model")
put("EceBaseVal", cv.loc["base", "ece"], "{:.3f}")
put("EceNoLSVal", cv.loc["reg_no_ls", "ece"], "{:.3f}")

# ---- robustness (worst level of each corruption)
rp = rob.pivot_table(index=["corruption", "level"], columns="model", values="accuracy")
cols = {"P": "Phase 1 CNN (re-trained)", "F": "Final deep model", "H": "H2 fusion", "Z": "H0 HOG + SVM"}
for cname, tag in (("gaussian_noise", "Noise"), ("brightness", "Bright"), ("contrast", "Contrast"),
                   ("shift", "Shift"), ("occlusion", "Occl")):
    worst = rp.loc[cname].index.max()
    for k, col in cols.items():
        put(f"Rob{tag}{k}", rp.loc[(cname, worst), col], "{:.3f}")
put("RobNoiseMidF", rp.loc[("gaussian_noise", 0.10), cols["F"]], "{:.3f}")
put("RobNoiseMidP", rp.loc[("gaussian_noise", 0.10), cols["P"]], "{:.3f}")

# ---- inference
put("GpuMsPone", inf.loc["Phase 1 CNN", "gpu_ms_per_image"], "{:.3f}")
put("GpuMsFinal", inf.loc["Final deep model", "gpu_ms_per_image"], "{:.3f}")
put("CpuMsPone", inf.loc["Phase 1 CNN", "cpu_ms_per_image"], "{:.2f}")
put("CpuMsFinal", inf.loc["Final deep model", "cpu_ms_per_image"], "{:.2f}")
put("MsHtwo", inf.loc["H2 fusion (end-to-end)", "cpu_ms_per_image"], "{:.2f}")
put("GpuSlowdown", inf.loc["Final deep model", "gpu_ms_per_image"] / inf.loc["Phase 1 CNN", "gpu_ms_per_image"], "{:.1f}")

(OUT / "numbers.tex").write_text("% Generated by report/build_phase2_report_assets.py. Do not edit by hand.\n"
                                 + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in N.items()))
(RES / "report_numbers.json").write_text(json.dumps({"formatted": N, "raw": RAW}, indent=1))


# ----------------------------------------------------------------------------- compact tables
def table(stem, caption, label, header, rows, colfmt, wide=False, size="\\footnotesize"):
    env = "table*" if wide else "table"
    lines = ["% Generated by report/build_phase2_report_assets.py. Do not edit by hand.", f"\\begin{{{env}}}[t]",
             "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}", size,
             f"\\begin{{tabular}}{{{colfmt}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        lines.append("\\midrule" if r is None else " & ".join(str(c) for c in r) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}", ""]
    (OUT / f"{stem}.tex").write_text("\n".join(lines))


esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
yn = lambda b: "yes" if b else "no"

names = {"p1_cnn_p1_recipe": "Phase 1 CNN", "arch_plain": "FashionNet, no shortcut", "base": "FashionNet base",
         "arch_deeper": "FashionNet deeper (final)", "arch_narrow": "FashionNet narrow",
         "arch_no_bn": "FashionNet, no BatchNorm"}
table("rt_architectures", "Architecture variants. Dense share is the fraction of parameters in fully connected "
      "layers; the receptive field is that of the last feature map. Validation macro-F1 is the mean $\\pm$ "
      "standard deviation over three seeds.", "tab:p2arch",
      ["Model", "3$\\times$3 conv layers", "Parameters", "Dense share", "Receptive field", "Val macro-F1"],
      [[names[n], int(r.conv_layers), f"{int(r.params_total):,}", f"{pp(r.dense_share):.1f}\\%",
        f"{int(r.receptive_field)} px", ms(r.val_macro_f1_mean, r.val_macro_f1_std)] for n, r in arch.iterrows()],
      "lrrrrr", wide=True)

order = [("roadmap", "p1_cnn_p1_recipe", "Phase 1 CNN, Phase 1 recipe"),
         ("roadmap", "p1_cnn_p2_recipe", "Phase 1 CNN, Phase 2 recipe"), None,
         ("base", "base", "FashionNet base (reference)"), None]
import sys
sys.path.insert(0, str(ROOT))
from src.ablation import ABLATIONS  # noqa: E402
READABLE = {"arch_plain": "no residual shortcut", "arch_no_bn": "no batch normalisation",
            "arch_deeper": "2 blocks per stage (13 conv layers)", "arch_narrow": "widths 16/32/64",
            "reg_dropout0": "dropout 0.0 (base 0.3)", "reg_dropout05": "dropout 0.5", "reg_no_wd": "no weight decay",
            "reg_no_ls": "no label smoothing", "aug_none": "no augmentation", "aug_no_flip": "no horizontal flip",
            "aug_no_erase": "no random erasing", "opt_sgd": "SGD + Nesterov momentum",
            "opt_constant_lr": "constant learning rate"}
assert set(READABLE) == set(ABLATIONS)
prev = None
for name, (group, _factor, _change, _) in ABLATIONS.items():
    if prev is not None and group != prev:
        order.append(None)
    order.append((group, name, READABLE[name]))
    prev = group
rows = []
for item in order:
    if item is None:
        rows.append(None)
        continue
    group, name, label = item
    s = summ.loc[name]
    if name == "base":
        d = ["--", "--", "--"]
    else:
        dd = delt.loc[name]
        d = [f"{pp(dd.delta_mean):+.2f}", f"{pp(dd.delta_min):+.2f} to {pp(dd.delta_max):+.2f}", yn(dd.consistent_sign)]
    rows.append([group, esc(label), ms(s.val_macro_f1_mean, s.val_macro_f1_std), *d, f"{pp(s.gap_mean):.2f}",
                 f"{s.shirt_recall_mean:.3f}"])
table("rt_ablations", "Validation results of all 16 configurations (three seeds each). $\\Delta$ is the paired "
      "change in macro-F1 relative to the base in percentage points (variant minus base, same seed); a negative "
      "value means that the factor present in the base helps. Gap is training accuracy minus validation accuracy.",
      "tab:p2abl", ["Group", "Change relative to base", "Val macro-F1", "$\\Delta$ (pp)", "Range over seeds (pp)",
                    "Seeds agree", "Gap (pp)", "Shirt recall"], rows, "llrrrrrr", wide=True)

lab = {"h0_hog_svm": "H0: PCA-HOG + SVM (classical)", "p1_cnn": "Phase 1 CNN, re-trained", "final_dl": "Final deep model",
       "h1_emb_svm": "H1: CNN embedding + SVM", "h2_fusion_svm": "H2: embedding + HOG fusion SVM",
       "h3_cascade": "H3: confidence-gated cascade"}
table("rt_test", "Test results of the frozen Phase 2 models: mean $\\pm$ standard deviation over three training "
      "seeds (H0 does not depend on the seed). The interval is a 95\\% bootstrap interval over test images for "
      "seed 42. In-group confusion is the share of T-shirt, Pullover, Coat and Shirt images assigned to another "
      "class of that group.", "tab:p2test",
      ["Model", "Accuracy", "Macro-F1", "95\\% CI (seed 42)", "Shirt recall", "In-group confusion"],
      [[lab[n], ms(r.accuracy_mean, r.accuracy_std), ms(r.macro_f1_mean, r.macro_f1_std),
        f"[{r.ci95_low:.4f}, {r.ci95_high:.4f}]", f"{r.shirt_recall_mean:.3f}",
        f"{pp(r.in_group_confusion_mean):.1f}\\%"] for n, r in test.iterrows()], "lrrrrr", wide=True)

short = {"final_dl": "Final deep model", "p1_cnn (re-trained)": "Phase 1 CNN, re-trained", "h1_emb_svm": "H1",
         "h2_fusion_svm": "H2", "h3_cascade": "H3", "Phase 1 CNN (original checkpoint)": "Phase 1 CNN, original",
         "Phase 1 RBF-SVM on pixels": "Phase 1 pixel SVM"}
fp = lambda p: "$<10^{-4}$" if p < 1e-4 else f"{p:.3f}"
table("rt_mcnemar", f"McNemar tests on the test set for seed 42 (A vs.\\ B). The corrected $p$-value is "
      f"Bonferroni-adjusted for {len(mcn)} planned comparisons.", "tab:p2mcnemar",
      ["Model A", "Model B", "Only A right", "Only B right", "$p$", "$p$ corrected", "Significant"],
      [[short[r.model_a], short[r.model_b], int(r.a_right_b_wrong), int(r.a_wrong_b_right), fp(r.p_value),
        fp(r.p_bonferroni), yn(r["significant_0.05"])] for _, r in mcn.iterrows()], "llrrrrr", wide=True)

rows = [[int(s), f"{acc.loc[s, 'final_dl']:.4f}", f"{acc.loc[s, 'h1_emb_svm']:.4f}", f"{acc.loc[s, 'h2_fusion_svm']:.4f}",
         f"{acc.loc[s, 'h3_cascade']:.4f}", f"{pp(d_h2.loc[s]):+.2f}"] for s in SEEDS]
rows += [None, ["mean", *(f"{acc[k].mean():.4f}" for k in ("final_dl", "h1_emb_svm", "h2_fusion_svm", "h3_cascade")),
                f"{pp(d_h2.mean()):+.2f}"]]
table("rt_hybrid", "Test accuracy per training seed: final deep model and the hybrids built on it. The last "
      "column is H2 minus the deep model in percentage points.", "tab:p2hybrid",
      ["Seed", "Deep", "H1", "H2", "H3", "H2 $-$ deep"], rows, "lrrrrr")

table("rt_calibration", "Calibration: expected calibration error (15 bins), mean confidence and accuracy. Test "
      "rows are means over three seeds; validation rows are seed 42.", "tab:p2cal",
      ["Model", "Split", "ECE", "Confidence", "Accuracy"],
      [["Phase 1 CNN", "test", f"{ct.loc['p1_cnn', 'ece']:.3f}", f"{ct.loc['p1_cnn', 'mean_confidence']:.3f}", f"{ct.loc['p1_cnn', 'accuracy']:.3f}"],
       ["Final deep model", "test", f"{ct.loc['final_dl', 'ece']:.3f}", f"{ct.loc['final_dl', 'mean_confidence']:.3f}", f"{ct.loc['final_dl', 'accuracy']:.3f}"],
       None,
       ["Base, smoothing 0.1", "val", f"{cv.loc['base', 'ece']:.3f}", f"{cv.loc['base', 'mean_confidence']:.3f}", f"{cv.loc['base', 'accuracy']:.3f}"],
       ["Base, no smoothing", "val", f"{cv.loc['reg_no_ls', 'ece']:.3f}", f"{cv.loc['reg_no_ls', 'mean_confidence']:.3f}", f"{cv.loc['reg_no_ls', 'accuracy']:.3f}"]],
      "llrrr")

cl = {"gaussian_noise": "Gaussian noise (std)", "brightness": "Brightness (added)", "contrast": "Contrast (reduction)",
      "shift": "Shift (pixels)", "occlusion": "Occlusion (square side)"}
rows = []
for i, cname in enumerate(cl):
    if i:
        rows.append(None)
    for level in sorted(rp.loc[cname].index):
        r = rp.loc[(cname, level)]
        rows.append([cl[cname] if level == 0 else "", f"{level:g}", *(f"{r[cols[k]]:.3f}" for k in "PFHZ")])
table("rt_robustness", "Accuracy on corrupted test images (stratified subset of 2{,}000 images, seed 42). "
      "Level 0 is the clean subset.", "tab:p2rob",
      ["Corruption", "Level", "Phase 1 CNN", "Final deep", "H2 fusion", "H0 HOG+SVM"], rows, "llrrrr", wide=True)

table("rt_inference", "Model size and inference time per image on the Colab T4 runtime (median of repeated "
      "passes, batch size 256). The H2 row is end-to-end from raw images, including HOG extraction on the CPU.",
      "tab:p2inf", ["Model", "Parameters", "CPU (ms)", "GPU (ms)"],
      [[n.replace("H2 fusion (end-to-end)", "H2 fusion, end-to-end"), f"{int(r.params):,}", f"{r.cpu_ms_per_image:.2f}",
        "--" if np.isnan(r.gpu_ms_per_image) else f"{r.gpu_ms_per_image:.3f}"] for n, r in inf.iterrows()], "lrrr")

print(f"{len(N)} numbers -> {OUT / 'numbers.tex'}")
for k in ("TestFinal", "TestPone", "TestGainFinalVsPone", "HtwoGain", "RoutedTest", "ErrorReduction", "RecipeOnPoneDelta"):
    print(f"  {k} = {N[k]}")
