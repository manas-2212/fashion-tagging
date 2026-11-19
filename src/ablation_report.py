"""Stage 2: turn the stored training runs into the ablation study (owner: Tanish).

Reads every run record, builds the multi-seed summary, the paired ablation deltas, the architecture table and the
figures, and applies the frozen selection rule for the final deep model. Validation data only.
"""
from __future__ import annotations


import pandas as pd
import torch

from src import ablation
from src.augment import plot_augmentation_examples
from src.dl_evaluation import (ablation_deltas, plot_ablation_deltas, plot_generalisation_gap,
                               plot_learning_curves_compare, plot_roadmap, select_final_config, summarise_runs)
from src.dl_models import build_model, embedding_dim, parameter_breakdown, receptive_field
from src.phase2_reporting import flatten_records, mean_std, save_json, save_table
from src.phase2_context import Phase2Context


def architecture_summary(ctx: Phase2Context, summary: pd.DataFrame) -> pd.DataFrame:
    """Parameters, where they sit, receptive field and validation score for the architecture variants."""
    rows = []
    for name in ("p1_cnn_p1_recipe", "arch_plain", "base", "arch_deeper", "arch_narrow", "arch_no_bn"):
        cfg = next((c for c in ctx.configs if c.name == name), None)
        if cfg is None or name not in set(summary["name"]):
            continue
        model = build_model(cfg.arch, **cfg.arch_kwargs)
        pb = parameter_breakdown(model)
        s = summary.set_index("name").loc[name]
        rows.append({"name": name, "conv_layers": sum(isinstance(m, torch.nn.Conv2d) and m.kernel_size[0] == 3
                                                      for m in model.modules()),
                     "params_total": pb["total"], "params_conv": pb["conv"], "params_norm": pb["norm"],
                     "params_dense": pb["dense"], "dense_share": pb["dense"] / pb["total"],
                     "receptive_field": receptive_field(model), "embedding_dim": embedding_dim(model),
                     "val_macro_f1_mean": s["val_macro_f1_mean"], "val_macro_f1_std": s["val_macro_f1_std"]})
    return pd.DataFrame(rows)


def stage_summary(ctx: Phase2Context) -> dict:
    """Aggregate the stored runs, write tables and figures, and select the final deep model (validation only)."""
    records = ctx.store.records(names=set(ctx.config_names), seeds=set(ctx.seeds))
    expected = len(ctx.configs) * len(ctx.seeds)
    if len(records) != expected:
        raise RuntimeError(f"{len(records)} of {expected} runs are stored; finish stage_train first")
    summary = summarise_runs(records)
    save_table(flatten_records(records), "run_records", ctx.results_dir)

    view = summary.copy()
    view["val_macro_f1"] = [mean_std(m, s) for m, s in zip(view["val_macro_f1_mean"], view["val_macro_f1_std"])]
    view["val_accuracy"] = [mean_std(m, s) for m, s in zip(view["val_accuracy_mean"], view["val_accuracy_std"])]
    latex = view[["name", "val_macro_f1", "val_accuracy", "gap_mean", "shirt_recall_mean", "n_params"]].rename(
        columns={"name": "configuration", "val_macro_f1": "val macro-F1", "val_accuracy": "val accuracy",
                 "gap_mean": "train-val gap", "shirt_recall_mean": "Shirt recall", "n_params": "parameters"})
    save_table(summary, "ablation_summary", ctx.results_dir, ctx.tables_dir, wide=True, latex_df=latex,
               caption=f"Validation results of every Phase 2 configuration (mean +- std over {len(ctx.seeds)} seeds).",
               label="tab:p2_ablation_summary")

    out = {"summary": summary, "records": records}
    if ablation.BASE_NAME in set(summary["name"]):
        deltas = ablation_deltas(records, ablation.BASE_NAME)
        if not deltas.empty:
            deltas.insert(1, "group", deltas["name"].map(lambda n: ablation.ABLATIONS.get(n, ("roadmap",))[0]))
            deltas.insert(2, "change", deltas["name"].map(lambda n: ablation.ABLATIONS.get(n, ("", "", n))[2]))
            d_acc = ablation_deltas(records, ablation.BASE_NAME, metric="val_accuracy").set_index("name")
            deltas["delta_accuracy_mean"] = deltas["name"].map(d_acc["delta_mean"])
            lat = deltas[deltas["name"].isin(ablation.ABLATIONS)][
                ["group", "change", "delta_mean", "delta_min", "delta_max", "consistent_sign"]].rename(
                columns={"delta_mean": "delta F1 (mean)", "delta_min": "min", "delta_max": "max",
                         "consistent_sign": "all seeds agree"})
            save_table(deltas, "ablation_deltas", ctx.results_dir, ctx.tables_dir, wide=True, latex_df=lat,
                       caption="One-factor ablations: paired change in validation macro-F1 relative to the base "
                               "(variant minus base, same seed).", label="tab:p2_ablation_deltas")
            if deltas["name"].isin(ablation.ABLATIONS).any():
                ctx.savefig(plot_ablation_deltas(deltas, ablation.ABLATIONS), "ablation_deltas.png")
            out["deltas"] = deltas

    roadmap = [c.name for c in ablation.roadmap_configs() if c.name in set(summary["name"])]
    if len(roadmap) >= 2:
        ctx.savefig(plot_roadmap(summary, roadmap, ablation.ROADMAP_LABELS), "architecture_roadmap.png")
        save_table(summary[summary["name"].isin(roadmap)], "architecture_roadmap", ctx.results_dir)
    gap_names = [n for n in ("p1_cnn_p1_recipe", "base", "aug_none", "reg_dropout0", "reg_no_wd", "reg_no_ls",
                             "arch_no_bn") if n in set(summary["name"])]
    if len(gap_names) >= 2:
        ctx.savefig(plot_generalisation_gap(summary, gap_names), "generalisation_gap.png")
    primary = {r["name"]: r for r in records if r["seed"] == ctx.seeds[0]}
    curves = {label: primary[n]["history"] for n, label in
              (("p1_cnn_p1_recipe", "Phase 1 CNN"), ("aug_none", "base, no augmentation"), ("base", "base"))
              if n in primary}
    if curves:
        ctx.savefig(plot_learning_curves_compare(curves, f"Learning curves (seed {ctx.seeds[0]})"),
                    "learning_curves.png")
    ctx.savefig(plot_augmentation_examples(ctx.data["X_train"], ctx.data["mean"], ctx.data["std"]),
                "augmentation_examples.png")
    arch = architecture_summary(ctx, summary)
    if not arch.empty:
        save_table(arch, "architecture_summary", ctx.results_dir, ctx.tables_dir, wide=True,
                   caption="Architecture variants: size, where the parameters sit, receptive field and validation "
                           "macro-F1.", label="tab:p2_architectures")

    final_name = select_final_config(summary, required_seeds=len(ctx.seeds))
    row = summary.set_index("name").loc[final_name]
    save_json({"final_config": final_name, "rule": "highest mean validation macro-F1 over seeds; ties: higher "
               "mean accuracy, then fewer parameters, then name; diverged or incomplete configurations excluded",
               "val_macro_f1_mean": row["val_macro_f1_mean"], "val_macro_f1_std": row["val_macro_f1_std"],
               "val_accuracy_mean": row["val_accuracy_mean"], "n_params": int(row["n_params"])},
              ctx.results_dir / "selection_preview.json")
    print(f"selected final deep model: {final_name} (val macro-F1 {row['val_macro_f1_mean']:.4f} "
          f"+- {row['val_macro_f1_std']:.4f}, {int(row['n_params']):,} parameters)")
    out["final_name"] = final_name
    return out
