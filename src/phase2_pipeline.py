"""Phase 2 pipeline: the stages in protocol order (owner: Manas, integration).

  1. stage_train     (this file)            train every configuration x seed, resumable     [train + validation]
  2. stage_summary   (ablation_report.py)   multi-seed tables, ablations, pick the final DL [validation]
  3. stage_hybrid    (hybrid_stage.py)      tune and fit the hybrid models and the cascade  [train + validation]
  4. stage_freeze    (this file)            write the freeze file
  5. stage_test_gate (this file)            one-shot evaluation on the official test set    [test, once]
  6. stage_analysis  (phase2_analysis.py)   CIs, McNemar, calibration, robustness, cost     [describes gate outputs]

Each stage reads what it needs from disk, so a stage can be re-run after a restart.
"""
from __future__ import annotations

import json
import time


from src.config import SEED
from src.dl_config import (BASE_ARCH, P1_RECIPE, P2_RECIPE)
from src.dl_train import run_single
from src.phase2_experiments import (freeze_phase2, run_phase2_test_gate)
from src.phase2_context import (GATE_KINDS, H0_NAME, Phase2Context, get_final_name, gate_name, load_test)
from src.ablation_report import stage_summary
from src.hybrid_stage import make_predict_all, stage_hybrid
from src.phase2_analysis import stage_analysis

# ----------------------------------------------------------------------------- stage 1: training
def stage_train(ctx: Phase2Context, verbose: bool = True) -> list[dict]:
    """Train every (configuration, seed) pair that is not stored yet. Safe to interrupt and re-run."""
    jobs = [(cfg, seed) for cfg in ctx.configs for seed in ctx.seeds]
    todo = [(c, s) for c, s in jobs if not ctx.store.is_done(c, s, len(ctx.data["y_tr"]))]
    print(f"{len(jobs)} runs in total, {len(jobs) - len(todo)} already stored, {len(todo)} to train")
    t0, done, records = time.perf_counter(), 0, []
    for cfg, seed in jobs:
        was_done = ctx.store.is_done(cfg, seed, len(ctx.data["y_tr"]))
        rec = ctx.store.run_or_load(cfg, seed, ctx.data, ctx.device, run_single, verbose=verbose)
        records.append(rec)
        if not was_done:
            done += 1
            elapsed = time.perf_counter() - t0
            left = (len(todo) - done) * elapsed / done
            print(f"  -> val acc {rec['val_accuracy']:.4f} macro-F1 {rec['val_macro_f1']:.4f} | best epoch "
                  f"{rec['best_epoch']}/{rec['epochs_run']} | {rec['train_seconds']:.0f}s | "
                  f"{done}/{len(todo)} new runs, about {left / 60:.0f} min left")
    return records


# ----------------------------------------------------------------------------- stage 4: freeze
def stage_freeze(ctx: Phase2Context) -> str:
    """Freeze the Phase 2 selection. After this call nothing may be tuned any more."""
    final_name = get_final_name(ctx)
    spec_path = ctx.results_dir / "hybrid_spec.json"
    if not spec_path.is_file():
        raise RuntimeError("run stage_hybrid first (hybrid_spec.json is missing)")
    spec = json.loads(spec_path.read_text())
    if spec["final_config"] != final_name:
        raise RuntimeError("hybrid_spec.json was built for a different final model; re-run stage_hybrid")
    gate_models = [gate_name(kind, seed) for seed in ctx.seeds for kind in GATE_KINDS] + [H0_NAME]
    final_cfg = next(c for c in ctx.configs if c.name == final_name)
    selection = {
        "phase": 2, "final_config": final_name, "final_config_dict": final_cfg.to_dict(),
        "selection_preview": json.loads((ctx.results_dir / "selection_preview.json").read_text()),
        "seeds": [int(s) for s in ctx.seeds], "primary_seed": int(ctx.seeds[0]), "hybrid": spec,
        "gate_models": gate_models, "p1_recipe": P1_RECIPE, "p2_recipe": P2_RECIPE, "base_arch": BASE_ARCH,
        "n_train": int(len(ctx.data["y_tr"])), "n_val": int(len(ctx.data["y_val"])),
        "split_hash": ctx.data["split_hash"], "smoke": bool(ctx.smoke),
        "configurations_trained": ctx.config_names,
    }
    sha = freeze_phase2(selection, ctx.results_dir)
    print(f"Phase 2 selection frozen: {len(gate_models)} gate models, sha256 = {sha}")
    return sha


# ----------------------------------------------------------------------------- stage 5: test gate
def stage_test_gate(ctx: Phase2Context) -> dict:
    """Open the test set once for the frozen models."""
    return run_phase2_test_gate(make_predict_all(ctx), load_test, ctx.results_dir,
                                test_subset_n=ctx.settings["test_n"], seed=SEED)


# ----------------------------------------------------------------------------- everything
STAGES = ("train", "summary", "hybrid", "freeze", "gate", "analysis")


def run_stages(ctx: Phase2Context, stages=STAGES, verbose: bool = True) -> dict:
    """Run the requested stages in protocol order."""
    out = {}
    for stage in STAGES:
        if stage not in stages:
            continue
        print(f"\n===== Phase 2 stage: {stage} =====")
        t0 = time.perf_counter()
        if stage == "train":
            out[stage] = stage_train(ctx, verbose=verbose)
        elif stage == "summary":
            out[stage] = stage_summary(ctx)
        elif stage == "hybrid":
            out[stage] = stage_hybrid(ctx)
        elif stage == "freeze":
            out[stage] = stage_freeze(ctx)
        elif stage == "gate":
            out[stage] = stage_test_gate(ctx)
        elif stage == "analysis":
            out[stage] = stage_analysis(ctx)
        print(f"===== stage {stage} finished in {(time.perf_counter() - t0) / 60:.1f} min =====")
    return out
