# Phase 2 runbook (Manas)

How to run Phase 2 from a clean machine to a finished results folder.

## 1. What Phase 2 adds

| Criterion | Where it is addressed |
|---|---|
| Architecture logic | `src/dl_models.py`, `docs/phase2/architecture_notes.md`, roadmap figure, architecture table |
| Ablation studies | `src/ablation.py`, `src/ablation_report.py`: 13 one-factor ablations x 3 seeds |
| DL dataset and regularisation | `src/augment.py`, dropout / weight decay / label smoothing / BN ablations, generalisation-gap figure |
| Hybrid innovation | `src/hybrid.py`, `src/hybrid_stage.py`: fusion SVM with two controls, confidence-gated cascade |
| Technical validation | 3 seeds, freeze + one-shot gate, bootstrap CIs, McNemar, calibration, robustness, inference cost |
| Theoretical rigor | `docs/phase2/*.md` (theory per component, viva questions per owner) |
| Code quality | 9 new test files (about 110 tests), single config file, resumable runs |

## 2. Run on Google Colab (recommended)

1. Open `notebook_phase2.ipynb` in Colab. Runtime > Change runtime type > **T4 GPU**.
2. Run the cells from top to bottom. The first cell asks for the project zip (or set `REPO_URL` to clone GitHub).
3. Step 1 runs the unit tests and a smoke run (tiny subsets). Do not continue if it fails.
4. Step 2 trains 48 runs (16 configurations x 3 seeds). Expect roughly 2.5 to 4 hours on a T4 (an estimate of the time left is printed after every run).
   **If Colab disconnects:** run Step 1 again, then Step 2: finished runs are skipped (results are on Google Drive).
5. Steps 3 and 4 use validation data only. Look at the tables and figures before going on.
6. Step 5 freezes the selection and opens the test set **once**. It refuses to run a second time.
7. Step 7 downloads `phase2_results.zip`. Unzip it into the repository root: it creates `results/phase2/`,
   `figures/phase2/` and `report/phase2_tables/`.

Short on GPU time? `make_context(seeds=(42, 43))` cuts the training time by a third and still gives a spread;
say so in the report.

## 3. Run from a terminal

    python -m pytest tests -q                 # unit tests (Phase 2 tests need no dataset)
    python run_phase2.py --smoke              # every stage on tiny subsets -> results/phase2_smoke/
    python run_phase2.py                      # full study -> results/phase2/, figures/phase2/, report/phase2_tables/
    python run_phase2.py --stages train       # only training; re-run to resume
    python run_phase2.py --stages summary hybrid          # validation-only stages
    python run_phase2.py --stages freeze gate analysis    # freeze, one-shot test gate, analysis

## 4. Stages, inputs and outputs

| Stage | Module | Data | Main outputs in `results/phase2/` |
|---|---|---|---|
| train | `phase2_pipeline.stage_train` | train, val | `runs/*.json`, `checkpoints/*.pt`, `val_probs/*.npy` |
| summary | `ablation_report.stage_summary` | val | `run_records.csv`, `ablation_summary.csv`, `ablation_deltas.csv`, `architecture_summary.csv`, `selection_preview.json` |
| hybrid | `hybrid_stage.stage_hybrid` | train, val | `hybrid_tuning.csv`, `hybrid_validation.csv`, `hybrid_complementarity_val.csv`, `cascade_thresholds.csv`, `hybrid_spec.json` |
| freeze | `phase2_pipeline.stage_freeze` | none | `selected_configs_phase2.json` |
| gate | `phase2_pipeline.stage_test_gate` | **test, once** | `test_gate_outputs_phase2.npz`, `test_gate_phase2_done.flag` |
| analysis | `phase2_analysis.stage_analysis` | gate outputs | `test_metrics.csv`, `test_metrics_per_seed.csv`, `mcnemar_phase2.csv`, `per_class_recall_phase2.csv`, `calibration.csv`, `robustness.csv`, `inference_phase2.csv`, `phase2_summary.json` |

Figures (`figures/phase2/`): `architecture_roadmap`, `ablation_deltas`, `generalisation_gap`, `learning_curves`,
`augmentation_examples`, `cascade_threshold`, `test_summary`, `per_class_recall_phase2`, `cm_final_dl`,
`reliability`, `robustness`.

## 5. What is fixed before any run

* Split: the Phase 1 split (54,000 / 6,000, seed 42); the pipeline checks its SHA-256 against Phase 1.
* Seeds 42, 43, 44 for every configuration.
* Base model and recipe: `src/dl_config.py` (`BASE_ARCH`, `P2_RECIPE`).
* Selection rule: highest mean validation macro-F1 over seeds; ties: accuracy, then fewer parameters.
* Hybrid hyper-parameters and the cascade threshold: chosen on validation with seed 42, re-used for the other seeds.

## 6. Disclosed limitations (carry these into the report)

1. The test set was already used at the Phase 1 gate (see `protocol_notes.md`).
2. Three seeds give a spread, not a significance test.
3. The recipe is a fixed standard choice; no hyper-parameter search was run for the base model.
4. Hybrid SVMs are tuned on a 10,000-image subset; the SVM is fitted on embeddings of images the CNN was trained on.
5. One-factor ablations do not measure interactions between components.
6. 28x28 grayscale product images; the robustness test is a first indication, not evidence about real photographs.
7. Timings depend on the Colab hardware of the day.
