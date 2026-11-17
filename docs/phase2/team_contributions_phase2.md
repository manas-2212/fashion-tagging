# Team contributions: Phase 2

Each member owns code, its tests and its design notes, and must be able to explain all three in the viva.
Every member has core technical work; nobody only writes slides or the report.

| Member | Enrollment | Technical ownership | Files |
|---|---|---|---|
| Manas Selukar | 2401010259 | Architecture design (FashionNet), training engine, pipeline integration, Colab run | `src/dl_models.py`, `src/dl_train.py`, `src/phase2_context.py`, `src/phase2_pipeline.py`, `run_phase2.py`, `notebook_phase2.ipynb`, `tests/test_dl_models.py`, `tests/test_dl_train.py`, `tests/test_phase2_leakage.py`, `docs/phase2/architecture_notes.md`, `docs/phase2/PHASE2_RUNBOOK.md` |
| Anant Singh | 2401010067 | Hybrid ML + DL: embeddings, HOG PCA fusion SVM, controls, confidence-gated cascade | `src/hybrid.py`, `src/hybrid_stage.py`, `tests/test_hybrid.py`, `docs/phase2/hybrid_notes.md` |
| Akhil Sharma | 2401020084 | DL data pipeline: batched augmentation; robustness stress test | `src/augment.py`, `src/robustness.py`, `tests/test_augment.py`, `tests/test_robustness.py`, `docs/phase2/augmentation_notes.md` |
| Tanish Rao | 2401010474 | Ablation study design, regularisation theory, ablation tables and model selection stage | `src/ablation.py`, `src/ablation_report.py`, `tests/test_ablation.py`, `docs/phase2/ablation_design.md` |
| Rohan Kumar | 2401020103 | Multi-seed statistics, calibration, significance tests, error analysis, test analysis stage | `src/dl_evaluation.py`, `src/phase2_analysis.py`, `tests/test_dl_evaluation.py`, `docs/phase2/evaluation_notes.md`, `docs/viva_qa.md` (HOG answers corrected) |
| Puneet Takhar | 2401010358 | Experiment configuration, resumable run store, freeze and one-shot test gate, reporting tables, DL literature | `src/dl_config.py`, `src/phase2_experiments.py`, `src/phase2_reporting.py`, `tests/test_phase2_experiments.py`, `docs/phase2/protocol_notes.md`, `README.md`, `.gitignore`, `src/config.py` (comment fix) |

Shared, after the full run: the Phase 2 report (`report/`), slides and the results interpretation. Each member
writes the report subsection for their own component.

**Draft.** This table was prepared from the task split. Each member should correct it if their actual work differs.
