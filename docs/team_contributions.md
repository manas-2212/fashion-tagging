# Team Contributions

**DRAFT.** Each paragraph below was drafted by Manas from the repository history and the task split, only to save time.
Each member must rewrite their own paragraph in their own words and correct anything that is wrong.

## Manas
CNN (`src/cnn.py`), training loop with early stopping, parameter counting and inference benchmarking (`src/benchmarking.py`),
their tests, the integrated notebook (`notebook.ipynb`), the integration log, the CNN theory notes and report section, and a first draft
of `src/evaluation.py` to unblock integration (to be reviewed by Rohan).

## Akhil
Data loading, verification and splitting (`src/data.py`), the exploratory data analysis and data-quality checks
(`docs/eda_notes.md`, EDA figures, `report/dataset_eda.tex`), the sanity baselines, and the presentation.

## Anant
Feature extraction and caching (`src/features.py`), the HOG study design and decision rule (`docs/feature_notes.md`).

## Tanish
Classical models, tuning, runtime planning and the overfit check (`src/classical_models.py`, `docs/classical_notes.md`), and the theory document (`docs/theory.md`).

## Rohan
Verified CNN shapes and the benchmarking protocol. Writing the metrics theory notes. Will review the evaluation module and own the Results and Confusion Matrices and Error Analysis sections of the report after the test gate.

## Puneet
Repository, shared configuration, experiment orchestration with the freeze and one-shot test gate (`src/config.py`, `src/experiments.py`, `src/reporting.py`), literature review, README and the LaTeX report.
