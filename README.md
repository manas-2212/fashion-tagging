# Inventory Image Tagging via Handcrafted Features and a Small CNN

Phase 1 (machine-learning evaluation only) on Fashion-MNIST.

## Overview

We test whether a small CNN, which learns spatial features, recognises similar clothing classes better than classical
models (logistic regression, k-NN, random forest, RBF-SVM) on flattened pixels or HOG features. The outcome is open:
we report what we measure. The final comparison is 9 models: 4 classical models x 2 representations + 1 CNN.

## Objectives

1. Review the relevant literature (`docs/literature_review.md`).
2. Describe and explore the dataset (EDA, data-quality checks).
3. Build and study handcrafted features (pixels, HOG).
4. Apply and compare classical models and a small CNN under one protocol.
5. Report accuracy, macro-F1, confidence intervals, error analysis and inference cost.
6. Deliver clean code, tests, an IEEE-style report, slides and a viva question bank.

## Dataset

Fashion-MNIST (Xiao et al., 2017): 28x28 grayscale images, 10 classes, 60,000 official training and 10,000 official test
images. It is downloaded with torchvision into `data/` (git-ignored). Features are cached as `data/cache/*.npy`.

## Repository structure

    fashion-tagging/
      README.md  requirements.txt  .gitignore  notebook.ipynb
      src/        config.py  data.py  features.py  classical_models.py  evaluation.py
                  cnn.py  benchmarking.py  experiments.py  reporting.py
      tests/      test_<module>.py  test_leakage_and_integration.py
      results/    tuning, fit info, frozen selection, final tables
      figures/    EDA and result figures
      docs/       literature review, notes, team contributions
      report/     main.tex  related_work.tex  references.bib
      data/       (git-ignored) dataset and feature cache

All nine modules exist. `docs/` holds notes and the question banks (`viva_qa.md`, `slides_outline.md`, `todo_by_owner.md`, `integration_log.md`).

## Setup

### Google Colab

Use Runtime > Change runtime type > GPU for the CNN. Then in the first cell:

    !git clone https://github.com/manas-2212/fashion-tagging.git
    %cd fashion-tagging
    !pip install -r requirements.txt

The notebook also clones the repository by itself if it is opened standalone in Colab. This has not yet been tested on a fresh Colab runtime.

### Local (macOS / Linux)

    git clone https://github.com/manas-2212/fashion-tagging.git
    cd fashion-tagging
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

`requirements.txt` is not version-pinned.

## How to run

1. Run `notebook.ipynb` top to bottom (total runtime depends on the hardware and has not been measured end to end; Tanish's dry run in `docs/classical_notes.md` found every final classical fit under 4 minutes on the 54,000-image split). Sections:
   0 setup and environment (`results/environment.json`) | 1 load and integrity checks | 2 EDA and data quality | 3 split and audit |
   4 features | 5 classical tuning | 6 classical final fits and overfit check | 7 CNN training (train and validation only) |
   **8 freeze, then the one-shot test gate** | 9 final evaluation (bootstrap CIs, McNemar) | 10 inference benchmarks and parameter counts |
   11 error analysis | 12 results table, LaTeX tables, comparison plot | 13 limitations and reproducibility.
   The held-out images are first loaded in section 8. Run sections 0-7 first and check them before section 8: the gate can run only once
   (delete `results/test_gate_done.flag` only if you intend to redo the protocol).
2. Smoke tests, from the repository root:

        python -m pytest tests -q

   Three tests in `tests/test_data.py` are skipped unless the real dataset can be loaded (torchvision, or the four IDX files in `data/idx_fallback/`).

## Experimental protocol

- Official split only. Validation = stratified 10% of the 60,000 official training images: 54,000 train / 6,000
  validation, seed 42. The 10,000 test images are used only inside `run_test_gate`.
- Nothing is fitted on validation or test data (scalers, mean/std, PCA, selection).
- Selection: highest validation macro-F1; validation accuracy breaks ties. The SVM is tuned on a stratified
  10,000-image subset of the training split.
- Runtime gate: before every final classical fit the fit time is estimated; if it exceeds 30 minutes the model is fitted
  on a disclosed stratified subset. `n_train_used` is printed and saved in `results/fit_info.json`.
- Final models are trained on the 54,000-image train split only (no refit on train + validation).
- Freeze and test gate: `results/selected_configs.json` is written first; all 9 models are then evaluated on the test
  images exactly once. The gate refuses to run without the freeze file or if `results/test_gate_done.flag` exists.

## Report

    cd report && latexmk -pdf main.tex

The notebook writes `report/results_table.tex` and `report/inference_table.tex` and the result figures in `figures/`; `main.tex` includes them automatically and shows placeholders until they exist.

## Generated files (after a full run)

`results/`: `tuning_<model>_<rep>.csv`, `fit_info.json`, `selected_configs.json`, `test_gate_done.flag`, `test_gate_outputs.npz`, `metrics_with_ci.csv`, `mcnemar_cnn_vs_classical.csv`, `per_class_recall.csv`, `confusion_matrices.npz`, `benchmarks.json`, `results_table.csv/.md`, `environment.json`, `split_audit.json`, `data_quality.json`.

## Verification status

Unit and integration tests pass on the merged code. The whole notebook has been executed end to end only on a scaled-down dataset (see `docs/integration_log.md`); the full run on 54,000/6,000/10,000 images, the GPU path and a fresh Colab runtime are still to be verified.

## Disclosed deviations

To be filled in after the experiments (TBD). Known by design: the CNN hyperparameters are not tuned while the classical models get small grids; hyperparameter grids have at most four values; SVM hyperparameters are tuned on a stratified subset of
10,000 of the 54,000 training images. Any final fit that hits the runtime gate, with its exact subset size, will be
listed here from `results/fit_info.json`.

## Results summary

Test results will be filled in from `results/results_table.md` (generated by the notebook) after the single test-gate run.

| Model | Representation | Accuracy | Macro-F1 |
|---|---|---|---|
| Logistic regression | pixels | TBD | TBD |
| k-NN | pixels | TBD | TBD |
| Random forest | pixels | TBD | TBD |
| RBF-SVM | pixels | TBD | TBD |
| Logistic regression | HOG | TBD | TBD |
| k-NN | HOG | TBD | TBD |
| Random forest | HOG | TBD | TBD |
| RBF-SVM | HOG | TBD | TBD |
| Small CNN | learned | TBD | TBD |

## Limitations

TBD after results. Known: one dataset (Fashion-MNIST, 28x28 grayscale), one seed, one official split, and a small CNN
only; results do not necessarily transfer to real inventory photos.

## References

[1] H. Xiao, K. Rasul and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
Algorithms," arXiv:1708.07747, 2017. https://arxiv.org/abs/1708.07747

[2] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition,"
Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998. DOI 10.1109/5.726791
