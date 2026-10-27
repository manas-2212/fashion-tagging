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

The final notebook is `fashion_tagging_phase1_final.ipynb`; `notebook.ipynb` is the earlier project notebook. The final notebook also clones the repository by itself if opened standalone in Colab. The recorded full experiment run is saved in the final notebook outputs; an independent fresh-runtime rerun was not evidenced in this repository snapshot.

### Local (macOS / Linux)

    git clone https://github.com/manas-2212/fashion-tagging.git
    cd fashion-tagging
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

`requirements.txt` is not version-pinned.

## How to run

1. Open and use `fashion_tagging_phase1_final.ipynb` (not the older `notebook.ipynb`) for the completed experiment. The final classical fit times ranged from 0.02 s to 393.61 s; HOG RBF-SVM was runtime-gated to 11,770 training images. Do not rerun the official test gate merely to regenerate report outputs; restore the saved `results/test_gate_done.flag` and prediction artifacts from the completed Colab run first. Sections:
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

The report includes `report/results_table.tex` and `report/inference_table.tex`, plus the model-comparison, CNN confusion-matrix, per-class-recall, and learning-curve figures. The notebook output is the source for the reported values. Some machine-generated artifacts (including the saved prediction arrays and test-gate completion flag) were not included in the ZIP snapshot; restore them from the completed Colab runtime for full reproducibility.

## Generated files (after a full run)

`results/`: `tuning_<model>_<rep>.csv`, `fit_info.json`, `selected_configs.json`, `test_gate_done.flag`, `test_gate_outputs.npz`, `metrics_with_ci.csv`, `mcnemar_cnn_vs_classical.csv`, `per_class_recall.csv`, `confusion_matrices.npz`, `benchmarks.json`, `results_table.csv/.md`, `environment.json`, `split_audit.json`, `data_quality.json`.

## Verification status

The final notebook output records a full 54,000/6,000/10,000 run and a Tesla T4 CNN training run. However, this ZIP snapshot does not include all generated outputs and the integration log still contains earlier scaled-down-run history. An independent fresh-Colab verification and a clean full test-suite run are not evidenced by this archive; check these before claiming independent reproducibility.

## Disclosed deviations

The CNN uses one fixed configuration (Adam, learning rate 0.001, batch size 128, dropout 0.5) rather than a hyperparameter search. Classical searches use small grids; the SVM search uses a stratified 10,000-image subset of the 54,000-image training split. The HOG RBF-SVM final fit was runtime-gated to 11,770 training images because the estimated full fit exceeded 30 minutes. Pixel logistic regression reached its 500-iteration limit with a convergence warning. These deviations are retained and disclosed rather than silently changing the frozen protocol.

## Results summary

The following results were recorded in the final notebook output after the one-shot test gate. Confidence intervals are 95% percentile bootstrap intervals from 1,000 resamples.

| Model | Representation | Accuracy (95% CI) | Macro-F1 (95% CI) |
|---|---|---:|---:|
| Logistic regression | pixels | 0.8411 (0.8338–0.8482) | 0.8403 (0.8336–0.8474) |
| k-NN | pixels | 0.8500 (0.8434–0.8567) | 0.8493 (0.8430–0.8559) |
| Random forest | pixels | 0.8756 (0.8693–0.8818) | 0.8741 (0.8682–0.8800) |
| RBF-SVM | pixels | 0.8953 (0.8887–0.9014) | 0.8951 (0.8889–0.9008) |
| Logistic regression | HOG | 0.8698 (0.8632–0.8766) | 0.8702 (0.8637–0.8764) |
| k-NN | HOG | 0.8653 (0.8583–0.8718) | 0.8621 (0.8555–0.8682) |
| Random forest | HOG | 0.8636 (0.8571–0.8703) | 0.8618 (0.8554–0.8682) |
| RBF-SVM | HOG | 0.8817 (0.8753–0.8879) | 0.8816 (0.8755–0.8877) |
| Small CNN | learned | 0.9274 (0.9219–0.9327) | 0.9269 (0.9214–0.9321) |

## Limitations

The evaluation covers one dataset (Fashion-MNIST, 28×28 grayscale), one official split and one seed; training-seed variability is not measured. The CNN hyperparameters are not tuned, classical search grids are small, pixel logistic regression emitted a convergence warning at the 500-iteration cap, and the HOG RBF-SVM was fit on 11,770 rather than 54,000 training images due to the runtime gate. Inference timings are single timed passes after warm-up and are preliminary; CNN tensor timings exclude preprocessing and are not directly comparable to classical end-to-end timings. Results may not transfer to real inventory photographs.

## References

[1] H. Xiao, K. Rasul and R. Vollgraf, "Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning
Algorithms," arXiv:1708.07747, 2017. https://arxiv.org/abs/1708.07747

[2] Y. LeCun, L. Bottou, Y. Bengio and P. Haffner, "Gradient-based learning applied to document recognition,"
Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998. DOI 10.1109/5.726791
