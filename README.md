# Inventory Image Tagging via Handcrafted Features and a Small CNN

Phase 1 (machine-learning evaluation) and Phase 2 (deep learning, ablations and a hybrid ML + DL model) on Fashion-MNIST.
The Phase 1 sections below are unchanged; Phase 2 is described in the section "Phase 2" at the end of this file.

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

---

# Phase 2: deep learning, ablation studies and a hybrid model

Phase 1 ended with a 3-block CNN at 0.9274 test accuracy, one training seed, no augmentation and no tuning.
Phase 2 asks four questions and answers each with a controlled experiment:

1. **Architecture.** Does a deeper network with batch normalisation, residual blocks and global average pooling
   separate the confusable classes (T-shirt, Pullover, Coat, Shirt) better, at a similar parameter budget?
2. **Regularisation and data.** Which of augmentation, dropout, weight decay, label smoothing and the learning-rate
   schedule actually contribute? 13 one-factor ablations, 3 seeds each.
3. **Hybrid.** Do handcrafted HOG features add information to the learned CNN embedding? A fusion SVM is compared
   with two controls, and a confidence-gated cascade sends only uncertain images to the hybrid.
4. **Validation.** Are the differences larger than seed-to-seed variation and test-set sampling noise?

## Phase 2 layout

    src/dl_config.py            Phase 2 constants: seeds, recipes, grids                     (Puneet)
    src/dl_models.py            FashionNet family, parameter and receptive-field helpers     (Manas)
    src/dl_train.py             training engine: AdamW/SGD, cosine schedule, early stopping  (Manas)
    src/augment.py              batched crop / flip / random erasing                         (Akhil)
    src/robustness.py           corruption stress test                                       (Akhil)
    src/ablation.py             roadmap and one-factor ablation design                       (Tanish)
    src/ablation_report.py      stage 2: multi-seed tables, figures, model selection         (Tanish)
    src/hybrid.py               embeddings, HOG PCA, fusion SVM, cascade, complementarity    (Anant)
    src/hybrid_stage.py         stage 3: tune and fit hybrids; test-time predictor           (Anant)
    src/dl_evaluation.py        multi-seed statistics, calibration, figures                  (Rohan)
    src/phase2_analysis.py      stage 6: test metrics, CIs, McNemar, robustness, cost        (Rohan)
    src/phase2_experiments.py   run store (resume), freeze file, one-shot test gate          (Puneet)
    src/phase2_reporting.py     CSV / Markdown / LaTeX tables                                (Puneet)
    src/phase2_context.py       data, device and folders shared by all stages                (Manas)
    src/phase2_pipeline.py      stages in protocol order                                     (Manas)
    run_phase2.py               command-line entry point                                     (Manas)
    notebook_phase2.ipynb       Colab notebook for the full study                            (Manas)
    tests/test_<module>.py      one test file per Phase 2 module
    docs/phase2/                runbook, design notes and viva questions per owner

## How to run Phase 2

    python -m pytest tests -q            # unit tests
    python run_phase2.py --smoke         # every stage on tiny subsets, a few minutes
    python run_phase2.py                 # full study (GPU recommended)

On Google Colab open `notebook_phase2.ipynb` and follow `docs/phase2/PHASE2_RUNBOOK.md`.

## Phase 2 protocol

- Same 54,000 / 6,000 split as Phase 1 (hash-checked). Every configuration is trained with seeds 42, 43 and 44.
- Selection of the final deep model, hybrid hyper-parameters and the cascade threshold use validation data only.
- `results/phase2/selected_configs_phase2.json` is frozen before `run_phase2_test_gate` opens the test set once
  for the frozen list of models. The test set had already been used once at the Phase 1 gate; this is disclosed.
- Reported: mean +- standard deviation over seeds, bootstrap confidence intervals, Bonferroni-corrected McNemar
  tests, per-class recall, calibration (ECE), robustness to corruptions, parameter counts and inference time.

## Phase 2 results

Full run on a Colab Tesla T4, 16 November 2026: 48 training runs, test gate opened once on all 10,000 test
images. Tables: `results/phase2/`; figures: `figures/phase2/`; report: `report/Phase2Report.pdf`.

| Model (test set, mean +- std over 3 seeds) | Accuracy | Macro-F1 | 95% CI (seed 42) | Shirt recall | In-group confusion |
|---|---|---|---|---|---|
| H0: PCA-HOG + SVM | 0.9048 +- 0.0000 | 0.9047 | [0.8990, 0.9102] | 0.708 | 15.3% |
| Phase 1 CNN (re-trained) | 0.9252 +- 0.0034 | 0.9250 | [0.9231, 0.9332] | 0.757 | 12.2% |
| Final deep model | 0.9488 +- 0.0015 | 0.9487 | [0.9445, 0.9532] | 0.834 | 8.5% |
| H1: CNN embedding + SVM | 0.9495 +- 0.0009 | 0.9494 | [0.9454, 0.9540] | 0.833 | 8.4% |
| H2: embedding + HOG fusion SVM | 0.9502 +- 0.0020 | 0.9501 | [0.9455, 0.9539] | 0.839 | 8.3% |
| H3: confidence-gated cascade | 0.9502 +- 0.0021 | 0.9501 | [0.9455, 0.9539] | 0.839 | 8.3% |

For reference, the single Phase 1 run of the small CNN scored 0.9274 and the Phase 1 pixel RBF-SVM 0.8953.

Findings, as measured:

1. **Architecture.** The final deep model (13 convolution layers, 696,042 parameters) is
   +2.36 pp above the re-trained Phase 1 CNN and makes 32% fewer errors
   (McNemar p < 0.0001). Shirt recall rises from 0.757 to 0.834.
2. **Recipe vs architecture.** The Phase 2 training recipe alone made the Phase 1 CNN slightly worse
   (-0.28 pp validation macro-F1); the new architecture under the same recipe adds +1.96 pp.
3. **Ablations (validation, paired over seeds).** Narrow network -1.10 pp, no BatchNorm -0.70 pp,
   no augmentation -0.72 pp, constant learning rate -0.68 pp, no label smoothing -0.29 pp,
   no random erasing -0.16 pp. No measurable effect: horizontal flip, dropout, weight decay, residual
   shortcuts at 7 layers, SGD instead of AdamW.
4. **Hybrid.** The embedding + HOG fusion SVM is above the deep model in all three seeds by +0.15 pp on
   average, which is not statistically significant (McNemar p = 0.20; fusion vs embedding-only SVM
   p = 0.90). The confidence-gated cascade reaches the same accuracy while routing 5.7% of the
   images to the hybrid.
5. **Costs.** The final model is under-confident (ECE 0.067 vs 0.013, caused by label smoothing),
   7.1x slower on the GPU, and far less robust to Gaussian noise than the Phase 1 CNN
   (0.381 vs 0.855 accuracy at noise std 0.10). It is more robust to shifts and occlusion.

Limitations are listed in `docs/phase2/PHASE2_RUNBOOK.md` and in the report (test set already used at the Phase 1
gate; three seeds; untuned recipe; one-factor ablations around the 7-layer base).
