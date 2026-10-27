# Slides outline (Akhil assembles; each owner supplies their own slide text)

Rules: every number on a slide is copied from a file in `results/` (never retyped); anything not yet measured stays **TBD**; one message per slide; figures are the saved PNGs in `figures/`.

| # | Title | Owner | Content | Source |
|---|---|---|---|---|
| 1 | Title and team | Puneet | Project title, names, roles | `docs/team_contributions.md` |
| 2 | Problem and question | Puneet | Does a small CNN recognise similar clothing classes better than classical models on pixels or HOG? Outcome open. | README |
| 3 | Literature | Puneet | Fashion-MNIST (Xiao et al.); learned vs hand-designed features (LeCun et al.); neither paper ran our experiment | `docs/literature_review.md` |
| 4 | Dataset, EDA and features | Akhil, Anant | 60k/10k, balanced; 54k/6k stratified split; 0 exact duplicates; PCA overlap of Shirt/Pullover/Coat; pixels (784) vs HOG (1,296) | `docs/eda_notes.md`, `docs/feature_notes.md`, `figures/eda_*.png`, `figures/hog_example.png` |
| 5 | Classical models | Tanish | Slide text below | `docs/classical_notes.md` |
| 6 | CNN | Manas | Slide text below | `docs/cnn_notes.md`, `figures/cnn_learning_curves.png` |
| 7 | Protocol | Puneet | Official split, no fitting on val/test, selection by validation macro-F1, freeze then one-shot test gate, runtime gate, SVM tuned on 10k subset | README, `results/selected_configs.json` |
| 8 | Results | Rohan | Table of 9 models with 95% CIs (TBD) | `results/metrics_with_ci.csv`, `report/results_table.tex` |
| 9 | Is the CNN really different? | Rohan | McNemar CNN vs each classical model, Bonferroni-adjusted (TBD) | `results/mcnemar_cnn_vs_classical.csv` |
| 10 | Error analysis | Rohan | Confusion matrix, top confusions, per-class recall (TBD) | `figures/cm_cnn.png`, `figures/per_class_recall.png` |
| 11 | Inference cost | Manas | ms/image predict-only vs end-to-end, CPU/GPU, parameter counts (TBD, with hardware) | `results/benchmarks.json`, `report/inference_table.tex` |
| 12 | Limitations | Puneet | One dataset, one seed, 28x28 grayscale, small grids, untuned CNN, SVM tuned on a subset | `docs/error_analysis_notes.md` section 6 |
| 13 | Conclusion and future work | Puneet | Answer the question from the measured table only (TBD) | results files |
| 14 | Team contributions | all | One line each | `docs/team_contributions.md` |

## Slide 4 text (Akhil, Anant)
- Fashion-MNIST: 28x28 grayscale, 60,000 train / 10,000 test images, 10 balanced classes (6,000 and 1,000 per class)
- Our split: 54,000 train / 6,000 validation (600 per class, seed 42); the test images are used only for integrity checks and a report-only duplicate check, never to select anything
- Data quality: 0 exact duplicates within train, within test, train vs test, and train vs validation (exact copies only; near-duplicates not checked)
- Pixels: mean 0.2862, 50.21% of pixels are zero; PCA: first 10 components explain 71.98% of the variance, first 50 explain 86.25%
- Shirt, Pullover and Coat overlap heavily in the 2D PCA plot; closest class means: Pullover-Coat 2.473, Pullover-Shirt 2.677, Coat-Shirt 2.990 (descriptive only, not a prediction of errors)
- Floors on validation: majority class accuracy 0.100, nearest class mean accuracy 0.692
- Flattened pixels: 784 values in [0,1]; the classical models cannot use which pixels are neighbours
- HOG: gradients -> orientation histograms per 4x4 cell -> overlapping 2x2-cell block normalisation -> 1,296 features
- HOG configuration chosen by the validation study: TBD
- Figures: `figures/eda_pca_scatter.png`, `figures/eda_confusable_panel.png`

## Slide 5 text (Tanish)
- Four classical models x two inputs (raw pixels, HOG) = 8 experiments
- Logistic regression: linear, softmax probabilities; k-NN: vote of nearest images; Random forest: many decision trees averaged; RBF-SVM: curved boundary with a margin
- Chosen on validation macro-F1; test set touched once
- SVM tuned on a 10,000-image subset (disclosed limitation)
- Results: TBD

## Slide 6 text (Manas)
- Small CNN: 3 x (3x3 conv -> ReLU -> 2x2 max-pool), then 2 dense layers
- 241,546 trainable parameters, verified in code
- Trained on 54k images; validation loss decides when to stop (best epoch restored)
- Convolutions look at neighbouring pixels, so edges and shapes are kept; flattened pixels lose that layout
- Result vs the 8 classical models: TBD

## Speaker-note rules
- Say "we measured" only for numbers in `results/`; say "hypothesis" for explanations.
- For any difference between models, state the confidence interval and the McNemar result.
