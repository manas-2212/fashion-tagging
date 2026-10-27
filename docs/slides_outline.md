# Slides outline (Akhil assembles; each owner supplies their own slide text)

Rules: every number on a slide is copied from a file in `results/` (never retyped); report the measured values below and explicitly label preliminary timing; one message per slide; figures are the saved PNGs in `figures/`.

| # | Title | Owner | Content | Source |
|---|---|---|---|---|
| 1 | Title and team | Puneet | Project title, names, roles | `docs/team_contributions.md` |
| 2 | Problem and question | Puneet | Does a small CNN recognise similar clothing classes better than classical models on pixels or HOG? Outcome open. | README |
| 3 | Literature | Puneet | Fashion-MNIST (Xiao et al.); learned vs hand-designed features (LeCun et al.); neither paper ran our experiment | `docs/literature_review.md` |
| 4 | Dataset, EDA and features | Akhil, Anant | 60k/10k, balanced; 54k/6k stratified split; 0 exact duplicates; PCA overlap of Shirt/Pullover/Coat; pixels (784) vs selected HOG (6,084) | `docs/eda_notes.md`, `docs/feature_notes.md`, `figures/eda_*.png`, `figures/hog_example.png` |
| 5 | Classical models | Tanish | Slide text below | `docs/classical_notes.md` |
| 6 | CNN | Manas | Slide text below | `docs/cnn_notes.md`, `figures/cnn_learning_curves.png` |
| 7 | Protocol | Puneet | Official split, no fitting on val/test, selection by validation macro-F1, freeze then one-shot test gate, runtime gate, SVM tuned on 10k subset | README, `results/selected_configs.json` |
| 8 | Results | Rohan | CNN leads: accuracy 0.9274 (95% CI 0.9219–0.9327), macro-F1 0.9269 (0.9214–0.9321); pixel RBF-SVM runner-up at 0.8953 accuracy | `results/metrics_with_ci.csv`, `report/results_table.tex` |
| 9 | Is the CNN really different? | Rohan | All 8 CNN-vs-classical McNemar tests remain significant after Bonferroni correction; adjusted p-values range from 1.14e-28 to 1.31e-111 | `results/mcnemar_cnn_vs_classical.csv` |
| 10 | Error analysis | Rohan | CNN made 726/10,000 errors; top confusion: Shirt→T-shirt/top 99 (9.9% of Shirts), then T-shirt/top→Shirt 65 | `figures/cm_cnn.png`, `figures/per_class_recall.png` |
| 11 | Inference cost | Manas | Single-pass preliminary timings: classical CPU predict-only and end-to-end; CNN 0.4402 ms/image CPU and 0.0116 ms/image GPU on pre-normalised tensors (not directly comparable); 241,546 CNN parameters | `results/benchmarks.json`, `report/inference_table.tex` |
| 12 | Limitations | Puneet | One dataset, one seed, 28x28 grayscale, small grids, untuned CNN, SVM tuned on a subset | `docs/error_analysis_notes.md` section 6 |
| 13 | Conclusion and future work | Puneet | CNN leads on Fashion-MNIST; do not generalise to real inventory photos without validation | results files |
| 14 | Team contributions | all | One line each | `docs/team_contributions.md` |

## Slide 4 text (Akhil, Anant)
- Fashion-MNIST: 28x28 grayscale, 60,000 train / 10,000 test images, 10 balanced classes (6,000 and 1,000 per class)
- Our split: 54,000 train / 6,000 validation (600 per class, seed 42); the test images are used only for integrity checks and a report-only duplicate check, never to select anything
- Data quality: 0 exact duplicates within train, within test, train vs test, and train vs validation (exact copies only; near-duplicates not checked)
- Pixels: mean 0.2862, 50.21% of pixels are zero; PCA: first 10 components explain 71.98% of the variance, first 50 explain 86.25%
- Shirt, Pullover and Coat overlap heavily in the 2D PCA plot; closest class means: Pullover-Coat 2.473, Pullover-Shirt 2.677, Coat-Shirt 2.990 (descriptive only, not a prediction of errors)
- Floors on validation: majority class accuracy 0.100, nearest class mean accuracy 0.692
- Flattened pixels: 784 values in [0,1]; the classical models cannot use which pixels are neighbours
- HOG: gradients -> orientation histograms per 2x2-pixel cell -> overlapping 2x2-cell block normalisation -> 6,084 features
- HOG configuration selected on validation: 9 orientations, 2x2 pixels/cell, 2x2 cells/block, L2-Hys; macro-F1 0.868348 vs 0.836344 baseline (+0.032004)
- Figures: `figures/eda_pca_scatter.png`, `figures/eda_confusable_panel.png`

## Slide 5 text (Tanish)
- Four classical models x two inputs (raw pixels, HOG) = 8 experiments
- Logistic regression: linear, softmax probabilities; k-NN: vote of nearest images; Random forest: many decision trees averaged; RBF-SVM: curved boundary with a margin
- Chosen on validation macro-F1; test set touched once
- SVM tuned on a 10,000-image subset (disclosed limitation)
- Best classical: pixel RBF-SVM (accuracy 0.8953, macro-F1 0.8951); CNN: accuracy 0.9274, macro-F1 0.9269. HOG SVM used 11,770 training images due to the runtime gate.

## Slide 6 text (Manas)
- Small CNN: 3 x (3x3 conv -> ReLU -> 2x2 max-pool), then 2 dense layers
- 241,546 trainable parameters, verified in code
- Trained on 54k images; validation loss decides when to stop (best epoch restored)
- Convolutions look at neighbouring pixels, so edges and shapes are kept; flattened pixels lose that layout
- CNN: 92.74% test accuracy, 0.9269 macro-F1; best classical pixel RBF-SVM: 89.53% accuracy, 0.8951 macro-F1. Best validation epoch 24; 29 epochs run; 241,546 parameters.

## Speaker-note rules
- Say "we measured" only for numbers in `results/`; say "hypothesis" for explanations.
- For any difference between models, state the confidence interval and the McNemar result.
