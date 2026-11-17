# Evaluation and statistics notes (Rohan)

Files: `src/dl_evaluation.py`, test `tests/test_dl_evaluation.py`. Phase 1 helpers re-used unchanged from
`src/evaluation.py`: `evaluate_predictions`, `bootstrap_ci`, `mcnemar_test`, `plot_confusion_matrix`.

## 1. Two different kinds of uncertainty

| Source | Question | How we measure it | Phase 1 | Phase 2 |
|---|---|---|---|---|
| Test-set sampling | would the score change with another 10,000 test images? | bootstrap CI over test images | yes | yes |
| Training randomness | would the score change if we trained again? | several seeds, mean +- std | **no (one seed)** | yes (3 seeds) |

Phase 1 stated this limitation; Phase 2 closes it. Every configuration is trained with seeds 42, 43, 44. The seed
changes the initial weights, the batch order and the augmentation draws.

* `summarise_runs`: mean and **sample** standard deviation (ddof = 1) over seeds.
* `ablation_deltas`: paired differences per seed (see `ablation_design.md`).
* `aggregate_seed_metrics`: the same for test metrics after the gate.

With n = 3 the standard deviation itself is uncertain, so we use it descriptively ("the difference is several
times the spread" or "within the spread"), not for a formal test.

## 2. Selection rule (`select_final_config`)

Highest mean validation macro-F1 over seeds; ties: higher mean accuracy, then fewer parameters, then name.
Diverged or incomplete configurations are excluded. The rule is code, it is tested, and its outcome is written to
the freeze file before the test gate.

## 3. Tests on the test set (after the gate)

* **Bootstrap 95% CI** for accuracy, 1,000 resamples of the test images, for the primary seed (42).
* **McNemar's test** for pairs of models on the same images. It only looks at the images where exactly one model
  is right: b = A right and B wrong, c = A wrong and B right. Statistic (|b - c| - 1)^2 / (b + c), chi-square with
  1 degree of freedom; exact binomial when b + c < 25. Planned comparisons, Bonferroni-corrected:
  final deep model vs Phase 1 CNN (re-trained and the original checkpoint), vs the Phase 1 pixel SVM;
  H1, H2, H3 vs the final deep model; H2 vs H1.
* McNemar on one seed tests "these two trained models differ on this test set". It does not include training
  randomness; the multi-seed table covers that. We report both and do not mix them.

## 4. Calibration (`expected_calibration_error`, `reliability_curve`)

A model is calibrated if, among predictions made with confidence 0.8, 80% are correct.
ECE = sum over 15 confidence bins of (share of images in the bin) x |accuracy in bin - mean confidence in bin|.
Why we need it: (1) label smoothing changes confidence, and the ablation `reg_no_ls` lets us show how;
(2) the cascade H3 routes by confidence, which only makes sense if confidence is informative.
In the reliability diagram, points below the diagonal mean over-confidence, above mean under-confidence.

## 5. Structure of the errors

* `in_group_confusion_rate`: share of T-shirt / Pullover / Coat / Shirt images assigned to a *different* class of
  that group. Phase 1: 11.8% for the CNN, 16.6-23.1% for classical models. It measures the specific failure the
  brief asks about, independent of the easy classes.
* Per-class recall (Shirt was 0.747 in Phase 1), confusion matrix of the final deep model, top-10 confusions.

## 6. Honest reading rules for the results

1. A gain smaller than the seed standard deviation is not a gain.
2. A McNemar p-value above 0.05 after correction means "no evidence of a difference", not "the models are equal".
3. The test set was already used at the Phase 1 gate. Phase 2 test numbers are therefore slightly optimistic
   estimates of performance on new data; relative comparisons between Phase 2 models are still fair because all of
   them were selected on validation only.
4. Fashion-MNIST has known label noise in the Shirt group; accuracy cannot reach 100%.

## 7. Viva questions

**Why macro-F1 when classes are balanced?** It weights every class equally and punishes a model that sacrifices
Shirt to gain elsewhere; with balanced classes it stays close to accuracy, which is a useful sanity check.
**Why ddof = 1?** Three seeds are a sample of all possible training runs; dividing by n - 1 gives an unbiased
variance estimate.
**Why Bonferroni?** With 7 tests at 0.05 each, the chance of at least one false alarm would be up to 30%.
Multiplying each p-value by 7 keeps the family-wise error at 0.05.
**Can ECE be zero for a bad model?** Yes: a model that always says "10% for every class" is calibrated and useless.
Calibration and accuracy are different properties.
**What does the bootstrap CI not tell you?** How much the score would move with another training run.

## Results for this component (full run, 16 November 2026)

| Model | Test accuracy | Macro-F1 | Shirt recall | In-group confusion |
|---|---|---|---|---|
| H0: PCA-HOG + SVM | 0.9048 | 0.9047 | 0.708 | 15.3% |
| Phase 1 CNN, re-trained | 0.9252 +- 0.0034 | 0.9250 | 0.757 | 12.2% |
| Final deep model | 0.9488 +- 0.0015 | 0.9487 | 0.834 | 8.5% |
| H1 | 0.9495 +- 0.0009 | 0.9494 | 0.833 | 8.4% |
| H2 | 0.9502 +- 0.0020 | 0.9501 | 0.839 | 8.3% |
| H3 | 0.9502 +- 0.0021 | 0.9501 | 0.839 | 8.3% |

What to say in the viva:

* **Training randomness is now measured.** The three re-trained Phase 1 CNNs scored 0.9215 to 0.9281; the single
  Phase 1 number (0.9274) sits inside that range. One run can move by about 0.7 pp from the seed alone.
* **The architecture gain is real under both kinds of uncertainty:** +2.36 pp against standard
  deviations of 0.0034 and 0.0015; bootstrap CI for the final model [0.9445,
  0.9532]; McNemar 367 vs 158 discordant images, p < 0.0001 after Bonferroni
  (7 comparisons).
* **The hybrid gain is not:** p = 0.20 for H2 vs the deep model, p = 0.90 for H2 vs H1.
  "No evidence of a difference" is the correct wording, not "they are equal".
* **Errors moved where we wanted:** mean errors 748 -> 512; in-group confusion
  12.2% -> 8.5%. Shirt is still the hardest class, most often called T-shirt/top.
* **Calibration got worse.** Test ECE 0.067 for the final model against 0.013 for the Phase 1
  CNN; mean confidence 0.886 at 0.9488 accuracy, so the model is *under*-confident. The ablation
  shows the cause is label smoothing (validation ECE 0.083 with, 0.010 without). Remedy not yet
  applied: temperature scaling fitted on validation.
