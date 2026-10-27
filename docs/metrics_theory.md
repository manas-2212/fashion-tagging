# Evaluation metrics theory (Rohan's section; draft written by Manas, Rohan to review and own)

Legend: **[F]** established fact (definition or mathematics) · **[H]** hypothesis for this project, to be checked against measured results. No result is stated here.

The code that implements everything below is `src/evaluation.py` (tested in `tests/test_evaluation.py`).

## 1. Confusion matrix
**[F]** For K = 10 classes, entry `C[i, j]` counts images whose true class is `i` and predicted class is `j`. Rows = true class, columns = predicted class (project convention, `labels = range(10)`, order = `CLASS_NAMES`). The diagonal holds correct predictions; the row sum is the number of images of that true class (support).

## 2. Accuracy
**[F]** `accuracy = trace(C) / sum(C)`. Simple, but it hides which classes fail.

## 3. Per-class recall, precision, F1
**[F]** For class `c`: TP = `C[c,c]`, FN = row sum − TP, FP = column sum − TP.
- recall = TP / (TP + FN) (share of true `c` images found)
- precision = TP / (TP + FP) (share of "predicted `c`" that are right)
- F1 = 2·TP / (2·TP + FP + FN) (harmonic mean of the two; it equals 2TP / (row sum + column sum))

## 4. Macro-F1 and why it is the selection metric
**[F]** Macro-F1 is the unweighted mean of the ten F1 values, so each class counts equally. A class that is never predicted has F1 = 0 (not skipped); this is the `zero_division=0` convention and the code uses it.
**[F]** Both Fashion-MNIST test and validation sets are balanced here (1,000 and 600 images per class), so **macro-averaged recall equals accuracy exactly**. Macro-F1 differs from accuracy only through precision, i.e. through *which* wrong class absorbs the mistakes.
**[H]** Weak, mutually confusable classes (Shirt / T-shirt/top / Pullover / Coat) should pull macro-F1 down together with accuracy. Check this on the confusion matrices; do not assume it.
Selection rule (project protocol): highest validation macro-F1, validation accuracy breaks ties.

## 5. Top confusions
**[F]** `top_confusions(cm, k)` lists the k largest off-diagonal cells with `rate_of_true_class = count / row sum`. Report counts and rates together: a large count in a big class and a large rate in a small class mean different things (here the classes have equal size, so the two rankings coincide).

## 6. Bootstrap confidence interval
**[F]** Percentile bootstrap: draw 10,000 test indices with replacement, recompute the metric, repeat 1,000 times (seed 42), take the 2.5 % and 97.5 % quantiles (Efron, 1979). It describes the sampling noise of *this* test set for *this* trained model.
**[F]** It does **not** include training randomness (other seeds, other initialisations) or hyperparameter-selection noise. Intervals therefore understate the total uncertainty of the claim "method A beats method B in general".
**[F]** For orientation only: with n = 10,000 and accuracy near 0.90 the binomial standard error is √(0.9·0.1/10,000) = 0.003, i.e. a 95 % interval of about ±0.6 percentage points. Use the measured intervals in the report, not this arithmetic.
**[H]** Models whose intervals overlap heavily may not be reliably ordered by this test set; the McNemar test below is the sharper tool for a pair.

## 7. McNemar's test (paired comparison)
**[F]** For two models on the same images, count the discordant pairs: `n10` = A correct and B wrong, `n01` = A wrong and B correct. Images where both are right or both wrong carry no information about the difference. Under H0 (equal error rates) each discordant image is equally likely to fall in either cell, so `n10 ~ Binomial(n10 + n01, 0.5)`.
- If `n10 + n01 < 25`: exact two-sided binomial test.
- Otherwise: `χ² = (|n01 − n10| − 1)² / (n01 + n10)` with 1 degree of freedom (continuity correction).
- If there are no discordant pairs: p = 1.
**[F]** We compare the CNN with each of the 8 classical models, so 8 tests are run; the notebook reports a Bonferroni-adjusted p-value `min(1, 8·p)`.
**[F]** Limits (Dietterich, 1998): the test uses one trained model per method, ignores training variance, and says nothing about other datasets. A significant p-value means "the error rates differ on this test set", not "the method is better in general".

## 8. Confidence scores
**[F]** `predict_with_confidence` returns the maximum predicted class probability when the model offers `predict_proba`. Logistic regression and the CNN give softmax probabilities, k-NN gives the fraction of neighbours voting for the winning class, random forest averages tree votes. The RBF-SVM is fitted without `probability=True` (cost), so its confidence is NaN by design.
**[F]** These numbers are not calibrated probabilities and are not comparable across model families; they are used only to choose which examples to display (most confident mistakes).

## 9. Inference timing (definitions)
**[F]** Same 10,000 test images, fixed order, model loading/training/plotting excluded. Classical models: "predict only" (precomputed features; an in-pipeline scaler is included because it is part of `pipeline.predict`) and "end-to-end" (feature extraction from raw images + predict), median of 3 repeats after one untimed warm-up call. CNN: pre-normalised tensors on the device, `eval()` + `inference_mode()`, batch 256, 3 warm-up batches, CUDA synchronised before and after, median of 5 repeats. CPU always; GPU if present. Hardware and thread counts come from `results/environment.json` and `results/benchmarks.json`, never typed by hand.

## References (verify before submission, see `report/references.bib`)
Efron (1979) bootstrap; McNemar (1947); Dietterich (1998).
