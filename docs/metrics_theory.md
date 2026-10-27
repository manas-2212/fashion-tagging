# Metrics theory (Rohan)

## 1. Why these metrics
The task is 10-class classification on Fashion-MNIST. The validation split has 600 images per class, so the classes are balanced and accuracy is meaningful. We also report macro-F1 because it shows whether a model is weak on particular classes (for example shirt vs T-shirt).

## 2. Accuracy
Accuracy = correct predictions / all predictions. It treats every image equally and hides which classes fail.

## 3. Precision, recall and F1
For one class: precision = TP / (TP + FP), recall = TP / (TP + FN), F1 = 2PR / (P + R), the harmonic mean. Recall is the fraction of a class's images that were found; precision is the fraction of its predictions that were right.

## 4. Macro-F1
Macro-F1 is the unweighted mean of the 10 per-class F1 scores, so every class counts the same. It is the model-selection metric; validation accuracy only breaks ties.

## 5. Confusion matrix
A 10 x 10 table: rows are true classes, columns are predicted classes. The diagonal holds correct predictions. Large off-diagonal cells show which classes are confused, and we compare these with the EDA class-centroid distances.

## 6. Bootstrap confidence interval
A score from 10,000 test images is one noisy estimate. We resample the test predictions with replacement many times, recompute the metric each time, and take the spread (for example the 2.5th and 97.5th percentiles for a 95% interval). It captures test-sample variability only, not training randomness (we use one seed). The exact number of resamples and the interval type must be confirmed in src/evaluation.py.

## 7. McNemar test (paired model comparison)
Two models are evaluated on the same test images. Let b = images only model A got right and c = images only model B got right. If both models have the same error rate, b and c should be about equal. The statistic is (|b - c| - 1)^2 / (b + c) with continuity correction, compared with a chi-square distribution with 1 degree of freedom (an exact binomial version is used when b + c is small). A small p-value means the difference in errors is unlikely to be chance. The exact variant used must be confirmed in src/evaluation.py.

## 8. Limits to state in the report
One dataset, one seed, one official split. Overlapping confidence intervals or a non-significant McNemar test mean we cannot claim one model is better. The test set is used exactly once, inside run_test_gate.
