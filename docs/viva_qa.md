# Viva question bank

Compiled by Manas from each owner's answers (`docs/classical_notes.md`, `docs/eda_notes.md`, `docs/feature_notes.md`, `docs/cnn_notes.md`, `docs/metrics_theory.md`). **Rohan owns this file**: edit the answers into your own words and add the questions you expect.

Rules for answering: answer in your own words; say "we measured X" only if the file in `results/` shows it; separate what is established from what is a hypothesis; if you do not know, say what you would check. Result-dependent answers are marked **TBD** below and must be filled from `results/*.csv` after the test gate.

## A. Project, protocol and leakage (Puneet)

**Q1. What question does the project ask?**
Whether a small CNN, which learns spatial features, recognises similar clothing classes better than classical models (logistic regression, k-NN, random forest, RBF-SVM) on flattened pixels or HOG features, on Fashion-MNIST. The outcome is open; we report what we measure.

**Q2. Why exactly nine models?**
Four classical models on two representations give eight, plus one CNN that learns its own features. Every model is trained and evaluated under the same protocol.

**Q3. How are the splits made?**
Official split only. The 60,000 official training images are split into 54,000 train and 6,000 validation (stratified, 600 per class, seed 42). The 10,000 official test images are used for the final evaluation only.

**Q4. What is data leakage and how do you prevent it?**
Information from validation or test data influencing training or selection. Nothing is fitted on validation or test data: scalers sit inside the pipelines and are fitted on the training split, CNN mean/std come from the training split, PCA is fitted on training data, and selection uses validation only.

**Q5. What does the freeze and the test gate do?**
`freeze_selection` writes the chosen configurations (with a timestamp and git commit) to `results/selected_configs.json` and returns its hash. `run_test_gate` refuses to run without that file and writes a flag file, so the test images can be evaluated only once.

**Q6. Why are final models trained on 54,000 images only, with no refit on train plus validation?**
It keeps the protocol simple and the validation set clean; the price is that models see 6,000 fewer images than they could. This is a stated limitation.

**Q7. What is the runtime gate?**
Before each final classical fit the fit time is estimated; if it exceeds 30 minutes the model is fitted on a disclosed stratified subset and `n_train_used` is printed and saved. Nothing may be subsetted silently.

**Q8. Is anything subsampled by design?**
Yes: the SVM hyperparameters are tuned on a stratified 10,000-image subset of the 54,000 training images, because SVM tuning is costly. This is disclosed.

**Q9. How is the work reproducible?**
Seed 42 everywhere, deterministic splits, cached features, saved results files, recorded software and hardware information, and unit tests. Single-seed CNN runs may still differ across hardware (GPU kernels are not guaranteed deterministic).

## B. Data and EDA (Akhil)

**Q10. What is in Fashion-MNIST?**
70,000 grayscale 28x28 images of fashion products in 10 classes (Xiao et al., 2017), with an official 60,000/10,000 train/test split. The classes are exactly balanced: 6,000 per class in train, 1,000 per class in test.

**Q11. Why not just use MNIST?**
It is the same format but a harder task; MNIST is considered too easy (the dataset's authors present Fashion-MNIST as a drop-in replacement for this reason).

**Q12. What data-quality checks did you run?**
Shape, dtype, pixel range and label-count assertions, and a hash-based exact-duplicate check, which found 0 exact duplicates within train, within test, between train and test, and between our train and validation splits. The check finds exact copies only, not near-duplicates.

**Q13. What do the sanity baselines tell you?**
A majority-class predictor gets accuracy 0.100 (as expected for a balanced 600-per-class validation set) and a nearest-class-mean classifier gets 0.692 accuracy on validation. A model must clearly beat these to be worth reporting.

**Q14. What does PCA tell you and what does it not?**
Its first two components explain 28.99% and 17.77% of the variance, and the 2D plot shows heavy overlap among Shirt, Pullover and Coat. PCA is linear and unsupervised, so overlap in two components does not prove the classes are inseparable, and separation would not prove easy classification.

**Q15. Why is the validation split stratified?**
So every class has exactly 600 validation images and class-wise estimates are comparable; a random split could skew class proportions.

## C. Features (Anant)

**Q16. What do flattened pixels capture and miss?**
They capture absolute grey level at each location (silhouette area, brightness). The models treat each pixel as an independent column, so permuting the columns identically for all images leaves logistic regression, k-NN, the RBF-SVM and the random forest essentially unchanged: they cannot use the fact that two pixels are neighbours.

**Q17. Explain HOG in a few sentences.**
Compute the gradient at each pixel, vote its direction (9 unsigned bins) weighted by magnitude into histograms per 4x4-pixel cell, normalise overlapping 2x2-cell blocks, and concatenate. It describes local edge directions and is less sensitive to local contrast, but it discards absolute grey level.

**Q18. Where does 1,296 come from?**
28/4 = 7 cells per axis, 7 - 2 + 1 = 6 blocks per axis, and each block has 2 x 2 x 9 = 36 numbers, so 6 x 6 x 36 = 1,296. We verified it against the library output.

**Q19. Why can HOG be computed on all splits without leakage?**
It is a fixed function of each image; nothing is fitted. Only the StandardScaler is fitted, and it lives inside the pipeline and is fitted on training data.

**Q20. How was the HOG parameter choice protected from overfitting the test set?**
The decision rule (switch only if validation macro-F1 improves by more than 0.010) was written down before the study ran, the study used validation data only, and the parameters could change once, at a fixed milestone.

## D. Classical models (Tanish; his Q&A from docs/classical_notes.md)

**Q21. Why must validation images not be used to train the final model here?**
Our protocol selects on validation and trains the final models on the 54,000 training images only. If validation images were also trained on, the validation score would no longer be an honest estimate and selection would be biased toward models that memorised them.

**Q22. Which models use a scaler, and why?**
Logistic regression and the SVM have a StandardScaler inside the pipeline, fitted on training data only, because they are sensitive to feature scale (the RBF kernel is distance-based). k-NN and the random forest use none, as specified.

**Q23. Why is k-NN slow at inference?**
It stores all 54,000 training vectors and compares each query with every one; training is trivial and the cost moves to prediction.

**Q24. Why is a random forest invariant to monotone feature rescaling?**
Each split asks whether feature j is above a threshold; rescaling feature j only moves the threshold, so the same partition is found.

**Q25. Why select on macro-F1 rather than accuracy?**
Macro-F1 averages per-class F1 equally, so weak classes such as Shirt count as much as easy ones. Accuracy breaks ties.

**Q26. Why not GridSearchCV with cross-validation?**
We have an official fixed validation split and a runtime budget; cross-validation would multiply the fit cost, especially for the SVM and k-NN.

**Q27. Why is the SVM fitted without probability=True?**
It runs an internal cross-validation that multiplies training cost, so SVM confidence is NaN downstream.

**Q28. How many parameters does logistic regression have?**
10 x 784 + 10 = 7,850 on pixels and 10 x 1,296 + 10 = 12,970 on HOG. For the random forest we report total tree nodes and for the SVM the number of support vectors; these are not comparable to weights, and k-NN is non-parametric.

## E. CNN (Manas)

**Q29. Why a CNN here?**
Convolution reuses small filters across the image, so local patterns such as edges and collars are learned directly with few parameters. Whether that beats HOG plus a classical model is open, and we report what we measure.

**Q30. How are 241,546 parameters obtained?**
Conv layers cost c_out(c_in x 9 + 1): 320, 18,496 and 73,856. The dense layers cost 1152 x 128 + 128 = 147,584 and 128 x 10 + 10 = 1,290. The sum is 241,546, and the code counts the same number.

**Q31. Why does the 7x7 map become 3x3?**
Pooling by 2 floors odd sizes, so the last row and column are ignored.

**Q32. What does max-pooling do?**
It keeps the strongest response in each 2x2 window: it halves the size, adds no parameters and gives a little shift tolerance.

**Q33. How does dropout work, and why only in the head?**
It randomly zeros activations during training (and rescales the rest) and is the identity at evaluation. The head holds most of the parameters, so that is where overfitting is most likely.

**Q34. Why stop on validation loss and restore the best weights?**
Loss is smoother than accuracy. Restoring the best epoch avoids returning an already-overfit last epoch, and the test images play no part.

**Q35. Why can't the training curve be compared directly with the validation curve?**
Training numbers are running averages over the epoch in train mode with dropout on, so they are pessimistic. Validation is a clean evaluation-mode pass after the epoch.

**Q36. Why does the model output logits and not probabilities?**
`CrossEntropyLoss` applies log-softmax itself and expects raw logits and integer class indices; feeding it softmax output would apply softmax twice.

**Q37. Why is the CNN normalised with the training split's mean and standard deviation?**
Computing them on validation or test data would leak information about those data into the preprocessing.

**Q38. How is inference timed fairly?**
Same 10,000 images in the same order, warm-up first, median of several repeats, CUDA synchronised if on GPU, model loading/training/plotting excluded. Classical models are reported both as predict-only and end-to-end from raw images, and each result says which one it is.

**Q39. What did you not tune?**
The CNN's learning rate, batch size, dropout and architecture were not tuned, while the classical models got small grid searches. This asymmetry is a stated limitation.

## F. Evaluation (Rohan)

**Q40. What is macro-F1 and why use it?**
For each of the 10 classes we compute an F1 score, and macro-F1 is the plain average of those 10 numbers, so every class counts the same. Our test set is balanced, so macro-recall equals accuracy. Macro-F1 differs from accuracy because it also uses precision, which shows which class takes the wrong predictions.

**Q41. What does a bootstrap confidence interval tell you?**
We resample the 10,000 test predictions with replacement 1,000 times, compute the metric each time, and take the 2.5% and 97.5% values as a 95% interval. It shows how much the score changes just because of which test images we happened to have. It does not include training randomness, because we trained each model once with one seed.

**Q42. Why McNemar's test for model comparison?**
Both models are tested on the same images, so their results are paired. Only the images where one model is right and the other is wrong tell us anything. If there are fewer than 25 such images we use an exact binomial test, otherwise a chi-square test with continuity correction.

**Q43. Why do you correct for multiple comparisons?**
We compare the CNN with 8 classical models, so we run 8 tests. With 8 tests at the 5% level, the chance of at least one false positive is about 34% (1 - 0.95^8). We therefore multiply each p-value by 8 (Bonferroni), capped at 1.

**Q44. Does a significant McNemar result prove the CNN is better in general?**
No. It only says that the two error rates differ on this test set for these single trained models. It does not cover other seeds or other datasets.

**Q45. Why are the SVM's confidence values NaN?**
We fit the SVM without probability=True because that setting adds an internal cross-validation and makes training much slower. We only use confidence to choose which examples to display, never in a metric.

**Q46. Which examples are shown in the error figures?**
Correct examples are the first n in dataset order, so they are not cherry-picked. Incorrect examples are the n mistakes where the model was most confident. For the SVM, confidence is NaN, so its mistakes come in dataset order.

## G. Result-dependent questions (fill after the single test-gate run; do not answer before)

**Q47. Which model had the highest test macro-F1, and is its lead over the runner-up outside the confidence intervals?**
TBD

**Q48. Did the CNN beat the best classical model on pixels and on HOG? What do the McNemar p-values say?**
TBD

**Q49. Which classes were confused most, and does this match the class-centroid distances in the EDA?**
TBD

**Q50. How do inference times compare (predict-only vs end-to-end vs CNN on CPU/GPU), and on which hardware?**
TBD

**Q51. Did any final fit hit the runtime gate? What was `n_train_used` for each model?**
TBD

**Q52. Did logistic regression converge within 500 iterations, and were any selected hyperparameters at the edge of their grid?**
TBD

**Q53. Was the HOG configuration changed by the sensitivity study?**
TBD
