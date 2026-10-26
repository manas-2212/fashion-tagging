# Classical models: design notes (Tanish)

Fashion-MNIST, 54,000 train / 6,000 validation (seed 42), 10,000 test used only in the test gate.
Selection rule: highest validation macro-F1, validation accuracy breaks ties. All results are TBD until measured.

## Hyperparameters and scaling (two sentences each)

**Logistic regression, C in {0.1, 1.0}.** C is the inverse regularisation strength, so a small C shrinks weights and may reduce overfitting on 784 or 1,296 correlated features. max_iter=500 gives lbfgs room to converge; any ConvergenceWarning is reported, not hidden.
**Logistic regression, StandardScaler.** The optimiser and the L2 penalty treat all weights alike, so features on different scales would be penalised unevenly and slow convergence. The scaler sits inside the pipeline and is fitted on the training split only.

**k-NN, k in {3, 5, 7, 11}.** Small k gives low bias and high variance (noisy boundaries); larger k smooths decisions but blurs similar classes such as Shirt vs T-shirt. We try an odd spread of k values because k is the only knob that sets this trade-off.
**k-NN, no scaler, brute force.** Raw pixels share one scale [0,1], and for HOG the block normalisation already equalises scale, so extra standardisation is not required by the method. Brute force is exact and avoids tree-index slowdown in high dimensions; its cost lands at predict time.

**Random forest, min_samples_leaf in {1, 3}.** Leaf size controls tree depth: 1 grows trees until pure leaves (low bias, high variance per tree), 3 gives slightly smoother trees. 200 trees is a fixed budget chosen for stable averaging against runtime, not tuned.
**Random forest, no scaler.** Splits compare one feature to a threshold, so any monotone rescaling of a feature leaves the chosen partitions unchanged. Scaling would add a step with no effect.

**RBF-SVM, C in {1, 10}.** C trades margin width against training errors: larger C punishes margin violations more and fits the training data more tightly. gamma="scale" sets the kernel width from the data variance, so we do not tune a second parameter and keep the grid small.
**RBF-SVM, StandardScaler.** The RBF kernel depends on Euclidean distances, so features with larger variance would dominate the kernel. Scaling puts every feature on an equal footing before distances are computed.
**RBF-SVM, tuned on 10,000.** SVM training cost grows faster than linearly in n, so tuning uses a stratified 10,000-image subset (SVM_TUNE_N). LIMITATION: C chosen at 10k may not be optimal at 54k, or on a disclosed subset used for the final fit.

## k-NN train accuracy is inflated
When k-NN predicts a training point, that point is in its own neighbour set. With k=1 it is always its own nearest neighbour, so train accuracy is exactly 1.0; for k>1 it is still inflated because the point votes for itself. So the k-NN train-vs-validation gap in overfit_check is not evidence of the same kind as the gap for other models and must be read with this caveat.

## overfit_check disclosure
Train accuracy uses a stratified subsample of n_eval=5,000 images of the data the pipeline was fitted on; validation accuracy uses all 6,000 validation images. gap = train_acc - val_acc.

## Runtime gate
Before every final fit, estimate_fit_seconds probes on 5,000 images and extrapolates (SVM quadratic, others linear; the SVM extrapolation is an assumption and probably optimistic). If the estimate exceeds MAX_FIT_MINUTES*60, fit on a stratified subset (e.g. 20,000) and record it. Every fit prints "n_train_used = X of 54000". Runtime plan table: TBD (needs real data).

---
# HANDOFFS

## Report section "Classical Machine Learning Models" (LaTeX-ready, for Puneet)
```latex
\section{Classical Machine Learning Models}
We compare four classifiers on two representations (flattened pixels, 784 features; HOG, 1296 features),
giving eight experiments. All models are trained on the 54{,}000-image training split; hyperparameters
are chosen by validation macro-F1 on the 6{,}000-image validation split.

\subsection{Logistic Regression}
For $K=10$ classes, the multinomial model gives
\begin{equation}
p(y=k\mid \mathbf{x}) = \frac{\exp(\mathbf{w}_k^{\top}\mathbf{x}+b_k)}{\sum_{j=1}^{K}\exp(\mathbf{w}_j^{\top}\mathbf{x}+b_j)},
\end{equation}
trained by minimising the L2-regularised cross-entropy
\begin{equation}
\min_{W,\mathbf{b}}\ \tfrac12\lVert W\rVert_F^2 + C\sum_{i=1}^{N}-\log p(y_i\mid \mathbf{x}_i).
\end{equation}
Features are standardised using training-split statistics only.

\subsection{k-Nearest Neighbours}
A test image is assigned the majority class among its $k$ nearest training images under Euclidean distance
$d(\mathbf{x},\mathbf{x}')=\lVert \mathbf{x}-\mathbf{x}'\rVert_2$. Small $k$ gives low bias and high variance.
Training stores the data; the cost is paid at prediction time.

\subsection{Random Forest}
Each of 200 trees is grown on a bootstrap sample, and each split considers a random subset of features.
The forest predicts by averaging tree class probabilities. Splits are threshold tests on single features,
so the model is invariant to monotone feature rescaling.

\subsection{RBF Support Vector Machine}
The kernel is $K(\mathbf{x},\mathbf{x}')=\exp(-\gamma\lVert \mathbf{x}-\mathbf{x}'\rVert^2)$ and the soft-margin problem is
\begin{equation}
\min_{\mathbf{w},b,\boldsymbol{\xi}}\ \tfrac12\lVert \mathbf{w}\rVert^2 + C\sum_i \xi_i
\quad \text{s.t. } y_i(\mathbf{w}^{\top}\phi(\mathbf{x}_i)+b)\ge 1-\xi_i,\ \xi_i\ge 0 .
\end{equation}
Multiclass prediction uses one-vs-one voting (45 binary classifiers for 10 classes).
Hyperparameters were tuned on a stratified 10{,}000-image subset (see limitations).
```

## Slide 5 text (for Akhil)
- Four classical models x two inputs (raw pixels, HOG) = 8 experiments
- Logistic regression: linear, softmax probabilities; k-NN: vote of nearest images; Random forest: many decision trees averaged; RBF-SVM: curved boundary with a margin
- Chosen on validation macro-F1; test set touched once
- SVM tuned on a 10,000-image subset (disclosed limitation)
- Results: TBD

## Viva Q&A (for Rohan)
1. **Why must the validation set not be used for training the final model here?** Our protocol is to select on validation and train the final models on the 54k train split only. If validation images were also trained on, the validation score would no longer be an honest estimate and selection would be biased toward models that memorised them.
2. **Why is scaling omitted for the SVM?** It is not omitted: the SVM pipeline has a StandardScaler. It is omitted for k-NN and RF only, and the scaler for the SVM is fitted on train data only, because the RBF kernel is distance-based and large-variance features would dominate.
3. **Why is k-NN slow at inference?** It stores all 54,000 training vectors and compares each query with every one, about N x d operations per image. Training is trivial; the cost moves to prediction.
4. **Why is a random forest scale-invariant?** Each split asks "is feature j above threshold t?", and monotone rescaling of feature j just moves t, so the same partition is found.
5. **Why select on macro-F1 rather than accuracy?** Macro-F1 averages per-class F1 equally, so weak classes such as Shirt count as much as easy ones. Accuracy breaks ties.
6. **Why not GridSearchCV with cross-validation?** We have an official fixed validation split and a runtime budget; CV would multiply fit cost, especially for SVM and k-NN.
7. **Why not probability=True for the SVM?** It runs an internal cross-validation that multiplies training cost, so SVM confidence is NaN downstream.
8. **Why is the logistic regression parameter count 7,850 for pixels?** 10 classes x 784 weights + 10 intercepts = 7,850; for HOG, 10 x 1,296 + 10 = 12,970.

---
## Runtime plan (probe estimates, 5,000-image stratified probe, first grid value per model)
Source: results/runtime_probe_tanish.txt. Estimates only; actual fit times TBD. Limit = MAX_FIT_MINUTES*60 = 1800 s.
Machine details: TBD (fill in own chip/RAM).

| Model | Representation | Estimated fit (s) | kNN predict val / test (s) | Decision |
|---|---|---|---|---|
| logreg | pixels | 6.4 | n/a | full 54,000 |
| knn | pixels | 0.0 | 6.1 / 10.2 | full 54,000 |
| rf | pixels | 7.4 | n/a | full 54,000 |
| svm | pixels | 149.8 | n/a | full 54,000 |
| logreg | hog | 5.1 | n/a | full 54,000 |
| knn | hog | 0.0 | 5.0 / 8.3 | full 54,000 |
| rf | hog | 10.7 | n/a | full 54,000 |
| svm | hog | 218.5 | n/a | full 54,000 |

Caveats: SVM estimates assume quadratic scaling (probably optimistic). Other grid values (SVM C=10, k=11, RF min_samples_leaf=3) were not probed. kNN predict times are extrapolated from a probe. Compare every estimate with the real fit time after the real runs.
