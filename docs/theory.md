# Theory: classical and learned classifiers for Fashion-MNIST

Dataset: Fashion-MNIST (Xiao et al., 2017), 28x28 grayscale, 10 classes, official 60,000 train / 10,000 test.
Numerical results in this document are TBD until measured.

## 1. Supervised multiclass classification
We observe pairs $(\mathbf{x}_i, y_i)$ with $\mathbf{x}_i\in\mathbb{R}^d$ (784 pixel values or 1,296 HOG values) and $y_i\in\{0,\dots,9\}$. A classifier is a function $f:\mathbb{R}^d\to\{0,\dots,9\}$ chosen to make few mistakes on unseen images. Training minimises a loss on the training images; what we care about is the error on new images drawn from the same distribution (generalisation).

## 2. Roles of train, validation and test
- Train (54,000, stratified 90% of the official train set): fits model parameters and any preprocessing statistics.
- Validation (6,000, the remaining 10%, seed 42): compares hyperparameters and models.
- Test (10,000, official test set): used once, at the end, to report performance. Using it earlier for any decision would make the reported number optimistic.

## 3. Hyperparameter selection
Hyperparameters (C, k, leaf size) are set before training and not learned by the fitting algorithm. We fit each candidate on the training split and score it on validation. Macro-F1 is $\frac1K\sum_k F1_k$ with $F1_k=\frac{2P_kR_k}{P_k+R_k}$, so every class counts equally; accuracy breaks ties.

## 4. Overfitting and generalisation
A model overfits when it fits training noise: training accuracy is high while validation accuracy is lower. We measure the gap = train accuracy - validation accuracy (train accuracy on a disclosed stratified subsample). A gap is an observation; its causes (deep trees, small k, large C) are hypotheses that need separate evidence. For k-NN, train accuracy is inflated because a point is its own neighbour.
Measured gaps: TBD (to be added after the real runs, observation first, hypotheses second).

## 5. Data leakage
Leakage is information from validation or test data influencing training. Concrete example from this project: if StandardScaler were fitted on train + validation images, its per-pixel mean and standard deviation would include validation statistics. Validation images would then be standardised with values partly computed from themselves, and validation scores would be slightly optimistic. We keep the scaler inside the Pipeline, so cross-split statistics cannot enter, and test that `scaler.mean_` equals the training-split mean and not the train+validation mean.

## 6. Reproducibility
Fixed seed 42 for the split and for models with randomness (random forest, SVM, logistic regression), a recorded environment (library versions), stratified subsets with a fixed seed, and any subsample size disclosed. Same seed on same data gives same predictions (tested for RF and SVM).

## 7. Logistic regression
Softmax model: $p(y=k\mid\mathbf{x})=\exp(\mathbf{w}_k^\top\mathbf{x}+b_k)/\sum_j\exp(\mathbf{w}_j^\top\mathbf{x}+b_j)$. Training minimises cross-entropy $-\sum_i\log p(y_i\mid\mathbf{x}_i)$ plus an L2 penalty; in scikit-learn, $C$ multiplies the loss term, so small C means stronger regularisation. The boundaries between classes are linear in the input features. Parameters: $K(d+1)$ = 7,850 (pixels) or 12,970 (HOG).

## 8. k-nearest neighbours
Prediction: find the $k$ training points with smallest Euclidean distance to the query and take a majority vote. Small $k$: flexible boundary, high variance, sensitive to noise; large $k$: smoother, higher bias. Cost: no training, but each prediction compares with all stored points (about $N\cdot d$ operations). Curse of dimensionality: in high dimensions, distances between points tend to become less contrasted, so "nearest" is less informative; this is a hypothesis about why raw pixel distance may be weaker than a better-designed feature space.

## 9. Random forest
Bagging: each tree is trained on a bootstrap sample of the training set, which reduces variance when predictions are averaged. Feature subsampling: each split considers only a random subset of features (the scikit-learn classifier default is the square root of d), which decorrelates trees. Predictions average the trees' class probabilities. Splits depend only on feature ordering, so the model is scale-invariant. Deeper trees (min_samples_leaf=1) have low bias and can fit training data perfectly.

## 10. RBF-SVM
The SVM finds a maximum-margin boundary; the soft-margin version allows violations $\xi_i$ at cost $C$: $\min \frac12\|\mathbf{w}\|^2+C\sum\xi_i$. The RBF kernel $K(\mathbf{x},\mathbf{x}')=\exp(-\gamma\|\mathbf{x}-\mathbf{x}'\|^2)$ implicitly maps inputs to a space where a linear margin corresponds to a curved boundary in the original space. Larger C fits training data more tightly; larger gamma makes the kernel more local. Support vectors are the training points that define the boundary; their number is a measure of model size. Multiclass: one-vs-one voting, 45 binary problems. Training is slower than linear in n; we tune on a 10,000-image subset and disclose this.

## 11. Why flattened pixels, HOG and CNN features may give different results (hypotheses, not claims)
- H1: Flattening discards 2D neighbourhood structure; models see pixels as unrelated coordinates, so a shifted shape looks very different.
- H2: HOG summarises local edge orientations in cells, which may be more tolerant of small shifts and lighting variation than raw intensity, and may help distance-based and linear models.
- H3: A CNN shares weights across positions and learns its own filters, which may capture local patterns that matter for visually similar classes (Shirt, T-shirt/top, Pullover, Coat).
- H4: Classical models may profit differently from each representation (e.g. k-NN from smoother features, RF from informative single features).
These are tested by the measured results; the outcome is open.

## 12. CNN theory
(To be merged from docs/cnn_notes.md, Manas.)

## 13. Evaluation metrics theory
(To be merged from docs/metrics_theory.md, Rohan.)
