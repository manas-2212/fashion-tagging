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
Measured gaps (dry run by Tanish: official 54,000/6,000 split, validation data only, test data not used; train accuracy on a stratified 5,000-image subsample of the training split; selected configuration per pair):

| Model | Rep | Selected params | Train acc | Val acc | Gap |
|---|---|---|---|---|---|
| logreg | pixels | C=0.1 | 0.8814 | 0.8608 | 0.0206 |
| knn | pixels | k=7 | 0.8850 | 0.8635 | 0.0215 |
| rf | pixels | min_samples_leaf=1 | 1.0000 | 0.8858 | 0.1142 |
| svm | pixels | C=10 | 0.9802 | 0.9093 | 0.0709 |
| logreg | hog | C=0.1 | 0.9302 | 0.8797 | 0.0505 |
| knn | hog | k=7 | 0.9124 | 0.8838 | 0.0286 |
| rf | hog | min_samples_leaf=3 | 0.9958 | 0.8772 | 0.1186 |
| svm | hog | C=10 | 1.0000 | 0.9155 | 0.0845 |

Observations. The two random forests have the largest gaps (0.1142 on pixels, 0.1186 on HOG), with training accuracy 1.0000 and 0.9958. The SVMs come next (0.0709 on pixels, 0.0845 on HOG); the HOG SVM reaches training accuracy 1.0000. Logistic regression (0.0206, 0.0505) and k-NN (0.0215, 0.0286) have the smallest gaps; for k-NN each training point is among its own neighbours, so its train accuracy is inflated and its gap is not directly comparable. Gap size does not rank validation accuracy: the SVMs have the highest validation accuracy (0.9093 on pixels, 0.9155 on HOG) despite larger gaps than logistic regression and k-NN, while the random forests, with the largest gaps, are in the middle (0.8858, 0.8772). The SVM final fits used 54,000 images, whereas C was tuned on 10,000, so tuning-time validation scores (0.8777, 0.8903 for C=10) are not comparable with these numbers.

Hypotheses (not tested here). H1: fully or nearly fully grown trees (min_samples_leaf 1 or 3) can fit training points almost exactly, and each training point is in-bag for about 63% of the trees, which would explain the near-perfect training accuracy of the forests. H2: C=10 with an RBF kernel gives a flexible boundary that fits the training data closely (training accuracy 1.0000 on HOG). H3: logistic regression is linear in its input features, so its capacity to fit the training data is limited, which may explain its small gaps. H4: voting over k=7 neighbours smooths the decision, which may keep the k-NN gap small. Testing these would require measuring train/validation gaps across the grid values (leaf size, C, k), which this dry run did not do.

Status: these numbers come from a dry run and must be checked against the official results table before the report is final.

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
(Merged from docs/cnn_notes.md.)

### CNN notes (contributed by Manas)

Legend: **[F]** established fact (textbook / mathematically exact) · **[H]** hypothesis for THIS project, to be checked against measured results. Nothing here is a result.

#### 1. Convolution
**[F]** A `Conv2d(c_in, c_out, 3, padding=1)` layer slides `c_out` learnable 3x3xc_in filters over the input; each output value is a weighted sum of a 3x3 neighbourhood plus a bias, and the *same* weights are reused at every position (weight sharing). Padding 1 keeps the spatial size. Parameters = `c_out*(c_in*9 + 1)`: 320, 18,496, 73,856 for our three layers.
**[F]** Consequence: the layer is translation-equivariant and has far fewer parameters than a dense layer over the same image (a dense 784->784 layer would have 615,440).

#### 2. ReLU
**[F]** `max(0, x)`. Non-linear (without it, stacked conv layers collapse into one linear map), cheap, gradient is 1 for x>0 and 0 otherwise, which avoids the saturation of sigmoid/tanh. **[F]** Units with always-negative input get no gradient ("dying ReLU").

#### 3. Max pooling
**[F]** `MaxPool2d(2)` keeps the maximum of each non-overlapping 2x2 window: halves height/width (28->14->7->3; 7->3 uses floor, so the last row/column of the 7x7 map is ignored), has no parameters, and gives small local translation tolerance and a larger receptive field for later layers.
**[H]** Discarding the last row/column at 7->3 costs little because Fashion-MNIST objects are centred; not measured.

#### 4. Dropout
**[F]** In training, each activation is zeroed with probability p and the rest are scaled by 1/(1-p) (inverted dropout); in eval mode it is the identity. It acts as a regulariser by preventing co-adaptation. We apply it (p=0.5) only in the dense head, where most of the parameters are (147,584 of 241,546).
**[H]** It will narrow the train/validation gap; this is checked by the overfit evidence in notebook section 7.

#### 5. Cross-entropy
**[F]** `CrossEntropyLoss` = log-softmax + negative log-likelihood on raw logits and int64 class indices; it is the negative log-probability assigned to the true class. Feeding softmax outputs into it would be a bug (double softmax); our model therefore returns logits. A model that outputs uniform probabilities has loss ln(10) ≈ 2.3026.

#### 6. Adam
**[F]** Adam keeps exponential moving averages of the gradient (first moment) and squared gradient (second moment), bias-corrects them, and scales each parameter's step by `m/(sqrt(v)+eps)`. Default lr=1e-3 is used here with no tuning (stated limitation).
**[H]** lr=1e-3 is adequate for this small network; not tuned, so other settings might do better.

#### 7. Early stopping
**[F]** Training is stopped when validation loss has not strictly improved for `patience` epochs; the weights of the best epoch are restored. This limits overfitting using the *validation* split only; the held-out images play no role.
**[F]** Because the best epoch is chosen on the validation set, the reported validation score is slightly optimistic; the held-out evaluation is the unbiased estimate.

#### 8. Why convolution preserves local spatial patterns and flattening does not
**[F]** Flattening a 28x28 image into 784 numbers keeps all values but gives a linear model / k-NN / RF no built-in notion that index i and i+1 are neighbours, or that i and i+28 are vertically adjacent; permuting the pixel columns identically for train and test leaves LogReg, k-NN (Euclidean), RF and RBF-SVM (up to tie-breaking and random feature sampling) essentially unchanged, i.e. they do not exploit pixel layout. A CNN's filters are defined on neighbourhoods, so edges, textures and shapes are detected wherever they occur and composed hierarchically.
**[H]** For Fashion-MNIST this should help most on visually similar classes (Shirt / T-shirt / Pullover / Coat). HOG also encodes local gradient structure, so it may close part of the gap for classical models. The outcome is open; we report what we measure.

## 13. Evaluation metrics theory
(To be merged from docs/metrics_theory.md, Rohan.)

