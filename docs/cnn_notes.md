# CNN notes (Manas) — for docs/theory.md (Tanish)

Legend: **[F]** established fact (textbook / mathematically exact) · **[H]** hypothesis for THIS project, to be checked against measured results. Nothing here is a result.

## 1. Convolution
**[F]** A `Conv2d(c_in, c_out, 3, padding=1)` layer slides `c_out` learnable 3x3xc_in filters over the input; each output value is a weighted sum of a 3x3 neighbourhood plus a bias, and the *same* weights are reused at every position (weight sharing). Padding 1 keeps the spatial size. Parameters = `c_out*(c_in*9 + 1)`: 320, 18,496, 73,856 for our three layers.
**[F]** Consequence: the layer is translation-equivariant and has far fewer parameters than a dense layer over the same image (a dense 784->784 layer would have 615,440).

## 2. ReLU
**[F]** `max(0, x)`. Non-linear (without it, stacked conv layers collapse into one linear map), cheap, gradient is 1 for x>0 and 0 otherwise, which avoids the saturation of sigmoid/tanh. **[F]** Units with always-negative input get no gradient ("dying ReLU").

## 3. Max pooling
**[F]** `MaxPool2d(2)` keeps the maximum of each non-overlapping 2x2 window: halves height/width (28->14->7->3; 7->3 uses floor, so the last row/column of the 7x7 map is ignored), has no parameters, and gives small local translation tolerance and a larger receptive field for later layers.
**[H]** Discarding the last row/column at 7->3 costs little because Fashion-MNIST objects are centred; not measured.

## 4. Dropout
**[F]** In training, each activation is zeroed with probability p and the rest are scaled by 1/(1-p) (inverted dropout); in eval mode it is the identity. It acts as a regulariser by preventing co-adaptation. We apply it (p=0.5) only in the dense head, where most of the parameters are (147,584 of 241,546).
**[H]** It will narrow the train/validation gap; this is checked by the overfit evidence in notebook section 7.

## 5. Cross-entropy
**[F]** `CrossEntropyLoss` = log-softmax + negative log-likelihood on raw logits and int64 class indices; it is the negative log-probability assigned to the true class. Feeding softmax outputs into it would be a bug (double softmax); our model therefore returns logits. A model that outputs uniform probabilities has loss ln(10) ≈ 2.3026.

## 6. Adam
**[F]** Adam keeps exponential moving averages of the gradient (first moment) and squared gradient (second moment), bias-corrects them, and scales each parameter's step by `m/(sqrt(v)+eps)`. Default lr=1e-3 is used here with no tuning (stated limitation).
**[H]** lr=1e-3 is adequate for this small network; not tuned, so other settings might do better.

## 7. Early stopping
**[F]** Training is stopped when validation loss has not strictly improved for `patience` epochs; the weights of the best epoch are restored. This limits overfitting using the *validation* split only; the held-out images play no role.
**[F]** Because the best epoch is chosen on the validation set, the reported validation score is slightly optimistic; the held-out evaluation is the unbiased estimate.

## 8. Why convolution preserves local spatial patterns and flattening does not
**[F]** Flattening a 28x28 image into 784 numbers keeps all values but gives a linear model / k-NN / RF no built-in notion that index i and i+1 are neighbours, or that i and i+28 are vertically adjacent; permuting the pixel columns identically for train and test leaves LogReg, k-NN (Euclidean), RF and RBF-SVM (up to tie-breaking and random feature sampling) essentially unchanged, i.e. they do not exploit pixel layout. A CNN's filters are defined on neighbourhoods, so edges, textures and shapes are detected wherever they occur and composed hierarchically.
**[H]** For Fashion-MNIST this should help most on visually similar classes (Shirt / T-shirt / Pullover / Coat). HOG also encodes local gradient structure, so it may close part of the gap for classical models. The outcome is open; we report what we measure.
