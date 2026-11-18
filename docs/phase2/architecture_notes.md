# Architecture notes (Manas)

Files: `src/dl_models.py`, `src/dl_train.py`, `src/phase2_pipeline.py`, `run_phase2.py`, `notebook_phase2.ipynb`.

## 1. What was wrong with the Phase 1 CNN

The Phase 1 CNN (3 x [conv 3x3, ReLU, max-pool], then Flatten 1,152 -> Dense 128 -> Dense 10) reached 0.9274 test
accuracy. Three measurable properties limit it (all computed by `src/dl_models.py`, checked in `tests/test_dl_models.py`):

| Property | Phase 1 CNN | Phase 2 base (FashionNet) |
|---|---|---|
| Trainable parameters | 241,546 | 308,074 |
| Parameters in dense layers | 148,874 (61.6%) | 1,290 (0.4%) |
| 3x3 convolution layers | 3 | 7 |
| Receptive field of the last feature map | 22 x 22 pixels | 34 x 34 (whole image) |
| Normalisation | none | BatchNorm after every convolution |
| Head | Flatten + 2 dense layers | global average pooling + 1 dense layer |

1. **Most weights are in one dense layer.** `Linear(1152, 128)` alone has 147,584 weights. A dense layer has no notion
   of position: it must learn "collar at the top" separately for every location. Convolutions share weights across positions.
2. **One convolution per resolution.** Each resolution (28, 14, 7) gets a single non-linear step. The classes that
   Phase 1 confused (T-shirt, Pullover, Coat, Shirt: 11.8% in-group confusion) differ in small local patterns such
   as collars, button plackets and sleeve length, which need several stacked filters to describe.
3. **The last feature map does not see the whole garment.** Its receptive field is 22 pixels; the image is 28.

## 2. FashionNet: the Phase 2 family

```
stem   : conv3x3(1 -> 32) + BN + ReLU                              28 x 28
stage 1: ConvBlock(32 -> 32)                      then MaxPool 2   28 -> 14
stage 2: ConvBlock(32 -> 64)                      then MaxPool 2   14 -> 7
stage 3: ConvBlock(64 -> 128)                                      7 x 7
head   : global average pool -> Dropout(0.3) -> Linear(128 -> 10)  logits
ConvBlock(x) = ReLU( BN(conv3x3( ReLU(BN(conv3x3(x))) )) + shortcut(x) )
```

Design decisions and the reason for each:

* **Two 3x3 convolutions per block.** Two stacked 3x3 layers cover a 5x5 area with 18 weights per channel pair
  instead of 25, and add a second non-linearity (the VGG argument).
* **Batch normalisation** after every convolution keeps the scale of activations stable from layer to layer, so a
  higher learning rate can be used and the loss surface is smoother. With BN the convolution bias is redundant
  (BN subtracts the mean), so `bias=False`.
* **Residual shortcuts.** The block learns a correction `F(x)` to its input instead of a full mapping. The gradient
  of `x + F(x)` with respect to `x` contains the identity, so the error signal reaches early layers without
  shrinking. When the channel count changes, a 1x1 convolution + BN matches the shapes. Whether shortcuts matter at
  only 7 layers is an empirical question: ablation `arch_plain` removes them and nothing else.
* **Global average pooling** replaces Flatten + Dense(1152, 128). It has no parameters, forces the last convolution
  to produce class-relevant feature maps, and the 128 pooled values are the **embedding** used by the hybrid models.
* **Width doubles when resolution halves** (32, 64, 128): the cost per layer stays roughly constant.
* **No pooling after stage 3.** 7 is odd; pooling to 3x3 would drop a row and a column (as the Phase 1 CNN did).

## 3. The architecture roadmap (how we separate "recipe" from "architecture")

| Step | Architecture | Recipe | What the step isolates |
|---|---|---|---|
| 1 | Phase 1 CNN | Phase 1 (Adam, constant LR, no augmentation) | reference, now with 3 seeds |
| 2 | Phase 1 CNN | Phase 2 | effect of the training recipe alone |
| 3 | FashionNet plain (no shortcut) | Phase 2 | effect of depth + BN + GAP |
| 4 | FashionNet residual (base) | Phase 2 | effect of the shortcuts alone |

Without step 2, a gain from step 1 to step 4 could not be attributed: it could be the augmentation.

## 4. The training engine (`src/dl_train.py`)

* **AdamW**: Adam with *decoupled* weight decay. In Adam with L2 regularisation the penalty gradient is divided by
  the adaptive step size, so weights with large gradients are decayed less. AdamW applies `w <- w - lr * wd * w`
  directly. Biases and BatchNorm parameters are excluded from weight decay (`split_decay_groups`).
* **Warm-up + cosine decay**: the learning rate rises linearly during epoch 1 (BatchNorm statistics and Adam's
  moment estimates are unreliable at the start), then follows `lr * 0.5 * (1 + cos(pi * t))` to zero. Large steps
  early explore; small steps late settle into a minimum.
* **Early stopping** on validation cross-entropy (patience 8) with the best weights restored. The validation loss
  is plain cross-entropy even when the training loss uses label smoothing, so runs are comparable.
* **Mixed precision** on GPU: forward pass in float16, weights in float32, gradients rescaled by `GradScaler`.
* **Reproducibility**: `set_global_seed(seed)` before building the model (initial weights), a seeded generator for
  the batch order and another for augmentation. Same seed -> same run on the same hardware (tested on CPU).

## 5. Viva questions

**Why not a pretrained ResNet-18?** The brief asks for a small CNN, the images are 28x28 grayscale (ImageNet
weights expect 224x224 colour), and a from-scratch model lets us ablate every component.
**Why is the final model selected by validation macro-F1 and not test accuracy?** Selecting on test would make the
test number optimistic. The rule is fixed in `select_final_config` and recorded in the freeze file.
**What is the receptive field and why does it matter?** The input region that can influence one output unit. A unit
that sees 22 pixels cannot combine "long sleeve on the left" with "long sleeve on the right" of a 28-pixel image.
**How many parameters does a 3x3 conv from 64 to 128 channels have?** 3*3*64*128 = 73,728 (no bias with BN).
**What does BatchNorm do at test time?** It uses running averages of mean and variance collected during training,
not the statistics of the test batch, so a prediction does not depend on which other images are in the batch.
**Why does the embedding come from the pooled layer?** It is the input of the linear classifier: the classes are
(almost) linearly separable there, which is what a classical classifier needs.

## Results for this component (full run, 16 November 2026)

All numbers are means over seeds 42, 43, 44; source files are in `results/phase2/`.

| Roadmap step (validation macro-F1) | Mean +- std | Train-val gap |
|---|---|---|
| Phase 1 CNN, Phase 1 recipe | 0.9343 +- 0.0023 | 2.74 pp |
| Phase 1 CNN, Phase 2 recipe | 0.9314 +- 0.0011 | 0.29 pp |
| FashionNet without shortcuts | 0.9510 +- 0.0024 | 1.74 pp |
| FashionNet base (7 conv layers) | 0.9523 +- 0.0007 | 1.49 pp |
| FashionNet deeper (13 conv layers), **selected** | 0.9557 +- 0.0010 | 2.53 pp |

What to say in the viva:

* **The gain is the architecture, not the recipe.** The Phase 2 recipe on the Phase 1 CNN changed macro-F1 by
  -0.28 pp (worse in all three seeds). Its training accuracy dropped to
  0.9347: the small network under-fits once augmentation and label smoothing make the task
  harder. With the same recipe, the new architecture adds +1.96 pp.
* **Shortcuts did not matter at 7 layers** (+0.13 pp, seeds disagree on the sign). That is the expected
  result: residual connections solve an optimisation problem of deep networks, and 7 layers is not deep. We did not
  test the 13-layer network without shortcuts, so we cannot say whether they matter there.
* **Depth helped** (+0.34 pp in every seed), so the selection rule picked the 13-layer variant
  (696,042 parameters, receptive field 62 px) as the final deep model. **Width mattered most**
  (-1.10 pp for the narrow network) and **BatchNorm** -0.70 pp.
* **Test:** final deep model 0.9488 +- 0.0015 vs re-trained Phase 1 CNN 0.9252 +- 0.0034
  (+2.36 pp, 32% fewer errors, McNemar p < 0.0001). Shirt recall
  0.757 -> 0.834.
* **Cost:** 0.077 ms per image on the GPU vs 0.011 ms (7.1x slower), 2.49 ms vs
  0.50 ms on the CPU. All 48 runs took 105 minutes of training on one T4.
