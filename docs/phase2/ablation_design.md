# Ablation design and regularisation theory (Tanish)

Files: `src/ablation.py`, test `tests/test_ablation.py`. Results: `results/phase2/ablation_summary.csv`,
`ablation_deltas.csv`, `figures/phase2/ablation_deltas.png`, `generalisation_gap.png`.

## 1. What an ablation study is and the rules we follow

An ablation removes or changes **one** component of a working system and measures what happens. It answers
"which parts of the model actually earn their place?" Our rules:

1. **Fixed base, decided before any run**: residual FashionNet + the Phase 2 recipe (`src/dl_config.py`).
2. **One factor at a time.** `tests/test_ablation.py` fails if an ablation differs from the base in more than its
   declared field. Two declared exceptions: SGD needs its own learning rate and weight-decay scale (2e-3 is far too
   small for SGD), and a constant schedule has no warm-up.
3. **Same seeds for every configuration** (42, 43, 44). Differences are computed **per seed** (variant minus base)
   and then averaged: a paired comparison removes the part of the noise that the two runs share.
4. **Validation only.** No ablation is ever evaluated on the test set.
5. **Report the spread.** With 3 seeds we report mean, min and max of the paired difference and whether all seeds
   agree in sign. A difference smaller than the seed-to-seed spread is reported as "no measurable effect".

Known limit of one-factor designs: they miss interactions (for example, dropout may matter more without
augmentation). A full factorial design over 9 factors would need 2^9 = 512 configurations per seed.

## 2. The 13 ablations

| Group | Name | Change | Question |
|---|---|---|---|
| Architecture | `arch_plain` | no residual shortcut | do shortcuts help at 7 layers? |
| | `arch_no_bn` | no BatchNorm | what does normalisation contribute? |
| | `arch_deeper` | 2 blocks per stage (13 conv layers) | is the base too shallow? |
| | `arch_narrow` | widths 16/32/64 (about 4x fewer weights) | how much capacity is needed? |
| Regularisation | `reg_dropout0` | dropout 0.0 | is dropout needed on top of the rest? |
| | `reg_dropout05` | dropout 0.5 | does stronger dropout under-fit? |
| | `reg_no_wd` | weight decay 0 | does decoupled weight decay matter? |
| | `reg_no_ls` | label smoothing 0 | accuracy and calibration without smoothing |
| Augmentation | `aug_none` | no augmentation | total value of augmentation |
| | `aug_no_flip` | crop + erase | value of the flip |
| | `aug_no_erase` | crop + flip | value of random erasing |
| Optimisation | `opt_sgd` | SGD + Nesterov momentum | is AdamW responsible for the result? |
| | `opt_constant_lr` | constant LR, no warm-up | value of the cosine schedule |

How to read `ablation_deltas.png`: a bar to the **left** of zero means validation macro-F1 fell when the component
was removed, so the component helps. A bar to the right means the base would be better without it.

## 3. Theory of each regulariser (bias-variance view)

Regularisation trades a little bias (the model fits the training set less perfectly) for less variance (it changes
less when the training sample changes). The **generalisation gap** = training accuracy (eval mode) minus validation
accuracy is our direct measurement of variance; `generalisation_gap.png` shows it per configuration.

**Dropout (p = 0.3 before the classifier).** During training each of the 128 pooled features is set to zero with
probability p and the rest are scaled by 1/(1-p). The classifier cannot rely on any single feature, which is
similar to averaging many thinned networks. At test time nothing is dropped. Too much dropout adds noise the model
cannot fit: under-fitting (higher bias).

**Weight decay (AdamW, 1e-2).** Each step multiplies the weights by (1 - lr x wd), pulling them towards zero. Small
weights mean a smoother function. In a network with BatchNorm the scale of a convolution's weights does not change
its output (BN re-normalises), so weight decay acts mainly by keeping the *effective learning rate* from shrinking
as weights grow. We do not decay biases or BN parameters.

**Label smoothing (0.1).** The target is 0.91 for the true class and 0.01 for each other class instead of 1 and 0.
Cross-entropy with hard targets is minimised only by infinite logits, which pushes the network towards
over-confidence. Smoothing bounds the optimal logit gap. Side effect to check: predictions become *less* confident,
which can make the model under-confident (see the calibration table and `reliability.png`).

**Batch normalisation.** For each channel, subtract the batch mean and divide by the batch standard deviation, then
apply a learned scale and shift. It stabilises the distribution of layer inputs, allows larger learning rates, and
adds slight noise (batch statistics differ between batches), a mild regulariser. Dropout placed *inside* BN blocks
shifts the variance BN sees between training and test, which is why our only dropout sits after global pooling.

**Data augmentation.** Enlarges the effective training set with label-preserving transformations: the strongest
regulariser for image models because it adds information (invariances) instead of only restricting weights.

**Early stopping.** Stops when validation loss has not improved for 8 epochs and restores the best weights. It
limits how far the weights can move from their small initial values, an implicit capacity control.

## 4. Optimisation ablations

**SGD with Nesterov momentum** uses one global learning rate; **AdamW** scales each weight's step by a running
estimate of its gradient magnitude. Adam usually converges faster; well-tuned SGD often generalises as well or
better on vision tasks. Our SGD setting is a standard one, not tuned, so the comparison says "our result does not
depend on AdamW" rather than "which optimiser is best".

**Cosine decay vs constant.** A constant learning rate keeps the weights bouncing around a minimum with a noise
level proportional to the learning rate. Decaying it shrinks that noise, so the final weights sit lower in the basin.

## 5. Viva questions

**Why paired differences instead of comparing the two means?** Runs with the same seed share initial weights and
batch order, so part of their noise is common and cancels in the difference.
**Is 3 seeds enough for a significance test?** No. We report descriptive statistics and say so.
**If an ablation improves on the base, did the design fail?** No, that is a finding. The final deep model is the
configuration with the highest mean validation macro-F1, whichever it is; the rule is fixed in advance.
**Why is the no-BatchNorm run not re-tuned?** An ablation keeps everything else fixed. A lower learning rate might
rescue it; that would be a different experiment ("BN-free network with its own recipe").
**What does a larger generalisation gap tell you?** More over-fitting: the model fits training images that it has
effectively memorised better than new ones.
**Why does label smoothing 0.1 give targets 0.91 / 0.01?** (1 - 0.1) + 0.1/10 = 0.91 for the true class and
0.1/10 = 0.01 for the others.

## Results for this component (full run, 16 November 2026)

Paired change in validation macro-F1 relative to the base (0.9523 +- 0.0007), three seeds. Full table:
`results/phase2/ablation_deltas.csv`; figure: `figures/phase2/ablation_deltas.png`.

| Group | Change | Delta (pp) | All seeds agree? |
|---|---|---|---|
| Architecture | narrow widths 16/32/64 | -1.10 | yes |
| | no BatchNorm | -0.70 | yes |
| | no residual shortcut | -0.13 | no |
| | deeper (13 conv layers) | +0.34 | yes |
| Regularisation | no label smoothing | -0.29 | yes |
| | dropout 0.5 | -0.18 | yes |
| | no weight decay | -0.10 | no |
| | no dropout | +0.02 | no |
| Augmentation | none | -0.72 | yes |
| | no random erasing | -0.16 | yes |
| | no horizontal flip | +0.00 | no |
| Optimisation | constant learning rate | -0.68 | yes |
| | SGD + Nesterov | -0.09 | no |

What to say in the viva:

* **Four things carry the model:** capacity (width), BatchNorm, augmentation and the cosine schedule, each worth
  about 0.7 to 1.1 pp. Label smoothing and random erasing are worth 0.2 to 0.3 pp.
* **Three things did nothing measurable:** dropout, weight decay and the flip. With augmentation, label smoothing
  and BatchNorm already regularising, dropout 0.3 on 128 pooled features has nothing left to do. This is an
  interaction that a one-factor design can only hint at.
* **Regularisation is not free for a small model.** The Phase 2 recipe made the Phase 1 CNN worse
  (-0.28 pp): its train-val gap was already only 0.29 pp. Bias-variance view: we
  added bias to a model that had no variance to trade.
* **An ablation that beats the base is a finding, not a failure.** "Deeper" gained +0.34 pp in every seed
  and became the final model by the pre-registered rule. Consequence to admit: the other 12 ablations describe the
  7-layer base, not the 13-layer final model.
* **BatchNorm also stabilises training:** without it the standard deviation over seeds rises from 0.07 to
  0.50 pp.
* **Label smoothing has a hidden cost:** +0.29 pp macro-F1, but calibration error rises from 0.010 to
  0.083 (see `evaluation_notes.md`).
