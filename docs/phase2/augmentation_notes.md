# Augmentation and robustness notes (Akhil)

Files: `src/augment.py`, `src/robustness.py`, tests `tests/test_augment.py`, `tests/test_robustness.py`.

## 1. Why augment Fashion-MNIST at all

The training split has 54,000 images, 5,400 per class. In Phase 1 the CNN's validation loss stopped
improving at epoch 24 while its training loss kept falling: it had started to memorise. Augmentation shows the network a slightly different version of
each image in every epoch, so memorising pixels stops paying off. It is a regulariser that encodes prior knowledge:
"the label does not change if the garment moves by two pixels".

Rule used for every operation: **it must not change the label, and it should produce images that could plausibly
occur.** That excludes vertical flips and large rotations (an upside-down shirt is not in the catalogue).

## 2. The three operations (`BatchAugmenter`)

| Operation | What it does | Why it is plausible here | Ablation |
|---|---|---|---|
| Random crop | pad 2 px with background, cut a random 28x28 window (shift of -2..+2 px) | items are roughly, not exactly, centred | part of `aug_none` |
| Horizontal flip | mirror with probability 0.5 | garments are near-symmetric; a mirrored shoe is still a shoe | `aug_no_flip` |
| Random erasing | with probability 0.25 overwrite a random rectangle (2-20% of the area) with the dataset mean | occlusion, folds, tags; stops reliance on one region | `aug_no_erase` |

Honest expectation for the flip: all footwear in the dataset points the same way, so flipped shoes are images the
test set never contains, and symmetric garments barely change. The ablation decides; we do not assume it helps.

Implementation details that matter:

* Augmentation runs on the **normalised** tensor on the training device, on a whole batch at once (no Python loop).
* Padding uses the normalised value of a black pixel, `(0 - mean) / std`, not 0. After normalisation 0 means
  "average gray", which would draw a gray frame around the image.
* Erasing fills with 0, which after normalisation is the dataset mean (the standard choice of Zhong et al.).
* Validation and test images are **never** augmented. Mean and std come from the training split only.
* A seeded `torch.Generator` makes the augmentation reproducible.

## 3. Robustness stress test (`src/robustness.py`)

Fashion-MNIST is clean; real inventory photos are not. After the test gate we corrupt a stratified subset of 2,000
test images and measure accuracy for four models (Phase 1 CNN re-trained, final deep model, hybrid H2, HOG + SVM).

| Corruption | Levels | What it imitates |
|---|---|---|
| Gaussian noise | std 0.05, 0.10, 0.20 | sensor noise, compression |
| Brightness | +0.1, +0.2, +0.3 | over-exposure, gray background instead of black |
| Contrast | x0.75, x0.5, x0.25 | flat lighting |
| Shift | 1, 2, 3 px | imperfect cropping |
| Occlusion | 5, 8, 11 px square | tags, hands, overlapping items |

Things to look for in `figures/phase2/robustness.png`:

* **Shift and occlusion** are what crop and erasing train for; the augmented model should degrade more slowly.
* **Brightness** changes the background from black to gray, something no training image shows. A CNN normalised
  with fixed mean/std has never seen that input range. HOG uses gradients, and adding a constant does not change a
  gradient, so the HOG model should be almost unaffected until pixels saturate at 255.
* **Contrast** scales gradients; HOG's block normalisation divides that scale out.

This test is descriptive. Nothing is tuned on it, and it runs only after the gate.

## 4. Viva questions

**Is augmentation a form of data leakage?** No. It uses only training images and only label-preserving changes.
**Why not augment the validation set?** Validation must estimate performance on real, unmodified images.
**Why pad with -0.81 and not 0?** With mean 0.286 and std 0.353, a black pixel becomes (0 - 0.286)/0.353 = -0.81.
**Does augmentation increase the dataset size?** Not per epoch: each epoch still has 54,000 images, but different
versions each time, so over 30 epochs the network sees far more distinct inputs.
**Why can augmentation lower training accuracy?** The training task becomes harder; the gap to validation shrinks.
**Why is HOG robust to brightness but the CNN is not?** HOG is built from differences between neighbouring pixels;
the CNN sees absolute values that leave the range it was trained on.

## Results for this component (full run, 16 November 2026)

**Augmentation ablations** (paired change in validation macro-F1 when the piece is removed, 3 seeds):

| Removed | Change | Seeds agree? | Reading |
|---|---|---|---|
| All augmentation | -0.72 pp | yes | train accuracy reaches 0.9990, gap 5.35 pp: memorisation |
| Random erasing | -0.16 pp | yes | small but consistent benefit |
| Horizontal flip | +0.00 pp | no | no effect, as predicted for near-symmetric garments |

**Robustness** (accuracy at the strongest level; 2,000 test images, seed 42; `results/phase2/robustness.csv`):

| Corruption | Phase 1 CNN | Final deep model | H2 fusion | H0 HOG + SVM |
|---|---|---|---|---|
| Shift 3 px | 0.664 | 0.942 | 0.940 | 0.416 |
| Occlusion 11 px | 0.836 | 0.916 | 0.914 | 0.794 |
| Gaussian noise 0.20 | 0.546 | 0.167 | 0.172 | 0.329 |
| Brightness +0.3 | 0.459 | 0.553 | 0.601 | 0.890 |
| Contrast -75% | 0.793 | 0.868 | 0.869 | 0.899 |

What to say in the viva:

* **Augmentation did what it was designed for.** Shift and occlusion barely hurt the final model, because random
  crops and random erasing show it exactly those changes during training.
* **It did nothing for what it does not cover.** Under Gaussian noise the final model is much *worse* than the
  Phase 1 CNN (0.381 vs 0.855 at std 0.10). No training image contains noise. We observed this;
  we did not test why the deeper network is more sensitive.
* **Brightness breaks every CNN, not HOG.** Adding a constant leaves gradients unchanged, so HOG + SVM stays at
  0.890 while the final model falls to 0.553.
* **Honest conclusion:** the better benchmark model is not the more robust model. Next step: add noise and
  brightness augmentation and repeat this table.
