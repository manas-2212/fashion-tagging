# Hybrid ML + DL notes (Anant)

Files: `src/hybrid.py`, test `tests/test_hybrid.py`. Stage: `stage_hybrid` in `src/phase2_pipeline.py`.

## 1. The question the hybrid answers

The project title is "handcrafted features **and** a small CNN". Phase 1 compared them as rivals. Phase 2 asks
whether they are **complementary**: does a description of edge orientation that was designed by hand (HOG) add
anything to a description that was learned from data (the CNN embedding)?

Evidence that motivates the question (Phase 1): the CNN made 726 test errors, the pixel SVM 1,047, and McNemar tests
showed that the models disagree on many images. If two models fail on different images, a combination can in
principle beat both. `complementarity()` measures this directly: the *oracle accuracy* is the share of images that
at least one of two models gets right, an upper bound for any combination of the two.

## 2. The four models and why each exists

| Name | Input to the RBF-SVM | Role |
|---|---|---|
| H0 | PCA(HOG), 64 or 128 dims | control: handcrafted features only |
| H1 | CNN embedding, 128 dims | control: learned features with a classical classifier |
| H2 | [embedding, alpha x PCA(HOG)] | **the hybrid**: feature-level fusion |
| H3 | CNN softmax if confidence >= tau, else H2 | **cascade**: pay for the hybrid only on uncertain images |

The controls make the claim falsifiable:

* H2 > CNN but H2 = H1 -> the gain comes from replacing the softmax layer by an SVM, **not** from HOG.
* H2 > H1 -> the handcrafted features carry information the embedding lacks.
* H2 = CNN -> the CNN has already learned everything HOG offers on this dataset; we report that plainly.

## 3. How the fusion vector is built (`HybridFeatures`)

1. **Embedding block**: 128 pooled CNN features, standardised (zero mean, unit variance per dimension).
2. **HOG block**: 6,084 HOG values -> PCA with whitening to `hog_dim` components. Whitening gives every component
   unit variance, so both blocks are on the same scale.
3. **alpha** multiplies the HOG block. The RBF kernel depends on the squared distance
   `||z_a - z_b||^2 = ||emb_a - emb_b||^2 + alpha^2 ||hog_a - hog_b||^2`, so alpha sets how much HOG differences
   count. alpha = 0 would be H1.
4. RBF-SVM with `gamma="scale"`, the best classical model of Phase 1.

Tuned on validation: `C` in {1, 10}, `hog_dim` in {64, 128}, `alpha` in {0.5, 1, 2}; the SVM is tuned on a
stratified 10,000-image training subset (as in Phase 1) and then fitted on all 54,000 images. Hyper-parameters are
chosen with seed 42 and re-used for seeds 43 and 44.

## 4. The cascade (H3)

`cascade_predict`: if the CNN's largest softmax probability is at least tau, keep the CNN answer; otherwise ask H2.
tau is chosen on validation from a fixed grid (highest macro-F1; ties go to the smallest routed share).
Why it is interesting even when accuracy is equal to H2: HOG extraction costs milliseconds per image while the CNN
costs microseconds on a GPU. The cascade pays the HOG cost only for the routed fraction.

## 5. Leakage rules and one honest limitation

* Scaler, PCA and SVM are fitted on the training split only; `C`, `hog_dim`, `alpha`, tau are chosen on validation;
  test data appears only inside the Phase 2 gate. The PCA is unsupervised and fitted on a 20,000-image stratified
  training subset to bound memory.
* **Limitation.** The CNN was trained on the same 54,000 images whose embeddings train the SVM. A network fits its
  training images better than new ones, so training embeddings are "cleaner" than validation embeddings. The SVM
  sees an easier problem during fitting than at prediction time. The textbook fix is out-of-fold embeddings
  (train the CNN K times), which would multiply the GPU budget. We keep the simple version and let validation and
  the test gate judge it.

## 6. Viva questions

**Why PCA before fusing?** 6,084 HOG dimensions next to 128 embedding dimensions would dominate every distance;
PCA keeps the directions with most variance and whitening equalises scales.
**Why an SVM and not another neural layer?** The task asks for an ML + DL hybrid; the RBF-SVM was the best classical
model in Phase 1 and it maximises the margin in the fused space instead of minimising cross-entropy.
**What would make you say the hybrid failed?** If H2 does not beat the CNN beyond seed-to-seed variation, or if
H1 matches H2. Both outcomes are reported with the same tables.
**Why is the oracle accuracy not achievable?** It uses the true label to pick the right model for every image.
**What does tau = 0 mean?** Nothing is routed: the cascade is the plain CNN.
**Is the hybrid slower?** Yes: it adds HOG extraction, a PCA projection and kernel evaluations against the support
vectors. `results/phase2/inference_phase2.csv` quantifies it.

## Results for this component (full run, 16 November 2026)

Selected on validation (seed 42): H2 uses C = 1, 128 HOG components (48.3% of the HOG variance),
alpha = 2; cascade threshold tau = 0.7.

| Model | Test accuracy (mean +- std, 3 seeds) |
|---|---|
| H0: PCA-HOG + SVM | 0.9048 |
| Final deep model | 0.9488 +- 0.0015 |
| H1: embedding + SVM | 0.9495 +- 0.0009 |
| H2: embedding + HOG fusion | 0.9502 +- 0.0020 |
| H3: cascade | 0.9502 +- 0.0021 |

What to say in the viva (this is the part that decides the hybrid marks, so say it exactly as measured):

* **Complementarity is real.** HOG + SVM is right on 181 of the 510 test images the deep
  model gets wrong (35.5%). A perfect selector would reach 0.9671.
* **The fusion gain is small and not statistically significant.** H2 beats the deep model in all 3 seeds, by
  +0.15 pp on average (+0.11 to +0.22; +0.21 pp on validation). The seed-to-seed
  standard deviation is the same size, and McNemar gives p = 0.20 (H2 fixes 36 errors and
  makes 25 new ones).
* **The control says HOG adds little beyond the SVM head.** H1 (no HOG) gains +0.08 pp; H2 vs H1 has
  p = 0.90.
* **Why complementarity does not turn into accuracy.** HOG is also wrong on 623 images the CNN gets right. A
  combiner must know *which* model to trust for each image, and nothing in the features tells it reliably.
* **Where the hybrid work does have value.** (1) The cascade reaches the same accuracy as H2 while sending only
  5.7% of the images to the HOG + SVM path, which costs 5.56 ms per image against 0.077 ms
  for the CNN. (2) PCA made the classical HOG model trainable on all 54,000 images: H0 reaches 0.9048,
  against 0.8817 for the Phase 1 HOG SVM fitted on 11,770 images. (3) Under brightness shift H0 keeps
  0.890 accuracy while the CNN falls to 0.553; the fusion did not inherit that
  (0.601), which points to routing by input statistics as the next design.
* **One sentence:** "We built the hybrid with controls so that it could fail, and it mostly did: the direction is
  consistent, the size is within noise, and we report it that way."
