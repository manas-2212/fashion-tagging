# Feature notes (Anant) — flattened pixels and HOG

## 0. DECISION RULE (written BEFORE the study was run on real data)

Evidence: `hog_sensitivity_study` — 4 HOG configurations, Logistic Regression (C=1.0, StandardScaler inside the
pipeline), trained on a stratified 10,000-image subset of the 54,000 training split, scored on the 6,000-image
validation split. **Test data is never used.** Seed 42.

1. Baseline = `orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), block_norm="L2-Hys"` (1,296 features).
2. **Switch** to another configuration only if its validation macro-F1 exceeds the baseline's by **more than
   0.010 (1 percentage point)**.
3. If several configurations pass, take the one with the highest validation macro-F1; if two are within 0.001,
   take the one with fewer features (smaller `hog_dim`).
4. Otherwise **keep the baseline**. HOG parameters may change **once**, frozen at milestone M2.
5. Limits (stated in advance): one model (LogReg), one 10k subset, one seed, one run. The study is *evidence for a
   choice*, not proof of the best descriptor. A descriptor that is best for LogReg need not be best for k-NN, RF or
   RBF-SVM. With 6,000 validation images the sampling noise of an accuracy near 0.85–0.90 is roughly ±0.4 pp
   (one standard error, binomial approximation), so a 1 pp threshold is deliberately conservative.

**Decision (after running):** SWITCH to `orientations=9, pixels_per_cell=(2,2), cells_per_block=(2,2), block_norm="L2-Hys"` (6,084 features). Validation macro-F1 was 0.868348 versus 0.836344 for the baseline, an improvement of 0.032004 (+3.20 percentage points), exceeding the pre-written 0.010 threshold. The 6-orientation configuration also passed the threshold (0.848631; improvement +0.012287), but its improvement was smaller. The 7×7 configuration did not pass. This is evidence for the Logistic Regression descriptor choice, not proof that it is optimal for every classifier.

## 1. Flattened pixels

**Construction (fact).** `X.reshape(N,-1).astype(float32)/255` → 784 features per image in [0,1], row-major
(row `r`, column `c` → index `28r+c`).

**What it captures (fact).** Absolute grey level at every location: silhouette area, overall brightness, and the
coarse position of bright regions (e.g. a sandal's bare area vs. a coat's large filled area).

**What it lacks (fact / property of the model, not of the pixels).** Each pixel is an independent column. Logistic
Regression, k-NN (Euclidean), RBF-SVM and Random Forest give the *same* results if all images' columns are permuted in
the same way, so these models cannot use the fact that two pixels are neighbours. It is also sensitive to
translation: shifting an object by one pixel changes many features at once.

## 2. HOG (Histogram of Oriented Gradients)

Original method: Dalal and Triggs, CVPR 2005 (**Puneet must verify the full reference before it enters the report**).
Implementation: `skimage.feature.hog`, scikit-image 0.26.0 (the version on both machines used so far; see §4), grayscale via
`channel_axis=None`.

**Pipeline (fact).**
1. Image → float32 in [0,1].
2. Gradients: horizontal and vertical differences per pixel → gradient magnitude and orientation (unsigned, 0–180°).
3. Cells: each `pixels_per_cell` patch gets a histogram of `orientations` bins; each pixel votes into its
   orientation bin(s) weighted by gradient magnitude.
4. Blocks: groups of `cells_per_block` cells, sliding with a stride of one cell (so blocks overlap); each block's
   concatenated histograms are normalised (`block_norm`).
5. All normalised blocks are concatenated into one vector.

**Dimension (verified by code against a real `hog` call).**
cells per axis = 28/4 = 7; blocks per axis = 7 − 2 + 1 = 6; length = 6 · 6 · (2 · 2) · 9 = **1,296**.
The same formula gives 324 (ppc 7), 6,084 (ppc 2) and 864 (orientations 6); all four matched `skimage` output.
If `28` is not divisible by `pixels_per_cell`, skimage floors the cell count and ignores the leftover border pixels.

**Parameters.**
- `orientations`: number of angle bins over 180°. More bins = finer edge-direction resolution, longer vector.
- `pixels_per_cell`: spatial size of the histogram area. Small = fine spatial detail but noisier, larger vector;
  large = coarser, more robust to small shifts.
- `cells_per_block`: how many cells are normalised together. Larger blocks = more context per normalisation,
  more redundancy (overlap) in the vector.
- `block_norm="L2-Hys"`: L2-normalise, clip values at 0.2, renormalise (implementation detail; check skimage docs
  when citing).

**Why block normalisation helps (fact in principle, magnitude here unmeasured).** Raw cell histograms scale with local
contrast. Normalising over overlapping blocks makes the descriptor much less sensitive to brightness/contrast
changes (e.g. a dark vs. light garment of the same shape), and overlap lets each cell be seen under several
normalisation contexts. Flat blocks produce an all-zero (not NaN) block vector — tested.

**Why HOG at 28×28 may help or hurt (HYPOTHESES — not tested).**
- Fabric texture, stitching, buttons and collar details occupy only a few pixels at 28×28; 4×4 cells (each 16 pixels)
  may contain too little structure for a stable histogram, so Shirt / T-shirt / Pullover / Coat may stay confused.
- HOG discards absolute grey level; if brightness or filled area is informative for a class pair, information is lost.
- Shape/edge classes (Trouser, Bag, Sneaker, Sandal) are plausible gains because edges are explicit.
These are explanations to check against the confusion matrices at M6, not findings.

## 3. Caching

`cached_features(X, kind, split_name, cache_dir)` saves `<kind>_<split>_N<N>[_<HOG params>]_<sha1-10>.npy`.
The key contains kind, split name, N, HOG parameters (for HOG) and — beyond the spec — a 10-character content hash, so
a different array under the same split name and N cannot load stale features. HOG parameters are read from
`config.HOG_PARAMS` at call time. Files are written to `*.tmp` and renamed. HOG is a fixed per-image function (nothing is
fitted), so computing it on train, val and test creates no leakage; only the StandardScaler (fitted on train inside the
pipeline) needs care.

## 4. Runtime (ESTIMATE)

Measured: serial HOG (ppc 4x4, o9) on 1,000 **mock random-noise** images, 3 runs per machine.
The HOG code is serial, so it used a single core even though the M3 reports 8 logical CPUs.

| Machine (as read from the system) | Runs (s) | Median (s) | ms/image | Linear estimate, 70,000 images |
|---|---|---|---|---|
| Apple M3, `hw.ncpu` = 8; numpy 2.5.4, scikit-image 0.26.0, Python 3.14.2 | 0.201, 0.203, 0.202 | 0.202 | 0.202 | ~14 s |
| Development sandbox: Intel(R) Xeon(R) Processor @ 2.80GHz, `os.cpu_count()` = 1; numpy 2.4.4, scikit-image 0.26.0, Python 3.12.3 | 0.596, 0.656, 0.638 | 0.638 | 0.638 | ~45 s |

These are **estimates** from linear extrapolation on random-noise images, not measured on real Fashion-MNIST.
Real images should cost about the same because the work per image is fixed, but this is unverified.
Flattening 1,000 images takes well under a millisecond (0.00029 s on the M3).
Caching still pays off: the notebook is re-run repeatedly and the study extracts several HOG variants.
Finer configs (ppc 2x2) will cost more; not yet measured.
Timing on real data and the 6,084-feature fit time: TBD.

## 5. Sensitivity table (validation only, 10k train subset, LogReg C=1.0)

| config | hog_dim | extract_seconds | val_accuracy | val_macro_f1 |
|---|---:|---:|---:|---:|
| o9, ppc 4×4, cpb 2×2 (baseline) | 1296 | 4.254 | 0.835167 | 0.836344 |
| o9, ppc 7×7, cpb 2×2 | 324 | 1.692 | 0.834333 | 0.834319 |
| o9, ppc 2×2, cpb 2×2 | 6084 | 18.165 | 0.867833 | 0.868348 |
| o6, ppc 4×4, cpb 2×2 | 864 | 4.323 | 0.847500 | 0.848631 |

The 2×2-cell configuration exceeded the baseline macro-F1 by 0.032004, passing the pre-registered +0.010 switch threshold. The 6-orientation configuration improved by 0.012287; the 7×7-cell configuration decreased by 0.002025. The selected configuration has 6,084 features, so its fitting time and memory costs must be assessed before final classical-model runs.

**Environment:** Python 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1, scikit-image 0.24.0 (active terminal environment; confirm the study used this same environment).

**Timing scope:** `extract_seconds` measures serial HOG extraction for the 10,000-image stratified training subset plus the 6,000-image validation set. It excludes model fitting and prediction. The 2×2-cell configuration took 18.165 seconds for these 16,000 images, versus 4.254 seconds for the baseline.
