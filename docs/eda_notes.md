# EDA notes (Akhil)

Every statement is labelled OBSERVATION (backed by a number or figure) or HYPOTHESIS (an explanation not tested here). TBD = not yet measured or checked.

## 1. Dataset facts
- OBSERVATION: official train (60,000) and test (10,000) images are 28x28 uint8 grayscale; pixel range 0-255; labels 0-9.
- OBSERVATION: classes are exactly balanced: 6,000 per class in train, 1,000 per class in test.
- OBSERVATION: the validation split is 6,000 images taken from the official train images (600 per class, seed 42), leaving 54,000 for training. The test images are read only for the integrity checks in the loading step and for the report-only duplicate check (run after the final evaluation); all other statistics below use the 54,000-image train split.

## 2. Data quality
- OBSERVATION: 0 exact duplicates within train, within test, between train and test, and between our 54k train and 6k validation splits.
- LIMIT: the check hashes raw pixel bytes, so it finds exact copies only. Near-duplicates (for example a one-pixel shift) are not checked.

## 3. Pixel statistics (54k train split, scale 0-1)
- OBSERVATION: mean pixel value 0.2862, standard deviation 0.3532, and 50.21% of all pixels are exactly zero (black background).
- OBSERVATION: per-class mean intensity ranges from 0.137 (Sandal) and 0.168 (Sneaker) up to 0.377 (Pullover) and 0.385 (Coat); Shirt is 0.333 and T-shirt/top is 0.326 (full table in results/data_quality.json).
- HYPOTHESIS: footwear has the lowest intensity because it covers less of the image area, but this was not tested.

## 4. PCA (fit on the 54k train split only)
- OBSERVATION: the first 10 components explain 71.98% of the variance and the first 50 explain 86.25%.
- OBSERVATION: PC1 explains 28.99% and PC2 explains 17.77% of the variance (46.76% together), so a 2D plot shows less than half of the variance.
- OBSERVATION (figures/eda_pca_scatter.png, 5,000 stratified train images): Shirt, Pullover and Coat points overlap heavily in the upper right of the plot, with no visible boundary between them. T-shirt/top lies mostly in the lower right, where it overlaps with Shirt and Dress. Trouser forms a compact cluster at the bottom and mixes with Dress only at its edge. Sandal and Sneaker overlap heavily at the far left, Ankle boot lies next to them toward the upper left, and Bag is spread widely across the top and middle.
- OBSERVATION (figures/eda_confusable_panel.png, 8 random train images per class): Shirt images vary a lot in pattern and sleeve length. By eye, some plain Shirt images resemble T-shirt/top and some plain long-sleeved Coat images resemble Pullover. This is a visual impression from 32 images; how often it happens was not counted.
- WHAT PCA DOES NOT SHOW: PCA is linear and unsupervised, so it ignores labels. Overlap in two components does not prove the classes are inseparable, because the other components and nonlinear boundaries are not shown. Clear separation in two components would not prove easy classification either.
- HYPOTHESIS: if these four classes overlap in the plot, a likely reason is similar overall silhouette and brightness, but this was not tested.

## 5. Class-centroid distances (train split, Euclidean on 0-1 pixels)
- OBSERVATION: the five closest class pairs are Pullover-Coat 2.473, Pullover-Shirt 2.677, Coat-Shirt 2.990, T-shirt/top-Shirt 3.685 and Sandal-Sneaker 3.823.
- HYPOTHESIS: pairs with similar mean images may be harder to tell apart. This is not a claim about classifier errors. Rohan will compare these distances with the real confusion matrices.

## 6. Sanity baselines (validation only; not part of the 9-model comparison)
- OBSERVATION: majority-class baseline gets accuracy 0.100 and macro-F1 0.0182. This matches the expected 600/6000 = 0.10 from the balanced validation set.
- OBSERVATION: nearest-class-mean baseline gets accuracy 0.692 and macro-F1 0.6882. A model must beat these floors to be worth reporting.

## 7. Limits of 28x28 grayscale
- OBSERVATION: every image is a single channel of 28x28 = 784 values (verified by shape and dtype checks).
- HYPOTHESIS: with only 784 grey values, fine details such as buttons, collars, seams and fabric texture are mostly lost, which may make Shirt, T-shirt/top, Pullover and Coat look alike.
- HYPOTHESIS: no colour information means colour cannot help separate classes.
- HYPOTHESIS: the images were downscaled and centred by the dataset creators, so real inventory photos (varied size, background, angle) may behave differently. Phase 1 does not test this.
