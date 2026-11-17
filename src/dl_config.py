"""Phase 2 (deep learning + hybrid) constants. Single source of truth; import from here (owner: Puneet).

Phase 1 constants stay in src/config.py and are NOT changed by Phase 2.
Everything that defines an experiment (recipes, seeds, grids, thresholds) lives in this file so that the
freeze file can record it and a reader can audit it in one place.
"""
import os
from pathlib import Path

from src.config import FIGURES_DIR, RESULTS_DIR, SEED

# ----------------------------------------------------------------------------- seeds and folders
P2_SEEDS = (42, 43, 44)          # every configuration is trained once per seed (Phase 1 used one seed only)
PRIMARY_SEED = SEED              # seed used for single-model analyses (bootstrap CI, McNemar, figures)

# The output root can be redirected (e.g. to Google Drive on Colab) with the FASHION_P2_RESULTS variable.
P2_RESULTS_DIR = Path(os.environ.get("FASHION_P2_RESULTS", RESULTS_DIR / "phase2"))
P2_FIGURES_DIR = Path(os.environ.get("FASHION_P2_FIGURES", FIGURES_DIR / "phase2"))

# ----------------------------------------------------------------------------- training recipes
# Phase 1 recipe: exactly what src/cnn.train_cnn did (Adam, constant LR, no augmentation, patience 5).
P1_RECIPE = dict(optimizer="adam", lr=1e-3, weight_decay=0.0, schedule="constant", warmup_epochs=0,
                 max_epochs=30, patience=5, batch_size=128, label_smoothing=0.0, aug="none")

# Phase 2 recipe, fixed BEFORE any Phase 2 run (the ablations then remove one piece at a time).
P2_RECIPE = dict(optimizer="adamw", lr=2e-3, weight_decay=1e-2, schedule="cosine", warmup_epochs=1,
                 max_epochs=30, patience=8, batch_size=128, label_smoothing=0.1, aug="crop_flip_erase")

# Phase 2 base architecture (FashionNet with residual blocks); see src/dl_models.py.
BASE_ARCH = dict(widths=(32, 64, 128), blocks_per_stage=1, use_bn=True, residual=True, dropout=0.3)
P1_ARCH = dict(dropout=0.5)      # the Phase 1 FashionCNN, unchanged

SGD_LR = 0.05                    # learning rate of the SGD ablation (Nesterov momentum 0.9)
SGD_WEIGHT_DECAY = 5e-4

# ----------------------------------------------------------------------------- augmentation
AUG_PAD = 2                      # random crop: pad 2 pixels on each side, crop back to 28x28 (shift of -2..+2)
FLIP_PROB = 0.5
ERASE_PROB = 0.25
ERASE_SCALE = (0.02, 0.20)       # erased area as a fraction of the image
ERASE_RATIO = (0.3, 3.3)         # aspect ratio range of the erased rectangle

# ----------------------------------------------------------------------------- hybrid (ML + DL)
HYBRID_TUNE_N = 10_000           # stratified train subset used to tune SVM hyper-parameters (as in Phase 1)
HOG_PCA_FIT_N = 20_000           # stratified train subset used to fit the (unsupervised) HOG PCA
HYBRID_GRID = dict(C=(1.0, 10.0), hog_dim=(64, 128), alpha=(0.5, 1.0, 2.0))
HOG_ONLY_GRID = dict(C=(1.0, 10.0), hog_dim=(64, 128))
EMB_ONLY_GRID = dict(C=(1.0, 10.0))
CASCADE_TAUS = (0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99)   # tau = 0 means "never route"

# ----------------------------------------------------------------------------- evaluation
ECE_BINS = 15
IN_GROUP = (0, 2, 4, 6)          # T-shirt/top, Pullover, Coat, Shirt: the confusable upper-body group
ROBUSTNESS_N = 2_000             # stratified test subset used for the corruption study (HOG is slow)
N_BOOT = 1_000
