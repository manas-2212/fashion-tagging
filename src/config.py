"""Single source of truth for constants. Import from here; never redefine."""
from pathlib import Path
import random
import numpy as np

SEED = 42
VAL_FRACTION = 0.10
CLASS_NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
NUM_CLASSES = 10
N_TRAIN_FULL, N_TEST = 60_000, 10_000
N_TRAIN, N_VAL = 54_000, 6_000
PIXEL_DIM = 784

HOG_PARAMS = dict(orientations=9, pixels_per_cell=(4, 4),
                  cells_per_block=(2, 2), block_norm="L2-Hys")
HOG_DIM = 1296  # 7x7 cells -> 6x6 blocks * 4 cells * 9 bins (verify by code)

SVM_TUNE_N = 10_000
MAX_FIT_MINUTES = 30
BENCH_BATCH_SIZE = 256

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
