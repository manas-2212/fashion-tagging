# Error analysis notes (Rohan; framework written by Manas)

Purpose: turn the notebook's section 11 outputs into the report section "Confusion Matrices and Error Analysis"
(`report/main.tex`, label `sec:errors`) **without inventing anything**. Everything below is a procedure or a hypothesis; every finding must be
read off the measured files in `results/` after the single test-gate run.

## 0. Rules
1. Describe only what the files show. Do not write a sentence about a class pair before you have looked at its row in `results/confusion_matrices.npz`.
2. Hypotheses go in a separate paragraph, labelled as hypotheses, and are marked "supported" or "not supported" by a number you cite.
3. No cherry-picking: "correct" examples are the first n in dataset order; "incorrect" examples are the n most confident mistakes (the code says so in its docstring; state this in the caption).
4. The test set is used once. Error analysis uses the gate's saved predictions (`results/test_gate_outputs.npz`) and nothing is re-evaluated.
5. Use the completed notebook outputs and report tables as the measured source. In this ZIP snapshot, the generated CSV/NPZ artifacts are missing; restore them from the completed Colab run rather than regenerating the one-shot test gate.

## 1. Files produced by the notebook (section 9, 11, 12)
| File | Content |
|---|---|
| `results/metrics_with_ci.csv` | accuracy, macro-F1 and 95 % bootstrap CIs per model |
| `results/per_class_recall.csv` | 9 models × 10 classes |
| `results/confusion_matrices.npz` | one 10×10 int64 matrix per model (keys like `svm__hog`, `cnn`) |
| `results/mcnemar_cnn_vs_classical.csv` | CNN vs each classical model, with Bonferroni-adjusted p |
| `figures/cm_cnn.png`, `figures/per_class_recall.png`, `figures/model_comparison.png` | report figures |

## 2. Questions to answer, in this order (each answer cites a file)
1. Which classes have the lowest recall for the CNN, and for the best classical model? (`per_class_recall.csv`)
2. What are the 10 largest confusions for the CNN? Are the same pairs the top confusions for the classical models? (`top_confusions`)
3. Do the CNN and the classical models fail on the same images? How many images do *all* nine models get wrong? (`compare_errors`: `n_shared_by_all`, `n_unique_errors`)
4. Does HOG change which classes are confused compared with pixels, for the same classical model? (compare two confusion matrices)
5. Are the CNN's most confident mistakes visually ambiguous (label noise, near-identical garments) or clearly wrong? (`show_prediction_examples(mode="incorrect")`; describe what you *see*, e.g. "6 of 12 are Shirt predicted as T-shirt/top", and count them)
6. For each CNN-vs-classical comparison, is the McNemar adjusted p below 0.05? Say it with the discordant counts (`n01`, `n10`).

## 3. Hypotheses to test (not findings)
- **H1** The most confused pairs are among Shirt, T-shirt/top, Pullover and Coat (similar silhouettes at 28×28). *Supported if* these classes hold most of the top-10 confusions.
- **H2** HOG helps shape-defined classes (Trouser, Bag, Sneaker, Sandal) more than texture-defined ones. *Supported if* per-class recall gain pixels→HOG is larger for those classes than for the four above.
- **H3** The CNN reduces the confusions between the four upper-body classes relative to the classical models. *Supported if* the row-normalised confusion rates between them are lower for the CNN, with the McNemar result as supporting evidence.
(Each hypothesis may turn out false; write that if it does.)

## 4. Snippet to read the saved files (run in a notebook cell after the gate)
```python
import numpy as np, pandas as pd
from src.config import CLASS_NAMES, RESULTS_DIR
from src.evaluation import top_confusions
cms = np.load(RESULTS_DIR / "confusion_matrices.npz")
print(list(cms.keys()))                       # e.g. 'logreg__pixels', ..., 'cnn'
display(top_confusions(cms["cnn"], k=10))
recall = pd.read_csv(RESULTS_DIR / "per_class_recall.csv", index_col=0)
display(recall.round(3))
```

## 5. Completed report section summary
1. One paragraph: overall ranking by macro-F1 with CIs (Table `tab:results`); say which differences are inside overlapping CIs.
2. One paragraph: per-class recall (Fig. `fig:recall`): weakest and strongest classes.
3. One paragraph: top confusions of the CNN (Fig. `fig:confusion`) and of the best classical model.
4. One paragraph: McNemar results (cite adjusted p and discordant counts); remind the reader of the limits (one seed, one test set).
5. One paragraph: hypotheses H1–H3, each marked supported / not supported with a number.

## 6. Discussion and conclusion checks (results now recorded in the report)
- Answer the research question in one sentence using the measured table only; if the CNN is not clearly better (overlapping CIs, non-significant McNemar), say so.
- Limitations to state: one dataset (28×28 grayscale), one seed, one official split, small hyperparameter grids (≤ 4 values; some selected values may sit at a grid edge, see `docs/classical_notes.md`), CNN hyperparameters not tuned, SVM tuned on a 10,000-image subset, LogReg convergence warnings if they occur (`max_iter = 500`), results may not transfer to real inventory photos.
- Future work: more seeds, larger grids, data augmentation, deeper CNNs, real inventory images.
