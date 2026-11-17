# Protocol, bookkeeping and reporting notes (Puneet)

Files: `src/dl_config.py`, `src/phase2_experiments.py`, `src/phase2_reporting.py`,
test `tests/test_phase2_experiments.py`.

## 1. One file defines the experiment (`src/dl_config.py`)

Seeds, both training recipes, the base architecture, augmentation strengths, hybrid grids, cascade thresholds and
evaluation constants. Phase 1 constants in `src/config.py` are untouched. The same train/validation split is used:
`load_phase2_data` recomputes the split and compares its SHA-256 with `results/split_audit.json` from Phase 1.

## 2. RunStore: resumable, auditable training runs

48 training runs on a free GPU will be interrupted. Each (configuration, seed) pair writes
`runs/<name>__seed<k>.json` (configuration, metrics, per-epoch history, timing), a checkpoint and the validation
probabilities. A run counts as done only if the stored configuration **and** training-set size equal the request,
so a result produced with other settings can never be mistaken for the current one. The JSON record is written last.

## 3. Freeze and one-shot test gate

Order enforced by code: train -> summary (selection) -> hybrid (tuning) -> **freeze** -> **gate** -> analysis.

* `freeze_phase2` writes `results/phase2/selected_configs_phase2.json`: the selected configuration, the selection
  rule and its inputs, hybrid hyper-parameters, tau, both recipes, the split hash, the exact list of gate models,
  a UTC timestamp, the git commit and a SHA-256.
* `run_phase2_test_gate` loads the test set **inside** the gate. It refuses when the freeze file is missing, when
  `test_gate_phase2_done.flag` exists, or when the set of predicted models differs from the frozen list (nothing
  extra, nothing missing). A refused attempt does not burn the flag.
* 16 models pass the gate: for each of 3 seeds the re-trained Phase 1 CNN, the final deep model, H1, H2, H3; plus H0.

**Disclosed limitation: second use of the test set.** The same 10,000 images were evaluated at the Phase 1 gate
and the Phase 1 error analysis informed Phase 2 design. All Phase 2 choices are made on validation and the gate is
opened once more for a frozen list, but the test set is no longer "never seen by the project". The report states this.

## 4. Reporting (`src/phase2_reporting.py`)

Every table exists as CSV (analysis), Markdown (README) and LaTeX (`report/phase2_tables/*.tex`, booktabs,
IEEEtran-compatible). The report `\input`s the generated files, so no number is typed by hand.

## 5. Deep learning literature map (to be expanded into `report/` in the report stage)

| Topic | Reference to verify and cite | Used for |
|---|---|---|
| Convolutional networks | LeCun et al., 1998 | weight sharing, pooling |
| Dataset | Xiao et al., 2017 | Fashion-MNIST, published baselines |
| Deeper stacks of 3x3 filters | Simonyan and Zisserman, 2015 | two convolutions per block |
| Batch normalisation | Ioffe and Szegedy, 2015 | BN after every convolution |
| Residual learning | He et al., 2016 | shortcuts, projection shortcut |
| Global average pooling | Lin et al., 2014 | head without a large dense layer |
| Dropout | Srivastava et al., 2014 | regularisation of the classifier |
| Adam / AdamW | Kingma and Ba, 2015; Loshchilov and Hutter, 2019 | optimiser, decoupled weight decay |
| Cosine schedule | Loshchilov and Hutter, 2017 | learning-rate decay |
| Label smoothing | Szegedy et al., 2016; Mueller et al., 2019 | regularisation, calibration effect |
| Random erasing | Zhong et al., 2020 | augmentation |
| Calibration | Guo et al., 2017 | ECE, reliability diagrams |
| HOG | Dalal and Triggs, 2005 | handcrafted block of the hybrid |
| CNN features + SVM | Razavian et al., 2014; Tang, 2013 | precedent for the hybrid |
| McNemar for classifiers | Dietterich, 1998 | paired significance test |

Every entry must be checked against the original paper before it enters `references.bib` (the Phase 1 audit noted
that references had not all been verified).

## 6. Viva questions

**Why a freeze file if the code already fixes the rule?** It proves *when* the selection was made relative to the
gate (timestamp, commit, hash) and what exactly was allowed through.
**What stops someone from deleting the flag and re-running?** Nothing technical; the flag makes a second run a
deliberate, visible act, and the git history shows it.
**Why is the model list checked both ways?** A missing model would hide a result; an extra model would be an
unplanned test-set evaluation.
**Why store validation probabilities per run?** Calibration and the cascade threshold need them, and they let us
recompute any validation metric without re-training.
**Why are checkpoints not in git?** They are binary and reproducible; `.gitignore` excludes `*.pt`.

## Results for this component (full run, 16 November 2026)

* 48 training runs (16 configurations x 3 seeds), 105 minutes of training on a
  Tesla T4; no run diverged; records in `results/phase2/runs/`.
* Freeze file `results/phase2/selected_configs_phase2.json` written at 05:04:01 UTC on 16 November 2026; the test
  gate ran at 05:06:58 UTC on all 10,000 test images for 16 frozen models (freeze SHA-256 starts with
  `a6eb05ea62db`).
* The run started from an uploaded zip, not from a git clone, so the freeze file says `git_commit: unavailable`.
  State this if asked; the timestamp and the hash still fix the order "freeze before gate".
* Report numbers: `python report/build_phase2_report_assets.py` regenerates `report/phase2_tables/numbers.tex`
  (one LaTeX macro per quoted number) and the `rt_*.tex` tables from the CSV files. The report source is
  `report/phase2_main.tex` with one file per section in `report/phase2_sections/`; the compiled PDF is
  `report/Phase2Report.pdf`.
* References added for Phase 2 are in `report/references.bib` under the "Phase 2 additions" comment. Authors, titles,
  venues and years were checked; page numbers are present only where confirmed.
