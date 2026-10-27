# Original pre-run task checklist (historical; not the current status)

This checklist was written before the final run and is retained as historical context. The final test gate has since been run once (see the saved notebook output). Do not execute the old run instructions or rerun the test gate just to regenerate omitted artifacts. Restore the original generated files from the completed Colab runtime. Several owner assignments below may now be stale.

## Critical path (do in this order)
1. **Merge everything** on `main` (all modules exist now; see `docs/integration_log.md`).
2. **Anant: HOG study** on real data, validation only → `results/hog_sensitivity.csv`, decision recorded in `docs/feature_notes.md` section 0. If the rule says "switch", Manas coordinates ONE commit touching `src/config.py`, `src/features.py`, `tests/test_features.py`, docs. **[BLOCKS]**
3. **Tanish + Manas + Puneet decide before the freeze:** `max_iter` for logistic regression (ConvergenceWarning at 500) and the grid edges (6 of 8 selections at a grid edge). Either extend the grids / raise `max_iter` once and document it in the integration log, or keep them and disclose in the limitations. After the freeze nothing may change. **[BLOCKS]**
4. **Manas: run `notebook.ipynb` in fresh Colab (GPU runtime) through section 7 only.** Stop before section 8's gate cell until sections 0-7 pass. Record the real time of one epoch.
5. **Puneet: fresh-runtime verification** (second person, independent run through section 7).
6. Historical task: the freeze and test gate were run once. Do not rerun. If later cells crash, restore `results/test_gate_outputs.npz` from the completed Colab run and debug only the post-gate cells.
7. Historical task: fill result placeholders from `results/*`. The report, README, slide outline, and viva answers have since been updated from the final notebook outputs; some generated CSV/NPZ files still need restoring from Colab.
8. **Final review** (everyone): `python -m pytest tests -q`, compile the report, read the README top to bottom.

---
## Manas
- Run the notebook in Colab (steps 4-6 above); confirm section 1 prints `classes_aligned_with_torchvision: True`.
- Save the notebook **with outputs** before committing (`File > Save a copy in GitHub` or download the .ipynb after the run).
- Stretch (only if everything else is verified): CNN over seeds 42/43/44, mean ± std, reported separately from the headline single-seed result (`docs/cnn_notes.md` is the theory; document the protocol in the report).
- Keep `docs/integration_log.md` current; arbitrate any interface change.
- Verifiers for your files: parameter count → Tanish; early stopping → Anant; shapes and benchmark protocol → Rohan; fresh-runtime run → Puneet; no held-out use before section 8 → Akhil.

## Puneet
- Verify every entry added at the end of `report/references.bib` (written from memory of well-known papers, **not checked**): authors, venue, pages, DOI. Delete any you cannot verify and remove its `\cite`.
- Fill authors' full names, affiliations, emails in `report/main.tex`; write abstract, discussion, conclusion, acknowledgment **after** results exist (guide: `docs/error_analysis_notes.md` section 6).
- Historical task completed in the current report: CNN best epoch/training time, selected HOG configuration, and classical fit/sample counts have been filled from the completed notebook output.
- Fill `README.md` "Disclosed deviations", "Results summary", "Limitations" from the results; remove statements that stop being true.
- Pin versions in `requirements.txt` if you want exact reproducibility (currently unpinned; record the versions printed by the notebook, `results/environment.json`).
- Be the independent runner of the notebook in a fresh Colab runtime; compile the report (`cd report && latexmk -pdf main.tex`); after the gate, commit `results/selected_configs.json` and, deliberately, `results/test_gate_done.flag` (`git add -f`, it is gitignored so that fresh clones can run) as evidence that the gate ran once, then never re-run the gate.
- Verifier for your files: `experiments.py` → Manas.

## Akhil
- Assemble the slides from `docs/slides_outline.md` (14 slides; slide 5 = Tanish's text, slide 6 = Manas's text; all numbers from `results/`).
- In `report/dataset_eda.tex` use the agreed sentence about the 10,000 test images: *"The 10,000 official test images are read only for the integrity checks in the loading step (shapes, data types, pixel range and class counts), from which nothing is selected or tuned, and for the report-only duplicate check, which runs after the final evaluation. Every other statistic in this section uses the official training images: the class distribution uses all 60,000, and the pixel statistics, PCA and class-mean analyses use the 54,000-image training split."*
- Check the class-distribution caption says "official 60,000 training images" (not 54k).
- Re-run the held-out search on the final notebook: `python -c "import nbformat;nb=nbformat.read('notebook.ipynb',4);print([i for i,c in enumerate(nb.cells) if c.cell_type=='code' and 'test' in c.source.lower()])"`. Indices below the first held-out cell (currently 29) must not appear.
- Compare the class-centroid distances (EDA) with the real confusion matrices after the gate, together with Rohan.

## Anant
- **Run the HOG sensitivity study on real data (validation only), then fill the table in `docs/feature_notes.md` section 5 and `results/hog_sensitivity.csv`:**
  ```python
  from src.data import load_dataset, prepare_splits
  from src.features import hog_sensitivity_study, DEFAULT_STUDY_CONFIGS
  X, y, _, _ = load_dataset()                       # the held-out arrays are discarded immediately
  s = prepare_splits(X, y); del X, y, _
  df = hog_sensitivity_study(s["X_train"], s["y_train"], s["X_val"], s["y_val"], DEFAULT_STUDY_CONFIGS)
  df.to_csv("results/hog_sensitivity.csv", index=False); print(df)
  ```
  Apply your pre-written decision rule (switch only if validation macro-F1 exceeds the baseline by more than 0.010) and record the decision in section 0 of `docs/feature_notes.md`. Tell Manas the same day.
- Measure HOG extraction time on real Fashion-MNIST (your notes only have mock-noise estimates).
- Verify the Dalal and Triggs (2005) reference together with Puneet.
- Verifier for Manas's early stopping: run `python -m pytest tests/test_cnn.py -k early -s` and report what you saw.

## Tanish
- Decide and document `max_iter` and the grid edges (critical path item 3).
- Confirm the "test" time estimates in `docs/classical_notes.md` are extrapolated from validation (10/6 ratio) and that no held-out image was used.
- Review `report/classical_models.tex` (assembled from your LaTeX block, with citations and a grid table added) and `docs/theory.md` (section 13 now contains the metrics theory).
- Verifier for Manas's parameter count: `python -m pytest tests/test_cnn.py::test_param_count_and_forward_shape -s` must print `trainable params: 241546`.

## Rohan
- **Review and take ownership of `src/evaluation.py` and `tests/test_evaluation.py`** (drafted by Manas to unblock integration; 10 tests pass). Check every function against your understanding of the spec; change what you disagree with and tell Manas (signatures are a contract).
- Edit `docs/metrics_theory.md`, `docs/viva_qa.md` (53 questions; answers in your own words), `docs/error_analysis_notes.md` and own the report sections "Results" and "Confusion Matrices and Error Analysis" after the gate.
- Verifier for Manas's shapes and benchmark protocol: run `python -m pytest tests/test_cnn.py tests/test_benchmarking.py -q`.
- After the gate, answer the hypotheses H1-H3 in `docs/error_analysis_notes.md` with numbers.
