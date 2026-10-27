# Repository audit and report-completion notes

Audit date: 2026-10-28  
Audited snapshot: final recovered submission archive, including `results.zip` and `figures.zip` downloaded from Colab.

## Summary

The report source had several unfinished sections and its HOG description still referred to the old 1,296-feature configuration. The completed notebook output records the final HOG configuration as 6,084 features and contains the full nine-model evaluation. The report and README have now been updated using the values visible in that saved notebook output.

## Changes made in this reviewed copy

- Completed the abstract, results discussion, error analysis, limitations, conclusion, and acknowledgment in `report/main.tex`.
- Completed the results and inference tables in `report/results_table.tex` and `report/inference_table.tex`.
- Updated `report/features.tex` to the selected 6,084-dimensional HOG representation and the validation-only sensitivity-study result.
- Completed the final-fit and runtime-gate disclosures in `report/classical_models.tex`.
- Added the recorded CNN best epoch, validation performance, training time, device, and final test metrics to `report/cnn_architecture.tex`.
- Extracted the notebook's saved output figures for the report: `figures/cnn_learning_curves.png`, `figures/cm_cnn.png`, `figures/per_class_recall.png`, and `figures/model_comparison.png`.
- Completed the README results table and disclosed deviations/limitations.
- Filled result-dependent answers in `docs/viva_qa.md` and updated the results slide outline.
- Marked old pre-run task/runtime notes as historical where their claims conflict with the final run.
- Added result summaries in `results/results_table.csv`, `results/results_table.md`, `results/metrics_with_ci.csv`, and `results/mcnemar_cnn_vs_classical.csv`, reconstructed from the notebook's rendered outputs.

## Results used

| Model | Representation | Test accuracy | Test macro-F1 | Training images used |
|---|---|---:|---:|---:|
| Logistic regression | pixels | 0.8411 | 0.8403 | 54,000 |
| k-NN | pixels | 0.8500 | 0.8493 | 54,000 |
| Random forest | pixels | 0.8756 | 0.8741 | 54,000 |
| RBF-SVM | pixels | 0.8953 | 0.8951 | 54,000 |
| Logistic regression | HOG | 0.8698 | 0.8702 | 54,000 |
| k-NN | HOG | 0.8653 | 0.8621 | 54,000 |
| Random forest | HOG | 0.8636 | 0.8618 | 54,000 |
| RBF-SVM | HOG | 0.8817 | 0.8816 | 11,770 |
| Small CNN | learned | 0.9274 | 0.9269 | 54,000 |

The confidence intervals in the report are 95% percentile bootstrap intervals from 1,000 resamples, as shown in the notebook. The CNN's highest test macro-F1 is 0.9269. The strongest classical configuration is pixel RBF-SVM at 0.8951 macro-F1.

## Important findings and verification status in the final recovered copy

1. **Original Colab artifacts have now been restored.** The final archive includes `results/test_gate_done.flag`, `results/test_gate_outputs.npz`, `results/fit_info.json`, `results/frozen_selection.json`, `results/selected_configs.json`, `results/benchmarks.json`, `results/confusion_matrices.npz`, `results/per_class_recall.csv`, `results/environment.json`, the tuning CSVs, and `results/cnn_best.pt`. The test-gate flag records `n_test: 10000`; the saved prediction archive contains nine prediction arrays, each with 10,000 entries. These files were copied from the user's downloaded Colab ZIPs; the official test gate was not rerun.

2. **The HOG SVM used a reduced training subset.** It used 11,770 of 54,000 training images because the estimated full fit exceeded the runtime gate. This is disclosed in the report and README.

3. **Logistic regression convergence is not established for the pixel configuration.** The final run hit the 500-iteration cap with a convergence warning. The report records this limitation rather than claiming convergence.

4. **Inference timings are preliminary.** The final notebook benchmark output contains one timed pass after warm-up, not the three/five-repeat median described in earlier protocol prose. The report states the actual timing scope. CNN timing uses pre-normalised tensors and excludes preprocessing, so it is not directly comparable to classical end-to-end timing.

5. **Bibliography build could not be fully verified in this environment.** `pdflatex` generated a PDF, but the bibliography stage could not finish because the `bibtex` executable is unavailable in the audit environment. Citation and cross-reference warnings therefore remain unverified. Citation keys referenced by the report are present in `references.bib`; this does not independently verify every bibliographic detail.

6. **The test suite did not finish within the audit timeout.** `python -m pytest tests -q` produced many passing-test dots and three skips, but did not exit within the allotted time. This audit does not claim the full suite passes; identify the slow/hanging test with a verbose run in the project's intended environment.

7. **Author metadata is incomplete in the source snapshot.** The report retains the supplied first names and uses “Student project team”; full author names, institution, and contact information were not invented. Replace the author block with the team's official details before submission.

## Before submission

- Preserve the restored one-shot test-gate evidence and predictions; do not rerun the official test gate.
- Compile with a complete TeX distribution including BibTeX and review the resulting PDF for layout, citation, and reference warnings.
- Run the test suite to completion and investigate the test that stalls.
- Confirm author names/affiliation and verify the bibliography entries against the original publications.
- Confirm the course rubric's remaining deliverables (e.g. slides) are actually produced; this audit completed the outline, not a slide deck.
