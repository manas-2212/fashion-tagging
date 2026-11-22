"""Protocol tests for Phase 2 (owner: Manas): where test data may appear, and an optional end-to-end smoke run."""
import inspect
import os
import subprocess
import sys

import pytest

from src import (ablation, ablation_report, augment, dl_evaluation, dl_models, dl_train, hybrid, hybrid_stage,
                 phase2_analysis, phase2_context, phase2_pipeline, phase2_reporting, robustness)

TEST_LOADERS = ("load_test_dataset", "load_dataset(", "verify_test_dataset", "train=False")


@pytest.mark.parametrize("module", [ablation, ablation_report, augment, dl_evaluation, dl_models, dl_train, hybrid,
                                    hybrid_stage, phase2_reporting, robustness], ids=lambda m: m.__name__)
def test_modules_never_load_test_data(module):
    source = inspect.getsource(module)
    assert not any(token in source for token in TEST_LOADERS)


def test_only_gate_and_post_gate_analysis_touch_test_data():
    for fn in (phase2_context.load_phase2_data, phase2_context.make_context, phase2_pipeline.stage_train,
               ablation_report.stage_summary, hybrid_stage.stage_hybrid, hybrid_stage.make_predict_all,
               phase2_pipeline.stage_freeze):
        src = inspect.getsource(fn)
        assert "load_test" not in src, fn.__name__
    context_src = inspect.getsource(phase2_context)
    assert context_src.count("load_test_dataset(") == 1                 # one reader of the test set: load_test()
    assert "load_test_dataset(" in inspect.getsource(phase2_context.load_test)
    assert "load_test" in inspect.getsource(phase2_pipeline.stage_test_gate)
    analysis = inspect.getsource(phase2_analysis.stage_analysis)
    assert analysis.index("load_gate_outputs") < analysis.index("load_test()")     # refuses before the gate ran


def test_stage_order_is_the_protocol_order():
    assert phase2_pipeline.STAGES == ("train", "summary", "hybrid", "freeze", "gate", "analysis")


def test_gate_names_do_not_use_the_tuple_separator():
    name = phase2_pipeline.gate_name("final_dl", 42)
    assert name == "final_dl@42" and "|" not in name     # '|' keys are turned into tuples by load_selection


@pytest.mark.skipif(os.environ.get("RUN_PHASE2_SMOKE") != "1",
                    reason="slow end-to-end run; enable with RUN_PHASE2_SMOKE=1 (needs the dataset)")
def test_smoke_pipeline_end_to_end(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    done = subprocess.run([sys.executable, "run_phase2.py", "--smoke", "--quiet", "--results-dir", str(tmp_path)],
                          cwd=root, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    for name in ("ablation_summary.csv", "ablation_deltas.csv", "hybrid_validation.csv", "test_metrics.csv",
                 "mcnemar_phase2.csv", "robustness.csv", "selected_configs_phase2.json",
                 "test_gate_phase2_done.flag"):
        assert (tmp_path / name).is_file(), name
