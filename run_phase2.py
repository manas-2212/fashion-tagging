"""Command-line entry point for Phase 2 (owner: Manas).

Examples
  python run_phase2.py --smoke                      # 2-3 minute dry run of EVERY stage on tiny subsets
  python run_phase2.py                              # the full study (GPU strongly recommended)
  python run_phase2.py --stages train summary       # only some stages; finished runs are skipped
  python run_phase2.py --seeds 42 43                # fewer seeds if the GPU budget is short

The smoke run writes to results/phase2_smoke/ and never touches the real results or the real test-gate flag.
"""
import argparse

import matplotlib

matplotlib.use("Agg")

from src.phase2_context import make_context  # noqa: E402
from src.phase2_pipeline import STAGES, run_stages  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Phase 2: deep learning, ablations and hybrid models")
    p.add_argument("--smoke", action="store_true", help="tiny subsets, 2 epochs, separate output folder")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None, help="override the epoch budget of every configuration")
    p.add_argument("--configs", nargs="+", default=None, help="train only these configuration names")
    p.add_argument("--results-dir", default=None)
    p.add_argument("--quiet", action="store_true", help="do not print one line per epoch")
    args = p.parse_args()
    ctx = make_context(smoke=args.smoke, seeds=args.seeds, max_epochs=args.max_epochs, config_names=args.configs,
                       results_dir=args.results_dir)
    run_stages(ctx, stages=args.stages, verbose=not args.quiet)


if __name__ == "__main__":
    main()
