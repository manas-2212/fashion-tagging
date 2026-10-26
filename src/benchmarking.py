"""Parameter counting and inference benchmarking (Manas).

Protocol (spec): same 10k test images, no shuffling; warm-up; median of repeats; exclude model loading/training/plotting.
Both benchmark functions return a dict with the keys:
  device, batch_size, n_images, total_s_median (seconds), per_image_ms (milliseconds), repeats_s (list, seconds),
  includes_preprocessing (bool)
plus extras: torch_threads, os_cpu_count, (cnn) device_name, (classical) n_jobs, label.
This module only TIMES forward passes; it never reads labels.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch

from src.config import BENCH_BATCH_SIZE


def count_trainable_params(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarise(device, batch_size, n_images, repeats_s, includes_preprocessing, **extra) -> dict:
    med = float(np.median(repeats_s))
    out = dict(device=str(device), batch_size=int(batch_size), n_images=int(n_images), total_s_median=med,
               per_image_ms=med / n_images * 1000.0, repeats_s=[float(r) for r in repeats_s],
               includes_preprocessing=bool(includes_preprocessing),
               torch_threads=torch.get_num_threads(), os_cpu_count=os.cpu_count())
    out.update(extra)
    return out


def benchmark_cnn_inference(model, X_t, device, batch_size=BENCH_BATCH_SIZE, warmup_batches=3, repeats=5) -> dict:
    """Time forward passes over X_t (already normalised, no shuffling) in eval mode under inference_mode.

    X_t is moved to `device` BEFORE timing (pre-normalised tensors on device). CUDA is synchronised before and after
    each timed repeat. The model is moved back to its original device afterwards.
    """
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    orig_device = next(model.parameters()).device
    model.to(device).eval()
    X_dev = X_t.to(device)
    n = len(X_dev)
    batches = [X_dev[i:i + batch_size] for i in range(0, n, batch_size)]  # views; fixed order
    with torch.inference_mode():
        for b in (batches * ((warmup_batches // max(len(batches), 1)) + 1))[:warmup_batches]:  # untimed warm-up
            model(b)
        _sync(device)
        times = []
        for _ in range(repeats):
            _sync(device)
            t0 = time.perf_counter()
            for b in batches:
                model(b)
            _sync(device)
            times.append(time.perf_counter() - t0)
    model.to(orig_device)
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
    return _summarise(device, batch_size, n, times, False, device_name=name, label="cnn forward only")


def benchmark_classical_inference(pipeline, X_input, repeats=3, feature_fn=None) -> dict:
    """Time pipeline.predict on the whole of X_input (one call per repeat -> batch_size == n_images).

    feature_fn is None  -> X_input = PRECOMPUTED features: "predict only" (any scaler inside the pipeline IS included,
                           because it is part of pipeline.predict); includes_preprocessing=False.
    feature_fn given    -> X_input = RAW uint8 images: "end-to-end from raw images" = feature_fn(X) + predict;
                           includes_preprocessing=True.
    One untimed warm-up call on a small slice (<=256 rows) precedes timing. Unit: seconds / ms.
    """
    n = len(X_input)
    warm = X_input[:min(n, 256)]
    pipeline.predict(feature_fn(warm) if feature_fn is not None else warm)   # untimed warm-up
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        feats = feature_fn(X_input) if feature_fn is not None else X_input
        pipeline.predict(feats)
        times.append(time.perf_counter() - t0)
    clf = getattr(pipeline, "named_steps", {}).get("clf")
    n_jobs = clf.get_params().get("n_jobs") if clf is not None and hasattr(clf, "get_params") else None
    label = "end-to-end from raw images" if feature_fn is not None else "predict only (precomputed features)"
    return _summarise("cpu", n, n, times, feature_fn is not None, n_jobs=n_jobs, label=label)
