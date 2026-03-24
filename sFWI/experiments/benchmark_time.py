#!/usr/bin/env python3
"""Inference-time benchmark script (sFWI vs baselines).

Goals:
1) Produce reproducible timing stats (mean / p50 / p90 / std).
2) Persist gss_topg-related sampling settings
   (sampling_method / n_candidates / group_top_g).

Notes:
- Reuses environment setup and samplers from `evaluation_exp.py`
  to keep algorithmic parity.
- Timing excludes one-time setup by default; setup costs are recorded
  separately as metadata.
"""

import argparse
import contextlib
import csv
import io
import json
import os
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm

try:
    from sFWI.experiments.evaluation_exp import (
        _build_gss_topg_light_cache,
        _run_sfwi_sample,
        _run_sfwi_sample_gss,
        load_baselines_from_config,
        setup_environment,
    )
except ImportError:
    from evaluation_exp import (  # type: ignore
        _build_gss_topg_light_cache,
        _run_sfwi_sample,
        _run_sfwi_sample_gss,
        load_baselines_from_config,
        setup_environment,
    )


def _sync_if_cuda(device: str):
    if device == "cuda":
        torch.cuda.synchronize()


def _time_call(fn, device: str) -> float:
    _sync_if_cuda(device)
    t0 = time.perf_counter()
    fn()
    _sync_if_cuda(device)
    return float(time.perf_counter() - t0)


def _parse_gt_indices(args, dataset_size: int):
    if args.gt_indices:
        out = []
        for idx in args.gt_indices:
            if idx < 0 or idx >= dataset_size:
                raise ValueError(f"gt index out of range: {idx}, dataset_size={dataset_size}")
            out.append(int(idx))
        return out

    n = max(1, min(int(args.n_gt), dataset_size))
    return list(range(n))


def _safe_percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _stats(values):
    if not values:
        return {
            "n": 0,
            "mean_s": float("nan"),
            "p50_s": float("nan"),
            "p90_s": float("nan"),
            "std_s": float("nan"),
            "min_s": float("nan"),
            "max_s": float("nan"),
            "total_s": float("nan"),
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean_s": float(arr.mean()),
        "p50_s": _safe_percentile(values, 50),
        "p90_s": _safe_percentile(values, 90),
        "std_s": float(arr.std(ddof=0)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "total_s": float(arr.sum()),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description="sFWI inference-time benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--master_seed", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--eval_patches_path", type=str, default=None)

    parser.add_argument(
        "--sampling_method",
        type=str,
        default="gss_topg_light",
        choices=["rss", "gss", "gss_topg", "gss_topg_light"],
        help="sFWI sampling method",
    )
    parser.add_argument("--n_candidates", type=int, default=50)
    parser.add_argument("--group_top_g", type=int, default=20)
    parser.add_argument("--gss_light_eval_bs", type=int, default=32)
    parser.add_argument("--sm_path", type=str, default=None)

    parser.add_argument("--baseline_config", type=str, default=None)

    parser.add_argument(
        "--gt_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit GT list. If omitted, use first --n_gt samples.",
    )
    parser.add_argument("--n_gt", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=1)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument(
        "--verbose_sampling",
        action="store_true",
        help="Enable verbose prints inside sampler calls.",
    )
    return parser


def _prepare_sm_and_cache(args, env):
    setup_info = {
        "sm_load_s": 0.0,
        "light_cache_s": 0.0,
        "light_cache_source": "",
        "light_cache_rep_n": 0,
    }

    if args.sampling_method not in ("gss", "gss_topg", "gss_topg_light"):
        return setup_info

    if not args.sm_path or not os.path.isfile(args.sm_path):
        raise FileNotFoundError(
            f"--sampling_method={args.sampling_method} requires a valid --sm_path, got: {args.sm_path}"
        )
    if args.group_top_g < 1:
        raise ValueError("--group_top_g must be >= 1")

    t0 = time.perf_counter()
    sm_data = torch.load(args.sm_path, weights_only=False)
    setup_info["sm_load_s"] = float(time.perf_counter() - t0)
    for key in ("similarity_matrix", "k", "centroid_indices", "x0hat_batch"):
        if key not in sm_data:
            raise KeyError(f"SM file missing required key: {key}")
    env["sm_info"] = sm_data

    if args.sampling_method != "gss_topg_light":
        return setup_info

    if args.gss_light_eval_bs < 1:
        raise ValueError("--gss_light_eval_bs must be >= 1")
    t0 = time.perf_counter()
    light_cache = _build_gss_topg_light_cache(
        sm_info=env["sm_info"],
        operator=env["operator"],
        device=env["cfg"].device,
        eval_bs=int(args.gss_light_eval_bs),
    )
    setup_info["light_cache_s"] = float(time.perf_counter() - t0)
    setup_info["light_cache_source"] = str(light_cache.get("source", "unknown"))
    setup_info["light_cache_rep_n"] = int(light_cache["rep_indices"].numel())
    env["gss_topg_light_cache"] = light_cache
    env["gss_topg_light_eval_bs"] = int(args.gss_light_eval_bs)
    return setup_info


def _run_sfwi_once(env, args, gt_idx):
    def _call():
        if args.sampling_method in ("gss", "gss_topg", "gss_topg_light"):
            if "sm_info" not in env:
                raise RuntimeError("GSS mode requires sm_info. Please check --sm_path.")
            _run_sfwi_sample_gss(
                env=env,
                gt_seed=gt_idx,
                master_seed=args.master_seed,
                n_candidates=args.n_candidates,
                sm_info=env["sm_info"],
                sampling_method=args.sampling_method,
                group_top_g=args.group_top_g,
            )
            return

        _run_sfwi_sample(
            env=env,
            gt_seed=gt_idx,
            master_seed=args.master_seed,
            n_candidates=args.n_candidates,
        )

    if args.verbose_sampling:
        _call()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            _call()


def _write_case_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "gt_idx",
                "method",
                "elapsed_s",
                "repeat_id",
                "is_warmup",
            ]
        )
        writer.writerows(rows)


def _write_summary_csv(path, summary_rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "n",
                "mean_s",
                "p50_s",
                "p90_s",
                "std_s",
                "min_s",
                "max_s",
                "total_s",
                "sampling_method",
                "n_candidates",
                "group_top_g",
                "gss_light_eval_bs",
                "sm_path",
                "cache_source",
            ]
        )
        writer.writerows(summary_rows)


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 60)
    print("sFWI Inference-Time Benchmark")
    print("=" * 60)
    print(f"  sampling_method: {args.sampling_method}")
    print(f"  n_candidates:    {args.n_candidates}")
    print(f"  group_top_g:     {args.group_top_g}")
    print(f"  gss_light_eval_bs: {args.gss_light_eval_bs}")
    if args.sm_path:
        print(f"  sm_path:         {args.sm_path}")
    if args.eval_patches_path:
        print(f"  eval_patches_path: {args.eval_patches_path}")

    t_env0 = time.perf_counter()
    env = setup_environment(args)
    t_env = float(time.perf_counter() - t_env0)
    device = env["cfg"].device
    data = env["data"]
    operator = env["operator"]

    gt_indices = _parse_gt_indices(args, len(data))
    if args.warmup >= len(gt_indices):
        raise ValueError("--warmup must be smaller than number of GT samples")
    if args.repeats < 1:
        raise ValueError("--repeats must be >= 1")

    print(f"[setup] GT count: {len(gt_indices)}, warmup={args.warmup}, repeats={args.repeats}")
    print(f"[setup] device: {device}")

    setup_info = _prepare_sm_and_cache(args, env)
    if args.sampling_method == "gss_topg_light":
        print(
            "[setup] gss_topg_light cache: "
            f"rep_n={setup_info['light_cache_rep_n']}, "
            f"source={setup_info['light_cache_source']}, "
            f"time={setup_info['light_cache_s']:.2f}s"
        )

    baselines = OrderedDict()
    if args.baseline_config and os.path.isfile(args.baseline_config):
        baselines = load_baselines_from_config(args.baseline_config, device)
    print(f"[setup] baselines: {list(baselines.keys())}")

    timing = OrderedDict()
    timing["sFWI"] = []
    for name in baselines:
        timing[name] = []

    case_rows = []
    method_wall = {name: 0.0 for name in timing}

    for pos, gt_idx in enumerate(tqdm(gt_indices, desc="benchmark")):
        gt = data.get_data(1, 0, seed=gt_idx).to(device)
        measurement = operator(gt)
        is_warmup = int(pos < args.warmup)

        for rep in range(args.repeats):
            elapsed = _time_call(
                lambda: _run_sfwi_once(env, args, gt_idx),
                device=device,
            )
            method_wall["sFWI"] += elapsed
            case_rows.append([gt_idx, "sFWI", f"{elapsed:.6f}", rep, is_warmup])
            if not is_warmup:
                timing["sFWI"].append(elapsed)

        for bl_name, bl_model in baselines.items():
            for rep in range(args.repeats):
                elapsed = _time_call(lambda: bl_model.predict(measurement), device=device)
                method_wall[bl_name] += elapsed
                case_rows.append([gt_idx, bl_name, f"{elapsed:.6f}", rep, is_warmup])
                if not is_warmup:
                    timing[bl_name].append(elapsed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = args.output_dir or os.path.join(env["output_dir"], "benchmark_time")
    os.makedirs(out_dir, exist_ok=True)

    case_csv = os.path.join(out_dir, f"benchmark_time_case{tag}_{stamp}.csv")
    summary_csv = os.path.join(out_dir, f"benchmark_time_summary{tag}_{stamp}.csv")
    meta_json = os.path.join(out_dir, f"benchmark_time_meta{tag}_{stamp}.json")

    _write_case_csv(case_csv, case_rows)

    summary_rows = []
    for method, arr in timing.items():
        st = _stats(arr)
        summary_rows.append(
            [
                method,
                st["n"],
                f"{st['mean_s']:.6f}",
                f"{st['p50_s']:.6f}",
                f"{st['p90_s']:.6f}",
                f"{st['std_s']:.6f}",
                f"{st['min_s']:.6f}",
                f"{st['max_s']:.6f}",
                f"{st['total_s']:.6f}",
                args.sampling_method if method == "sFWI" else "",
                args.n_candidates if method == "sFWI" else "",
                args.group_top_g if method == "sFWI" else "",
                args.gss_light_eval_bs if method == "sFWI" else "",
                args.sm_path if method == "sFWI" else "",
                setup_info["light_cache_source"] if method == "sFWI" else "",
            ]
        )
    _write_summary_csv(summary_csv, summary_rows)

    meta = {
        "timestamp": stamp,
        "device": device,
        "dataset_source": env.get("dataset_source", ""),
        "gt_indices": gt_indices,
        "warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "sampling": {
            "sampling_method": args.sampling_method,
            "n_candidates": int(args.n_candidates),
            "group_top_g": int(args.group_top_g),
            "gss_light_eval_bs": int(args.gss_light_eval_bs),
            "sm_path": args.sm_path,
        },
        "setup_time_s": {
            "environment": t_env,
            "sm_load": setup_info["sm_load_s"],
            "light_cache": setup_info["light_cache_s"],
        },
        "light_cache": {
            "source": setup_info["light_cache_source"],
            "rep_n": setup_info["light_cache_rep_n"],
        },
        "method_wall_time_s": method_wall,
        "timing_count_no_warmup": {k: len(v) for k, v in timing.items()},
        "baseline_config": args.baseline_config,
    }
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\nMethod                 n      mean(s)      p50(s)      p90(s)      std(s)")
    print("-" * 76)
    for method, arr in timing.items():
        st = _stats(arr)
        print(
            f"{method:<20}"
            f"{st['n']:>4d}"
            f"{st['mean_s']:>13.4f}"
            f"{st['p50_s']:>12.4f}"
            f"{st['p90_s']:>12.4f}"
            f"{st['std_s']:>12.4f}"
        )

    print("\n[sampling] sFWI sampling settings:")
    print(f"  sampling_method={args.sampling_method}")
    print(f"  n_candidates={args.n_candidates}")
    print(f"  group_top_g={args.group_top_g}")
    print(f"  gss_light_eval_bs={args.gss_light_eval_bs}")
    print(f"  sm_path={args.sm_path}")
    if args.sampling_method == "gss_topg_light":
        print(
            f"  light_cache_source={setup_info['light_cache_source']}, "
            f"rep_n={setup_info['light_cache_rep_n']}"
        )

    print("\n[save] case_csv:", case_csv)
    print("[save] summary_csv:", summary_csv)
    print("[save] meta_json:", meta_json)


if __name__ == "__main__":
    main()
