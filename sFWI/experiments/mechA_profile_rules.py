"""
Rule-based DAPS annealing profile labeling (Experiment A).

输入:
  - daps_mech_outer_summary*.csv (可多个)

输出:
  - profile_detail*.csv: 每个 run 的 profile 指标 + 标签
  - profile_summary*.csv: 标签分布与关键统计

典型用法:
  python sFWI/experiments/mechA_profile_rules.py \
    --outer_csv "code/csv_temp/mechA_showcase_grid/*/daps_mech_outer_summary_model-*.csv" \
    --out_dir code/csv_temp/mechA_showcase_grid \
    --tag grid
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable


DEFAULT_RUN_ID_COLS = [
    "gt_idx",
    "repeat_id",
    "seed",
    "candidate_seed",
    "candidate_group",
]


@dataclass
class ProfileRuleConfig:
    misfit_col: str = "misfit_x0hat"
    min_steps: int = 4
    monotonic_eps: float = 0.0
    drop_eps_abs: float = 0.01
    flat_eps_abs: float = 0.0
    rebound_abs_min: float = 0.01
    rebound_ratio_min: float = 0.35
    late_drop_min_pos_ratio: float = 0.70
    steady_mono_min: float = 0.70


def _ts() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _to_float(v) -> float:
    if v is None:
        return float("nan")
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if s == "":
        return float("nan")
    try:
        return float(s)
    except Exception:
        return float("nan")


def _is_finite(x: float) -> bool:
    return isinstance(x, float) and math.isfinite(x)


def _resolve_csv_paths(specs: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for spec in specs:
        m = sorted(glob.glob(spec))
        if m:
            paths.extend(m)
            continue
        if os.path.isfile(spec):
            paths.append(spec)
    uniq = sorted(dict.fromkeys(paths))
    if not uniq:
        raise FileNotFoundError("未解析到任何 CSV，请检查 --outer_csv 参数。")
    return uniq


def _load_outer_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = dict(row)
                item["_source_csv"] = p
                item["_config_id"] = os.path.basename(os.path.dirname(p))
                rows.append(item)
    if not rows:
        raise RuntimeError("CSV 已读取但无数据行。")
    return rows


def _make_run_key(row: dict, run_id_cols: list[str]) -> tuple:
    return tuple([row.get("_config_id", "")] + [row.get(c, "") for c in run_id_cols])


def _compute_profile_features(series: list[float], monotonic_eps: float) -> dict:
    n = len(series)
    if n <= 0:
        return {
            "n_steps": 0,
            "first": float("nan"),
            "last": float("nan"),
            "min": float("nan"),
            "min_step": -1,
            "min_step_pos_ratio": float("nan"),
            "net_drop": float("nan"),
            "best_drop": float("nan"),
            "rebound_amp": float("nan"),
            "rebound_ratio": float("nan"),
            "monotonic_ratio": float("nan"),
            "drop_step_fraction": float("nan"),
            "rise_step_fraction": float("nan"),
        }

    first = float(series[0])
    last = float(series[-1])
    min_val = float(min(series))
    min_idx = int(series.index(min_val))

    net_drop = first - last
    best_drop = first - min_val
    rebound_amp = max(0.0, last - min_val)
    rebound_ratio = rebound_amp / max(1e-12, abs(best_drop))
    min_pos_ratio = min_idx / float(max(1, n - 1))

    if n == 1:
        monotonic_ratio = 1.0
        drop_frac = 0.0
        rise_frac = 0.0
    else:
        non_inc = 0
        drop_cnt = 0
        rise_cnt = 0
        for i in range(n - 1):
            d = series[i + 1] - series[i]
            if d <= monotonic_eps:
                non_inc += 1
            if d < -monotonic_eps:
                drop_cnt += 1
            if d > monotonic_eps:
                rise_cnt += 1
        den = float(n - 1)
        monotonic_ratio = non_inc / den
        drop_frac = drop_cnt / den
        rise_frac = rise_cnt / den

    return {
        "n_steps": n,
        "first": first,
        "last": last,
        "min": min_val,
        "min_step": min_idx,
        "min_step_pos_ratio": min_pos_ratio,
        "net_drop": net_drop,
        "best_drop": best_drop,
        "rebound_amp": rebound_amp,
        "rebound_ratio": rebound_ratio,
        "monotonic_ratio": monotonic_ratio,
        "drop_step_fraction": drop_frac,
        "rise_step_fraction": rise_frac,
    }


def classify_profile(features: dict, cfg: ProfileRuleConfig) -> str:
    n = int(features["n_steps"])
    if n < cfg.min_steps:
        return "insufficient_steps"

    net_drop = float(features["net_drop"])
    best_drop = float(features["best_drop"])
    rebound = float(features["rebound_amp"])
    rebound_ratio = float(features["rebound_ratio"])
    min_pos = float(features["min_step_pos_ratio"])
    mono = float(features["monotonic_ratio"])

    if net_drop <= cfg.flat_eps_abs:
        if best_drop >= cfg.drop_eps_abs and rebound >= cfg.rebound_abs_min and rebound_ratio >= cfg.rebound_ratio_min:
            return "rebound"
        return "flat_or_worse"

    if min_pos >= cfg.late_drop_min_pos_ratio and net_drop >= cfg.drop_eps_abs:
        return "late_drop"

    if rebound >= cfg.rebound_abs_min and rebound_ratio >= cfg.rebound_ratio_min:
        return "rebound"

    if mono >= cfg.steady_mono_min:
        return "steady_drop"

    return "noisy_drop"


def build_profile_rows(
    outer_rows: list[dict],
    run_id_cols: list[str],
    cfg: ProfileRuleConfig,
) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in outer_rows:
        grouped[_make_run_key(r, run_id_cols)].append(r)

    out_rows: list[dict] = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(_to_float(x.get("outer_step", 0))))
        series = [_to_float(r.get(cfg.misfit_col)) for r in rows]
        series = [x for x in series if _is_finite(x)]
        feats = _compute_profile_features(series, cfg.monotonic_eps)
        label = classify_profile(feats, cfg)

        base = dict(rows[0])
        out = {
            "_config_id": base.get("_config_id", ""),
            "_source_csv": base.get("_source_csv", ""),
            "profile_label": label,
            "misfit_col": cfg.misfit_col,
            "n_steps": feats["n_steps"],
            "first": feats["first"],
            "last": feats["last"],
            "min": feats["min"],
            "min_step": feats["min_step"],
            "min_step_pos_ratio": feats["min_step_pos_ratio"],
            "net_drop": feats["net_drop"],
            "best_drop": feats["best_drop"],
            "rebound_amp": feats["rebound_amp"],
            "rebound_ratio": feats["rebound_ratio"],
            "monotonic_ratio": feats["monotonic_ratio"],
            "drop_step_fraction": feats["drop_step_fraction"],
            "rise_step_fraction": feats["rise_step_fraction"],
        }
        for c in run_id_cols:
            out[c] = base.get(c, "")
        out_rows.append(out)
    return out_rows


def _save_csv(path: str, rows: list[dict], headers: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _build_summary(profile_rows: list[dict]) -> list[dict]:
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in profile_rows:
        by_label[str(r["profile_label"])].append(r)

    total = max(1, len(profile_rows))
    rows: list[dict] = []
    for label in sorted(by_label.keys()):
        items = by_label[label]
        net_drop = [_to_float(x["net_drop"]) for x in items if _is_finite(_to_float(x["net_drop"]))]
        mono = [_to_float(x["monotonic_ratio"]) for x in items if _is_finite(_to_float(x["monotonic_ratio"]))]
        rebound = [_to_float(x["rebound_amp"]) for x in items if _is_finite(_to_float(x["rebound_amp"]))]
        rows.append(
            {
                "profile_label": label,
                "n_runs": len(items),
                "ratio": len(items) / float(total),
                "mean_net_drop": mean(net_drop) if net_drop else float("nan"),
                "median_net_drop": median(net_drop) if net_drop else float("nan"),
                "mean_monotonic_ratio": mean(mono) if mono else float("nan"),
                "mean_rebound_amp": mean(rebound) if rebound else float("nan"),
            }
        )
    return rows


def _parse_args():
    p = argparse.ArgumentParser(description="DAPS 机理 profile 自动标注（基于 outer_summary）")
    p.add_argument(
        "--outer_csv",
        nargs="+",
        required=True,
        help="CSV 路径或 glob，可多个。",
    )
    p.add_argument("--out_dir", required=True, help="输出目录")
    p.add_argument("--tag", type=str, default="", help="输出文件名附加标记")
    p.add_argument("--run_id_cols", nargs="+", default=DEFAULT_RUN_ID_COLS, help="用于识别一次 run 的列")
    p.add_argument("--misfit_col", type=str, default="misfit_x0hat", help="用于 profile 判别的 misfit 列名")
    p.add_argument("--min_steps", type=int, default=4)
    p.add_argument("--monotonic_eps", type=float, default=0.0)
    p.add_argument("--drop_eps_abs", type=float, default=0.01)
    p.add_argument("--flat_eps_abs", type=float, default=0.0)
    p.add_argument("--rebound_abs_min", type=float, default=0.01)
    p.add_argument("--rebound_ratio_min", type=float, default=0.35)
    p.add_argument("--late_drop_min_pos_ratio", type=float, default=0.70)
    p.add_argument("--steady_mono_min", type=float, default=0.70)
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = ProfileRuleConfig(
        misfit_col=args.misfit_col,
        min_steps=int(args.min_steps),
        monotonic_eps=float(args.monotonic_eps),
        drop_eps_abs=float(args.drop_eps_abs),
        flat_eps_abs=float(args.flat_eps_abs),
        rebound_abs_min=float(args.rebound_abs_min),
        rebound_ratio_min=float(args.rebound_ratio_min),
        late_drop_min_pos_ratio=float(args.late_drop_min_pos_ratio),
        steady_mono_min=float(args.steady_mono_min),
    )

    csv_paths = _resolve_csv_paths(args.outer_csv)
    outer_rows = _load_outer_rows(csv_paths)
    profile_rows = build_profile_rows(outer_rows, list(args.run_id_cols), cfg)
    summary_rows = _build_summary(profile_rows)

    suffix = f"_{args.tag}" if args.tag else ""
    detail_path = os.path.join(args.out_dir, f"mechA_profile_detail{suffix}_{_ts()}.csv")
    summary_path = os.path.join(args.out_dir, f"mechA_profile_summary{suffix}_{_ts()}.csv")

    detail_headers = [
        "_config_id",
        "_source_csv",
        *list(args.run_id_cols),
        "profile_label",
        "misfit_col",
        "n_steps",
        "first",
        "last",
        "min",
        "min_step",
        "min_step_pos_ratio",
        "net_drop",
        "best_drop",
        "rebound_amp",
        "rebound_ratio",
        "monotonic_ratio",
        "drop_step_fraction",
        "rise_step_fraction",
    ]
    summary_headers = [
        "profile_label",
        "n_runs",
        "ratio",
        "mean_net_drop",
        "median_net_drop",
        "mean_monotonic_ratio",
        "mean_rebound_amp",
    ]
    _save_csv(detail_path, profile_rows, detail_headers)
    _save_csv(summary_path, summary_rows, summary_headers)

    cnt = Counter([r["profile_label"] for r in profile_rows])
    print("=" * 72)
    print("Mechanism Profile Rules")
    print("=" * 72)
    print(f"outer_csv_count: {len(csv_paths)}")
    print(f"run_count: {len(profile_rows)}")
    print(f"misfit_col: {cfg.misfit_col}")
    print("label_counts:")
    for k in sorted(cnt.keys()):
        print(f"  {k}: {cnt[k]}")
    print(f"detail_csv: {detail_path}")
    print(f"summary_csv: {summary_path}")


if __name__ == "__main__":
    main()

