"""
Mechanism-A statistics summarizer for DAPS-FWI (table-ready).

输入（可多组）:
  - run_summary CSV(s):   daps_mech_run_summary_model-*.csv
  - outer_summary CSV(s): daps_mech_outer_summary_model-*.csv (可选，用于 phase 与 profile)
  - profile CSV(s):       mechA_profile_detail*.csv (可选；若不给则可由 outer 自动生成)

输出:
  - tableA_overall*.csv   (run-level 指标总表，支持 difficulty 分层)
  - tableB_phase*.csv     (early/mid/late 分阶段统计；需要 outer)
  - tableC_profiles*.csv  (profile 分布 + 分 profile 指标)
  - tidy_merged*.csv      (可选，run 行附 difficulty/profile 便于追踪)
  - 可选 .tex 文件

示例:
  python sFWI/experiments/mechA_stats_summary.py \
    --run_csv "code/csv_temp/mechA_showcase_grid/*/daps_mech_run_summary_model-*.csv" \
    --outer_csv "code/csv_temp/mechA_showcase_grid/*/daps_mech_outer_summary_model-*.csv" \
    --out_dir code/csv_temp/mechA_showcase_grid \
    --tag showcase_grid \
    --difficulty_mode tertile \
    --bootstrap_iters 2000 \
    --write_tex
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, median, pstdev
from typing import Iterable


DEFAULT_RUN_ID_COLS = [
    "gt_idx",
    "repeat_id",
    "seed",
    "candidate_seed",
    "candidate_group",
]

DEFAULT_RUN_METRICS = [
    "x0_gt_nrmse",
    "x0hat_gt_nrmse_improve",
    "x0y_gt_nrmse_improve",
    "x0hat_misfit_ratio",
    "x0y_misfit_ratio",
    "x0hat_structural_pass",
    "x0y_structural_pass",
    "mean_noise_to_drift_ratio",
    "mean_noise_to_grad_ratio",
    "mean_prior_over_data_loss",
    "mean_cos_data_to_gt",
    "mean_cos_prior_to_gt",
    "mean_cos_update_to_gt",
    "mean_delta_gt_rmse_drift",
    "mean_delta_gt_rmse_full",
    "frac_drift_toward_gt",
    "frac_full_toward_gt",
    "run_seconds",
    "x0hat_improve_flag",
    "x0y_improve_flag",
    "x0hat_misfit_improve_flag",
    "x0y_misfit_improve_flag",
]

DEFAULT_PHASE_METRICS = [
    "misfit_x0hat",
    "misfit_x0y",
    "mean_measurement_misfit",
    "mean_data_loss",
    "mean_prior_loss",
    "mean_prior_over_data_loss",
    "mean_noise_to_drift_ratio",
    "mean_noise_to_grad_ratio",
    "mean_cos_update_to_gt",
    "mean_delta_gt_rmse_full",
    "frac_full_toward_gt",
]

DEFAULT_PROFILE_METRICS = [
    "x0hat_gt_nrmse_improve",
    "x0hat_misfit_ratio",
    "x0hat_structural_pass",
    "mean_cos_update_to_gt",
    "mean_delta_gt_rmse_full",
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


def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    v = sorted(vals)
    if len(v) == 1:
        return float(v[0])
    q = min(1.0, max(0.0, q))
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(v[lo])
    w = pos - lo
    return float(v[lo] * (1.0 - w) + v[hi] * w)


def _bootstrap_ci_mean(vals: list[float], iters: int, ci: float, seed: int) -> tuple[float, float]:
    if (iters is None) or (iters <= 0) or (not vals):
        return float("nan"), float("nan")
    rng = random.Random(seed)
    n = len(vals)
    means: list[float] = []
    for _ in range(int(iters)):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / float(n))
    alpha = max(0.0, min(1.0, (100.0 - float(ci)) / 200.0))
    lo = _percentile(means, alpha)
    hi = _percentile(means, 1.0 - alpha)
    return lo, hi


def _summary_from_values(vals: list[float], b_iters: int, b_ci: float, b_seed: int) -> dict:
    if not vals:
        return {
            "n_effective": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    ci_low, ci_high = _bootstrap_ci_mean(vals, b_iters, b_ci, b_seed)
    return {
        "n_effective": len(vals),
        "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "median": median(vals),
        "q25": _percentile(vals, 0.25),
        "q75": _percentile(vals, 0.75),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def _resolve_csv_paths(specs: Iterable[str]) -> list[str]:
    if not specs:
        return []
    paths: list[str] = []
    for spec in specs:
        m = sorted(glob.glob(spec))
        if m:
            paths.extend(m)
            continue
        if os.path.isfile(spec):
            paths.append(spec)
    return sorted(dict.fromkeys(paths))


def _load_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for p in paths:
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                item = dict(row)
                item["_source_csv"] = p
                item["_config_id"] = os.path.basename(os.path.dirname(p))
                rows.append(item)
    return rows


def _parse_metrics(text: str, fallback: list[str]) -> list[str]:
    if text is None:
        return list(fallback)
    vals = [x.strip() for x in str(text).split(",") if x.strip()]
    return vals if vals else list(fallback)


def _save_csv(path: str, rows: list[dict], headers: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _escape_tex(s: str) -> str:
    return str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _fmt(v):
    if isinstance(v, float):
        if not math.isfinite(v):
            return ""
        return f"{v:.6g}"
    return str(v)


def _save_tex(path: str, rows: list[dict], headers: list[str], caption: str, label: str) -> None:
    align = "l" * len(headers)
    lines = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_escape_tex(caption)}}}")
    lines.append(f"\\label{{{_escape_tex(label)}}}")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\hline")
    lines.append(" & ".join([_escape_tex(h) for h in headers]) + " \\\\")
    lines.append("\\hline")
    for r in rows:
        vals = [_escape_tex(_fmt(r.get(h, ""))) for h in headers]
        lines.append(" & ".join(vals) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _make_run_key(row: dict, run_id_cols: list[str]) -> tuple:
    return tuple([row.get("_config_id", "")] + [row.get(c, "") for c in run_id_cols])


def _add_derived_run_metrics(run_rows: list[dict]) -> None:
    for r in run_rows:
        nrmse_imp = _to_float(r.get("x0hat_gt_nrmse_improve"))
        nrmse_imp_y = _to_float(r.get("x0y_gt_nrmse_improve"))
        mis_ratio = _to_float(r.get("x0hat_misfit_ratio"))
        mis_ratio_y = _to_float(r.get("x0y_misfit_ratio"))
        r["x0hat_improve_flag"] = 1.0 if _is_finite(nrmse_imp) and nrmse_imp > 0.0 else 0.0
        r["x0y_improve_flag"] = 1.0 if _is_finite(nrmse_imp_y) and nrmse_imp_y > 0.0 else 0.0
        r["x0hat_misfit_improve_flag"] = 1.0 if _is_finite(mis_ratio) and mis_ratio < 1.0 else 0.0
        r["x0y_misfit_improve_flag"] = 1.0 if _is_finite(mis_ratio_y) and mis_ratio_y < 1.0 else 0.0


def _assign_difficulty_map(run_rows: list[dict], mode: str) -> tuple[dict[str, str], float, float]:
    if mode != "tertile":
        return {}, float("nan"), float("nan")

    by_gt = defaultdict(list)
    for r in run_rows:
        gt = str(r.get("gt_idx", ""))
        x = _to_float(r.get("x0_gt_nrmse"))
        if gt != "" and _is_finite(x):
            by_gt[gt].append(x)

    gt_median = {}
    for gt, vals in by_gt.items():
        if vals:
            gt_median[gt] = float(median(vals))
    base_vals = sorted(gt_median.values())
    if not base_vals:
        return {}, float("nan"), float("nan")

    q33 = _percentile(base_vals, 0.33)
    q67 = _percentile(base_vals, 0.67)

    m: dict[str, str] = {}
    for gt, v in gt_median.items():
        if v <= q33:
            m[gt] = "easy"
        elif v <= q67:
            m[gt] = "medium"
        else:
            m[gt] = "hard"
    return m, q33, q67


def _annotate_difficulty(rows: list[dict], diff_map: dict[str, str]) -> None:
    for r in rows:
        gt = str(r.get("gt_idx", ""))
        r["_difficulty"] = diff_map.get(gt, "all")


def _collect_metric_values(
    rows: list[dict],
    metric: str,
    aggregate_by_gt_median: bool,
) -> list[float]:
    if aggregate_by_gt_median:
        by_gt = defaultdict(list)
        for r in rows:
            gt = str(r.get("gt_idx", ""))
            v = _to_float(r.get(metric))
            if gt != "" and _is_finite(v):
                by_gt[gt].append(v)
        out = []
        for vals in by_gt.values():
            if vals:
                out.append(float(median(vals)))
        return out

    out = []
    for r in rows:
        v = _to_float(r.get(metric))
        if _is_finite(v):
            out.append(v)
    return out


def _build_scope_rows(rows: list[dict], scope: str) -> list[dict]:
    if scope == "all":
        return rows
    return [r for r in rows if str(r.get("_difficulty", "all")) == scope]


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
        mono = 1.0
    else:
        non_inc = 0
        for i in range(n - 1):
            d = series[i + 1] - series[i]
            if d <= monotonic_eps:
                non_inc += 1
        mono = non_inc / float(n - 1)

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
        "monotonic_ratio": mono,
    }


def _classify_profile(features: dict, cfg: ProfileRuleConfig) -> str:
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


def _build_profile_rows_from_outer(
    outer_rows: list[dict],
    run_id_cols: list[str],
    cfg: ProfileRuleConfig,
) -> list[dict]:
    grouped = defaultdict(list)
    for r in outer_rows:
        grouped[_make_run_key(r, run_id_cols)].append(r)

    out = []
    for key, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(_to_float(x.get("outer_step", 0))))
        seq = [_to_float(r.get(cfg.misfit_col)) for r in rows]
        seq = [x for x in seq if _is_finite(x)]
        feat = _compute_profile_features(seq, cfg.monotonic_eps)
        label = _classify_profile(feat, cfg)
        base = dict(rows[0])
        rec = {
            "_config_id": base.get("_config_id", ""),
            "_source_csv": base.get("_source_csv", ""),
            "profile_label": label,
            "n_steps": feat["n_steps"],
            "first": feat["first"],
            "last": feat["last"],
            "min": feat["min"],
            "min_step": feat["min_step"],
            "min_step_pos_ratio": feat["min_step_pos_ratio"],
            "net_drop": feat["net_drop"],
            "best_drop": feat["best_drop"],
            "rebound_amp": feat["rebound_amp"],
            "rebound_ratio": feat["rebound_ratio"],
            "monotonic_ratio": feat["monotonic_ratio"],
        }
        for c in run_id_cols:
            rec[c] = base.get(c, "")
        out.append(rec)
    return out


def _build_phase_rows(
    outer_rows: list[dict],
    run_id_cols: list[str],
    phase_metrics: list[str],
    split1: float,
    split2: float,
) -> list[dict]:
    grouped = defaultdict(list)
    for r in outer_rows:
        grouped[_make_run_key(r, run_id_cols)].append(r)

    out = []
    for _, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(_to_float(x.get("outer_step", 0))))
        n = len(rows)
        if n == 0:
            continue

        phase_bucket = {"early": [], "mid": [], "late": []}
        for i, r in enumerate(rows):
            ratio = i / float(max(1, n - 1))
            if ratio < split1:
                ph = "early"
            elif ratio < split2:
                ph = "mid"
            else:
                ph = "late"
            phase_bucket[ph].append(r)

        base = dict(rows[0])
        for ph in ["early", "mid", "late"]:
            rs = phase_bucket[ph]
            if not rs:
                continue
            rec = {
                "_config_id": base.get("_config_id", ""),
                "_source_csv": base.get("_source_csv", ""),
                "phase": ph,
            }
            for c in run_id_cols:
                rec[c] = base.get(c, "")
            for m in phase_metrics:
                vals = [_to_float(x.get(m)) for x in rs]
                vals = [v for v in vals if _is_finite(v)]
                rec[m] = mean(vals) if vals else float("nan")
            out.append(rec)
    return out


def _parse_phase_splits(text: str) -> tuple[float, float]:
    s = [x.strip() for x in text.split(",") if x.strip()]
    if len(s) != 2:
        raise ValueError("--phase_splits 需要两个逗号分隔值，例如 0.33,0.67")
    a = float(s[0])
    b = float(s[1])
    if not (0.0 < a < b < 1.0):
        raise ValueError("--phase_splits 必须满足 0 < a < b < 1")
    return a, b


def _build_tableA(
    run_rows: list[dict],
    scopes: list[str],
    metrics: list[str],
    aggregate_by_gt_median: bool,
    b_iters: int,
    b_ci: float,
    b_seed: int,
) -> list[dict]:
    rows = []
    for sc in scopes:
        rs = _build_scope_rows(run_rows, sc)
        for m in metrics:
            if not rs or all((m not in r) for r in rs):
                continue
            vals = _collect_metric_values(rs, m, aggregate_by_gt_median)
            sm = _summary_from_values(vals, b_iters, b_ci, b_seed)
            rows.append(
                {
                    "table": "A_overall",
                    "scope": sc,
                    "metric": m,
                    "aggregate_by_gt_median": int(bool(aggregate_by_gt_median)),
                    "n_raw_rows": len(rs),
                    "n_unique_gt": len({str(x.get("gt_idx", "")) for x in rs if str(x.get("gt_idx", "")) != ""}),
                    "n_effective": sm["n_effective"],
                    "mean": sm["mean"],
                    "std": sm["std"],
                    "median": sm["median"],
                    "q25": sm["q25"],
                    "q75": sm["q75"],
                    "ci_low": sm["ci_low"],
                    "ci_high": sm["ci_high"],
                }
            )
    return rows


def _build_tableB(
    phase_rows: list[dict],
    scopes: list[str],
    metrics: list[str],
    aggregate_by_gt_median: bool,
    b_iters: int,
    b_ci: float,
    b_seed: int,
) -> list[dict]:
    rows = []
    for sc in scopes:
        rs_sc = _build_scope_rows(phase_rows, sc)
        for ph in ["early", "mid", "late"]:
            rs = [r for r in rs_sc if str(r.get("phase", "")) == ph]
            for m in metrics:
                if not rs or all((m not in r) for r in rs):
                    continue
                vals = _collect_metric_values(rs, m, aggregate_by_gt_median)
                sm = _summary_from_values(vals, b_iters, b_ci, b_seed)
                rows.append(
                    {
                        "table": "B_phase",
                        "scope": sc,
                        "phase": ph,
                        "metric": m,
                        "aggregate_by_gt_median": int(bool(aggregate_by_gt_median)),
                        "n_raw_rows": len(rs),
                        "n_unique_gt": len({str(x.get("gt_idx", "")) for x in rs if str(x.get("gt_idx", "")) != ""}),
                        "n_effective": sm["n_effective"],
                        "mean": sm["mean"],
                        "std": sm["std"],
                        "median": sm["median"],
                        "q25": sm["q25"],
                        "q75": sm["q75"],
                        "ci_low": sm["ci_low"],
                        "ci_high": sm["ci_high"],
                    }
                )
    return rows


def _build_tableC(
    run_rows: list[dict],
    scopes: list[str],
    profile_metrics: list[str],
    aggregate_by_gt_median: bool,
    b_iters: int,
    b_ci: float,
    b_seed: int,
) -> list[dict]:
    rows = []
    for sc in scopes:
        rs = _build_scope_rows(run_rows, sc)
        labels = [str(r.get("_profile_label", "unknown")) for r in rs]
        cnt = Counter(labels)
        total = max(1, len(labels))
        for lb in sorted(cnt.keys()):
            rows.append(
                {
                    "table": "C_profiles",
                    "row_type": "distribution",
                    "scope": sc,
                    "profile_label": lb,
                    "metric": "count",
                    "aggregate_by_gt_median": int(bool(aggregate_by_gt_median)),
                    "n_raw_rows": len(rs),
                    "n_unique_gt": len({str(x.get("gt_idx", "")) for x in rs if str(x.get("gt_idx", "")) != ""}),
                    "n_effective": cnt[lb],
                    "mean": float(cnt[lb]),
                    "std": 0.0,
                    "median": float(cnt[lb]),
                    "q25": float(cnt[lb]),
                    "q75": float(cnt[lb]),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "ratio": cnt[lb] / float(total),
                    "phase": "",
                }
            )

        for lb in sorted(cnt.keys()):
            rs_lb = [r for r in rs if str(r.get("_profile_label", "unknown")) == lb]
            for m in profile_metrics:
                if not rs_lb or all((m not in r) for r in rs_lb):
                    continue
                vals = _collect_metric_values(rs_lb, m, aggregate_by_gt_median)
                sm = _summary_from_values(vals, b_iters, b_ci, b_seed)
                rows.append(
                    {
                        "table": "C_profiles",
                        "row_type": "metric",
                        "scope": sc,
                        "profile_label": lb,
                        "metric": m,
                        "aggregate_by_gt_median": int(bool(aggregate_by_gt_median)),
                        "n_raw_rows": len(rs_lb),
                        "n_unique_gt": len(
                            {str(x.get("gt_idx", "")) for x in rs_lb if str(x.get("gt_idx", "")) != ""}
                        ),
                        "n_effective": sm["n_effective"],
                        "mean": sm["mean"],
                        "std": sm["std"],
                        "median": sm["median"],
                        "q25": sm["q25"],
                        "q75": sm["q75"],
                        "ci_low": sm["ci_low"],
                        "ci_high": sm["ci_high"],
                        "ratio": float("nan"),
                        "phase": "",
                    }
                )
    return rows


def _parse_args():
    p = argparse.ArgumentParser(description="Mechanism-A 统计汇总（A/B/C 表）")
    p.add_argument("--run_csv", nargs="+", required=True, help="run_summary CSV 路径或 glob")
    p.add_argument("--outer_csv", nargs="+", default=[], help="outer_summary CSV 路径或 glob")
    p.add_argument("--profile_csv", nargs="+", default=[], help="profile_detail CSV 路径或 glob")
    p.add_argument("--out_dir", required=True, help="输出目录")
    p.add_argument("--tag", type=str, default="", help="输出文件附加标记")
    p.add_argument("--run_id_cols", nargs="+", default=DEFAULT_RUN_ID_COLS)

    p.add_argument("--difficulty_mode", choices=["none", "tertile"], default="none")
    p.add_argument("--aggregate_by_gt_median", dest="aggregate_by_gt_median", action="store_true")
    p.add_argument("--no_aggregate_by_gt_median", dest="aggregate_by_gt_median", action="store_false")
    p.set_defaults(aggregate_by_gt_median=True)

    p.add_argument("--bootstrap_iters", type=int, default=0)
    p.add_argument("--bootstrap_ci", type=float, default=95.0)
    p.add_argument("--bootstrap_seed", type=int, default=8)
    p.add_argument("--phase_splits", type=str, default="0.33,0.67")

    p.add_argument("--run_metrics", type=str, default=",".join(DEFAULT_RUN_METRICS))
    p.add_argument("--phase_metrics", type=str, default=",".join(DEFAULT_PHASE_METRICS))
    p.add_argument("--profile_metrics", type=str, default=",".join(DEFAULT_PROFILE_METRICS))

    p.add_argument("--profile_misfit_col", type=str, default="misfit_x0hat")
    p.add_argument("--profile_min_steps", type=int, default=4)
    p.add_argument("--profile_monotonic_eps", type=float, default=0.0)
    p.add_argument("--profile_drop_eps_abs", type=float, default=0.01)
    p.add_argument("--profile_flat_eps_abs", type=float, default=0.0)
    p.add_argument("--profile_rebound_abs_min", type=float, default=0.01)
    p.add_argument("--profile_rebound_ratio_min", type=float, default=0.35)
    p.add_argument("--profile_late_drop_min_pos_ratio", type=float, default=0.70)
    p.add_argument("--profile_steady_mono_min", type=float, default=0.70)

    p.add_argument("--write_tex", action="store_true", help="额外保存 tex 表")
    p.add_argument("--save_tidy", action="store_true", help="保存合并后的逐 run 追踪表")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    run_paths = _resolve_csv_paths(args.run_csv)
    if not run_paths:
        raise FileNotFoundError("未解析到 run_summary CSV。")
    outer_paths = _resolve_csv_paths(args.outer_csv)
    profile_paths = _resolve_csv_paths(args.profile_csv)

    run_rows = _load_rows(run_paths)
    _add_derived_run_metrics(run_rows)

    diff_map, q33, q67 = _assign_difficulty_map(run_rows, args.difficulty_mode)
    _annotate_difficulty(run_rows, diff_map)
    scopes = ["all"]
    if args.difficulty_mode == "tertile" and diff_map:
        scopes.extend(["easy", "medium", "hard"])

    run_metrics = _parse_metrics(args.run_metrics, DEFAULT_RUN_METRICS)
    phase_metrics = _parse_metrics(args.phase_metrics, DEFAULT_PHASE_METRICS)
    profile_metrics = _parse_metrics(args.profile_metrics, DEFAULT_PROFILE_METRICS)
    split1, split2 = _parse_phase_splits(args.phase_splits)

    tableA_rows = _build_tableA(
        run_rows=run_rows,
        scopes=scopes,
        metrics=run_metrics,
        aggregate_by_gt_median=bool(args.aggregate_by_gt_median),
        b_iters=int(args.bootstrap_iters),
        b_ci=float(args.bootstrap_ci),
        b_seed=int(args.bootstrap_seed),
    )

    tableB_rows: list[dict] = []
    phase_rows: list[dict] = []
    if outer_paths:
        outer_rows = _load_rows(outer_paths)
        _annotate_difficulty(outer_rows, diff_map)
        phase_rows = _build_phase_rows(
            outer_rows=outer_rows,
            run_id_cols=list(args.run_id_cols),
            phase_metrics=phase_metrics,
            split1=split1,
            split2=split2,
        )
        _annotate_difficulty(phase_rows, diff_map)
        tableB_rows = _build_tableB(
            phase_rows=phase_rows,
            scopes=scopes,
            metrics=phase_metrics,
            aggregate_by_gt_median=bool(args.aggregate_by_gt_median),
            b_iters=int(args.bootstrap_iters),
            b_ci=float(args.bootstrap_ci),
            b_seed=int(args.bootstrap_seed),
        )

    profile_rows: list[dict] = []
    if profile_paths:
        profile_rows = _load_rows(profile_paths)
    elif outer_paths:
        outer_rows_for_profile = _load_rows(outer_paths)
        cfg = ProfileRuleConfig(
            misfit_col=args.profile_misfit_col,
            min_steps=int(args.profile_min_steps),
            monotonic_eps=float(args.profile_monotonic_eps),
            drop_eps_abs=float(args.profile_drop_eps_abs),
            flat_eps_abs=float(args.profile_flat_eps_abs),
            rebound_abs_min=float(args.profile_rebound_abs_min),
            rebound_ratio_min=float(args.profile_rebound_ratio_min),
            late_drop_min_pos_ratio=float(args.profile_late_drop_min_pos_ratio),
            steady_mono_min=float(args.profile_steady_mono_min),
        )
        profile_rows = _build_profile_rows_from_outer(
            outer_rows=outer_rows_for_profile,
            run_id_cols=list(args.run_id_cols),
            cfg=cfg,
        )

    profile_map = {}
    if profile_rows:
        _annotate_difficulty(profile_rows, diff_map)
        for r in profile_rows:
            profile_map[_make_run_key(r, list(args.run_id_cols))] = str(r.get("profile_label", "unknown"))

    for r in run_rows:
        r["_profile_label"] = profile_map.get(_make_run_key(r, list(args.run_id_cols)), "unknown")

    tableC_rows = _build_tableC(
        run_rows=run_rows,
        scopes=scopes,
        profile_metrics=profile_metrics,
        aggregate_by_gt_median=bool(args.aggregate_by_gt_median),
        b_iters=int(args.bootstrap_iters),
        b_ci=float(args.bootstrap_ci),
        b_seed=int(args.bootstrap_seed),
    )

    suffix = f"_{args.tag}" if args.tag else ""
    stamp = _ts()
    tableA_path = os.path.join(args.out_dir, f"mechA_tableA_overall{suffix}_{stamp}.csv")
    tableB_path = os.path.join(args.out_dir, f"mechA_tableB_phase{suffix}_{stamp}.csv")
    tableC_path = os.path.join(args.out_dir, f"mechA_tableC_profiles{suffix}_{stamp}.csv")

    headers_common = [
        "table",
        "row_type",
        "scope",
        "phase",
        "profile_label",
        "metric",
        "aggregate_by_gt_median",
        "n_raw_rows",
        "n_unique_gt",
        "n_effective",
        "mean",
        "std",
        "median",
        "q25",
        "q75",
        "ci_low",
        "ci_high",
        "ratio",
    ]

    tableA_rows_out = []
    for r in tableA_rows:
        x = dict(r)
        x["row_type"] = "metric"
        x["phase"] = ""
        x["profile_label"] = ""
        x["ratio"] = float("nan")
        tableA_rows_out.append(x)
    _save_csv(tableA_path, tableA_rows_out, headers_common)

    if tableB_rows:
        tableB_rows_out = []
        for r in tableB_rows:
            x = dict(r)
            x["row_type"] = "metric"
            x["profile_label"] = ""
            x["ratio"] = float("nan")
            tableB_rows_out.append(x)
        _save_csv(tableB_path, tableB_rows_out, headers_common)

    _save_csv(tableC_path, tableC_rows, headers_common)

    tidy_path = ""
    if args.save_tidy:
        tidy_path = os.path.join(args.out_dir, f"mechA_tidy_merged{suffix}_{stamp}.csv")
        tidy_headers = sorted(set().union(*[set(r.keys()) for r in run_rows])) if run_rows else []
        _save_csv(tidy_path, run_rows, tidy_headers)

    if args.write_tex:
        _save_tex(
            tableA_path.replace(".csv", ".tex"),
            tableA_rows_out,
            headers_common,
            caption="Mechanism-A Overall Statistics",
            label="tab:mecha_overall",
        )
        if tableB_rows:
            _save_tex(
                tableB_path.replace(".csv", ".tex"),
                tableB_rows_out,
                headers_common,
                caption="Mechanism-A Phase Statistics",
                label="tab:mecha_phase",
            )
        _save_tex(
            tableC_path.replace(".csv", ".tex"),
            tableC_rows,
            headers_common,
            caption="Mechanism-A Profile Statistics",
            label="tab:mecha_profile",
        )

    print("=" * 72)
    print("Mechanism-A Statistics Summary")
    print("=" * 72)
    print(f"run_csv_count: {len(run_paths)}  run_rows: {len(run_rows)}")
    print(f"outer_csv_count: {len(outer_paths)}  phase_rows: {len(phase_rows)}")
    print(f"profile_csv_count: {len(profile_paths)}  profile_rows: {len(profile_rows)}")
    print(f"difficulty_mode: {args.difficulty_mode}")
    if args.difficulty_mode == "tertile":
        print(f"q33={q33:.6f}, q67={q67:.6f}")
    print(f"aggregate_by_gt_median: {bool(args.aggregate_by_gt_median)}")
    print(f"bootstrap_iters: {int(args.bootstrap_iters)}")
    print(f"tableA_csv: {tableA_path}")
    if tableB_rows:
        print(f"tableB_csv: {tableB_path}")
    else:
        print("tableB_csv: <skipped, outer_csv missing>")
    print(f"tableC_csv: {tableC_path}")
    if tidy_path:
        print(f"tidy_csv: {tidy_path}")


if __name__ == "__main__":
    main()

