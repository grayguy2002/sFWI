#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash sFWI/experiments/run_mechA_postprocess.sh \
    --output_dir /content/drive/MyDrive/score_sde_inverseSolving/outputs/mechA_batch100_test_seed8

Optional:
  --inner_csv <path>           # 可不传，默认 output_dir 下最新 inner_detail
  --outer_csv <path>           # 可不传，默认 output_dir 下最新 outer_summary
  --run_csv <path>             # 可不传，默认 output_dir 下最新 run_summary
  --stats_out_dir <dir>        # 默认同 output_dir
  --tag <str>                  # 默认自动从 run_csv 时间戳生成，如 post_20260222_083257
  --difficulty_mode <none|tertile>   # 默认 tertile
  --bootstrap_iters <int>      # 默认 2000
  --bootstrap_ci <float>       # 默认 95
  --write_tex                  # 传入则输出 tex 表
  --no_save_tidy               # 不输出 tidy_merged

Output:
  1) mechA_profile_detail_*.csv / mechA_profile_summary_*.csv
  2) mechA_tableA_overall_*.csv / mechA_tableB_phase_*.csv / mechA_tableC_profiles_*.csv
  3) mechA_conclusion_<tag>.txt   (论文可写结论草案)
EOF
}

OUTPUT_DIR=""
INNER_CSV=""
OUTER_CSV=""
RUN_CSV=""
STATS_OUT_DIR=""
TAG=""
DIFFICULTY_MODE="tertile"
BOOTSTRAP_ITERS="2000"
BOOTSTRAP_CI="95"
WRITE_TEX=0
SAVE_TIDY=1
PYTHON_BIN=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output_dir) OUTPUT_DIR="${2:-}"; shift 2 ;;
    --inner_csv) INNER_CSV="${2:-}"; shift 2 ;;
    --outer_csv) OUTER_CSV="${2:-}"; shift 2 ;;
    --run_csv) RUN_CSV="${2:-}"; shift 2 ;;
    --stats_out_dir) STATS_OUT_DIR="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    --difficulty_mode) DIFFICULTY_MODE="${2:-}"; shift 2 ;;
    --bootstrap_iters) BOOTSTRAP_ITERS="${2:-}"; shift 2 ;;
    --bootstrap_ci) BOOTSTRAP_CI="${2:-}"; shift 2 ;;
    --write_tex) WRITE_TEX=1; shift 1 ;;
    --no_save_tidy) SAVE_TIDY=0; shift 1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$OUTPUT_DIR" ]]; then
  echo "[ERROR] --output_dir is required."
  usage
  exit 1
fi

if command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "[ERROR] python/python3 not found in PATH."
  exit 1
fi

if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "[ERROR] output_dir not found: $OUTPUT_DIR"
  exit 1
fi

if [[ -z "$STATS_OUT_DIR" ]]; then
  STATS_OUT_DIR="$OUTPUT_DIR"
fi
mkdir -p "$STATS_OUT_DIR"

find_latest_or_empty() {
  local pattern="$1"
  local latest
  latest="$(ls -1t $pattern 2>/dev/null | head -n 1 || true)"
  echo "$latest"
}

if [[ -z "$INNER_CSV" ]]; then
  INNER_CSV="$(find_latest_or_empty "$OUTPUT_DIR/daps_mech_inner_detail_model-*.csv")"
fi
if [[ -z "$OUTER_CSV" ]]; then
  OUTER_CSV="$(find_latest_or_empty "$OUTPUT_DIR/daps_mech_outer_summary_model-*.csv")"
fi
if [[ -z "$RUN_CSV" ]]; then
  RUN_CSV="$(find_latest_or_empty "$OUTPUT_DIR/daps_mech_run_summary_model-*.csv")"
fi

if [[ -z "$OUTER_CSV" || ! -f "$OUTER_CSV" ]]; then
  echo "[ERROR] outer_summary csv missing."
  exit 1
fi
if [[ -z "$RUN_CSV" || ! -f "$RUN_CSV" ]]; then
  echo "[ERROR] run_summary csv missing."
  exit 1
fi
if [[ -n "$INNER_CSV" && ! -f "$INNER_CSV" ]]; then
  echo "[WARN] inner_csv not found, ignore: $INNER_CSV"
  INNER_CSV=""
fi

if [[ -z "$TAG" ]]; then
  bn="$(basename "$RUN_CSV")"
  # 尝试提取 YYYYMMDD_HHMMSS
  stamp="$(echo "$bn" | sed -E 's/.*_([0-9]{8}_[0-9]{6})\.csv/\1/' || true)"
  if [[ "$stamp" =~ ^[0-9]{8}_[0-9]{6}$ ]]; then
    TAG="post_${stamp}"
  else
    TAG="post_$(date +%Y%m%d_%H%M%S)"
  fi
fi

echo "========================================================================"
echo "MechA Postprocess"
echo "========================================================================"
echo "output_dir:     $OUTPUT_DIR"
echo "stats_out_dir:  $STATS_OUT_DIR"
echo "inner_csv:      ${INNER_CSV:-<none>}"
echo "outer_csv:      $OUTER_CSV"
echo "run_csv:        $RUN_CSV"
echo "tag:            $TAG"
echo "difficulty:     $DIFFICULTY_MODE"
echo "bootstrap:      iters=$BOOTSTRAP_ITERS, ci=$BOOTSTRAP_CI"
echo "python_bin:     $PYTHON_BIN"

echo
echo "[1/3] Run mechA_profile_rules.py ..."
"$PYTHON_BIN" sFWI/experiments/mechA_profile_rules.py \
  --outer_csv "$OUTER_CSV" \
  --out_dir "$STATS_OUT_DIR" \
  --tag "$TAG"

PROFILE_CSV="$(find_latest_or_empty "$STATS_OUT_DIR/mechA_profile_detail_${TAG}_*.csv")"
if [[ -z "$PROFILE_CSV" || ! -f "$PROFILE_CSV" ]]; then
  echo "[ERROR] profile detail csv not found."
  exit 1
fi

echo
echo "[2/3] Run mechA_stats_summary.py ..."
stats_args=(
  --run_csv "$RUN_CSV"
  --outer_csv "$OUTER_CSV"
  --profile_csv "$PROFILE_CSV"
  --out_dir "$STATS_OUT_DIR"
  --tag "$TAG"
  --difficulty_mode "$DIFFICULTY_MODE"
  --bootstrap_iters "$BOOTSTRAP_ITERS"
  --bootstrap_ci "$BOOTSTRAP_CI"
)
if [[ "$SAVE_TIDY" -eq 1 ]]; then
  stats_args+=(--save_tidy)
fi
if [[ "$WRITE_TEX" -eq 1 ]]; then
  stats_args+=(--write_tex)
fi

"$PYTHON_BIN" sFWI/experiments/mechA_stats_summary.py "${stats_args[@]}"

TABLE_A="$(find_latest_or_empty "$STATS_OUT_DIR/mechA_tableA_overall_${TAG}_*.csv")"
TABLE_B="$(find_latest_or_empty "$STATS_OUT_DIR/mechA_tableB_phase_${TAG}_*.csv")"
TABLE_C="$(find_latest_or_empty "$STATS_OUT_DIR/mechA_tableC_profiles_${TAG}_*.csv")"

if [[ -z "$TABLE_A" || -z "$TABLE_B" || -z "$TABLE_C" ]]; then
  echo "[ERROR] table csv missing."
  echo "tableA=$TABLE_A"
  echo "tableB=$TABLE_B"
  echo "tableC=$TABLE_C"
  exit 1
fi

CONCLUSION_TXT="$STATS_OUT_DIR/mechA_conclusion_${TAG}.txt"

echo
echo "[3/3] Generate paper-ready conclusions ..."
"$PYTHON_BIN" - "$TABLE_A" "$TABLE_B" "$TABLE_C" "$CONCLUSION_TXT" <<'PY'
import csv
import math
import sys

table_a, table_b, table_c, out_txt = sys.argv[1:5]

def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")

def fmt(v, nd=4):
    x = to_float(v)
    if not math.isfinite(x):
        return "nan"
    return f"{x:.{nd}f}"

def pick(rows, **conds):
    for r in rows:
        ok = True
        for k, v in conds.items():
            if str(r.get(k, "")) != str(v):
                ok = False
                break
        if ok:
            return r
    return None

def effect_text(mean_v, lo_v, hi_v, positive_good=True):
    m, lo, hi = to_float(mean_v), to_float(lo_v), to_float(hi_v)
    if not (math.isfinite(m) and math.isfinite(lo) and math.isfinite(hi)):
        return "（CI 不可用）"
    if positive_good:
        if lo > 0:
            return "（稳健提升，CI 全部 > 0）"
        if hi < 0:
            return "（稳健退化，CI 全部 < 0）"
    else:
        if hi < 1:
            return "（稳健改善，CI 全部 < 1）"
        if lo > 1:
            return "（稳健退化，CI 全部 > 1）"
    return "（效果不稳健，CI 跨阈值）"

rows_a = read_csv(table_a)
rows_b = read_csv(table_b)
rows_c = read_csv(table_c)

lines = []
lines.append("=" * 80)
lines.append("论文可写结论（自动生成草案）")
lines.append("=" * 80)

r = pick(rows_a, table="A_overall", row_type="metric", scope="all", metric="x0hat_gt_nrmse_improve")
if r:
    lines.append(
        "1) Overall（x0hat 模型域）: "
        f"x0hat_gt_nrmse_improve={fmt(r.get('mean'))}, "
        f"95%CI=[{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}] "
        + effect_text(r.get("mean"), r.get("ci_low"), r.get("ci_high"), positive_good=True)
    )

r = pick(rows_a, table="A_overall", row_type="metric", scope="all", metric="x0hat_misfit_ratio")
if r:
    lines.append(
        "2) Overall（数据域）: "
        f"x0hat_misfit_ratio={fmt(r.get('mean'))}, "
        f"95%CI=[{fmt(r.get('ci_low'))}, {fmt(r.get('ci_high'))}] "
        + effect_text(r.get("mean"), r.get("ci_low"), r.get("ci_high"), positive_good=False)
    )

r = pick(rows_a, table="A_overall", row_type="metric", scope="all", metric="x0hat_improve_flag")
if r:
    lines.append(
        f"3) 提升样本占比: x0hat_improve_flag={fmt(r.get('mean'))} (~{fmt(to_float(r.get('mean')) * 100.0, 2)}%)"
    )

easy = pick(rows_a, table="A_overall", row_type="metric", scope="easy", metric="x0hat_gt_nrmse_improve")
med = pick(rows_a, table="A_overall", row_type="metric", scope="medium", metric="x0hat_gt_nrmse_improve")
hard = pick(rows_a, table="A_overall", row_type="metric", scope="hard", metric="x0hat_gt_nrmse_improve")
if easy and med and hard:
    lines.append(
        "4) 难度分层（x0hat_gt_nrmse_improve 均值）: "
        f"easy={fmt(easy.get('mean'))}, medium={fmt(med.get('mean'))}, hard={fmt(hard.get('mean'))}"
    )

e = pick(rows_b, table="B_phase", row_type="metric", scope="all", phase="early", metric="misfit_x0hat")
m = pick(rows_b, table="B_phase", row_type="metric", scope="all", phase="mid", metric="misfit_x0hat")
l = pick(rows_b, table="B_phase", row_type="metric", scope="all", phase="late", metric="misfit_x0hat")
if e and m and l:
    de = to_float(l.get("mean")) - to_float(e.get("mean"))
    trend = "上升(退化)" if de > 0 else "下降(改善)"
    lines.append(
        "5) 分阶段（misfit_x0hat）: "
        f"early={fmt(e.get('mean'))}, mid={fmt(m.get('mean'))}, late={fmt(l.get('mean'))}，"
        f"整体呈{trend}（late-early={fmt(de)}）"
    )

dist = [
    r for r in rows_c
    if str(r.get("table", "")) == "C_profiles"
    and str(r.get("row_type", "")) == "distribution"
    and str(r.get("scope", "")) == "all"
]
if dist:
    dist = sorted(dist, key=lambda x: to_float(x.get("ratio")), reverse=True)
    segs = []
    for r in dist:
        segs.append(f"{r.get('profile_label','unknown')}={fmt(to_float(r.get('ratio')) * 100.0, 2)}%")
    lines.append("6) Profile 分布: " + ", ".join(segs))

lines.append("")
lines.append("--- Results 可用模板 ---")
lines.append(
    "在当前测试集上，我们以统计表替代主观的下降曲线口头分类。"
    "总体指标与置信区间显示 DAPS-FWI 在样本间呈现异质动力学；"
    "分阶段统计揭示 early/mid/late 内部演化趋势；"
    "profile 分布进一步说明该过程并非单一单调下降机制，而是多模式并存。"
)

txt = "\n".join(lines)
print(txt)
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(txt + "\n")
PY

echo
echo "Done."
echo "profile_detail_csv: $PROFILE_CSV"
echo "tableA_csv:         $TABLE_A"
echo "tableB_csv:         $TABLE_B"
echo "tableC_csv:         $TABLE_C"
echo "conclusion_txt:     $CONCLUSION_TXT"
