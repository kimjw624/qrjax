#!/usr/bin/env bash
# Overnight fixes study for qrjax. Run from anywhere; it cd's to the repo root.
#
#   bash study_fixes/run_overnight.sh                 # full run, stops starting
#                                                     # new training at HOURS
#   SMOKE=1 bash study_fixes/run_overnight.sh         # ~10 min end-to-end check
#   STAGES="figures report" bash study_fixes/run_overnight.sh   # slides only
#
# Environment knobs (defaults in brackets):
#   HOURS [9]          wall-clock budget; no training run is started that would
#                      not finish RESERVE_MIN before it
#   RESERVE_MIN [45]   minutes kept for evaluation + report at the end
#   RESULTS [study_results]   output root (relative to the repo)
#   PAPER_DIR [auto]   run_final_study output used for the paper
#   SEEDS [0,1,2]   EPISODES [512]   STEPS [2000000]   TRAIN_GROUPS [A,B,C]
#                      (talk ablations: TRAIN_GROUPS=H,M, 18 runs, ~5 h)
#   STAGES ["tests provenance ablation train eval figures diagnose talk report"]
#   PY [python]
set -u
cd "$(dirname "$0")/.."

SMOKE=${SMOKE:-0}
PY=${PY:-python}
if [ "$SMOKE" = "1" ]; then
  RESULTS=${RESULTS:-study_results_smoke}
  HOURS=${HOURS:-1}; RESERVE_MIN=${RESERVE_MIN:-5}
  SEEDS=${SEEDS:-0}; EPISODES=${EPISODES:-32}; STEPS=${STEPS:-20000}
  MAX_JOBS=${MAX_JOBS:-3}; CHUNK=${CHUNK:-32}; SMOKE_FLAG="--smoke"
else
  RESULTS=${RESULTS:-study_results}
  HOURS=${HOURS:-9}; RESERVE_MIN=${RESERVE_MIN:-45}
  SEEDS=${SEEDS:-0,1,2}; EPISODES=${EPISODES:-512}; STEPS=${STEPS:-2000000}
  MAX_JOBS=${MAX_JOBS:-0}; CHUNK=${CHUNK:-256}; SMOKE_FLAG=""
fi
PAPER_DIR=${PAPER_DIR:-auto}
GROUPS_=${TRAIN_GROUPS:-A,B,C}
EVAL_SEED=${EVAL_SEED:-20260829}
STAGES=${STAGES:-"tests provenance ablation train eval figures diagnose talk report"}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-.80}
export PYTHONUNBUFFERED=1

START=$(date +%s)
DEADLINE=$(( START + HOURS * 3600 ))
mkdir -p "$RESULTS/logs"
LOG="$RESULTS/logs/overnight_$(date +%Y%m%d_%H%M).log"
STAGE_TSV="$RESULTS/logs/stages.tsv"

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
run_stage() {          # name, command...
  local name=$1; shift
  local t0=$(date +%s)
  say "=== stage $name ==="
  "$@" 2>&1 | tee -a "$LOG" "$RESULTS/logs/stage_${name}.log"
  local rc=${PIPESTATUS[0]}
  local mins=$(( ($(date +%s) - t0) / 60 ))
  printf "%s\t%s\t%s\n" "$name" "$rc" "$mins" >> "$STAGE_TSV"
  if [ "$rc" != "0" ]; then say "stage $name FAILED (exit $rc) -- continuing"; fi
  return 0
}
has() { case " $STAGES " in *" $1 "*) return 0;; esac; return 1; }

say "repo     : $(pwd)  ($(git rev-parse --short HEAD 2>/dev/null) + overlay)"
say "results  : $RESULTS"
say "budget   : $HOURS h (training stops starting at $(date -d @$(( DEADLINE - RESERVE_MIN*60 )) '+%H:%M' 2>/dev/null))"
say "settings : seeds $SEEDS, $EPISODES eval episodes, $STEPS steps/run, groups $GROUPS_, smoke=$SMOKE"
$PY -c "import jax; print('jax', jax.__version__, jax.devices())" 2>&1 | tee -a "$LOG"

COMMON=(--results "$RESULTS")
EVAL=(--episodes "$EPISODES" --eval_seed "$EVAL_SEED" --chunk "$CHUNK")

# Tests run on the CPU so they never compete with training for GPU memory.
has tests && run_stage tests env JAX_PLATFORMS=cpu $PY -m pytest -q tests \
    -p no:cacheprovider
has provenance && run_stage provenance $PY -m study_fixes.s0_provenance "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR"
has ablation && run_stage ablation $PY -m study_fixes.s1_ablation "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR" "${EVAL[@]}"
has train && run_stage train $PY -m study_fixes.s2_train "${COMMON[@]}" \
    --seeds "$SEEDS" --steps "$STEPS" --deadline "$DEADLINE" \
    --reserve_min "$RESERVE_MIN" --groups "$GROUPS_" --max_jobs "$MAX_JOBS" $SMOKE_FLAG
has eval && run_stage eval $PY -m study_fixes.s3_eval "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR" --seeds "$SEEDS" "${EVAL[@]}"
has figures && run_stage figures $PY -m study_fixes.s5_figures "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR" --seeds "$SEEDS" "${EVAL[@]}"
has diagnose && run_stage diagnose $PY -m study_fixes.s6_diagnose "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR" --seeds "$SEEDS" "${EVAL[@]}"
has talk && run_stage talk $PY -m study_fixes.s7_talk "${COMMON[@]}" \
    --paper_dir "$PAPER_DIR" --seeds "$SEEDS" "${EVAL[@]}"
has report && run_stage report $PY -m study_fixes.s4_report "${COMMON[@]}"

say "finished in $(( ($(date +%s) - START) / 60 )) min"
say "read $RESULTS/SUMMARY.md; send back the newest zip in $RESULTS/upload/"
