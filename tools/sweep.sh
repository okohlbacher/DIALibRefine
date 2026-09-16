#!/usr/bin/env bash
#
# The accuracy/runtime sweep: training size x convergence, for RT and CCS.
#
# Design (from the GPT-6-Astra protocol, 2026-09-16):
#  * One CONTINUOUS trajectory per training size with a 100-epoch horizon and
#    per-epoch evaluation, patience disabled -- so every stopping rule can be
#    replayed offline on the stored trajectory (stop_rule_sim.py) instead of
#    one run per threshold.
#  * Nested sizes: every smaller training set is a prefix of the larger one,
#    and val/test are identical across sizes (finetune.py freezes them first).
#  * The horizon confound, measured rather than assumed: a cosine schedule that
#    anneals to zero at epoch 100 is NOT the schedule a 20-epoch run gets, so
#    replaying "stop at 20" on a 100-horizon trajectory underestimates a real
#    20-epoch run. Two sizes also get explicit 20- and 40-epoch horizons.
#
#   sweep.sh <report.parquet> <outdir> [threads-per-head]
set -uo pipefail
report="${1:?usage: sweep.sh <report.parquet> <outdir> [threads]}"
out="${2:?usage: sweep.sh <report.parquet> <outdir> [threads]}"
thr="${3:-24}"
py="${ODIA_FINETUNE_PYTHON:-/scratch/kohlbach/odia/rtfinetune/env/bin/python}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ODIA_PEPTDEEP_MODELS="${ODIA_PEPTDEEP_MODELS:-/scratch/kohlbach/odia/ptm/generic}"
mkdir -p "$out"

SIZES=(500 1000 2000 4000 8000 16000 0)     # 0 = the whole pool
HORIZON_SIZES=(2000 0)                       # explicit short horizons here
HORIZONS=(20 40)

run_head() { # head
  local head=$1 log="$out/$head.log"
  echo "### $head sweep start $(date -Is)" >> "$log"
  for n in "${SIZES[@]}"; do
    local d="$out/$head/n${n}_h100"
    [[ -f "$d/${head}_provenance.json" ]] && { echo "skip $d (done)" >> "$log"; continue; }
    rm -rf "$d"
    echo "--- $head n=$n horizon=100 $(date -Is)" >> "$log"
    "$py" "$here/finetune.py" "$report" "$d" --head "$head" --train-size "$n" \
      --epochs 100 --warmup 10 --eval-every 1 --min-epochs 100 --patience 1000 \
      --rt-max-minutes 30 --threads "$thr" >> "$log" 2>&1
    echo "--- exit $? $(date -Is)" >> "$log"
  done
  for n in "${HORIZON_SIZES[@]}"; do for h in "${HORIZONS[@]}"; do
    local d="$out/$head/n${n}_h${h}"
    [[ -f "$d/${head}_provenance.json" ]] && continue
    rm -rf "$d"
    echo "--- $head n=$n horizon=$h $(date -Is)" >> "$log"
    "$py" "$here/finetune.py" "$report" "$d" --head "$head" --train-size "$n" \
      --epochs "$h" --warmup $(( h / 5 )) --eval-every 1 --min-epochs "$h" --patience 1000 \
      --rt-max-minutes 30 --threads "$thr" >> "$log" 2>&1
    echo "--- exit $? $(date -Is)" >> "$log"
  done; done
  echo "### $head sweep done $(date -Is)" >> "$log"
}

run_head rt  &
run_head ccs &
wait
echo "### SWEEP DONE $(date -Is)" >> "$out/sweep.log"
