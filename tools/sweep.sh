#!/usr/bin/env bash
#
# The accuracy/runtime sweep: training size x convergence, for RT and CCS.
#
# Design (GPT-6-Astra protocol + the scoping measurements, 2026-09-16):
#  * One CONTINUOUS trajectory per (head, training size) with a 100-epoch
#    horizon, per-epoch evaluation and patience OFF, so every stopping rule can
#    be replayed offline on the stored trajectory (stop_rule_sim.py).
#  * Nested sizes; val/test identical across sizes (finetune.py freezes them).
#  * Explicit 20/40-epoch horizons at two sizes, because "stop at 20" replayed
#    on a cosine schedule that anneals at 100 underestimates a real 20-run.
#  * 4 THREADS PER JOB, many jobs in parallel. Measured on this node: a 2k-PSM
#    epoch takes 1.2 s at 4 threads, 3.7-5.8 s at 24, 10 s at 48. And cost is
#    optimizer STEPS, not peptides -- peptdeep batches per peptide length, so a
#    500-PSM subsample still runs ~23 steps/epoch vs 38 for the full set.
#
#   sweep.sh <report.parquet> <outdir> [threads-per-job=4] [parallel-jobs=12]
set -uo pipefail
report="${1:?}"; out="${2:?}"; thr="${3:-4}"; par="${4:-12}"
py="${ODIA_FINETUNE_PYTHON:-/scratch/kohlbach/odia/rtfinetune/env/bin/python}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ODIA_PEPTDEEP_MODELS="${ODIA_PEPTDEEP_MODELS:-/scratch/kohlbach/odia/ptm/generic}"
mkdir -p "$out"

jobs=()   # "head size horizon"
for head in rt ccs; do
  for n in 500 1000 2000 4000 8000 16000 0; do jobs+=("$head $n 100"); done
  for n in 2000 0; do for h in 20 40; do jobs+=("$head $n $h"); done; done
done

run_one() { # head size horizon
  local head=$1 n=$2 h=$3 d="$out/$1/n${2}_h${3}" log
  log="$out/$1/n${2}_h${3}.log"; mkdir -p "$out/$1"
  [[ -f "$d/${head}_provenance.json" ]] && return 0
  rm -rf "$d"
  echo "### start $(date -Is) head=$head n=$n horizon=$h threads=$thr" > "$log"
  "$py" "$here/finetune.py" "$report" "$d" --head "$head" --train-size "$n" \
    --epochs "$h" --warmup $(( h >= 100 ? 10 : h / 5 )) --eval-every 1 --min-epochs "$h" --patience 100000 \
    --rt-max-minutes 30 --threads "$thr" >> "$log" 2>&1
  echo "### exit $? $(date -Is)" >> "$log"
}
export -f run_one; export out thr py here report ODIA_PEPTDEEP_MODELS
printf '%s\n' "${jobs[@]}" | xargs -P "$par" -I{} bash -c 'run_one {}'
echo "### SWEEP DONE $(date -Is)" > "$out/DONE"
