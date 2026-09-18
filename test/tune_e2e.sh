#!/usr/bin/env bash
# Bounded end-to-end fine-tune on a synthetic run, both heads, through the
# MERGED tool: a library and a reference report go in, a refined library comes
# out, and the tuned models are kept so the write-back can still be checked.
#
#   tune_e2e.sh <DIALibRefine> <models-dir> <proteins.fasta> [python]
# Exits 77 (ctest SKIP) when python/pyarrow/numpy are unavailable.
#
# The fixture covers MORE precursors than the report identifies. Those extras
# are the whole reason the stage exists -- refinement cannot touch them, only a
# tuned model can -- so a fixture where everything matched would test nothing.
set -u
BIN="$1"; MODELS="$2"; FASTA="$3"; PY="${4:-python3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fail() { echo "FAIL: $*" >&2; exit 1; }
TMP=$(mktemp -d "${TMPDIR:-/tmp}/dlr-tune-e2e.XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT

"$PY" -c "import pyarrow, numpy" 2>/dev/null || { echo "SKIP: python with pyarrow+numpy needed" >&2; exit 77; }
"$PY" "$HERE/synth_report.py" "$FASTA" "$TMP/report.parquet" --precursors 1600 \
      --library "$TMP/library.tsv" || fail "synthetic fixture"
for head in rt ccs; do
  [ -s "$MODELS/peptdeep_${head}_dynamic.onnx" ] || fail "no stock model for $head in $MODELS"
done

# One invocation: tune both heads, re-predict, refine, write a library.
# -q_global/-q_protein 1 disable the gates whose columns a synthetic report does
# not carry; the precursor gate still applies.
"$BIN" -in "$TMP/library.tsv" -ids "$TMP/report.parquet" -out "$TMP/refined.tsv" \
   -q_global 1 -q_protein 1 \
   -tune -tune_models "$MODELS" -tune_out_models "$TMP/tuned" \
   -filter:rt_max_minutes 30 \
   -train:epochs 20 -train:warmup 2 -stop:min_epochs 20 -stop:patience 100 -machine:threads 2 \
   > "$TMP/run.log" 2>&1 || { cat "$TMP/run.log" >&2; fail "DIALibRefine -tune exited non-zero"; }

# The deliverable is a LIBRARY. This is the claim the merge added.
[ -s "$TMP/refined.tsv" ] || fail "no refined library written"
[ "$(wc -l < "$TMP/refined.tsv")" -gt 1 ] || fail "the refined library has no rows"
[ -s "$TMP/refined.tsv.refine.json" ] || fail "no refine provenance sidecar"
grep -q '"tool"' "$TMP/refined.tsv.refine.json" || fail "the sidecar names no tool"

# The write-back guarantees, unchanged, against the kept models.
for head in rt ccs; do
  stock="$MODELS/peptdeep_${head}_dynamic.onnx"
  tuned="$TMP/tuned/peptdeep_${head}_dynamic.onnx"
  [ -s "$tuned" ] || fail "$head: no tuned ONNX kept"
  [ -s "$tuned.tune.json" ] || fail "$head: no provenance sidecar"
  cmp -s "$stock" "$tuned" && fail "$head: the tuned ONNX is byte-identical to the stock one"
  [ "$(stat -c %s "$stock" 2>/dev/null || stat -f %z "$stock")" = "$(stat -c %s "$tuned" 2>/dev/null || stat -f %z "$tuned")" ] \
     || fail "$head: the tuned ONNX changed size -- write-back must only touch raw_data"
  "$PY" - "$tuned.tune.json" "$head" <<'PYEOF' || exit 1
import json, sys
p = json.load(open(sys.argv[1])); head = sys.argv[2]
c = p["course"]; ev = p["evaluation"]
def die(m): print("FAIL: " + head + ": " + m, file=sys.stderr); sys.exit(1)
if c["updates"] < 1: die("no optimizer updates")
if not c["param_l2_change"] > 0: die("parameters did not move")
key = "calibrated_sd"
s, t = ev["stock"]["val"][key], ev["tuned"]["val"][key]
if not (t < s): die(f"validation {key} did not improve: stock {s:.5f} tuned {t:.5f}")
if p["cohorts"]["test"] < 10 or p["cohorts"]["val"] < 10: die("cohorts too small: " + str(p["cohorts"]))
print(f"ok   {head}: val {key} {s:.4f} -> {t:.4f} in {c['epochs_run']} epochs, {c['updates']} updates, TEST {ev['stock']['test'][key]:.4f} -> {ev['tuned']['test'][key]:.4f}")
PYEOF
done

# And that the tuning actually REACHED the library. Not that the stage ran --
# that it re-predicted precursors. predictRetentionTimes returns what it could
# NOT do, so a caller reading it as a success count reports 0 when everything
# worked, and every assertion above still passes.
"$PY" - "$TMP/refined.tsv.refine.json" <<'PYEOF2' || exit 1
import json, sys
p = json.load(open(sys.argv[1]))
t = p.get("tune") or {}
def die(m): print("FAIL: " + m, file=sys.stderr); sys.exit(1)
if not t: die("the refine sidecar has no tune section")
for head in ("rt", "ccs"):
    h = t.get(head) or die(f"no {head} section in the tune provenance")
    if h.get("repredicted", 0) < 1:
        die(f"{head}: re-predicted {h.get('repredicted')} precursors -- the stage ran and changed nothing")
    if not h.get("model_sha256") or h["model_sha256"] == h.get("stock_sha256"):
        die(f"{head}: the tuned model hash equals the stock one")
print("ok   re-predicted rt=%d ccs=%d precursors" % (t["rt"]["repredicted"], t["ccs"]["repredicted"]))
PYEOF2
echo "PASSED"
