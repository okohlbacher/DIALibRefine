#!/usr/bin/env bash
#
# Turn fine-tuned AlphaPeptDeep checkpoints into the ONNX triplet DIALibGen and
# ODIA consume. Generalises ODIA's export_finetuned_rt.sh: any of the three
# heads may be replaced, the rest are staged from the stock checkpoints.
#
#   export_finetuned.sh <outdir> [--rt rt.pth] [--ccs ccs.pth] [--ms2 ms2.pth]
#
# The exporter itself is OpenMS's own script, unmodified -- it produces the
# tensor names ODIA expects (input_sequences, mod_x -> rt_pred / ccs_pred).
set -euo pipefail

out="${1:?usage: export_finetuned.sh <outdir> [--rt pth] [--ccs pth] [--ms2 pth]}"; shift
rt="" ccs="" ms2=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --rt)  rt="$2";  shift 2 ;;
    --ccs) ccs="$2"; shift 2 ;;
    --ms2) ms2="$2"; shift 2 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done

env_py="${ODIA_FINETUNE_PYTHON:-/scratch/kohlbach/odia/rtfinetune/env/bin/python}"
src="${ODIA_OPENMS_SOURCE:?set ODIA_OPENMS_SOURCE to the OpenMS source tree (source ODIA/scripts/env.sh)}"
exporter="${src}/tools/scripts/export_peptdeep_models_to_onnx.py"
stock="${ODIA_PEPTDEEP_MODELS:?set ODIA_PEPTDEEP_MODELS to the unpacked generic/ checkpoint dir}"

[[ -x "${env_py}" ]]         || { echo "no fine-tuning python at ${env_py}" >&2; exit 1; }
[[ -f "${exporter}" ]]       || { echo "no exporter at ${exporter}" >&2; exit 1; }
[[ -f "${stock}/ms2.pth" ]]  || { echo "no stock checkpoints at ${stock}" >&2; exit 1; }

stage="$(mktemp -d)"; trap 'rm -rf "${stage}"' EXIT
for head in rt ccs ms2; do
  cp "${stock}/${head}.pth" "${stage}/${head}.pth"
done
[[ -n "${rt}"  ]] && cp "${rt}"  "${stage}/rt.pth"  && echo "==> rt:  ${rt}"
[[ -n "${ccs}" ]] && cp "${ccs}" "${stage}/ccs.pth" && echo "==> ccs: ${ccs}"
[[ -n "${ms2}" ]] && cp "${ms2}" "${stage}/ms2.pth" && echo "==> ms2: ${ms2}"

mkdir -p "${out}"
"${env_py}" "${exporter}" --pretrained-dir "${stage}" --out-dir "${out}"

# Record what actually got exported, and refuse to call a stock export "tuned".
"${env_py}" - "${out}" "${stock}" "${rt}" "${ccs}" "${ms2}" <<'PYEOF'
import hashlib, json, os, sys
out, stock, rt, ccs, ms2 = sys.argv[1:6]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
rec = {}
for head, given in (("rt", rt), ("ccs", ccs), ("ms2", ms2)):
    onnx = os.path.join(out, f"peptdeep_{head}_dynamic.onnx")
    rec[head] = {"onnx": onnx, "onnx_sha256": sha(onnx), "source_pth": os.path.abspath(given) if given else f"{stock}/{head}.pth (stock)"}
json.dump(rec, open(os.path.join(out, "export_manifest.json"), "w"), indent=2)
for h, r in rec.items():
    print(f"  {h}: {r['onnx_sha256'][:16]}...  <- {r['source_pth']}")
PYEOF
