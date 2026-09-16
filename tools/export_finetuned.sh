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

# Validate every requested checkpoint BEFORE staging anything. A missing tuned
# file must be an error here, not a silent fall-through to stock.
for given in "${rt}" "${ccs}" "${ms2}"; do
  [[ -z "${given}" || -s "${given}" ]] || { echo "checkpoint not found or empty: ${given}" >&2; exit 1; }
done

stage="$(mktemp -d)"; trap 'rm -rf "${stage}"' EXIT
for head in rt ccs ms2; do
  cp -- "${stock}/${head}.pth" "${stage}/${head}.pth"
done
# Plain if-blocks, not `[[ -n x ]] && cp && echo`: under `set -e` a failing
# command that is not LAST in an && list does not abort, so a failed copy of the
# tuned checkpoint would have exported STOCK and labelled it tuned.
if [[ -n "${rt}"  ]]; then cp -- "${rt}"  "${stage}/rt.pth";  echo "==> rt:  ${rt}";  fi
if [[ -n "${ccs}" ]]; then cp -- "${ccs}" "${stage}/ccs.pth"; echo "==> ccs: ${ccs}"; fi
if [[ -n "${ms2}" ]]; then cp -- "${ms2}" "${stage}/ms2.pth"; echo "==> ms2: ${ms2}"; fi

# The output directory must be EMPTY: the exporter's behaviour on pre-existing
# files is not established, and a stale ONNX beside a fresh manifest would be
# provenance that lies.
if [[ -d "${out}" ]] && [[ -n "$(ls -A "${out}")" ]]; then
  echo "output directory ${out} is not empty; refusing to export into it" >&2; exit 1
fi
mkdir -p "${out}"
# Hash what was actually staged, so the manifest records the bytes the exporter read.
for head in rt ccs ms2; do sha256sum -- "${stage}/${head}.pth" | cut -d' ' -f1 > "${stage}/${head}.staged.sha256"; done
"${env_py}" "${exporter}" --pretrained-dir "${stage}" --out-dir "${out}"

# Record what actually got exported. The manifest carries the hash of the STAGED
# checkpoint bytes the exporter read (not just the path that was requested), the
# stock hash for comparison, and refuses to call an export "tuned" when the
# staged bytes equal stock.
"${env_py}" - "${out}" "${stock}" "${stage}" "${rt}" "${ccs}" "${ms2}" <<'PYEOF'
import hashlib, json, os, sys
out, stock, stage, rt, ccs, ms2 = sys.argv[1:7]
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
rec, bad = {}, []
for head, given in (("rt", rt), ("ccs", ccs), ("ms2", ms2)):
    onnx = os.path.join(out, f"peptdeep_{head}_dynamic.onnx")
    staged = open(os.path.join(stage, f"{head}.staged.sha256")).read().strip()
    stock_sha = sha(os.path.join(stock, f"{head}.pth"))
    tuned = bool(given)
    if tuned and staged == stock_sha:
        bad.append(head)
    rec[head] = {"onnx": onnx, "onnx_sha256": sha(onnx),
                 "requested_pth": os.path.abspath(given) if given else None,
                 "staged_pth_sha256": staged, "stock_pth_sha256": stock_sha,
                 "tuned": tuned and staged != stock_sha}
json.dump(rec, open(os.path.join(out, "export_manifest.json"), "w"), indent=2)
for h, r in rec.items():
    print(f"  {h}: onnx {r['onnx_sha256'][:16]}...  staged {r['staged_pth_sha256'][:16]}...  "
          f"{'TUNED' if r['tuned'] else 'stock'}  <- {r['requested_pth'] or 'stock'}")
if bad:
    sys.exit(f"REFUSING: {bad} were requested as tuned but the staged bytes equal the stock checkpoint")
PYEOF

# Checkpoint -> ONNX numerical parity, on a fixed varied panel: the exported graph
# must predict what the checkpoint predicts, or the export -- not the model -- is
# what a downstream measurement would be scoring.
"${env_py}" - "${out}" "${stage}" "${rt}" "${ccs}" <<'PYEOF'
import json, os, sys
import numpy as np, pandas as pd, onnxruntime as ort, torch
out, stage, rt, ccs = sys.argv[1:5]
from peptdeep.pretrained_models import ModelManager
mgr = ModelManager(mask_modloss=False, device="cpu")
mgr.load_installed_models()
panel = pd.DataFrame({"sequence": ["PEPTIDEK", "LGGNEQVTR", "SAMPLECYSTEINER", "AAAAAAAAVPSAGPAGPAPTSAAGR", "LFLQFGAQGSPFLK", "YICDNQDTISSK"],
                      "mods": ["", "", "Carbamidomethyl@C", "", "", "Carbamidomethyl@C"],
                      "mod_sites": ["", "", "7", "", "", "3"], "charge": [2, 2, 2, 3, 2, 2]})
panel["nAA"] = panel["sequence"].str.len()
report = {}
for head, given in (("rt", rt), ("ccs", ccs)):
    if not given:
        continue
    model = mgr.rt_model if head == "rt" else mgr.ccs_model
    model.load(os.path.join(stage, f"{head}.pth"))
    ref = model.predict(panel.copy())[f"{head}_pred"].values.astype(np.float32)
    sess = ort.InferenceSession(os.path.join(out, f"peptdeep_{head}_dynamic.onnx"), providers=["CPUExecutionProvider"])
    from peptdeep.model.featurize import get_batch_aa_indices, get_batch_mod_feature
    got = []
    for _, r in panel.iterrows():
        one = pd.DataFrame([r])
        aa = get_batch_aa_indices([r["sequence"]]).astype(np.int64)
        mod = get_batch_mod_feature(one).astype(np.float32)
        feeds = {"input_sequences" if head == "rt" else "aa_indices": aa, "mod_x": mod}
        if head == "ccs":
            # The graph is the bare nn.Module, BELOW peptdeep's charge_factor scaling, so it
            # takes charge * 0.1 -- exactly what ODIA's PeptDeepEncoder packs (CHARGE_SCALE).
            feeds["charges"] = np.array([[r["charge"] * 0.1]], dtype=np.float32)
        got.append(float(sess.run(None, feeds)[0].ravel()[0]))
    got = np.array(got, np.float32)
    diff = float(np.max(np.abs(got - ref)))
    report[head] = {"max_abs_diff": diff, "checkpoint": ref.tolist(), "onnx": got.tolist()}
    print(f"  parity {head}: max |onnx - checkpoint| = {diff:.3e} over {len(panel)} peptides")
m = json.load(open(os.path.join(out, "export_manifest.json"))); m["parity"] = report
json.dump(m, open(os.path.join(out, "export_manifest.json"), "w"), indent=2)
PYEOF
