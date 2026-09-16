# Backlog

Things that are known, not done, and not forgotten. Each carries the reason it
is here rather than done.

- **Cross-run transfer of tuned models.** Every number is same-run. The
  measurement is one more DIA-NN arm: a library from models tuned on run A,
  searched on run B of the same method, at matched entrapment budget.
- **Provenance inside the tuned ONNX** (`metadata_props`). Deliberately absent
  so the file stays byte-for-byte the stock layout; would need a protobuf
  writer for one message type. The sidecar is the record.
- **Windows.** DIALibGen's `windows.yml` (OpenMS from source, SignPath) is the
  template; libtorch win-64 is a new dependency there.
- **CUDA bundle.** Built from source only (`DLR_LIBTORCH_DIR`); pytorch.org's
  CUDA zips ship only cuDNN's loader shim, so a bundle would have to carry the
  cuDNN 9 library set (~700 MB) or ship with `-machine:no_cudnn` as the default.
- **Docs generated from the binaries** (`--helphelp`, `-write_config`) with a
  drift job. DIALibGen has the pattern; the reference tables here are still
  hand-written.
- **`tools/finetune_rt.py` / `finetune_ccs.py`** are superseded by
  `tools/finetune.py` and can go once `build_unimod_index` /
  `parse_modified_sequence` live in their own module.
- **A Python-on-GPU control.** The GPU comparison in `docs/results.md` is C++
  on GPU vs C++ and Python on CPU; the Python environment used here is a CPU
  torch. Not needed for any claim made, listed for completeness.
- **`odia_refine` / `odia_tune` as an installed cmake package with a fetched
  DIALibGen** — blocked upstream: DIALibGen's exported targets bake an
  absolute ONNX Runtime path.
