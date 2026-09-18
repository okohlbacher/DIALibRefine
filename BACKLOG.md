# Backlog

Things that are known, not done, and not forgotten. Each carries the reason it
is here rather than done.

- **Cross-run transfer of tuned models.** Every number is same-run. The
  measurement is one more DIA-NN arm: a library from models tuned on run A,
  searched on run B of the same method, at matched entrapment budget.
- **Provenance inside the tuned ONNX** (`metadata_props`). Deliberately absent
  so the file stays byte-for-byte the stock layout; would need a protobuf
  writer for one message type. The sidecar is the record.
- **`-tune` on linux-arm64.** conda-forge's aarch64 libtorch 2.10.0
  segfaults in `at::_ops::lstm_input::call` (single-threaded, without oneDNN
  alike) and 2.11+ do not co-install with openms 3.5.0; pytorch.org has no
  aarch64 zip. The pip `torch` wheel's libtorch through `DLR_LIBTORCH_DIR` is
  the untested route (its auditwheel-renamed `libgomp` would sit beside
  conda's — two OpenMP runtimes). Until then the arm64 Linux bundle is built without
  `DLR_BUILD_FINETUNE` and has no `-tune` flag.
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
