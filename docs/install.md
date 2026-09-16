# Installing

## Binaries

Every tagged release carries `DIALibRefine-<platform>.tar.gz` for linux-x64,
linux-arm64, macos-arm64 and macos-x64, and a `.dmg` for the two macOS
platforms. Each holds `bin/DIALibraryRefiner`, `bin/DIALibTune`, the shared
libraries they need (`lib/`) and OpenMS's data tables (`share/OpenMS/`); they
run from wherever they are unpacked, with no environment set.

The macOS builds are signed with a Developer ID and notarized; the `.dmg` is
stapled, so it opens without a network check. Prefer it over the tarball on
macOS — a tarball cannot carry a notarization ticket, and Gatekeeper then
inspects every library on first launch.

`DIALibTune` in these bundles is the **CPU** build (libtorch from conda-forge).
A CUDA build is not released; see below.

## From source

Requirements:

- a C++23 compiler, CMake ≥ 3.21;
- OpenMS ≥ 3.5 (bioconda `openms=3.5.0` is what CI uses), with the Boost
  headers and Qt6 its cmake package asks for;
- Apache Arrow/Parquet (the version OpenMS pins), ONNX Runtime (C++ headers
  and library), nlohmann/json;
- for `DIALibTune`: libtorch, **CXX11 ABI** (the conda-forge `libtorch`
  package, or pytorch.org's `libtorch-shared-with-deps` zip, which has been
  CXX11 since 2.6). The pre-CXX11 zip cannot link against OpenMS and Arrow and
  is refused at configure time.

[DIALibGen](https://github.com/okohlbacher/DIALibGen) is fetched at the pinned
tag (`DLR_DIALIBGEN_TAG`, v0.10.0) unless an installed one is found.

The conda recipe CI builds with, in one line:

```bash
micromamba create -n dialibrefine -c conda-forge -c bioconda openms=3.5.0 onnxruntime-cpp \
  'libtorch=2.10.0=cpu*' libparquet libarrow-dataset libarrow-acero nlohmann_json xerces-c \
  libsvm eigen zlib bzip2 libzip hdf5 libcurl libboost-devel glpk coin-or-cbc coin-or-utils \
  qt6-main cmake ninja cxx-compiler python numpy pyarrow
```

then

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" -DQT_HOST_PATH="$CONDA_PREFIX" \
  -DDLR_BUILD_FINETUNE=ON -DTorch_DIR="$CONDA_PREFIX/share/cmake/Torch" \
  -DDLR_PEPTDEEP_ONNX_DIR=/path/to/models
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build --prefix /where/you/want/it
```

`DLR_PEPTDEEP_ONNX_DIR` points at the three stock `peptdeep_*_dynamic.onnx`
files (DIALibGen's share directory, or OpenMS's `share/OpenMS/models`); it
enables the parity and end-to-end tests. Without libtorch, leave
`DLR_BUILD_FINETUNE` off and only `DIALibraryRefiner` is built.

### Options

| option | default | |
|---|---|---|
| `DLR_BUILD_TOOL` | ON | build the executables |
| `DLR_BUILD_TESTS` | ON | build the test suite |
| `DLR_BUILD_FINETUNE` | OFF | build `DIALibTune` and `odia_tune` (needs libtorch) |
| `DLR_LIBTORCH_DIR` | — | link an unpacked libtorch zip directly, bypassing `TorchConfig.cmake` (see CUDA) |
| `DLR_INSTALL_TOOLS` | ON | install the executables |
| `DLR_INSTALL` | OFF | install the libraries and headers as a cmake package (needs an *installed* DIALibGen) |
| `DLR_DIALIBGEN_TAG` | v0.10.0 | DIALibGen tag to fetch |
| `FETCHCONTENT_SOURCE_DIR_DIALIBGEN` | — | build against a local DIALibGen checkout instead |

### CUDA

A GPU build links pytorch.org's CUDA zip, e.g.
`libtorch-shared-with-deps-2.14.0+cu130.zip` (2 GB; ships the CUDA runtime
libraries it needs, but not the driver). Point `DLR_LIBTORCH_DIR` at the
unpacked directory rather than `Torch_DIR`:

```bash
cmake -S . -B build-cuda -DDLR_BUILD_FINETUNE=ON -DDLR_LIBTORCH_DIR=/opt/libtorch-cu130/libtorch ...
```

`TorchConfig.cmake` from a CUDA zip enables the CUDA language and wants a
toolkit that matches the zip, although nothing in this project compiles CUDA
code; the direct link needs neither a toolkit nor `nvcc`. The resulting
binary starts on a machine without a GPU driver and trains on the CPU there;
`-machine:device cuda:0` selects the GPU and fails with a clear message when
CUDA is unavailable rather than silently falling back.

## Checking an install

```bash
DIALibraryRefiner --help      # prints "Version: <this tool's version> (OpenMS <version>)"
DIALibTune --help
```

`test/standalone_test.sh <DIALibraryRefiner> <DIALibTune> <version>` is the
gate CI runs against the installed tree: both tools start in an empty
environment, report their own version, keep OpenMS's update check off, and
export their parameter descriptions.
