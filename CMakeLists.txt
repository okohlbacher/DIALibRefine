cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

project(DIALibRefine
  VERSION 0.1.0
  DESCRIPTION "Reference-based refinement of DIA spectral libraries"
  HOMEPAGE_URL "https://github.com/okohlbacher/DIALibRefine"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  set(DLR_IS_TOP_LEVEL ON)
else()
  set(DLR_IS_TOP_LEVEL OFF)
endif()
option(DLR_BUILD_TOOL  "Build the DIALibRefine executable" ${DLR_IS_TOP_LEVEL})
option(DLR_BUILD_TESTS "Build the test suite" ${DLR_IS_TOP_LEVEL})
# Install rules are conditional on DIALibGen being an INSTALLED package. When it
# is fetched instead, its odia_library target belongs to no export set, and
# exporting a target that links it is a CMake error rather than a broken install
# discovered later.
option(DLR_INSTALL     "Generate install rules for the LIBRARIES and headers (needs an installed DIALibGen)" OFF)
option(DLR_INSTALL_TOOLS "Install the executables (works with a fetched DIALibGen too)" ON)

include(GNUInstallDirs)

# ---------------------------------------------------------------------------
# DIALibGen provides the library data model and the DIA-NN Parquet/TSV reader
# and writer. Refinement is the second half of the same job, so it consumes the
# first half rather than duplicating it.
#
# Note what is NOT here: mzPeak, SQLite and the extraction stack. This tool
# reads a library and a RESULTS TABLE, never the raw run -- which is what keeps
# it the same shape as DIALibGen rather than ODIA with a different main().
# ---------------------------------------------------------------------------
set(DLR_DIALIBGEN_TAG "v0.10.0" CACHE STRING
    "Tag, branch or commit of DIALibGen to build against")

find_package(DIALibGen QUIET)
if(DIALibGen_FOUND)
  message(STATUS "DIALibGen: ${DIALibGen_DIR} (installed)")
  if(DLR_IS_TOP_LEVEL)
    set(DLR_INSTALL ON CACHE BOOL "" FORCE)
  endif()
else()
  include(FetchContent)
  FetchContent_Declare(DIALibGen
    GIT_REPOSITORY https://github.com/okohlbacher/DIALibGen.git
    GIT_TAG        ${DLR_DIALIBGEN_TAG}
    GIT_SHALLOW    TRUE)
  set(DIALIBGEN_BUILD_TOOL  OFF CACHE BOOL "" FORCE)
  set(DIALIBGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(DIALIBGEN_INSTALL     OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(DIALibGen)
  message(STATUS "DIALibGen: ${dialibgen_SOURCE_DIR} (fetched, ${DLR_DIALIBGEN_TAG})")
endif()

find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
  find_path(NLOHMANN_JSON_INCLUDE_DIR NAMES nlohmann/json.hpp)
  if(NOT NLOHMANN_JSON_INCLUDE_DIR)
    message(FATAL_ERROR "nlohmann/json not found. Install nlohmann_json, or set "
                        "NLOHMANN_JSON_INCLUDE_DIR.")
  endif()
  add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
  set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NLOHMANN_JSON_INCLUDE_DIR}")
endif()

add_library(odia_refine STATIC src/refine/LibraryRefiner.cpp)
add_library(DIALibRefine::odia_refine ALIAS odia_refine)
target_include_directories(odia_refine PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(odia_refine PUBLIC DIALibGen::odia_library)
set_target_properties(odia_refine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DLR_BUILD_TOOL)
  add_executable(DIALibRefine src/DIALibRefine.cpp)
  target_link_libraries(DIALibRefine PRIVATE
    odia_refine nlohmann_json::nlohmann_json)
  target_compile_definitions(DIALibRefine PRIVATE DLR_VERSION="${PROJECT_VERSION}")
  if(DLR_INSTALL_TOOLS)
    install(TARGETS DIALibRefine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()

# ---------------------------------------------------------------------------
# Fine-tuning (optional). The RT and CCS models are re-trained on one run with
# libtorch -- ONNX Runtime cannot train (its training API was deprecated at
# 1.20) and the alternative was a Python dependency. The model is transcribed
# from peptdeep, weights are loaded from and written back into the stock ONNX
# file, so DIALibGen consumes the result unchanged.
#
# libtorch is found through find_package(Torch): point Torch_DIR (or
# CMAKE_PREFIX_PATH) at <libtorch>/share/cmake/Torch. It must be the CXX11-ABI
# build (pytorch.org's "libtorch-cxx11-abi-shared-with-deps" zip, or conda's
# libtorch); the pre-cxx11 zip cannot link against OpenMS and Arrow and is
# refused below rather than failing at link time with 300 undefined symbols.
# ---------------------------------------------------------------------------
option(DLR_BUILD_FINETUNE "Build DIALibTune and the fine-tuning library (needs libtorch)" OFF)
set(DLR_LIBTORCH_DIR "" CACHE PATH
    "Link an unpacked libtorch zip DIRECTLY (its include/ and lib/), bypassing TorchConfig.cmake. \
Use this for a CUDA zip: TorchConfig enables the CUDA language and wants a toolkit that matches \
the zip, although nothing here compiles CUDA code. Empty = find_package(Torch).")
if(DLR_BUILD_FINETUNE)
  if(DLR_LIBTORCH_DIR)
    # The zip's own layout: lib/libtorch.so pulls libtorch_cpu.so and, in a
    # CUDA build, libtorch_cuda.so. Both must be linked --no-as-needed or the
    # backends never register (TorchConfig does the same for libtorch.so).
    set(_dlr_tl "${DLR_LIBTORCH_DIR}/lib")
    foreach(_lib torch torch_cpu c10)
      if(NOT EXISTS "${_dlr_tl}/lib${_lib}.so" AND NOT EXISTS "${_dlr_tl}/lib${_lib}.dylib")
        message(FATAL_ERROR "DLR_LIBTORCH_DIR=${DLR_LIBTORCH_DIR} has no lib/lib${_lib}.so")
      endif()
    endforeach()
    add_library(dlr_torch INTERFACE)
    target_include_directories(dlr_torch INTERFACE
      "${DLR_LIBTORCH_DIR}/include" "${DLR_LIBTORCH_DIR}/include/torch/csrc/api/include")
    target_compile_definitions(dlr_torch INTERFACE _GLIBCXX_USE_CXX11_ABI=1)
    set(_dlr_torch_libs "${_dlr_tl}/libtorch.so" "${_dlr_tl}/libtorch_cpu.so")
    if(EXISTS "${_dlr_tl}/libtorch_cuda.so")
      list(APPEND _dlr_torch_libs "${_dlr_tl}/libtorch_cuda.so" "${_dlr_tl}/libc10_cuda.so")
      set(_dlr_torch_cuda "yes")
    else()
      set(_dlr_torch_cuda "no")
    endif()
    if(APPLE)
      target_link_libraries(dlr_torch INTERFACE ${_dlr_torch_libs} "${_dlr_tl}/libc10.dylib")
    else()
      target_link_libraries(dlr_torch INTERFACE
        -Wl,--no-as-needed ${_dlr_torch_libs} -Wl,--as-needed "${_dlr_tl}/libc10.so")
    endif()
    set(TORCH_LIBRARIES dlr_torch)
    set(TORCH_CXX_FLAGS "")
    file(READ "${DLR_LIBTORCH_DIR}/build-version" _dlr_torch_ver)
    string(STRIP "${_dlr_torch_ver}" _dlr_torch_ver)
    message(STATUS "libtorch: ${DLR_LIBTORCH_DIR} linked directly (${_dlr_torch_ver}, CUDA libs: ${_dlr_torch_cuda})")
  else()
    find_package(Torch REQUIRED)
    string(FIND "${TORCH_CXX_FLAGS}" "_GLIBCXX_USE_CXX11_ABI=0" _dlr_pre_cxx11)
    if(NOT _dlr_pre_cxx11 EQUAL -1)
      message(FATAL_ERROR "libtorch at ${Torch_DIR} is the pre-CXX11-ABI build "
                          "(_GLIBCXX_USE_CXX11_ABI=0); OpenMS and Arrow need the "
                          "CXX11 ABI. Use the libtorch-cxx11-abi-shared-with-deps "
                          "zip or a conda libtorch.")
    endif()
    message(STATUS "libtorch: ${Torch_DIR} (${Torch_VERSION}, CUDA: ${TORCH_CUDA_LIBRARIES})")
  endif()

  add_library(odia_tune STATIC
    src/finetune/OnnxWeights.cpp
    src/finetune/Trainer.cpp)
  add_library(DIALibRefine::odia_tune ALIAS odia_tune)
  target_include_directories(odia_tune PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
  target_link_libraries(odia_tune PUBLIC
    DIALibGen::odia_library ${TORCH_LIBRARIES} nlohmann_json::nlohmann_json)
  separate_arguments(_dlr_torch_flags UNIX_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(odia_tune PUBLIC ${_dlr_torch_flags})
  set_target_properties(odia_tune PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(DLR_BUILD_TOOL)
    add_executable(DIALibTune src/DIALibTune.cpp)
    target_link_libraries(DIALibTune PRIVATE odia_tune)
    target_compile_definitions(DIALibTune PRIVATE DLR_VERSION="${PROJECT_VERSION}")
    if(DLR_INSTALL_TOOLS)
      install(TARGETS DIALibTune RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    # A relocatable binary: libtorch's .so live next to it in ../lib when packaged.
    set_target_properties(DIALibTune PROPERTIES
      INSTALL_RPATH "$ORIGIN/../lib;$ORIGIN"
      BUILD_WITH_INSTALL_RPATH OFF)
    if(DLR_LIBTORCH_DIR)
      set_property(TARGET DIALibTune APPEND PROPERTY BUILD_RPATH "${DLR_LIBTORCH_DIR}/lib")
    endif()
  endif()
endif()

if(DLR_INSTALL)
  include(CMakePackageConfigHelpers)
  install(TARGETS odia_refine EXPORT DIALibRefineTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  if(DLR_BUILD_FINETUNE)
    install(TARGETS odia_tune EXPORT DIALibRefineTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  install(DIRECTORY include/odia DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  install(EXPORT DIALibRefineTargets
          FILE DIALibRefineTargets.cmake
          NAMESPACE DIALibRefine::
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIALibRefine)
  configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/DIALibRefineConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/DIALibRefineConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIALibRefine)
  write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/DIALibRefineConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
  install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/DIALibRefineConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/DIALibRefineConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/DIALibRefine)
endif()

if(DLR_BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()
