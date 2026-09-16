# Tests that need no data at all. The join key is the one piece of this tool
# that can fail silently, so it is the one thing tested without any fixture.
add_executable(dlr_canonical_modseq canonical_modseq.cpp)
target_link_libraries(dlr_canonical_modseq PRIVATE odia_refine)
add_test(NAME canonical_modseq COMMAND dlr_canonical_modseq)

if(TARGET DIALibraryRefiner)
  add_test(NAME DIALibraryRefiner_help COMMAND DIALibraryRefiner --help)
  add_test(NAME DIALibraryRefiner_write_ini
           COMMAND DIALibraryRefiner -write_ini "${CMAKE_CURRENT_BINARY_DIR}/DIALibraryRefiner.ini")
  add_test(NAME config_write_effective
           COMMAND DIALibraryRefiner -in x.parquet -ids y.parquet -out z.parquet
                   -write_config "${CMAKE_CURRENT_BINARY_DIR}/effective.json")
  set_tests_properties(config_write_effective PROPERTIES
    PASS_REGULAR_EXPRESSION "effective config")

  # An unknown config key is a typo that would otherwise become a default.
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/typo.json" "{\"q_precursour\": 0.05}\n")
  add_test(NAME config_rejects_unknown_key
           COMMAND DIALibraryRefiner -in x.parquet -ids y.parquet -out z.parquet
                   -config "${CMAKE_CURRENT_BINARY_DIR}/typo.json")
  set_tests_properties(config_rejects_unknown_key PROPERTIES WILL_FAIL TRUE)

  # Intensity replacement is measured as a wash by the paper; it must refuse
  # loudly rather than run as a no-op.
  add_test(NAME write_intensity_is_refused
           COMMAND DIALibraryRefiner -in x.parquet -ids y.parquet -out z.parquet -write_intensity)
  set_tests_properties(write_intensity_is_refused PROPERTIES WILL_FAIL TRUE)
endif()

# The fine-tuner rests on two things that are checked against the stock ONNX
# models: the libtorch transcription predicts what ONNX Runtime predicts, and
# load -> store is byte-identical. Needs the stock models: point
# DLR_PEPTDEEP_ONNX_DIR at a directory holding peptdeep_rt_dynamic.onnx and
# peptdeep_ccs_dynamic.onnx (DIALibGen ships them in its share directory).
if(TARGET odia_tune)
  add_executable(dlr_tune_parity tune_parity.cpp)
  target_link_libraries(dlr_tune_parity PRIVATE odia_tune)
  if(DLR_LIBTORCH_DIR)
    set_property(TARGET dlr_tune_parity APPEND PROPERTY BUILD_RPATH "${DLR_LIBTORCH_DIR}/lib")
  endif()
  set(DLR_PEPTDEEP_ONNX_DIR "" CACHE PATH "Directory with the stock peptdeep_{rt,ccs}_dynamic.onnx for the parity test")
  if(EXISTS "${DLR_PEPTDEEP_ONNX_DIR}/peptdeep_rt_dynamic.onnx" AND EXISTS "${DLR_PEPTDEEP_ONNX_DIR}/peptdeep_ccs_dynamic.onnx")
    add_test(NAME tune_parity COMMAND dlr_tune_parity
             "${DLR_PEPTDEEP_ONNX_DIR}/peptdeep_rt_dynamic.onnx"
             "${DLR_PEPTDEEP_ONNX_DIR}/peptdeep_ccs_dynamic.onnx")
  else()
    message(STATUS "tune_parity test not registered: set DLR_PEPTDEEP_ONNX_DIR to the stock models")
  endif()
  if(TARGET DIALibTune)
    add_test(NAME DIALibTune_help COMMAND DIALibTune --help)
  endif()
endif()

# The standalone gates: bare environment, own version, update check off,
# descriptor export. Cheap, and each has shipped broken past a green build in
# a sibling project.
if(TARGET DIALibraryRefiner)
  if(TARGET DIALibTune)
    set(_dlr_tune_bin $<TARGET_FILE:DIALibTune>)
  else()
    set(_dlr_tune_bin "")
  endif()
  add_test(NAME standalone
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/standalone_test.sh
                   $<TARGET_FILE:DIALibraryRefiner> "${_dlr_tune_bin}" ${PROJECT_VERSION})
  set_tests_properties(standalone PROPERTIES SKIP_RETURN_CODE 77)
endif()

# End to end: a synthetic report, both heads, bounded (~1 min on 2 threads).
# Needs the stock models and a python with pyarrow + numpy; skipped otherwise.
if(TARGET DIALibTune)
  set(DLR_EXAMPLE_FASTA "-" CACHE FILEPATH "A protein FASTA for the synthetic-report test; - = synthetic proteins")
  if(NOT DLR_EXAMPLE_FASTA)
    set(DLR_EXAMPLE_FASTA "-")
  endif()
  find_program(DLR_PYTHON NAMES python3 python)
  if(DLR_PYTHON AND EXISTS "${DLR_PEPTDEEP_ONNX_DIR}/peptdeep_rt_dynamic.onnx")
    add_test(NAME tune_e2e
             COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tune_e2e.sh
                     $<TARGET_FILE:DIALibTune> "${DLR_PEPTDEEP_ONNX_DIR}" "${DLR_EXAMPLE_FASTA}" "${DLR_PYTHON}")
    set_tests_properties(tune_e2e PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 900)
  else()
    message(STATUS "tune_e2e test not registered: needs python and DLR_PEPTDEEP_ONNX_DIR")
  endif()
endif()
