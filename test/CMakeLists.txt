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
