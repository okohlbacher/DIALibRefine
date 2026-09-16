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
