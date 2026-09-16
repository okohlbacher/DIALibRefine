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
option(DLR_BUILD_TOOL  "Build the DIALibraryRefiner executable" ${DLR_IS_TOP_LEVEL})
option(DLR_BUILD_TESTS "Build the test suite" ${DLR_IS_TOP_LEVEL})
# Install rules are conditional on DIALibGen being an INSTALLED package. When it
# is fetched instead, its odia_library target belongs to no export set, and
# exporting a target that links it is a CMake error rather than a broken install
# discovered later.
option(DLR_INSTALL     "Generate install rules" OFF)

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
set(DLR_DIALIBGEN_TAG "main" CACHE STRING
    "Tag, branch or commit of DIALibGen to build against")

find_package(DIALibraryGenerator QUIET)
if(DIALibraryGenerator_FOUND)
  message(STATUS "DIALibGen: ${DIALibraryGenerator_DIR} (installed)")
  if(DLR_IS_TOP_LEVEL)
    set(DLR_INSTALL ON CACHE BOOL "" FORCE)
  endif()
else()
  include(FetchContent)
  FetchContent_Declare(DIALibraryGenerator
    GIT_REPOSITORY https://github.com/okohlbacher/DIALibGen.git
    GIT_TAG        ${DLR_DIALIBGEN_TAG}
    GIT_SHALLOW    TRUE)
  set(DIALIBGEN_BUILD_TOOL  OFF CACHE BOOL "" FORCE)
  set(DIALIBGEN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(DIALIBGEN_INSTALL     OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(DIALibraryGenerator)
  message(STATUS "DIALibGen: ${dialibrarygenerator_SOURCE_DIR} (fetched, ${DLR_DIALIBGEN_TAG})")
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
target_link_libraries(odia_refine PUBLIC DIALibraryGenerator::odia_library)
set_target_properties(odia_refine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DLR_BUILD_TOOL)
  add_executable(DIALibraryRefiner src/DIALibraryRefiner.cpp)
  target_link_libraries(DIALibraryRefiner PRIVATE
    odia_refine nlohmann_json::nlohmann_json)
endif()

if(DLR_INSTALL)
  include(CMakePackageConfigHelpers)
  install(TARGETS odia_refine EXPORT DIALibRefineTargets
          ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  if(DLR_BUILD_TOOL)
    install(TARGETS DIALibraryRefiner RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
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
