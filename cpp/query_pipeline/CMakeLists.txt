cmake_minimum_required(VERSION 3.18)
project(resrank_kernels_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED ENV{CONDA_PREFIX})
  message(FATAL_ERROR "CONDA_PREFIX is not set. Please 'conda activate <env>' first.")
endif()

# -----------------------------
# CRITICAL: stop Torch/Caffe2 from pulling system MKLConfig.cmake
# -----------------------------
set(CMAKE_DISABLE_FIND_PACKAGE_MKL ON CACHE BOOL "" FORCE)
set(MKL_FOUND OFF CACHE BOOL "" FORCE)

set(CMAKE_FIND_USE_PACKAGE_REGISTRY OFF)
set(CMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY OFF)
set(CMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY ON)
set(CMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY ON)

list(APPEND CMAKE_IGNORE_PATH
  "/usr/local"
  "/usr/local/lib"
  "/usr/local/lib/cmake"
)

# -----------------------------
# Torch
# -----------------------------
find_package(Torch REQUIRED)
message(STATUS "TORCH_LIBRARIES=${TORCH_LIBRARIES}")
message(STATUS "TORCH_CXX_FLAGS=${TORCH_CXX_FLAGS}")
message(STATUS "TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}")

# -----------------------------
# ONLY target: enum+centroid stage benchmark (link torch to mimic real env)
# -----------------------------
add_executable(bench_enum_centroid_stage_only_v6_8_aligned
  bench_enum_centroid_stage_only_v6_8_aligned.cpp
  resrank_kernels.cpp
  torch_scorer.cpp
)

# Torch ABI match (keep if your torch needs it)
target_compile_definitions(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE _GLIBCXX_USE_CXX11_ABI=0)

# Torch flags
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE ${TORCH_CXX_FLAGS_LIST})

# Your kernels need AVX2/FMA
target_compile_options(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE -mavx2 -mfma)

# Link torch
if(TARGET Torch::Torch)
  target_link_libraries(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE Torch::Torch)
else()
  target_link_libraries(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE ${TORCH_LIBRARIES})
endif()

# OpenMP (GNU) - mimic torch+gomp environment
find_package(OpenMP REQUIRED)
target_link_libraries(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE OpenMP::OpenMP_CXX)

# RPATH: conda + torch
set(CONDA_LIB_DIR "$ENV{CONDA_PREFIX}/lib")
set(_rpaths "${CONDA_LIB_DIR}")
if(DEFINED TORCH_INSTALL_PREFIX)
  list(APPEND _rpaths "${TORCH_INSTALL_PREFIX}/lib")
endif()

set_target_properties(bench_enum_centroid_stage_only_v6_8_aligned PROPERTIES
  BUILD_RPATH "${_rpaths}"
  INSTALL_RPATH "${_rpaths}"
)

if(UNIX AND NOT APPLE)
  target_link_libraries(bench_enum_centroid_stage_only_v6_8_aligned PRIVATE dl pthread)
endif()