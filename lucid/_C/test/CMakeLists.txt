# lucid/_C/test/CMakeLists.txt
#
# Google Test unit tests for the Lucid C++ engine.
#
# Build with:
#   BUILD_TESTING=ON pip install -e . --no-build-isolation
# Or directly:
#   cmake -DBUILD_TESTING=ON ... && cmake --build .
# Run with:
#   ctest --test-dir <build_dir> --output-on-failure

cmake_minimum_required(VERSION 3.24)

# ── Python (required because TensorImpl.h pulls in pybind11 headers) ─────────
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ── Fetch Google Test ─────────────────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
  GIT_SHALLOW    TRUE
)
# Prevent overriding parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST          OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK            ON  CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ── Test sources ──────────────────────────────────────────────────────────────
# CONFIGURE_DEPENDS re-runs the glob on every build: without it a test file added
# after the first configure is not in ``lucid_test`` until someone re-runs cmake,
# and the gtest that was meant to guard a fix is silently absent.
file(GLOB_RECURSE LUCID_TEST_SRCS CONFIGURE_DEPENDS
    core/*.cpp
    ops/*.cpp
    nn/*.cpp
    backend/*.cpp
    autograd/*.cpp
    integration/*.cpp
)

if(NOT LUCID_TEST_SRCS)
    message(WARNING "[lucid_test] No test source files found — skipping test target.")
    return()
endif()

# ── Test executable ───────────────────────────────────────────────────────────
add_executable(lucid_test ${LUCID_TEST_SRCS})

target_link_libraries(lucid_test PRIVATE
    GTest::gtest_main
    GTest::gmock
    lucid_compile_options          # inherits C++20 standard + all -W flags
    lucid_core
    lucid_backend_cpu
    lucid_backend_gpu
    lucid_backend_init             # registers CPU+GPU backends at static-init
    lucid_ops
    lucid_autograd
    lucid_random
    lucid_compile                  # compile::Tracer — OpScope hooks referenced by ops
    Python3::Python                # pybind11 C API symbols (set_error etc.)
    "${LUCID_MLX_LIBRARY}"         # MLX symbols used by GpuBackend
    "-framework Accelerate"        # BLAS / LAPACK / vDSP / vForce
    "-framework Metal"             # Metal API (MetalAllocator, MetalKernelRunner)
    "-framework Foundation"        # NSString etc. (required by Metal on macOS)
    "-framework MetalPerformanceShaders"        # MPSGraph deps (via lucid_compile)
    "-framework MetalPerformanceShadersGraph"   # MPSGraph kernels (via lucid_compile)
)

# Include paths: helpers/, engine root, project root, and pybind11/Python headers
# (needed because TensorImpl.h includes pybind11/numpy.h)
target_include_directories(lucid_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/helpers
    ${CMAKE_CURRENT_SOURCE_DIR}/..        # lucid/_C — for #include "core/..."
)
target_include_directories(lucid_test SYSTEM PRIVATE
    ${LUCID_PYBIND11_INCLUDE_DIR}
    ${LUCID_PYTHON_INCLUDE_DIR}
    ${LUCID_MLX_INCLUDE_DIR}
)

# ── Register with CTest ───────────────────────────────────────────────────────
include(GoogleTest)
gtest_discover_tests(lucid_test
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    PROPERTIES        LABELS "lucid_unit"
)
