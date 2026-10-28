cmake_minimum_required(VERSION 3.24)

if(NOT APPLE)
    message(FATAL_ERROR "Lucid C++ engine supports macOS arm64 Apple Silicon only")
endif()

if(NOT CMAKE_OSX_ARCHITECTURES)
    set(CMAKE_OSX_ARCHITECTURES arm64 CACHE STRING "Target Apple Silicon only" FORCE)
endif()

project(lucid_engine LANGUAGES CXX OBJCXX)

# Always emit compile_commands.json.  clangd and the C/C++ extension both read
# it to learn where MLX, pybind11 and the Metal frameworks live; without one
# they fall back to guessing include paths and light the whole engine up with
# red squiggles that are not compile errors.  Costs nothing to write, and the
# editor configs in the repo root point at a tree configured by this file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    message(FATAL_ERROR "Lucid C++ engine supports macOS arm64 Apple Silicon only")
endif()

set(LUCID_BUILD_MODE "release" CACHE STRING "Lucid build mode: release, debug, debug-ubsan, debug-asan, debug-tsan")
set(LUCID_SANITIZE "" CACHE STRING "Comma-separated Clang sanitizers (e.g. address,undefined, thread)")
option(LUCID_COVERAGE "Enable LLVM coverage instrumentation" OFF)
set(LUCID_REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")

# Build paths (Python / pybind11 / MLX include+lib / extension suffix + output
# dir) are normally injected by setup.py.  When this file is configured directly
# — e.g. an IDE (VS Code CMake Tools with cmake.sourceDirectory = lucid/_C) —
# they are absent, which previously aborted with FATAL_ERROR and left the
# editor's CMake integration red.  Derive them from the active Python instead
# (MLX through tools/fetch_mlx.py, as setup.py does), so a standalone/IDE
# configure just works.
set(_lucid_need_derive FALSE)
foreach(required_var
        LUCID_PYTHON_INCLUDE_DIR
        LUCID_PYBIND11_INCLUDE_DIR
        LUCID_MLX_INCLUDE_DIR
        LUCID_MLX_LIBRARY_DIR
        LUCID_MLX_PIN
        LUCID_PYTHON_EXTENSION_SUFFIX
        LUCID_EXTENSION_OUTPUT_DIR)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        set(_lucid_need_derive TRUE)
    endif()
endforeach()

if(_lucid_need_derive)
    # Try candidate interpreters in priority order and keep the first that can
    # actually import pybind11: an explicit ``-DPython3_EXECUTABLE=`` /
    # ``-DLUCID_PYTHON=`` (an IDE may inject the selected interpreter), then the
    # first ``python3`` on PATH.  CMake's ``find_package(Python3)`` heuristic is
    # avoided (it picks a bare system Python without the build packages), and a
    # candidate that can't run — e.g. an IDE macro like ``${command:...}`` that
    # never expanded, so it still contains a ``$`` — is skipped, not fatal.
    set(_lucid_candidates "")
    if(DEFINED Python3_EXECUTABLE AND Python3_EXECUTABLE
       AND NOT "${Python3_EXECUTABLE}" MATCHES "[$]")
        list(APPEND _lucid_candidates "${Python3_EXECUTABLE}")
    endif()
    if(DEFINED LUCID_PYTHON AND LUCID_PYTHON
       AND NOT "${LUCID_PYTHON}" MATCHES "[$]")
        list(APPEND _lucid_candidates "${LUCID_PYTHON}")
    endif()
    find_program(_lucid_path_py NAMES python3 python)
    if(_lucid_path_py)
        list(APPEND _lucid_candidates "${_lucid_path_py}")
    endif()

    set(_lucid_py_res 1)
    set(_lucid_py "")
    foreach(_cand IN LISTS _lucid_candidates)
        execute_process(
            COMMAND "${_cand}" -c [==[
import sysconfig
import pybind11
print(sysconfig.get_paths()['include'])
print(pybind11.get_include())
print(sysconfig.get_config_var('EXT_SUFFIX'))
]==]
            OUTPUT_VARIABLE _lucid_py_out
            RESULT_VARIABLE _lucid_py_res
            ERROR_VARIABLE _lucid_py_err
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(_lucid_py_res EQUAL 0)
            set(_lucid_py "${_cand}")
            break()
        endif()
    endforeach()

    if(_lucid_py_res EQUAL 0)
        # The pinned MLX, staged (downloaded on first use) into the per-machine
        # cache; never the `mlx` Python package, whose libmlx has another ABI.
        foreach(_lucid_what dir version)
            execute_process(
                COMMAND "${_lucid_py}" "${LUCID_REPO_ROOT}/tools/fetch_mlx.py" --print-${_lucid_what}
                OUTPUT_VARIABLE _lucid_mlx_${_lucid_what}
                RESULT_VARIABLE _lucid_py_res
                ERROR_VARIABLE _lucid_py_err
                OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT _lucid_py_res EQUAL 0)
                break()
            endif()
        endforeach()
    endif()

    if(_lucid_py_res EQUAL 0)
        string(REPLACE "\n" ";" _lucid_paths "${_lucid_py_out}")
        list(GET _lucid_paths 0 _lucid_py_inc)
        list(GET _lucid_paths 1 _lucid_pybind_inc)
        list(GET _lucid_paths 2 _lucid_ext_suffix)
        set(_lucid_mlx_inc "${_lucid_mlx_dir}/include")
        set(_lucid_mlx_lib "${_lucid_mlx_dir}/lib")
        if(NOT LUCID_PYTHON_INCLUDE_DIR)
            set(LUCID_PYTHON_INCLUDE_DIR "${_lucid_py_inc}")
        endif()
        if(NOT LUCID_PYBIND11_INCLUDE_DIR)
            set(LUCID_PYBIND11_INCLUDE_DIR "${_lucid_pybind_inc}")
        endif()
        if(NOT LUCID_MLX_INCLUDE_DIR)
            set(LUCID_MLX_INCLUDE_DIR "${_lucid_mlx_inc}")
        endif()
        if(NOT LUCID_MLX_LIBRARY_DIR)
            set(LUCID_MLX_LIBRARY_DIR "${_lucid_mlx_lib}")
        endif()
        if(NOT LUCID_MLX_PIN)
            set(LUCID_MLX_PIN "${_lucid_mlx_version}")
        endif()
        if(NOT LUCID_PYTHON_EXTENSION_SUFFIX)
            set(LUCID_PYTHON_EXTENSION_SUFFIX "${_lucid_ext_suffix}")
        endif()
        if(NOT LUCID_EXTENSION_OUTPUT_DIR)
            set(LUCID_EXTENSION_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}")
        endif()
        message(STATUS
            "Lucid: build paths auto-derived from ${_lucid_py} "
            "(standalone/IDE configure; setup.py injects them for a real build)")
    else()
        message(FATAL_ERROR
            "Lucid build paths were not provided by setup.py and could not be "
            "auto-derived from Python (need an interpreter with pybind11, and "
            "tools/fetch_mlx.py must be able to stage the pinned MLX). Build the "
            "engine with `pip install -e .`.\n${_lucid_py_err}")
    endif()
endif()

# liblucid_mlx is the pinned MLX build renamed by tools/fetch_mlx.py, so dyld
# never confuses it with the @rpath/libmlx.dylib a Python `mlx` loads.  A plain
# variable, not find_library: a cached result would keep linking the previous
# pin's library after a bump changes LUCID_MLX_LIBRARY_DIR.
set(LUCID_MLX_LIBRARY "${LUCID_MLX_LIBRARY_DIR}/liblucid_mlx.dylib")
if(NOT EXISTS "${LUCID_MLX_LIBRARY}")
    message(FATAL_ERROR "${LUCID_MLX_LIBRARY} not found; stage it with tools/fetch_mlx.py")
endif()

add_library(lucid_compile_options INTERFACE)
target_compile_features(lucid_compile_options INTERFACE cxx_std_20)
target_include_directories(lucid_compile_options INTERFACE "${LUCID_REPO_ROOT}")
target_include_directories(
    lucid_compile_options
    SYSTEM
    INTERFACE
        "${LUCID_PYTHON_INCLUDE_DIR}"
        "${LUCID_PYBIND11_INCLUDE_DIR}"
        "${LUCID_MLX_INCLUDE_DIR}")
target_compile_definitions(lucid_compile_options INTERFACE LUCID_BUILDING_ENGINE=1)
# Apple deprecated the legacy CBLAS / CLAPACK Fortran-name interfaces in
# macOS 13.3 in favour of a new header layout (cblas_sgemm, sgetrf_, …).
# ACCELERATE_NEW_LAPACK opts into the new symbol routing without changing
# index width (LP64); add ACCELERATE_LAPACK_ILP64 alongside if 64-bit
# indices are ever required.
target_compile_definitions(lucid_compile_options INTERFACE ACCELERATE_NEW_LAPACK)
target_compile_options(
    lucid_compile_options
    INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        -Werror
        -Wno-unused-parameter
        -fvisibility=hidden)

if(LUCID_SANITIZE)
    target_compile_options(
        lucid_compile_options
        INTERFACE
            -O1
            -g
            -fno-omit-frame-pointer
            -fsanitize=${LUCID_SANITIZE})
    target_link_options(lucid_compile_options INTERFACE -fsanitize=${LUCID_SANITIZE})
elseif(LUCID_BUILD_MODE STREQUAL "release")
    target_compile_options(lucid_compile_options INTERFACE -O3)
    target_compile_definitions(lucid_compile_options INTERFACE NDEBUG)
elseif(LUCID_BUILD_MODE STREQUAL "debug")
    target_compile_options(lucid_compile_options INTERFACE -O0 -g -fno-omit-frame-pointer)
    target_link_options(lucid_compile_options INTERFACE -g)
elseif(LUCID_BUILD_MODE STREQUAL "debug-ubsan")
    target_compile_options(
        lucid_compile_options INTERFACE -O1 -g -fno-omit-frame-pointer -fsanitize=undefined)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=undefined)
elseif(LUCID_BUILD_MODE STREQUAL "debug-asan")
    target_compile_options(
        lucid_compile_options
        INTERFACE
            -O1
            -g
            -fno-omit-frame-pointer
            -fsanitize=address,undefined)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=address,undefined)
elseif(LUCID_BUILD_MODE STREQUAL "debug-tsan")
    target_compile_options(
        lucid_compile_options INTERFACE -O1 -g -fno-omit-frame-pointer -fsanitize=thread)
    target_link_options(lucid_compile_options INTERFACE -fsanitize=thread)
else()
    message(FATAL_ERROR "Unknown LUCID_BUILD_MODE=${LUCID_BUILD_MODE}")
endif()

if(LUCID_COVERAGE)
    target_compile_options(lucid_compile_options INTERFACE -fprofile-instr-generate -fcoverage-mapping)
    target_link_options(lucid_compile_options INTERFACE -fprofile-instr-generate -fcoverage-mapping)
endif()

function(lucid_object_library target)
    add_library(${target} OBJECT ${ARGN})
    set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(${target} PRIVATE lucid_compile_options)
endfunction()

lucid_object_library(
    lucid_core
    version.cpp
    core/Allocator.cpp
    core/AmpPolicy.cpp
    core/Determinism.cpp
    core/ErrorBuilder.cpp
    core/Error.cpp
    core/Generator.cpp
    core/GradMode.cpp
    core/MemoryStats.cpp
    core/OpRegistry.cpp
    core/Profiler.cpp
    core/SchemaGuard.cpp
    core/TensorImpl.cpp
    core/Validate.cpp)

file(GLOB LUCID_CPU_BACKEND_METHODS CONFIGURE_DEPENDS backend/cpu/methods/*.cpp)
lucid_object_library(
    lucid_backend_cpu
    backend/cpu/Blas.cpp
    backend/cpu/Lapack.cpp
    backend/cpu/Im2Col.cpp
    backend/cpu/Norm.cpp
    backend/cpu/Pool.cpp
    backend/cpu/Reduce.cpp
    backend/cpu/Shape.cpp
    backend/cpu/Vdsp.cpp
    backend/cpu/Vforce.cpp
    ${LUCID_CPU_BACKEND_METHODS})

# Phase 9.1 + 18: Objective-C++ files require ARC and the Metal framework.
# 3.4 Phase 1: gpu/mps/ subdirectory adds MPSGraph dispatch sources.
file(GLOB LUCID_GPU_BACKEND_METHODS CONFIGURE_DEPENDS backend/gpu/methods/*.cpp)
lucid_object_library(
    lucid_backend_gpu
    backend/gpu/MlxBridge.cpp
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm
    backend/gpu/mps/MpsBridge.mm
    ${LUCID_GPU_BACKEND_METHODS})
set_source_files_properties(
    backend/gpu/MetalAllocator.mm
    backend/gpu/MetalKernelRunner.mm
    backend/gpu/mps/MpsBridge.mm
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

lucid_object_library(lucid_backend_init backend/BackendInit.cpp)

# Zero-copy DLPack in the Metal dialect — see interop/Dlpack.h.  Plain C++:
# the Obj-C types it touches are already erased to void* by MpsBridge.
lucid_object_library(lucid_interop interop/Dlpack.cpp)

# Core ML model-package writer.  Plain C++: the protobuf/blob/bundle
# formats are byte layout, and only the runtime half needs Obj-C.
lucid_object_library(
    lucid_coreml
    coreml/BlobWriter.cpp
    coreml/ModelPackage.cpp
    coreml/MilProgram.cpp
    coreml/CoreMLRuntime.mm)
set_source_files_properties(coreml/CoreMLRuntime.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")

lucid_object_library(
    lucid_autograd
    autograd/Node.cpp
    autograd/Helpers.cpp
    autograd/AccumulateGrad.cpp
    autograd/Engine.cpp
    autograd/TensorHooks.cpp
    autograd/ModuleHookNode.cpp
    autograd/CustomFunction.cpp)   # Phase 12: PythonBackwardNode

lucid_object_library(
    lucid_ops
    ops/bfunc/Add.cpp
    ops/bfunc/Sub.cpp
    ops/bfunc/Mul.cpp
    ops/bfunc/Div.cpp
    ops/bfunc/Pow.cpp
    ops/bfunc/Maximum.cpp
    ops/bfunc/Minimum.cpp
    ops/bfunc/Matmul.cpp
    ops/bfunc/Compare.cpp
    ops/bfunc/Bitwise.cpp
    ops/bfunc/Dot.cpp
    ops/bfunc/Inner.cpp
    ops/bfunc/Outer.cpp
    ops/bfunc/Tensordot.cpp
    ops/bfunc/Floordiv.cpp
    ops/bfunc/Inplace.cpp
    ops/ufunc/Arith.cpp
    ops/ufunc/Exponential.cpp
    ops/ufunc/Trig.cpp
    ops/ufunc/Hyperbolic.cpp
    ops/ufunc/Activation.cpp
    ops/ufunc/ScalarParam.cpp
    ops/ufunc/Discrete.cpp
    ops/ufunc/Softmax.cpp
    ops/ufunc/UnaryGpu.cpp
    ops/ufunc/Reductions.cpp
    ops/ufunc/Transpose.cpp
    ops/ufunc/Var.cpp
    ops/ufunc/Trace.cpp
    ops/ufunc/Scan.cpp
    ops/ufunc/Inplace.cpp
    ops/ufunc/CubeRoot.cpp
    ops/ufunc/Predicate.cpp
    ops/gfunc/Gfunc.cpp
    ops/utils/View.cpp
    ops/utils/InplaceGraph.cpp
    ops/utils/Contiguous.cpp
    ops/utils/Concat.cpp
    ops/utils/Repeat.cpp
    ops/utils/Pad.cpp
    ops/utils/Layout.cpp
    ops/utils/Tri.cpp
    ops/utils/Select.cpp
    ops/utils/Sort.cpp
    ops/utils/Meshgrid.cpp
    ops/utils/Histogram.cpp
    ops/utils/Nextafter.cpp
    ops/composite/Math.cpp
    ops/composite/Reductions.cpp
    ops/composite/Matrix.cpp
    ops/composite/Logical.cpp
    ops/composite/Indexing.cpp
    ops/composite/Layout.cpp
    ops/composite/Stats.cpp
    ops/composite/Search.cpp
    ops/einops/Einsum.cpp
    ops/linalg/Inv.cpp
    ops/linalg/Det.cpp
    ops/linalg/Solve.cpp
    ops/linalg/Cholesky.cpp
    ops/linalg/Norm.cpp
    ops/linalg/QR.cpp
    ops/linalg/SVD.cpp
    ops/linalg/MatrixPower.cpp
    ops/linalg/Pinv.cpp
    ops/linalg/Eig.cpp
    ops/linalg/Eigh.cpp
    ops/ufunc/Astype.cpp
    ops/ufunc/ToDevice.cpp
    ops/linalg/LUFactor.cpp
    ops/linalg/SolveTriangular.cpp
    ops/linalg/LUSolve.cpp
    ops/linalg/HouseholderProduct.cpp
    ops/linalg/LDLFactor.cpp
    ops/fft/Fftn.cpp
    ops/fft/Ifftn.cpp
    ops/fft/Rfftn.cpp
    ops/fft/Irfftn.cpp
    ops/quantized/QuantizedMatmul.cpp
    ops/complex/Real.cpp
    ops/complex/Imag.cpp
    ops/complex/Complex.cpp
    ops/complex/Conj.cpp
    ops/diffeq/BroydenProbe.cpp
    ops/diffeq/RkCombine.cpp
    ops/diffeq/RkErrorNorm.cpp
    nn/Linear.cpp
    nn/Dropout.cpp
    nn/LayerNorm.cpp
    nn/RMSNorm.cpp
    nn/ConvNd.cpp
    nn/ConvTransposeNd.cpp
    nn/PoolNd.cpp
    nn/AdaptivePool.cpp
    nn/BatchNorm.cpp
    nn/GroupNorm.cpp
    nn/NormExt.cpp
    nn/Loss.cpp
    nn/Attention.cpp
    nn/LSTM.cpp
    nn/Embedding.cpp
    nn/Spatial.cpp
    nn/Interpolate.cpp
    nn/Vision.cpp)

lucid_object_library(lucid_random random/Random.cpp)

lucid_object_library(
    lucid_optim
    optim/Optimizer.cpp
    optim/SGD.cpp
    optim/Adam.cpp
    optim/Prop.cpp
    optim/Ada.cpp
    optim/LRScheduler.cpp)

# 3.5 Phase 1.1+1.2: lucid.compile() graph-capture path.
# Phase 1.1 = Tracer + TraceIR + I/O wiring.
# Phase 1.2 step 1 = MpsBuilder + ExecutableCache + first OpEmitter (Linear).
# Emitters are organised into six sub-packages: core/ linalg/ shape/
# reduce/ nn/ misc/.  Each subdir owns one bucket of MPSGraph emitters
# plus its own ``Registrar`` static initialiser that pushes them into
# the process-global registry on load.
set(LUCID_OP_EMITTERS
    # framework
    compile/OpEmitters/OpEmitter.cpp
    # elementwise/ — one-output ops where each output element only
    # depends on the same-position input element(s)
    compile/OpEmitters/elementwise/Activation.mm
    compile/OpEmitters/elementwise/Arith.mm
    compile/OpEmitters/elementwise/Compare.mm
    compile/OpEmitters/elementwise/DtypeCast.mm
    compile/OpEmitters/elementwise/Math.mm
    compile/OpEmitters/elementwise/Predicate.mm
    compile/OpEmitters/elementwise/Softmax.mm
    compile/OpEmitters/elementwise/Trig.mm
    # reduce/ — collapse one or more axes
    compile/OpEmitters/reduce/Reduction.mm
    compile/OpEmitters/reduce/Cumulative.mm
    compile/OpEmitters/reduce/ArgReduce.mm
    # linalg/ — matrix algebra primitives
    compile/OpEmitters/linalg/Linear.mm
    compile/OpEmitters/linalg/Matmul.mm
    compile/OpEmitters/linalg/MatrixOps.mm
    compile/OpEmitters/linalg/Decompositions.mm
    # shape/ — pure view + layout rearrangement (no data-dep indexing)
    compile/OpEmitters/shape/Concat.mm
    compile/OpEmitters/shape/Layout.mm
    compile/OpEmitters/shape/Permute.mm
    compile/OpEmitters/shape/Reshape.mm
    compile/OpEmitters/shape/Split.mm
    # index/ — data-dependent reads / writes
    compile/OpEmitters/index/Gather.mm
    compile/OpEmitters/index/Indexing.mm
    compile/OpEmitters/index/Scatter.mm
    # nn/ — neural network op families
    compile/OpEmitters/nn/Conv.mm
    compile/OpEmitters/nn/Dropout.mm
    compile/OpEmitters/nn/Pool.mm
    compile/OpEmitters/nn/Norm.mm
    compile/OpEmitters/nn/Loss.mm
    compile/OpEmitters/nn/Attention.mm
    compile/OpEmitters/nn/Embedding.mm
    compile/OpEmitters/nn/Spatial.mm
    compile/OpEmitters/nn/Rnn.mm
    # special/ — compile-pipeline plumbing (factories, complex split,
    # nan_to_num, eager-fallback stubs)
    compile/OpEmitters/special/Complex.mm
    compile/OpEmitters/special/NanToNum.mm
    compile/OpEmitters/special/Factory.mm
    compile/OpEmitters/special/Random.mm
    compile/OpEmitters/special/Stubs.mm)

# Manual VJP emitters (Phase 1-4): backward subgraph emission that
# replaces MPSGraph's gradientForPrimaryTensor: when LUCID_MANUAL_VJP=1.
# Mirrors LUCID_OP_EMITTERS layout — one bucket per op family.
set(LUCID_VJP_EMITTERS
    # framework
    compile/VjpEmitters/VjpEmitter.mm
    # elementwise
    compile/VjpEmitters/elementwise/Arith.mm
    compile/VjpEmitters/elementwise/Activation.mm
    compile/VjpEmitters/elementwise/Softmax.mm
    compile/VjpEmitters/elementwise/Math.mm
    # linalg
    compile/VjpEmitters/linalg/Linear.mm
    compile/VjpEmitters/linalg/Products.mm
    # reduce
    compile/VjpEmitters/reduce/Reduction.mm
    compile/VjpEmitters/reduce/Scan.mm
    # shape
    compile/VjpEmitters/shape/Reshape.mm
    compile/VjpEmitters/shape/Concat.mm
    compile/VjpEmitters/shape/Stack.mm
    compile/VjpEmitters/shape/Reorder.mm
    # nn
    compile/VjpEmitters/nn/Loss.mm
    compile/VjpEmitters/nn/Embedding.mm
    compile/VjpEmitters/nn/Norm.mm
    compile/VjpEmitters/nn/Conv.mm
    compile/VjpEmitters/nn/Dropout.mm
    compile/VjpEmitters/nn/Pool.mm
    compile/VjpEmitters/nn/Attention.mm
    compile/VjpEmitters/nn/Spatial.mm)

lucid_object_library(
    lucid_compile
    compile/Tracer.cpp
    compile/RngFeeds.cpp
    compile/ExecutableCache.cpp
    compile/CompiledExecutable.mm
    compile/MpsBuilder.mm
    ${LUCID_OP_EMITTERS}
    ${LUCID_VJP_EMITTERS})
# ObjC++ sources in the compile path need ARC, same as the other GPU
# bridge ``.mm`` files.  Strip the framework ``.cpp`` from the ARC
# list (it's pure C++) and apply to every emitter ``.mm``.
set(LUCID_OP_EMITTER_MM ${LUCID_OP_EMITTERS})
list(REMOVE_ITEM LUCID_OP_EMITTER_MM compile/OpEmitters/OpEmitter.cpp)
set_source_files_properties(
    compile/CompiledExecutable.mm
    compile/MpsBuilder.mm
    ${LUCID_OP_EMITTER_MM}
    ${LUCID_VJP_EMITTERS}
    PROPERTIES COMPILE_FLAGS "-fobjc-arc")

# 3.5+ tokenizer family — Python ``lucid.utils.tokenizer`` mirror.
# Each algorithm gets its own .cpp; the base ``Tokenizer.cpp`` carries
# the default optional overrides (encode_batch / decode_batch / etc.).
# Future tokenizers (Unigram / ByteLevelBPE-distinct-from-BPE) plug
# in by appending their .cpp to this list.
lucid_object_library(
    lucid_utils_tokenizer
    utils/tokenizer/Tokenizer.cpp
    utils/tokenizer/BPE.cpp
    utils/tokenizer/Basic.cpp        # Byte / Char / Whitespace / Word / Regex
    utils/tokenizer/WordPiece.cpp    # BERT-family greedy longest-match
    utils/tokenizer/Unigram.cpp)     # SentencePiece-flavor Viterbi + EM

lucid_object_library(
    lucid_bindings
    bindings/bind_errors.cpp
    bindings/bind_tensor.cpp
    bindings/bind_amp.cpp
    bindings/bind_profiler.cpp
    bindings/bind_op_registry.cpp
    bindings/bind_autograd.cpp
    bindings/bind_nn.cpp
    bindings/bind_random.cpp
    bindings/bind_optim.cpp
    bindings/bind_gfunc.cpp
    bindings/bind_bfunc.cpp
    bindings/bind_ufunc.cpp
    bindings/bind_utils.cpp
    bindings/bind_composite.cpp
    bindings/bind_linalg.cpp
    bindings/bind_einops.cpp
    bindings/bind_fft.cpp
    bindings/bind_diffeq.cpp
    bindings/bind_quantized.cpp
    bindings/bind_complex.cpp
    bindings/bind_compile.cpp        # 3.5 Phase 1.1: lucid.compile() bindings
    bindings/bind_interop.cpp        # kDLMetal DLPack producer + consumer
    bindings/bind_coreml.cpp         # Core ML package writer
    bindings/bind_tokenizer.cpp      # 3.5+ tokenizer surface (utils.tokenizer)
    bindings/bind.cpp)
# Only module init compares the loaded MLX against the pin, so only it sees the
# pin: a bump does not invalidate every other object through a definition.
set_source_files_properties(
    bindings/bind.cpp
    PROPERTIES COMPILE_DEFINITIONS "LUCID_MLX_PIN=\"${LUCID_MLX_PIN}\"")

option(LUCID_MLX_BUNDLED "The MLX runtime is copied next to the engine (wheel build)" OFF)
if(LUCID_MLX_BUNDLED)
    set(LUCID_ENGINE_RPATH "@loader_path")
else()
    set(LUCID_ENGINE_RPATH "@loader_path;${LUCID_MLX_LIBRARY_DIR}")
endif()

add_library(
    lucid_engine MODULE
    $<TARGET_OBJECTS:lucid_core>
    $<TARGET_OBJECTS:lucid_backend_cpu>
    $<TARGET_OBJECTS:lucid_backend_gpu>
    $<TARGET_OBJECTS:lucid_backend_init>
    $<TARGET_OBJECTS:lucid_autograd>
    $<TARGET_OBJECTS:lucid_ops>
    $<TARGET_OBJECTS:lucid_random>
    $<TARGET_OBJECTS:lucid_optim>
    $<TARGET_OBJECTS:lucid_compile>     # 3.5 Phase 1.1: graph-capture tracer
    $<TARGET_OBJECTS:lucid_interop>     # kDLMetal DLPack bridge
    $<TARGET_OBJECTS:lucid_coreml>      # Core ML writer
    $<TARGET_OBJECTS:lucid_utils_tokenizer>  # 3.5+ BPE / WordPiece / ...
    $<TARGET_OBJECTS:lucid_bindings>)

target_link_libraries(
    lucid_engine
    PRIVATE
        lucid_compile_options
        "${LUCID_MLX_LIBRARY}"
        "-framework Accelerate"
        "-framework Metal"                          # Phase 9.1: MetalAllocator
        "-framework Foundation"                     # required by Metal on macOS
        "-framework MetalPerformanceShaders"        # 3.4 Phase 1: MPSGraph deps
        "-framework MetalPerformanceShadersGraph"  # 3.4 Phase 1: graph kernels
        "-framework CoreML")                       # lucid.coreml: ANE execution
if(APPLE)
    target_link_options(lucid_engine PRIVATE -undefined dynamic_lookup)
endif()
set_target_properties(
    lucid_engine
    PROPERTIES
        PREFIX ""
        OUTPUT_NAME "engine"
        SUFFIX "${LUCID_PYTHON_EXTENSION_SUFFIX}"
        LIBRARY_OUTPUT_DIRECTORY "${LUCID_EXTENSION_OUTPUT_DIR}"
        # The path list dyld searches for the ``@rpath/liblucid_mlx.dylib``
        # load command, in order:
        #
        #   1. ``@loader_path`` — a wheel: setup.py copies liblucid_mlx,
        #      liblucid_jaccl and mlx.metallib next to engine*.so.
        #   2. ``${LUCID_MLX_LIBRARY_DIR}`` — an editable / in-place build,
        #      which copies nothing: the staged entry in the per-machine
        #      cache (tools/fetch_mlx.py).  Left out of a wheel
        #      (LUCID_MLX_BUNDLED), which must not name a build-machine path.
        #
        # Neither entry can reach a Python `mlx` package's libmlx.dylib: the
        # file name differs, so whichever MLX the user has loads beside it.
        INSTALL_RPATH "${LUCID_ENGINE_RPATH}"
        # Use INSTALL_RPATH at link time too, so the build artifact has
        # both entries (no separate install relink step).
        BUILD_WITH_INSTALL_RPATH ON
        # Don't auto-add the linker's search path to RPATH — that would
        # add an unintended third entry pointing into the build directory.
        INSTALL_RPATH_USE_LINK_PATH OFF)

# ── Optional C++ unit tests (Google Test) ────────────────────────────────────
option(BUILD_TESTING "Build C++ unit tests using Google Test" OFF)
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(test)
    if(TARGET lucid_test)
        # Same resolution as the engine's editable build: the staged MLX.
        set_target_properties(lucid_test PROPERTIES BUILD_RPATH "${LUCID_MLX_LIBRARY_DIR}")
    endif()
endif()
