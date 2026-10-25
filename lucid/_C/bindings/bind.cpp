// lucid/_C/bindings/bind.cpp
//
// Top-level entry point for the `lucid._C.engine` Python extension module.
// This file owns the single PYBIND11_MODULE(engine, m) definition and is
// responsible for:
//   1. Exporting the ABI / version constants (VERSION_MAJOR, VERSION_MINOR,
//      VERSION_PATCH, ABI_VERSION, __version__).
//   2. Calling every subsystem register_xxx() function in dependency order so
//      that base types (errors, enums, TensorImpl) are visible before the ops
//      that use them.
//   3. Creating the `nn`, `linalg`, and `einops` sub-modules.
//   4. Defining the two fused-kernel helpers (_fused_linear_relu,
//      _fused_linear_gelu) and the custom Metal kernel launcher
//      (_run_metal_kernel) that are too tightly coupled to the Dispatcher to
//      live in a separate file.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <string>
#include <string_view>

#include <mlx/version.h>

#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../core/fwd.h"
#include "../version.h"

namespace py = pybind11;

// Forward declarations of every subsystem binder.  Each is defined in its own
// bind_xxx.cpp translation unit and linked into the shared library.
namespace lucid::bindings {
void register_errors(py::module_& m);
void register_core(py::module_& m);
void register_tensor_impl(py::module_& m);
void register_interop(py::module_& m);
void register_coreml(py::module_& m);
void register_autograd(py::module_& m);
void register_amp(py::module_& m);
void register_profiler(py::module_& m);
void register_op_registry(py::module_& m);
void register_optim(py::module_& m);
void register_nn(py::module_& m);
void register_random(py::module_& m);
void register_gfunc(py::module_& m);
void register_bfunc(py::module_& m);
void register_ufunc(py::module_& m);
void register_utils(py::module_& m);
void register_composite(py::module_& m);
void register_linalg(py::module_& m);
void register_einops(py::module_& m);
void register_fft(py::module_& m);
void register_diffeq(py::module_& m);
void register_quantized(py::module_& m);
void register_complex(py::module_& m);
void register_compile(py::module_& m);
// 3.5+ tokenizer surface — lives under ``utils.tokenizer`` to mirror
// the Python ``lucid.utils.tokenizer`` package.  Registered against the
// ``utils`` sub-module after ``register_utils`` has built it.
void register_tokenizer(py::module_& utils);
}  // namespace lucid::bindings

#ifndef LUCID_MLX_PIN
#error "LUCID_MLX_PIN is set by CMake from pyproject.toml [tool.lucid.mlx] version"
#endif

namespace {

#define LUCID_STR_IMPL(x) #x
#define LUCID_STR(x) LUCID_STR_IMPL(x)
// The headers compiled against and the pin must name the same MLX: a stale
// include dir would otherwise build an engine whose layout assumptions belong
// to another release.
static_assert(std::string_view(LUCID_STR(MLX_VERSION_MAJOR) "." LUCID_STR(
                  MLX_VERSION_MINOR) "." LUCID_STR(MLX_VERSION_PATCH)) == LUCID_MLX_PIN,
              "MLX headers do not match [tool.lucid.mlx] version");
#undef LUCID_STR
#undef LUCID_STR_IMPL

// The MLX C++ ABI changes between patch releases, so an engine must run only
// on the MLX it was built for.  rpath resolution can still hand it another
// liblucid_mlx (a wheel whose copied dylib was swapped, a rebuilt cache
// entry); refuse to load rather than fail later on a missing symbol or a
// silently different inline layout.
void require_pinned_mlx() {
    const char* loaded = mlx::core::version();
    if (std::strcmp(loaded, LUCID_MLX_PIN) != 0) {
        throw py::import_error(std::string("lucid._C.engine was built for MLX ") + LUCID_MLX_PIN +
                               " but loaded MLX " + loaded +
                               " (stale build - rebuild the engine, e.g. `pip install -e .`)");
    }
}

enum class FusedActivation { Relu, Gelu };

// The ``_fused_linear_*`` inference kernels.  CpuBackend fuses the GEMM and
// the activation for F32 only; every other device and dtype computes the same
// function as the backend's own ``linear`` followed by ``relu`` / ``gelu``
// (tanh form, as the CPU kernel), so ``nn.FusedLinear`` runs wherever
// ``nn.Linear`` does.  Either way the tracer records one fused op, which the
// compile and Core ML emitters translate.  The operands are checked here,
// before any kernel reads them: the CPU kernel derives K from x's byte size
// and would read past a smaller weight.
lucid::TensorImplPtr fused_linear_forward(const char* name,
                                          FusedActivation act,
                                          const lucid::TensorImplPtr& x,
                                          const lucid::TensorImplPtr& w,
                                          const lucid::TensorImplPtr& b) {
    using lucid::Validator;
    const std::string op(name);
    Validator::input(x, op + ".x")
        .dtype_in({lucid::Dtype::F16, lucid::Dtype::BF16, lucid::Dtype::F32, lucid::Dtype::F64})
        .ndim_at_least(1);
    Validator::input(w, op + ".weight").non_null().ndim(2);
    Validator::input(b, op + ".bias").non_null().ndim(1);
    Validator::pair(x, w, op + ": x/weight").same_device().same_dtype();
    Validator::pair(x, b, op + ": x/bias").same_device().same_dtype();
    const auto& xs = x->shape();
    const auto& ws = w->shape();
    if (ws[1] != xs.back())
        throw lucid::ShapeMismatch(ws, xs, op + ": weight.shape[1] != x.shape[-1]");
    if (b->shape()[0] != ws[0])
        throw lucid::ShapeMismatch(b->shape(), ws, op + ": bias.shape[0] != weight.shape[0]");

    lucid::Shape out_shape(xs.begin(), xs.end() - 1);
    out_shape.push_back(ws[0]);
    const lucid::Dtype dt = x->dtype();
    auto& be = lucid::backend::Dispatcher::for_device(x->device());
    lucid::Storage out;
    if (x->device() == lucid::Device::CPU && dt == lucid::Dtype::F32) {
        out = act == FusedActivation::Relu
                  ? be.fused_linear_relu_forward(x->storage(), w->storage(), b->storage(),
                                                 out_shape, dt)
                  : be.fused_linear_gelu_forward(x->storage(), w->storage(), b->storage(),
                                                 out_shape, dt);
    } else {
        lucid::Storage pre =
            be.linear(x->storage(), w->storage(), b->storage(), xs, ws, out_shape, dt);
        out = act == FusedActivation::Relu ? be.relu(pre, out_shape, dt)
                                           : be.gelu(pre, out_shape, dt);
    }
    return lucid::compile::traced_derive(
        op, std::vector<lucid::TensorImplPtr>{x, w, b},
        std::make_shared<lucid::TensorImpl>(std::move(out), out_shape, dt, x->device(), false));
}

}  // namespace

PYBIND11_MODULE(engine, m) {
    m.doc() = "Lucid C++ engine (production rebuild).";
    require_pinned_mlx();

    // Version constants are read by lucid/__init__.py to enforce ABI
    // compatibility between the Python wheel and the compiled extension.
    m.attr("__version__") = LUCID_VERSION_STRING;
    m.attr("VERSION_MAJOR") = LUCID_VERSION_MAJOR;
    m.attr("VERSION_MINOR") = LUCID_VERSION_MINOR;
    m.attr("VERSION_PATCH") = LUCID_VERSION_PATCH;
    m.attr("ABI_VERSION") = LUCID_ABI_VERSION;

    // Registration order matters: error translators must be installed first so
    // that exceptions thrown during subsequent registrations are mapped
    // correctly.  Core enums (Device, Dtype) must precede TensorImpl which
    // depends on them.
    lucid::bindings::register_errors(m);
    lucid::bindings::register_core(m);
    lucid::bindings::register_tensor_impl(m);
    lucid::bindings::register_interop(m);
    lucid::bindings::register_coreml(m);
    lucid::bindings::register_amp(m);
    lucid::bindings::register_profiler(m);
    lucid::bindings::register_op_registry(m);
    lucid::bindings::register_autograd(m);
    // nn ops live in a dedicated sub-module (lucid._C.engine.nn) to avoid
    // polluting the top-level namespace.
    auto nn = m.def_submodule("nn", "Neural-network ops (linear, conv, norm, pool, ...).");
    lucid::bindings::register_nn(nn);
    lucid::bindings::register_random(m);
    lucid::bindings::register_optim(m);

    lucid::bindings::register_gfunc(m);
    lucid::bindings::register_bfunc(m);
    lucid::bindings::register_ufunc(m);
    lucid::bindings::register_utils(m);
    // Composite ops layered on top of the primitives — must come after
    // the primitive registrations so its forwards/aliases see them.
    lucid::bindings::register_composite(m);
    // linalg and einops each get their own sub-module so Python can import
    // them as lucid.linalg / lucid.einops without name collisions.
    auto linalg = m.def_submodule("linalg", "Linear-algebra ops.");
    lucid::bindings::register_linalg(linalg);
    auto einops = m.def_submodule("einops", "einops-style rearrange/reduce/repeat/einsum.");
    lucid::bindings::register_einops(einops);
    auto fft = m.def_submodule("fft", "Discrete Fourier transform ops (mlx::core::fft wrappers).");
    lucid::bindings::register_fft(fft);
    // Differential-equation primitives; the integrator loop lives in Python
    // (lucid.diffeq) because the right-hand side is a Python callable.
    auto diffeq = m.def_submodule("diffeq", "Differential-equation primitives (Runge-Kutta).");
    lucid::bindings::register_diffeq(diffeq);
    // Low-precision (int4/int8) GEMM primitives (mlx::core quantized kernels).
    auto quantized = m.def_submodule("quantized", "MLX quantized GEMM ops (GPU only).");
    lucid::bindings::register_quantized(quantized);

    // Complex viewing ops (real / imag / complex / conj) live at the
    // top level — they're general-purpose, not under any sub-module.
    lucid::bindings::register_complex(m);

    // 3.5 Phase 1.1: lucid.compile() graph-capture tracer.  The compile
    // surface lives in its own sub-module so the global namespace stays
    // clean (mirrors `nn` / `linalg` / `einops` / `fft`).
    auto compile_mod =
        m.def_submodule("compile", "Graph-capture tracer and compiled executable cache.");
    lucid::bindings::register_compile(compile_mod);

    // 3.5+ utils sub-module — exclusively for new packages whose Python
    // mirror sits under ``lucid.utils.*`` (currently: tokenizer).  Note
    // that ``register_utils`` (called earlier with ``m``) puts shape
    // manipulation ops at the top level for backwards compatibility;
    // this ``utils`` sub-module is a *separate* namespace dedicated to
    // tokenization / future helpers that mirror ``lucid.utils.*``.
    auto utils_mod =
        m.def_submodule("utils", "Helper sub-packages mirroring lucid.utils.* (tokenizer, ...).");
    lucid::bindings::register_tokenizer(utils_mod);

    // Fused kernel bindings that directly call IBackend dispatch and therefore
    // cannot be separated into their own translation units without exposing
    // internal Dispatcher headers to bind_nn.cpp.
    // The output never requires grad: the Python wrappers call these only
    // when no operand needs one, and take linear + activation otherwise.
    m.def(
        "_fused_linear_relu",
        [](const lucid::TensorImplPtr& x, const lucid::TensorImplPtr& w,
           const lucid::TensorImplPtr& b) {
            return fused_linear_forward("_fused_linear_relu", FusedActivation::Relu, x, w, b);
        },
        py::arg("x"), py::arg("weight"), py::arg("bias"),
        "Fused linear + ReLU inference kernel: y = max(0, x @ W.T + b), no autograd.");

    m.def(
        "_fused_linear_gelu",
        [](const lucid::TensorImplPtr& x, const lucid::TensorImplPtr& w,
           const lucid::TensorImplPtr& b) {
            return fused_linear_forward("_fused_linear_gelu", FusedActivation::Gelu, x, w, b);
        },
        py::arg("x"), py::arg("weight"), py::arg("bias"),
        "Fused linear + tanh-approximate GELU inference kernel: y = GELU(x @ W.T + b), "
        "no autograd.");

    // Custom Metal kernel launcher.  The dtype string is converted to a
    // Dtype enum here rather than in C++ generic code so that Python callers
    // can pass plain strings ("f32", "f16", etc.).  The output TensorImpl
    // always lives on Device::GPU and is created without autograd.
    m.def(
        "_run_metal_kernel",
        [](const std::string& kernel_source, const std::string& function_name,
           const std::vector<lucid::TensorImplPtr>& inputs,
           const std::vector<std::int64_t>& output_shape_list, const std::string& dtype_str,
           const std::array<std::size_t, 3>& grid,
           const std::array<std::size_t, 3>& threads) -> lucid::TensorImplPtr {
            lucid::Dtype dt = lucid::Dtype::F32;
            if (dtype_str == "f16")
                dt = lucid::Dtype::F16;
            else if (dtype_str == "f32")
                dt = lucid::Dtype::F32;
            else if (dtype_str == "f64")
                dt = lucid::Dtype::F64;
            else if (dtype_str == "i32")
                dt = lucid::Dtype::I32;
            else if (dtype_str == "i64")
                dt = lucid::Dtype::I64;

            lucid::Shape out_shape(output_shape_list.begin(), output_shape_list.end());

            std::vector<lucid::Storage> in_storages;
            in_storages.reserve(inputs.size());
            for (const auto& t : inputs) {
                if (!t)
                    throw std::invalid_argument("_run_metal_kernel: null input tensor");
                // A shared tensor hands over the buffer itself, which the
                // runner binds without a copy; its storage is the CPU view or
                // GPU alias, which the runner would copy.
                if (const auto* sh = t->metal_shared())
                    in_storages.push_back(lucid::Storage{*sh});
                else
                    in_storages.push_back(t->storage());
            }

            auto& be = lucid::backend::Dispatcher::for_device(lucid::Device::GPU);
            lucid::Storage out = be.run_custom_metal_kernel(
                kernel_source, function_name, in_storages, out_shape, dt, grid, threads);

            // MetalKernelRunner returns SharedStorage, which lives in MTLResourceStorageModeShared
            // memory — directly accessible from both the GPU and CPU without a copy.  Labelled
            // Device::CPU, the TensorImpl adopts it as a CPU view (see its constructor), so
            // ordinary ops run on it, and .to(device='metal') relabels it without a copy.
            return lucid::compile::traced_derive(
                "_run_metal_kernel", inputs,
                std::make_shared<lucid::TensorImpl>(std::move(out), out_shape, dt,
                                                    lucid::Device::CPU, false));
        },
        py::arg("kernel_source"), py::arg("function_name"), py::arg("inputs"),
        py::arg("output_shape"), py::arg("dtype") = "f32", py::arg("grid"), py::arg("threads"),
        "Execute a custom Metal Shading Language (MSL) compute kernel on the GPU.\n"
        "Inputs are bound as read-only buffers; the output buffer is appended last.");
}
