// lucid/_C/ops/bfunc/Tensordot.cpp
//
// Implements tensordot_op.  For gradient-tracked tensors the call is lowered to
// einsum_op.  For the CPU inference path, inputs are permuted and reshaped into
// 2-D matrices then contracted with a scalar GEMM loop.  GPU inference
// delegates to the backend tensordot primitive.

#include "Tensordot.h"

#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../einops/Einops.h"
#include "../ufunc/Astype.h"
#include "../utils/Promote.h"
#include "_Detail.h"

namespace lucid {

namespace {

using bfunc_detail::allocate_cpu;
using bfunc_detail::fresh;
using bfunc_detail::validate_pair;

// Build an einsum contraction string equivalent to the tensordot over the
// given axis pairs.
//
// Contracted axes in A and B are assigned matching uppercase letters starting
// at 'A'.  Free axes (not contracted) in A and B receive distinct lowercase
// letters starting at 'a', and appear in the output in A-free, B-free order.
//
// Example — tensordot of a 4-D and 3-D tensor contracting axis (2) of A with
// axis (0) of B:  tensordot_einsum_pattern(4, 3, {2}, {0}) → "abAd,Ace->abcde"
//   ('A' is the shared contracted label; free labels in A are {a,b,d} and in B
//    are {c,e}).
std::string tensordot_einsum_pattern(std::size_t na,
                                     std::size_t nb,
                                     const std::vector<int>& axes_a,
                                     const std::vector<int>& axes_b) {
    // Normalise negative axis indices to the range [0, n).
    auto norm = [](int ax, std::size_t n) { return ax < 0 ? ax + static_cast<int>(n) : ax; };
    std::set<int> ca_set, cb_set;
    for (auto a : axes_a)
        ca_set.insert(norm(a, na));
    for (auto b : axes_b)
        cb_set.insert(norm(b, nb));

    std::string a_lhs(na, '?'), b_lhs(nb, '?'), rhs;
    char free = 'a';    // Next free-axis label.
    char shared = 'A';  // Next contracted-axis label.

    // Assign shared (contracted) labels to paired axis positions.
    for (std::size_t i = 0; i < axes_a.size(); ++i) {
        int pa = norm(axes_a[i], na);
        int pb = norm(axes_b[i], nb);
        a_lhs[pa] = shared;
        b_lhs[pb] = shared;
        ++shared;
    }

    // Assign free labels to the remaining axes in A (appear first in rhs).
    for (std::size_t i = 0; i < na; ++i) {
        if (a_lhs[i] == '?') {
            a_lhs[i] = free;
            rhs.push_back(free);
            ++free;
        }
    }
    // Assign free labels to the remaining axes in B (appear after A's free dims).
    for (std::size_t i = 0; i < nb; ++i) {
        if (b_lhs[i] == '?') {
            b_lhs[i] = free;
            rhs.push_back(free);
            ++free;
        }
    }
    return a_lhs + "," + b_lhs + "->" + rhs;
}

// The contraction, for operands the entry below has already cast.
TensorImplPtr contraction(const TensorImplPtr& a,
                          const TensorImplPtr& b,
                          std::vector<int> axes_a,
                          std::vector<int> axes_b) {
    validate_pair(a, b, "tensordot");
    if (axes_a.size() != axes_b.size())
        ErrorBuilder("tensordot").fail("axes_a and axes_b must have equal length");

    // Autograd path: express the contraction as an einsum so that the einsum
    // backward node handles gradient computation.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(
            tensordot_einsum_pattern(a->shape().size(), b->shape().size(), axes_a, axes_b), {a, b});
    }

    const Dtype dt = a->dtype();
    const Device device = a->device();

    // Accelerate has no half GEMM, and the loop below is written out for
    // float and double only — so the pair widens at the door and the
    // result rounds back, which is the route every other CPU entry point
    // takes for these two formats.  Metal ran float16 and the CPU did
    // not, which made a half-precision contraction a device-dependent
    // proposition.
    if (device == Device::CPU && is_half_float(dt))
        return astype_op(
            contraction(astype_op(a, Dtype::F32), astype_op(b, Dtype::F32), axes_a, axes_b), dt);

    OpScopeFull scope{"tensordot", device, dt, Shape{}};
    {
        std::vector<std::int64_t> ax_a, ax_b;
        for (int v : axes_a)
            ax_a.push_back(static_cast<std::int64_t>(v));
        for (int v : axes_b)
            ax_b.push_back(static_cast<std::int64_t>(v));
        scope.set_attr("axes_a", std::move(ax_a));
        scope.set_attr("axes_b", std::move(ax_b));
    }

    // GPU inference path: delegate to the backend tensordot primitive.
    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).tensordot(
            a->storage(), b->storage(), a->shape(), b->shape(), Shape{}, axes_a, axes_b, dt);
        // Read the actual output shape from the MLX array because the backend
        // may have resolved trailing scalar squeezes.
        const auto& gs = storage_gpu(out_storage);
        Shape out_shape;
        for (auto d : gs.arr->shape())
            out_shape.push_back(static_cast<std::int64_t>(d));
        auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
        if (auto* trc = ::lucid::compile::current_tracer()) {
            trc->on_op_io({a, b}, result);
        }
        return result;
    }

    // CPU inference path: manually permute and reshape both tensors into 2-D
    // matrices [free_dims × contract_dims] and [contract_dims × free_dims],
    // then execute a scalar GEMM.
    //
    // contract() permutes the axes of t so that either the contracted axes come
    // first (put_first=true) or last (put_first=false), then flattens contracted
    // and free axes into a 2-D CpuStorage.  Returns the permuted storage, the
    // list of kept (free) dimension sizes, and the product of contracted dims.
    auto contract = [&](const TensorImplPtr& t, const std::vector<int>& axes_contract,
                        bool put_first) {
        const std::size_t nd = t->shape().size();
        std::vector<bool> is_c(nd, false);
        for (auto ax : axes_contract) {
            int p = ax < 0 ? ax + (int)nd : ax;
            if (p < 0 || p >= (int)nd)
                ErrorBuilder("tensordot").index_error("axis out of range");
            is_c[p] = true;
        }
        std::vector<int> perm;
        std::vector<std::int64_t> kept;
        std::int64_t contract_size = 1;
        if (put_first) {
            // Contracted axes first, then free axes — used for B so the
            // resulting matrix is [K × N_free].
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape()[p];
            }
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape()[d]);
                }
        } else {
            // Free axes first, then contracted axes — used for A so the
            // resulting matrix is [M_free × K].
            for (std::size_t d = 0; d < nd; ++d)
                if (!is_c[d]) {
                    perm.push_back((int)d);
                    kept.push_back(t->shape()[d]);
                }
            for (auto ax : axes_contract) {
                int p = ax < 0 ? ax + (int)nd : ax;
                perm.push_back(p);
                contract_size *= t->shape()[p];
            }
        }
        Shape src_shape = t->shape();
        const auto& cs = storage_cpu(t->storage());
        auto& be = backend::Dispatcher::for_device(Device::CPU);
        CpuStorage dst = be.permute_cpu(cs, src_shape, perm, dt);
        return std::tuple<CpuStorage, std::vector<std::int64_t>, std::int64_t>{
            std::move(dst), std::move(kept), contract_size};
    };

    // A is arranged as [M × K]; B is arranged as [K × N].
    auto [a_cpu, a_kept, a_contract] = contract(a, axes_a, false);
    auto [b_cpu, b_kept, b_contract] = contract(b, axes_b, true);
    if (a_contract != b_contract)
        ErrorBuilder("tensordot").fail("contracted dim sizes don't match");

    const std::int64_t M =
        std::accumulate(a_kept.begin(), a_kept.end(), (std::int64_t)1, std::multiplies<>());
    const std::int64_t K = a_contract;
    const std::int64_t N =
        std::accumulate(b_kept.begin(), b_kept.end(), (std::int64_t)1, std::multiplies<>());

    Shape out_shape(a_kept.begin(), a_kept.end());
    out_shape.insert(out_shape.end(), b_kept.begin(), b_kept.end());
    auto out_cpu = allocate_cpu(out_shape, dt);

    // Scalar GEMM: C[i,j] = Σ_k A[i,k] * B[k,j].
    // This is used only when gradient tracking is off; for autograd the call
    // is routed through einsum_op above.
    auto gemm = [&](auto* op, const auto* ap, const auto* bp) {
        using T = std::remove_pointer_t<decltype(op)>;
        for (std::int64_t i = 0; i < M; ++i)
            for (std::int64_t j = 0; j < N; ++j) {
                T s{};
                for (std::int64_t k = 0; k < K; ++k)
                    s = s + ap[i * K + k] * bp[k * N + j];
                op[i * N + j] = s;
            }
    };
    if (dt == Dtype::F32)
        gemm(reinterpret_cast<float*>(out_cpu.ptr.get()),
             reinterpret_cast<const float*>(a_cpu.ptr.get()),
             reinterpret_cast<const float*>(b_cpu.ptr.get()));
    else if (dt == Dtype::F64)
        gemm(reinterpret_cast<double*>(out_cpu.ptr.get()),
             reinterpret_cast<const double*>(a_cpu.ptr.get()),
             reinterpret_cast<const double*>(b_cpu.ptr.get()));
    else
        ErrorBuilder("tensordot").not_implemented("dtype not supported");

    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    // Recorded as the GPU path is.  Without it a CPU trace never saw the
    // contraction: a Core ML export took its result for a constant — the
    // example's answer, frozen — or, with nothing else reading the input,
    // refused it as unread.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({a, b}, result);
    }
    return result;
}

}  // namespace

TensorImplPtr tensordot_op(const TensorImplPtr& a,
                           const TensorImplPtr& b,
                           std::vector<int> axes_a,
                           std::vector<int> axes_b) {
    if (!a || !b)
        ErrorBuilder("tensordot").fail("null input");
    if (a->device() == Device::CPU)
        validate_pair(a, b, "tensordot");
    // The row: WIDEST under Metal autocast, LOWER under the CPU's.  The
    // contraction runs as an einsum or a GEMM, and neither applies its own
    // row inside the scope.
    const AmpScope amp{"tensordot", a->device()};
    const auto operands = amp.cast_widest({a, b});
    return contraction(operands[0], operands[1], std::move(axes_a), std::move(axes_b));
}

}  // namespace lucid
