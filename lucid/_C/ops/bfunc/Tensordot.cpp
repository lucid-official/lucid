// lucid/_C/ops/bfunc/Tensordot.cpp
//
// Implements tensordot_op.  For gradient-tracked tensors the call is lowered to
// einsum_op.  For the CPU inference path, inputs are permuted and reshaped into
// 2-D matrices then contracted with a scalar GEMM loop.  GPU inference
// delegates to the backend tensordot primitive.

#include "Tensordot.h"

#include <complex>
#include <cstdint>
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

void validate_tensordot_axes(const Shape& sa,
                             const Shape& sb,
                             std::vector<int>& axes_a,
                             std::vector<int>& axes_b) {
    if (axes_a.size() != axes_b.size())
        ErrorBuilder("tensordot").invalid_argument("axes_a and axes_b must have equal length");
    const int na = static_cast<int>(sa.size());
    const int nb = static_cast<int>(sb.size());
    std::vector<bool> seen_a(sa.size(), false);
    std::vector<bool> seen_b(sb.size(), false);
    for (std::size_t i = 0; i < axes_a.size(); ++i) {
        int pa = axes_a[i] < 0 ? axes_a[i] + na : axes_a[i];
        int pb = axes_b[i] < 0 ? axes_b[i] + nb : axes_b[i];
        if (pa < 0 || pa >= na || pb < 0 || pb >= nb)
            ErrorBuilder("tensordot").index_error("axis out of range");
        if (seen_a[static_cast<std::size_t>(pa)] || seen_b[static_cast<std::size_t>(pb)])
            ErrorBuilder("tensordot").invalid_argument("duplicate contracted axis");
        seen_a[static_cast<std::size_t>(pa)] = true;
        seen_b[static_cast<std::size_t>(pb)] = true;
        if (sa[static_cast<std::size_t>(pa)] != sb[static_cast<std::size_t>(pb)])
            throw ShapeMismatch(sa, sb, "tensordot: contracted dim sizes don't match");
        axes_a[i] = pa;
        axes_b[i] = pb;
    }
}

std::tuple<CpuStorage, std::vector<std::int64_t>, std::int64_t> contract_tensordot_operand(
    const TensorImplPtr& t, const std::vector<int>& axes_contract, bool put_first, Dtype dt) {
    const std::size_t nd = t->shape().size();
    std::vector<bool> is_c(nd, false);
    for (int ax : axes_contract)
        is_c[static_cast<std::size_t>(ax)] = true;
    std::vector<int> perm;
    std::vector<std::int64_t> kept;
    std::int64_t contract_size = 1;
    for (int ax : axes_contract)
        contract_size *= t->shape()[static_cast<std::size_t>(ax)];
    for (std::size_t d = 0; d < nd; ++d) {
        if (!is_c[d])
            kept.push_back(t->shape()[d]);
    }
    if (put_first) {
        perm.insert(perm.end(), axes_contract.begin(), axes_contract.end());
        for (std::size_t d = 0; d < nd; ++d)
            if (!is_c[d])
                perm.push_back(static_cast<int>(d));
    } else {
        for (std::size_t d = 0; d < nd; ++d)
            if (!is_c[d])
                perm.push_back(static_cast<int>(d));
        perm.insert(perm.end(), axes_contract.begin(), axes_contract.end());
    }
    CpuStorage dst = backend::Dispatcher::for_device(Device::CPU)
                         .permute_cpu(storage_cpu(t->storage()), t->shape(), perm, dt);
    return {std::move(dst), std::move(kept), contract_size};
}

template <typename T>
void gemm_loop(CpuStorage& out_cpu,
               const CpuStorage& a_cpu,
               const CpuStorage& b_cpu,
               std::int64_t M,
               std::int64_t K,
               std::int64_t N) {
    auto* op = reinterpret_cast<T*>(out_cpu.ptr.get());
    const auto* ap = reinterpret_cast<const T*>(a_cpu.ptr.get());
    const auto* bp = reinterpret_cast<const T*>(b_cpu.ptr.get());
    for (std::int64_t i = 0; i < M; ++i)
        for (std::int64_t j = 0; j < N; ++j) {
            T s{};
            for (std::int64_t k = 0; k < K; ++k)
                s = s + ap[i * K + k] * bp[k * N + j];
            op[i * N + j] = s;
        }
}

void run_cpu_tensordot_gemm(CpuStorage& out_cpu,
                            const CpuStorage& a_cpu,
                            const CpuStorage& b_cpu,
                            std::int64_t M,
                            std::int64_t K,
                            std::int64_t N,
                            Dtype dt) {
    switch (dt) {
    case Dtype::F32:
        gemm_loop<float>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::F64:
        gemm_loop<double>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::C64:
        gemm_loop<std::complex<float>>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::I8:
        gemm_loop<std::int8_t>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::I16:
        gemm_loop<std::int16_t>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::I32:
        gemm_loop<std::int32_t>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    case Dtype::I64:
        gemm_loop<std::int64_t>(out_cpu, a_cpu, b_cpu, M, K, N);
        break;
    default:
        ErrorBuilder("tensordot").not_implemented("dtype not supported");
    }
}

// The contraction, for operands the entry below has already cast.
TensorImplPtr contraction(const TensorImplPtr& a,
                          const TensorImplPtr& b,
                          std::vector<int> axes_a,
                          std::vector<int> axes_b) {
    validate_pair(a, b, "tensordot");
    const Dtype dt = a->dtype();
    if (dt == Dtype::Bool)
        ErrorBuilder("tensordot").not_implemented("not implemented for bool");
    validate_tensordot_axes(a->shape(), b->shape(), axes_a, axes_b);

    // Autograd path: express the contraction as an einsum so that the einsum
    // backward node handles gradient computation.
    if (GradMode::is_enabled() && (a->requires_grad() || b->requires_grad())) {
        return einsum_op(
            tensordot_einsum_pattern(a->shape().size(), b->shape().size(), axes_a, axes_b), {a, b});
    }

    const Device device = a->device();
    if (device == Device::CPU && is_half_float(dt))
        return astype_op(
            contraction(astype_op(a, Dtype::F32), astype_op(b, Dtype::F32), axes_a, axes_b), dt);

    OpScopeFull scope{"tensordot", device, dt, Shape{}};
    {
        std::vector<std::int64_t> ax_a(axes_a.begin(), axes_a.end());
        std::vector<std::int64_t> ax_b(axes_b.begin(), axes_b.end());
        scope.set_attr("axes_a", std::move(ax_a));
        scope.set_attr("axes_b", std::move(ax_b));
    }

    if (device == Device::GPU) {
        auto out_storage = backend::Dispatcher::for_device(device).tensordot(
            a->storage(), b->storage(), a->shape(), b->shape(), Shape{}, axes_a, axes_b, dt);
        const auto& gs = storage_gpu(out_storage);
        Shape out_shape(gs.arr->shape().begin(), gs.arr->shape().end());
        auto result = fresh(std::move(out_storage), std::move(out_shape), dt, device);
        if (auto* trc = ::lucid::compile::current_tracer())
            trc->on_op_io({a, b}, result);
        return result;
    }

    auto [a_cpu, a_kept, a_contract] = contract_tensordot_operand(a, axes_a, false, dt);
    auto [b_cpu, b_kept, b_contract] = contract_tensordot_operand(b, axes_b, true, dt);
    (void)b_contract;

    const std::int64_t M =
        std::accumulate(a_kept.begin(), a_kept.end(), (std::int64_t)1, std::multiplies<>());
    const std::int64_t N =
        std::accumulate(b_kept.begin(), b_kept.end(), (std::int64_t)1, std::multiplies<>());

    Shape out_shape(a_kept.begin(), a_kept.end());
    out_shape.insert(out_shape.end(), b_kept.begin(), b_kept.end());
    auto out_cpu = allocate_cpu(out_shape, dt);
    run_cpu_tensordot_gemm(out_cpu, a_cpu, b_cpu, M, a_contract, N, dt);

    auto result = fresh(Storage{std::move(out_cpu)}, std::move(out_shape), dt, device);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a, b}, result);
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
