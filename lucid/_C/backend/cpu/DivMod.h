// lucid/_C/backend/cpu/DivMod.h
//
// The CPU kernels of the floor-division family: ``//``, ``%`` and ``fmod``
// on equal-shape operands, exactly, in the operands' own dtype.  The rule
// is in ../DivMod.h.
//
// Integers run at their own width; Accelerate has no integer division.
// Floats take ``fmod`` from vForce (``vvfmodf`` / ``vvfmod``, exact and
// IEEE on inf, NaN and signed zeros) and finish element by element.  The
// 16-bit floats are not accepted: the caller widens them to float32 first.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

#include <Accelerate/Accelerate.h>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Storage.h"
#include "../DivMod.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

namespace divmod_detail {

// Elements per parallel chunk: an integer division costs a few cycles, so a
// chunk this size outweighs the dispatch.  Floating chunks are aligned to
// 16-element boundaries (``kBlock``) so ``vvfmodf`` / ``vvfmod`` always sees
// the same SIMD blocks at any core count (LCD-596).
constexpr std::size_t kGrain = std::size_t{1} << 16;
constexpr std::size_t kBlock = 16;

template <typename T>
void vector_fmod(T* out, const T* a, const T* b, std::size_t n) {
    // vForce counts in ``int``, rounded down to a multiple of ``kBlock`` so
    // any multi-chunk call keeps ``kBlock`` alignment.
    constexpr std::size_t kChunk =
        (static_cast<std::size_t>(std::numeric_limits<int>::max()) / kBlock) * kBlock;
    for (std::size_t off = 0; off < n; off += kChunk) {
        const int m = static_cast<int>(std::min(kChunk, n - off));
        if constexpr (std::is_same_v<T, float>)
            vvfmodf(out + off, a + off, b + off, &m);
        else
            vvfmod(out + off, a + off, b + off, &m);
    }
}

template <DivMod M, typename T>
void run(T* out, const T* a, const T* b, std::size_t n) {
    if constexpr (std::is_floating_point_v<T>) {
        const std::size_t blocks = (n + kBlock - 1) / kBlock;
        parallel_for(blocks, kGrain / kBlock, [&](std::size_t b_lo, std::size_t b_hi) {
            const std::size_t lo = b_lo * kBlock;
            const std::size_t hi = std::min(b_hi * kBlock, n);
            vector_fmod(out + lo, a + lo, b + lo, hi - lo);
            if constexpr (M != DivMod::Fmod) {
                for (std::size_t i = lo; i < hi; ++i)
                    out[i] = divmod_floating<M>(a[i], b[i], out[i]);
            }
        });
    } else {
        parallel_for(n, kGrain, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t i = lo; i < hi; ++i)
                out[i] = divmod_integer<M>(a[i], b[i]);
        });
    }
}

template <typename T>
bool any_zero(const T* p, std::size_t n) {
    return std::any_of(p, p + n, [](T v) { return v == T{0}; });
}

// ``fn(T{})`` with the element type of ``dt``: the integers and the two
// full-width floats.
template <typename Fn>
void visit(Dtype dt, const char* op, Fn&& fn) {
    switch (dt) {
    case Dtype::I8:
        fn(std::int8_t{});
        break;
    case Dtype::I16:
        fn(std::int16_t{});
        break;
    case Dtype::I32:
        fn(std::int32_t{});
        break;
    case Dtype::I64:
        fn(std::int64_t{});
        break;
    case Dtype::F32:
        fn(float{});
        break;
    case Dtype::F64:
        fn(double{});
        break;
    default:
        ErrorBuilder(op).not_implemented(std::string("dtype ") + std::string(dtype_name(dt)) +
                                         " is not supported by the CPU kernel");
    }
}

}  // namespace divmod_detail

// ``a (op) b`` element-wise over ``n`` elements of ``dt``.
inline CpuStorage
divmod(DivMod mode, const CpuStorage& a, const CpuStorage& b, std::size_t n, Dtype dt) {
    CpuStorage out;
    out.dtype = dt;
    out.nbytes = n * dtype_size(dt);
    out.ptr = allocate_aligned_bytes(out.nbytes, Device::CPU);
    divmod_detail::visit(dt, "divmod", [&](auto tag) {
        using T = decltype(tag);
        auto* o = reinterpret_cast<T*>(out.ptr.get());
        const auto* pa = reinterpret_cast<const T*>(a.ptr.get());
        const auto* pb = reinterpret_cast<const T*>(b.ptr.get());
        switch (mode) {
        case DivMod::FloorDiv:
            divmod_detail::run<DivMod::FloorDiv>(o, pa, pb, n);
            break;
        case DivMod::Remainder:
            divmod_detail::run<DivMod::Remainder>(o, pa, pb, n);
            break;
        case DivMod::Fmod:
            divmod_detail::run<DivMod::Fmod>(o, pa, pb, n);
            break;
        }
    });
    return out;
}

// Whether any of the ``n`` integer elements of ``s`` is zero.
inline bool any_zero_integer(const CpuStorage& s, std::size_t n, Dtype dt) {
    bool found = false;
    divmod_detail::visit(dt, "divmod", [&](auto tag) {
        using T = decltype(tag);
        found = divmod_detail::any_zero(reinterpret_cast<const T*>(s.ptr.get()), n);
    });
    return found;
}

}  // namespace lucid::backend::cpu
