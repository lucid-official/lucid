// lucid/_C/backend/cpu/SortSelect.h
//
// The CPU sort / top-k selection kernel and the scatter-add that is its
// adjoint (sort, argsort, topk and their backward, as_strided backward).
//
// The positions they produce and consume are int64, the index dtype every
// index-returning op answers with (argmax, nonzero, the reference's sort /
// argsort / topk).  The scatter still reads an int32 index too, because
// as_strided's backward builds one.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "Parallel.h"

namespace lucid::backend::cpu {

template <typename T>
struct SortKeyTrait;

template <>
struct SortKeyTrait<float> {
    using Key = std::uint32_t;
    static Key encode(float v, bool descending) noexcept {
        if (std::isnan(v))
            return descending ? 0u : 0xFFFFFFFFu;
        if (v == 0.0f)
            v = 0.0f;
        std::uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        const std::uint32_t k = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
        return descending ? ~k : k;
    }
};

template <>
struct SortKeyTrait<double> {
    using Key = std::uint64_t;
    static Key encode(double v, bool descending) noexcept {
        if (std::isnan(v))
            return descending ? 0ull : ~0ull;
        if (v == 0.0)
            v = 0.0;
        std::uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        const std::uint64_t k = (bits & (1ull << 63)) ? ~bits : (bits ^ (1ull << 63));
        return descending ? ~k : k;
    }
};

template <>
struct SortKeyTrait<std::int32_t> {
    using Key = std::uint32_t;
    static Key encode(std::int32_t v, bool descending) noexcept {
        const std::uint32_t k = static_cast<std::uint32_t>(v) ^ 0x80000000u;
        return descending ? ~k : k;
    }
};

template <>
struct SortKeyTrait<std::int64_t> {
    using Key = std::uint64_t;
    static Key encode(std::int64_t v, bool descending) noexcept {
        const std::uint64_t k = static_cast<std::uint64_t>(v) ^ (1ull << 63);
        return descending ? ~k : k;
    }
};

template <typename Key, typename Pos = std::uint32_t>
struct KeyPos {
    Key key;
    Pos pos;
};

template <typename T>
constexpr std::size_t radix_min_len() noexcept {
    return sizeof(typename SortKeyTrait<T>::Key) == 8 ? 512 : 256;
}

template <typename T, typename Pos = std::uint32_t>
const KeyPos<typename SortKeyTrait<T>::Key, Pos>*
radix_sort_line(const T* src_line,
                std::size_t L,
                std::size_t stride,
                bool descending,
                std::vector<KeyPos<typename SortKeyTrait<T>::Key, Pos>>& buf0,
                std::vector<KeyPos<typename SortKeyTrait<T>::Key, Pos>>& buf1) {
    using Key = typename SortKeyTrait<T>::Key;
    constexpr std::size_t kPasses = sizeof(Key);
    Pos hist[kPasses][256] = {};
    for (std::size_t i = 0; i < L; ++i) {
        const Key k = SortKeyTrait<T>::encode(src_line[i * stride], descending);
        buf0[i] = {k, static_cast<Pos>(i)};
        for (std::size_t p = 0; p < kPasses; ++p)
            ++hist[p][(k >> (p * 8)) & 0xFFu];
    }
    auto* cur = buf0.data();
    auto* nxt = buf1.data();
    const Key first_key = cur[0].key;
    for (std::size_t p = 0; p < kPasses; ++p) {
        const std::size_t shift = p * 8;
        if (hist[p][(first_key >> shift) & 0xFFu] == static_cast<Pos>(L))
            continue;
        Pos sum = 0;
        for (std::size_t b = 0; b < 256; ++b) {
            const Pos c = hist[p][b];
            hist[p][b] = sum;
            sum += c;
        }
        for (std::size_t i = 0; i < L; ++i)
            nxt[hist[p][(cur[i].key >> shift) & 0xFFu]++] = cur[i];
        std::swap(cur, nxt);
    }
    return cur;
}

template <typename T, typename Pos = std::uint32_t>
const KeyPos<typename SortKeyTrait<T>::Key, Pos>*
sort_line_keys(const T* src_line,
               std::size_t L,
               std::size_t stride,
               std::size_t K,
               bool descending,
               std::vector<KeyPos<typename SortKeyTrait<T>::Key, Pos>>& buf0,
               std::vector<KeyPos<typename SortKeyTrait<T>::Key, Pos>>& buf1) {
    if (K == L && L >= radix_min_len<T>())
        return radix_sort_line<T, Pos>(src_line, L, stride, descending, buf0, buf1);
    for (std::size_t i = 0; i < L; ++i)
        buf0[i] = {SortKeyTrait<T>::encode(src_line[i * stride], descending), static_cast<Pos>(i)};
    auto cmp = [](const auto& lhs, const auto& rhs) {
        return lhs.key != rhs.key ? (lhs.key < rhs.key) : (lhs.pos < rhs.pos);
    };
    if (K == L)
        std::sort(buf0.begin(), buf0.end(), cmp);
    else if (K > 0)
        std::partial_sort(buf0.begin(), buf0.begin() + static_cast<std::ptrdiff_t>(K), buf0.end(),
                          cmp);
    return buf0.data();
}

template <typename T, typename Entry>
void write_sorted_line(const T* src_line,
                       const Entry* sorted,
                       T* dst,
                       std::int64_t* idx_dst,
                       std::size_t o,
                       std::size_t j,
                       std::size_t K,
                       std::size_t inner) {
    if (inner == 1) {
        std::int64_t* row_idx = idx_dst + o * K;
        if (dst) {
            T* row_dst = dst + o * K;
            for (std::size_t k = 0; k < K; ++k) {
                const auto pos = sorted[k].pos;
                row_dst[k] = src_line[pos];
                row_idx[k] = static_cast<std::int64_t>(pos);
            }
        } else {
            for (std::size_t k = 0; k < K; ++k)
                row_idx[k] = static_cast<std::int64_t>(sorted[k].pos);
        }
        return;
    }
    for (std::size_t k = 0; k < K; ++k) {
        const std::size_t out_flat = (o * K + k) * inner + j;
        const auto pos = sorted[k].pos;
        if (dst)
            dst[out_flat] = src_line[static_cast<std::size_t>(pos) * inner];
        idx_dst[out_flat] = static_cast<std::int64_t>(pos);
    }
}

// ``outer`` / ``inner`` extents around ``axis`` of ``shape``.
inline std::pair<std::size_t, std::size_t> around_axis(const Shape& shape, int axis) {
    std::size_t outer = 1;
    std::size_t inner = 1;
    for (int d = 0; d < axis; ++d)
        outer *= static_cast<std::size_t>(shape[static_cast<std::size_t>(d)]);
    for (std::size_t d = static_cast<std::size_t>(axis) + 1; d < shape.size(); ++d)
        inner *= static_cast<std::size_t>(shape[d]);
    return {outer, inner};
}

// The first ``output_shape[axis]`` elements of each line along ``axis`` in
// sorted order, and the int64 position each came from.  Equal keys keep
// their input order, in a full sort and a top-k alike — what sort / argsort
// document, and what the GPU path does.
template <typename T>
std::pair<CpuStorage, CpuStorage> sort_select_cpu(const CpuStorage& input,
                                                  const Shape& input_shape,
                                                  const Shape& output_shape,
                                                  int axis,
                                                  Dtype dt,
                                                  bool descending,
                                                  bool need_values = true) {
    const std::size_t n_out = shape_numel(output_shape);
    CpuStorage values;
    if (need_values) {
        values = CpuStorage{allocate_aligned_bytes(n_out * dtype_size(dt), Device::CPU),
                            n_out * dtype_size(dt), dt};
    }
    CpuStorage indices{allocate_aligned_bytes(n_out * sizeof(std::int64_t), Device::CPU),
                       n_out * sizeof(std::int64_t), Dtype::I64};

    const auto* src = reinterpret_cast<const T*>(input.ptr.get());
    auto* dst = need_values ? reinterpret_cast<T*>(values.ptr.get()) : nullptr;
    auto* idx_dst = reinterpret_cast<std::int64_t*>(indices.ptr.get());
    const auto [outer, inner] = around_axis(input_shape, axis);
    const std::size_t L = static_cast<std::size_t>(input_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(output_shape[static_cast<std::size_t>(axis)]);
    if (K == 0 || outer == 0 || inner == 0)
        return {std::move(values), std::move(indices)};

    using Key = typename SortKeyTrait<T>::Key;
    const std::size_t total_lines = outer * inner;
    const std::size_t line_grain = std::max<std::size_t>(1, 4096 / std::max<std::size_t>(L, 1));
    const std::size_t buf1_len = K == L && L >= radix_min_len<T>() ? L : 0;
    parallel_for(total_lines, line_grain, [&](std::size_t begin, std::size_t end) {
        std::vector<KeyPos<Key>> buf0(L);
        std::vector<KeyPos<Key>> buf1(buf1_len);
        for (std::size_t line = begin; line < end; ++line) {
            const std::size_t o = line / inner;
            const std::size_t j = line % inner;
            const T* src_line = src + o * L * inner + j;
            const auto* sorted = sort_line_keys<T>(src_line, L, inner, K, descending, buf0, buf1);
            write_sorted_line(src_line, sorted, dst, idx_dst, o, j, K, inner);
        }
    });
    return {std::move(values), std::move(indices)};
}

// ``dst[..., idx[..., k, ...], ...] += grad[..., k, ...]`` along ``axis``, into
// ``dst`` of ``output_shape`` (zeroed by the caller).  A negative position
// counts from the end of the axis.
template <typename T, typename I>
void scatter_add_axis_lines(T* dst,
                            const T* grad,
                            const I* idx,
                            const Shape& output_shape,
                            const Shape& grad_shape,
                            int axis) {
    const auto [outer, inner] = around_axis(output_shape, axis);
    const auto L = static_cast<std::int64_t>(output_shape[static_cast<std::size_t>(axis)]);
    const std::size_t K = static_cast<std::size_t>(grad_shape[static_cast<std::size_t>(axis)]);
    for (std::size_t o = 0; o < outer; ++o) {
        for (std::size_t j = 0; j < inner; ++j) {
            for (std::size_t k = 0; k < K; ++k) {
                const std::size_t grad_flat = (o * K + k) * inner + j;
                std::int64_t src_k = static_cast<std::int64_t>(idx[grad_flat]);
                if (src_k < 0)
                    src_k += L;
                dst[(o * static_cast<std::size_t>(L) + static_cast<std::size_t>(src_k)) * inner +
                    j] += grad[grad_flat];
            }
        }
    }
}

// :func:`scatter_add_axis_lines` on the index's own width.
template <typename T>
void scatter_add_axis_any_index(T* dst,
                                const T* grad,
                                const CpuStorage& idx,
                                const Shape& output_shape,
                                const Shape& grad_shape,
                                int axis) {
    if (idx.dtype == Dtype::I64)
        scatter_add_axis_lines(dst, grad, reinterpret_cast<const std::int64_t*>(idx.ptr.get()),
                               output_shape, grad_shape, axis);
    else
        scatter_add_axis_lines(dst, grad, reinterpret_cast<const std::int32_t*>(idx.ptr.get()),
                               output_shape, grad_shape, axis);
}

}  // namespace lucid::backend::cpu
