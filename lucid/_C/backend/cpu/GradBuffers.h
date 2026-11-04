// lucid/_C/backend/cpu/GradBuffers.h
//
// Output buffers of a CPU fused backward, allocated only for the gradients
// the :class:`GradMask` wants.
//
// A CPU backward writes its gradients through raw typed pointers.  Every
// kernel used to size, allocate, zero and wrap each wanted buffer by hand
// and to hand a null pointer to the typed kernel for the rest.
// :class:`GradBuffers` is the single place that does it: ``data<T>(i)`` is
// the pointer for slot ``i`` (null when the slot is not wanted), ``zero()``
// clears the wanted buffers, ``take(i)`` / ``take_all()`` wrap them as
// :type:`Storage` and leave an empty one in an unwanted slot.

#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <memory>
#include <vector>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../GradMask.h"

namespace lucid {
namespace backend {
namespace cpu {

template <std::size_t N>
class GradBuffers {
public:
    // ``numel[i]`` is the element count of gradient ``i``; it is read only
    // when the mask wants that gradient.
    GradBuffers(const GradMask<N>& mask, Dtype dt, const std::array<std::size_t, N>& numel)
        : dt_(dt) {
        for (std::size_t i = 0; i < N; ++i) {
            wanted_[i] = mask.want(i);
            if (!wanted_[i])
                continue;
            nbytes_[i] = count_nbytes(numel[i], dtype_size(dt));
            ptr_[i] = allocate_aligned_bytes(nbytes_[i], Device::CPU);
        }
    }

    template <typename T>
    T* data(std::size_t i) const noexcept {
        return reinterpret_cast<T*>(ptr_[i].get());
    }

    void zero() const noexcept {
        for (std::size_t i = 0; i < N; ++i)
            if (nbytes_[i])
                std::memset(ptr_[i].get(), 0, nbytes_[i]);
    }

    Storage take(std::size_t i) {
        if (!wanted_[i])
            return Storage{};
        return Storage{CpuStorage{std::move(ptr_[i]), nbytes_[i], dt_}};
    }

    std::vector<Storage> take_all() {
        std::vector<Storage> out;
        out.reserve(N);
        for (std::size_t i = 0; i < N; ++i)
            out.push_back(take(i));
        return out;
    }

private:
    Dtype dt_;
    std::array<bool, N> wanted_{};
    std::array<std::size_t, N> nbytes_{};
    std::array<std::shared_ptr<std::byte[]>, N> ptr_{};
};

// The CPU storage behind ``s`` when some wanted gradient reads it, else null.
inline const CpuStorage* cpu_operand(bool needed, const Storage& s) {
    return needed ? &std::get<CpuStorage>(s) : nullptr;
}

// Typed read pointer of an optional operand; null when the operand is.
template <typename T>
const T* operand_data(const CpuStorage* s) noexcept {
    return s ? reinterpret_cast<const T*>(s->ptr.get()) : nullptr;
}

// ``out[r] += sum(rows[r, :])`` over ``n_rows`` rows of ``row_len``: the bias
// gradient of a convolution, one row per output channel.
template <typename T>
void add_row_sums(const T* rows, T* out, int n_rows, int row_len) noexcept {
    for (int r = 0; r < n_rows; ++r) {
        const T* row = rows + static_cast<std::size_t>(r) * row_len;
        T s = T(0);
        for (int j = 0; j < row_len; ++j)
            s += row[j];
        out[r] += s;
    }
}

}  // namespace cpu
}  // namespace backend
}  // namespace lucid
