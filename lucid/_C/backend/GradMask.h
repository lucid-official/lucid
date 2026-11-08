// lucid/_C/backend/GradMask.h
//
// Which gradients a fused backward kernel is asked to compute.
//
// A fused backward (``linear_backward``, ``layer_norm_backward``,
// ``conv_nd_backward`` ...) returns one gradient per differentiable input,
// but a node often needs only some of them: a frozen weight, an input that
// does not require grad.  The kernel skips the work for the rest and hands
// back an empty :type:`Storage` in that slot.  Autograd never reads a slot
// whose input did not ask for it.
//
// :class:`GradMask` is the one owner of that rule.  A kernel asks it
// ``want(i)`` / ``want_any(i, j)`` instead of indexing a ``std::array<bool,N>``
// and re-spelling ``mask[i] ? compute : Storage{}`` at every branch, and it
// computes a slot through ``make(i, f)`` so an unwanted slot costs nothing.
// The default mask wants every gradient, which is what a caller that passes
// no mask gets.
//
// The pointer helpers at the bottom serve the typed kernels below the
// backends, which receive a null output pointer for a gradient nobody wants.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "../core/Storage.h"

namespace lucid {
namespace backend {

// Wanted-gradient flags for a backward with ``N`` differentiable inputs.
template <std::size_t N>
class GradMask {
public:
    // Wants every gradient.
    constexpr GradMask() noexcept { bits_.fill(true); }

    // One flag per input, in the order the backward returns its gradients:
    // ``{needs_input_grad(0), needs_input_grad(1), ...}``.
    template <typename... Flags,
              typename = std::enable_if_t<sizeof...(Flags) == N &&
                                          (std::is_convertible_v<Flags, bool> && ...)>>
    constexpr GradMask(Flags... flags) noexcept : bits_{static_cast<bool>(flags)...} {}

    constexpr bool want(std::size_t i) const noexcept { return bits_[i]; }

    // True when at least one of the listed gradients is wanted.  Also the
    // question "does any wanted gradient read this operand".
    template <typename... Index>
    constexpr bool want_any(Index... i) const noexcept {
        return (bits_[i] || ...);
    }

    // True when at least one gradient at all is wanted.
    constexpr bool any() const noexcept {
        for (bool b : bits_)
            if (b)
                return true;
        return false;
    }

    // ``f()`` when gradient ``i`` is wanted, a default-constructed result
    // (an empty :type:`Storage`) when it is not.  ``f`` is not called then.
    template <typename F>
    auto make(std::size_t i, F&& f) const -> std::invoke_result_t<F> {
        if (bits_[i])
            return f();
        return {};
    }

    // What a backward returns when no gradient is wanted: ``N`` empty slots.
    static std::vector<Storage> none() { return std::vector<Storage>(N); }

private:
    std::array<bool, N> bits_;
};

// The two promises every backend relies on, pinned at compile time: a mask a
// caller did not pass wants every gradient (a default-argument ``GradMask3{}``
// in :class:`IBackend` must never silently skip one), and the constructor's
// flags land in slot order (``{dx, dW, db}`` is not ``{db, dW, dx}``).
namespace detail {
template <std::size_t N>
constexpr bool default_wants_all() noexcept {
    constexpr GradMask<N> mask{};
    for (std::size_t i = 0; i < N; ++i)
        if (!mask.want(i))
            return false;
    return true;
}
constexpr GradMask<3> kFlagOrderProbe{true, false, false};
}  // namespace detail

static_assert(detail::default_wants_all<2>() && detail::default_wants_all<3>() &&
                  detail::default_wants_all<4>() && detail::default_wants_all<5>(),
              "a default GradMask must want every gradient");
static_assert(detail::kFlagOrderProbe.want(0) && !detail::kFlagOrderProbe.want(1) &&
                  !detail::kFlagOrderProbe.want(2) && !detail::kFlagOrderProbe.want_any(1, 2),
              "GradMask flags are stored in the order they are written");

// Arity-named spellings keep a backward's signature to one token per mask.
using GradMask2 = GradMask<2>;
using GradMask3 = GradMask<3>;
using GradMask4 = GradMask<4>;
using GradMask5 = GradMask<5>;

// ── Optional output pointers ──────────────────────────────────────────────
// A typed kernel gets ``nullptr`` for a gradient it should not write.

template <typename... Ptr>
constexpr bool any_set(const Ptr*... p) noexcept {
    return ((p != nullptr) || ...);
}

template <typename... Ptr>
constexpr bool none_set(const Ptr*... p) noexcept {
    return ((p == nullptr) && ...);
}

// ``base + off``, or ``nullptr`` when ``base`` is (arithmetic on a null
// pointer is undefined, so an optional operand is offset through this).
template <typename T>
constexpr T* offset_or_null(T* base, std::size_t off) noexcept {
    return base ? base + off : nullptr;
}

// ``p[i]``, or zero when ``p`` is null.
template <typename T>
constexpr T at_or_zero(const T* p, std::size_t i) noexcept {
    return p ? p[i] : T{};
}

// ``n`` when ``p`` is set, else 0: the length of a scratch buffer that only
// a wanted gradient needs.
constexpr std::size_t size_if_set(const void* p, std::size_t n) noexcept {
    return p ? n : 0;
}

// ``dst[i] = acc[i]`` narrowed to ``T``; nothing when ``dst`` is null.
template <typename T, typename Acc>
void store_narrowed(T* dst, const std::vector<Acc>& acc) noexcept {
    if (dst)
        for (std::size_t i = 0; i < acc.size(); ++i)
            dst[i] = static_cast<T>(acc[i]);
}

// Zero ``n`` elements at ``p``; nothing when ``p`` is null.
template <typename T>
void zero_fill(T* p, std::size_t n) noexcept {
    if (p)
        std::fill_n(p, n, T{});
}

}  // namespace backend
}  // namespace lucid
