// lucid/_C/ops/utils/ScalarAxis.h
//
// A one-axis op applied to a 0-d tensor.
//
// The reference gives a 0-d tensor one notional axis wherever an op takes a
// dim: 0 and -1 both name it (``wrap_dim``), and the answer is 0-d again —
// sort and cumsum return the scalar, argsort and topk index 0, softmax 1.
// The kernels need a real axis, so the op runs on the scalar viewed as shape
// (1,) along axis 0 and each result is viewed back to 0-d.  Both views are
// reshapes, so autograd and the compile trace see ordinary ops.

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include "../../autograd/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "View.h"

namespace lucid {

// ``t`` viewed as shape (1,) when it is 0-d, else ``t`` itself.
inline TensorImplPtr scalar_as_1d(const TensorImplPtr& t) {
    return t->shape().empty() ? reshape_op(t, {1}) : t;
}

// ``run(a viewed as (1,), 0)`` for a 0-d ``a`` whose ``dim`` names its one
// notional axis, with every result viewed back to 0-d.  ``run`` returns a
// tensor or a vector of tensors.
template <class Run>
auto on_scalar_axis(const TensorImplPtr& a, int dim, const char* op, Run&& run) {
    wrap_dim(dim, 0, op);
    auto out = std::forward<Run>(run)(reshape_op(a, {1}), 0);
    if constexpr (std::is_same_v<decltype(out), std::vector<TensorImplPtr>>) {
        for (auto& t : out)
            t = reshape_op(t, {});
        return out;
    } else {
        return reshape_op(out, {});
    }
}

// The scatters' form of the rule: a 0-d operand counts as one element along
// a single axis, so a 0-d base scatters, and a 1-d base takes a 0-d index
// and src.  Runs ``run(base, index, src, 0)`` on the (1,) views and gives a
// 0-d base its 0-d result back.  Returns null when the rule does not apply
// (no 0-d operand, or a base of rank >= 2, which the op's rank check refuses).
template <class Run>
TensorImplPtr on_scalar_scatter(const char* op,
                                const TensorImplPtr& base,
                                const TensorImplPtr& indices,
                                const TensorImplPtr& src,
                                int dim,
                                Run&& run) {
    if (!base || !indices || !src || base->shape().size() > 1)
        return nullptr;
    if (!base->shape().empty() && !indices->shape().empty() && !src->shape().empty())
        return nullptr;
    Validator::pair(base, indices, op).same_device();
    Validator::pair(base, src, op).same_device();
    wrap_dim(dim, static_cast<int>(base->shape().size()), op);
    if (indices->numel() == 0) {
        if (!base->shape().empty())
            return nullptr;
        return reshape_op(
            std::forward<Run>(run)(reshape_op(base, {1}), indices, scalar_as_1d(src), 0), {});
    }
    auto out =
        std::forward<Run>(run)(scalar_as_1d(base), scalar_as_1d(indices), scalar_as_1d(src), 0);
    return base->shape().empty() ? reshape_op(out, {}) : out;
}

}  // namespace lucid
