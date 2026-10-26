// lucid/_C/ops/composite/Stats.cpp
//
// ``histc`` is a counts-only wrapper around ``histogram_op``.  When the
// range is omitted (lo == hi) it auto-derives from the input via min/max
// reductions, matching the reference framework's ``histc`` default.
//
// ``cartesian_prod`` builds the n-dim grid via ``meshgrid`` then flattens
// each grid component and stacks them along a new last axis.  Inputs must
// be 1-D — the reference framework rejects higher-rank inputs and so do we.

#include "Stats.h"

#include <cstring>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Reductions.h"
#include "../utils/Concat.h"
#include "../utils/Histogram.h"
#include "../utils/Meshgrid.h"
#include "../utils/View.h"

namespace lucid {

namespace {

// Materialise a 0-D reduction result as a host double.  The reduce kernel
// always produces a CPU- or GPU-backed scalar; ``to_cpu`` brings it home.
double scalar_to_double(const TensorImplPtr& s) {
    auto s_f = is_half_float(s->dtype()) ? astype_op(s, Dtype::F32) : s;
    auto host = backend::Dispatcher::for_device(s_f->device()).to_cpu(s_f->storage(), s_f->shape());
    if (s_f->dtype() == Dtype::F32) {
        float v = 0.0f;
        std::memcpy(&v, host.ptr.get(), sizeof(float));
        return static_cast<double>(v);
    }
    double v = 0.0;
    std::memcpy(&v, host.ptr.get(), sizeof(double));
    return v;
}

}  // namespace

TensorImplPtr histc_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi) {
    if (!a)
        ErrorBuilder("histc").fail("null input");
    // Integer input is refused, as the reference refuses it.
    //
    // It was accepted and answered ``[0, 0, 0, 0]`` for data with values
    // plainly inside the range — the auto-range bounds are read back
    // through a float path that an integer buffer does not survive.  A
    // histogram of zeros is the worst possible answer here: it is the
    // shape the caller expects, so nothing downstream can tell.
    if (!is_floating_point(a->dtype()))
        ErrorBuilder("histc").not_implemented("input must be a floating-point dtype, got " +
                                              std::string(dtype_name(a->dtype())));

    double effective_lo = lo;
    double effective_hi = hi;
    if (effective_lo == effective_hi && a->numel() == 0) {
        // Nothing to take a range from — and ``min`` / ``max`` of nothing are
        // refused.  Every bin is empty whatever the range; the reference
        // answers ``bins`` zeros the same way.
        effective_hi = effective_lo + 1.0;
    } else if (effective_lo == effective_hi) {
        // Auto-range: collapse all axes to scalars and read the bounds back.
        // ``min == max`` is a degenerate range; bump ``hi`` so the bins are
        // well-defined (matches the reference framework).
        effective_lo = scalar_to_double(min_op(a, {}, false));
        effective_hi = scalar_to_double(max_op(a, {}, false));
        if (effective_lo == effective_hi)
            effective_hi = effective_lo + 1.0;
    }
    auto counts = histogram_op(a, bins, effective_lo, effective_hi, false).front();

    // ``histogram`` counts in F64 by design; ``histc`` reports in the
    // input's dtype, which is what the reference does and what lets the
    // result stay where the input was.
    //
    // Always returning F64 had a second consequence, because MLX has no
    // float64: the counts could only live on the CPU.  So ``histc`` of a
    // Metal tensor came back on the CPU and the next op raised
    // DeviceMismatch — a device bug whose actual cause was a dtype.
    if (is_floating_point(a->dtype()) && a->dtype() != counts->dtype())
        counts = astype_op(counts, a->dtype());
    if (counts->device() != a->device() && a->device() == Device::GPU) {
        counts = helpers::fresh(backend::Dispatcher::for_device(Device::GPU)
                                    .from_cpu(storage_cpu(counts->storage()), counts->shape()),
                                counts->shape(), counts->dtype(), Device::GPU);
    }
    return counts;
}

TensorImplPtr cartesian_prod_op(const std::vector<TensorImplPtr>& tensors) {
    if (tensors.empty())
        ErrorBuilder("cartesian_prod").fail("requires at least one input");
    for (const auto& t : tensors) {
        if (!t)
            ErrorBuilder("cartesian_prod").fail("null input");
        if (t->shape().size() != 1)
            ErrorBuilder("cartesian_prod").fail("inputs must be 1-D");
    }
    if (tensors.size() == 1)
        return reshape_op(tensors[0], tensors[0]->shape());

    // ``meshgrid`` with ``ij`` indexing produces n-D grids whose flattened
    // form gives every combination in a stable order.
    auto grids = meshgrid_op(tensors, /*indexing_xy=*/false);
    std::vector<TensorImplPtr> flat;
    flat.reserve(grids.size());
    const std::int64_t total = static_cast<std::int64_t>(grids.front()->numel());
    for (auto& g : grids)
        flat.push_back(reshape_op(g, Shape{total}));
    // Stack the flattened components along a new trailing axis: shape (N, D).
    return stack_op(flat, -1);
}

}  // namespace lucid
