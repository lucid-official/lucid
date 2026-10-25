// lucid/_C/ops/composite/Indexing.cpp
//
// The ops below decompose into existing primitives, except ``scatter``, which
// dispatches the backend's overwrite kernel and carries its own backward.
// Index dtype is validated up front so error messages name the failing op
// rather than the underlying gather/scatter dispatch.

#include "Indexing.h"

#include <cstring>
#include <string>
#include <variant>
#include <vector>

#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"  // detail::ensure_grad_fn
#include "../gfunc/Gfunc.h"
#include "../ufunc/Astype.h"
#include "../utils/Concat.h"
#include "../utils/Layout.h"
#include "../utils/ScalarAxis.h"
#include "../utils/Select.h"
#include "../utils/Sort.h"
#include "../utils/View.h"

namespace lucid {

namespace {

// Resolve a possibly-negative ``dim`` against ``a``'s rank, which must be at
// least 1: these ops index ``shape()[dim]``.  The rule is :func:`wrap_dim`'s.
int wrap_axis_of(const TensorImplPtr& a, int dim, const char* op) {
    const int ndim = static_cast<int>(a->shape().size());
    if (ndim == 0)
        ErrorBuilder(op).index_error("dim out of range: the op needs a tensor of rank >= 1");
    return wrap_dim(dim, ndim, op);
}

// ``gather_op`` requires int32/int64 indices; we surface a per-op error so
// the user knows which call rejected the dtype.
void require_index_dtype(const TensorImplPtr& idx, const char* op) {
    if (idx->dtype() != Dtype::I32 && idx->dtype() != Dtype::I64)
        ErrorBuilder(op).index_error("indices must be int32 or int64");
}

}  // namespace

TensorImplPtr take_op(const TensorImplPtr& a, const TensorImplPtr& indices) {
    if (!a || !indices)
        ErrorBuilder("take").fail("null input");
    require_index_dtype(indices, "take");

    // ``gather`` along a freshly-flattened axis 0 lifts the multi-dim layout
    // into a single contiguous buffer; ReshapeBackward + GatherBackward
    // jointly carry the gradient back to the original shape.
    const std::int64_t total = static_cast<std::int64_t>(a->numel());
    auto flat = reshape_op(a, Shape{total});
    try {
        return gather_op(flat, indices, 0);
    } catch (LucidError& e) {
        e.prefix_message("take: ");
        throw;
    }
}

TensorImplPtr index_select_op(const TensorImplPtr& a, int dim, const TensorImplPtr& indices) {
    if (!a || !indices)
        ErrorBuilder("index_select").fail("null input");
    Validator::pair(a, indices, "index_select").same_device();
    require_index_dtype(indices, "index_select");
    // A 0-d index is one position, as in the reference.
    if (indices->shape().size() > 1)
        ErrorBuilder("index_select").index_error("indices must be 0-D or 1-D");
    if (a->shape().empty()) {
        if (indices->numel() != 1)
            ErrorBuilder("index_select")
                .index_error("Index to scalar can have only 1 value, got " +
                             std::to_string(indices->numel()) + " value(s)");
        return on_scalar_axis(a, dim, "index_select", [&](const TensorImplPtr& v, int ax) {
            return index_select_op(v, ax, scalar_as_1d(indices));
        });
    }
    const TensorImplPtr index = scalar_as_1d(indices);

    const int d = wrap_axis_of(a, dim, "index_select");
    const int ndim = static_cast<int>(a->shape().size());
    const std::int64_t k = index->shape()[0];

    // Reshape the 1-D index list to rank ``a`` with size ``k`` along ``d``
    // and 1 elsewhere; broadcast it to the source shape so ``gather_op``'s
    // same-rank-as-input contract holds.  ``broadcast_to``, not ``expand``:
    // on the CPU ``expand`` is a zero-stride view, which the gather then
    // copied element by element to make contiguous — 2.4 ms of a 4 ms
    // select of a (3, 375, 500) image.  ``broadcast_to`` materialises it in
    // runs.  The index carries no gradient, so nothing is lost.
    Shape idx_reshaped(static_cast<std::size_t>(ndim), 1);
    idx_reshaped[static_cast<std::size_t>(d)] = k;
    auto idx_r = reshape_op(index, idx_reshaped);
    Shape idx_target = a->shape();
    idx_target[static_cast<std::size_t>(d)] = k;
    auto idx_full = broadcast_to_op(idx_r, idx_target);
    try {
        return gather_op(a, idx_full, d);
    } catch (LucidError& e) {
        e.prefix_message("index_select: ");
        throw;
    }
}

TensorImplPtr narrow_op(const TensorImplPtr& a, int dim, std::int64_t start, std::int64_t length) {
    if (!a)
        ErrorBuilder("narrow").fail("null input");
    const int d = wrap_axis_of(a, dim, "narrow");
    const std::int64_t size = a->shape()[static_cast<std::size_t>(d)];
    if (start < 0 || length < 0 || start + length > size)
        ErrorBuilder("narrow").index_error("range out of bounds");

    // Full-axis slice — no split needed; return the input directly.
    if (start == 0 && length == size)
        return a;

    // Cut the axis at the window boundaries and pick the middle (or first /
    // last) piece.  ``split_at_op`` carries autograd via SplitSliceBackward.
    std::vector<std::int64_t> cuts;
    int wanted = 0;
    if (start > 0) {
        cuts.push_back(start);
        wanted = 1;
    }
    if (start + length < size)
        cuts.push_back(start + length);
    auto pieces = split_at_op(a, cuts, d);
    return pieces[static_cast<std::size_t>(wanted)];
}

namespace {

// The overwrite's backward.  ``out`` is ``base`` except where the index
// points, which holds ``src``: so ``base`` takes the gradient with those
// positions zeroed, and ``src`` takes the gradient read back from where each
// of its elements landed.  That is the reference's rule, duplicates
// included — every element aimed at one position gathers its gradient,
// though only one of them survived the write.
struct ScatterSetNode : Node {
    int dim_ = 0;
    TensorImplPtr saved_indices_;
    Shape base_shape_;
    Shape idx_shape_;
    Dtype dtype_ = Dtype::F32;
    Device device_ = Device::CPU;

    std::string node_name() const override { return "scatter_set"; }
    void release_saved() override { saved_indices_.reset(); }
    void for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const override {
        if (saved_indices_)
            fn(saved_indices_);
    }
    // The index is the one tensor read by value; base and src only route the
    // gradient.
    void validate_versions() override { check_saved_version(saved_indices_, 0, "scatter", 2); }

    std::vector<Storage> apply(Storage g) override {
        auto& be = backend::Dispatcher::for_device(device_);
        auto g_impl = std::make_shared<TensorImpl>(g, base_shape_, dtype_, device_, false);
        Storage grad_src = gather_op(g_impl, saved_indices_, dim_)->storage();
        Storage grad_base =
            be.scatter_set(g, saved_indices_->storage(), be.zeros(idx_shape_, dtype_), base_shape_,
                           idx_shape_, dim_, dtype_);
        return {std::move(grad_base), std::move(grad_src)};
    }

    // The same two halves as ops, so the gradient is differentiable again:
    // zeroing positions is itself an overwrite, linear in ``g``.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& g) override {
        auto zeros = zeros_op(idx_shape_, g->dtype(), g->device(), false);
        return {scatter_op(g, dim_, saved_indices_, zeros), gather_op(g, saved_indices_, dim_)};
    }
};

}  // namespace

TensorImplPtr scatter_op(const TensorImplPtr& base,
                         int dim,
                         const TensorImplPtr& indices,
                         const TensorImplPtr& src) {
    if (!base || !indices || !src)
        ErrorBuilder("scatter").fail("null input");
    if (auto out = on_scalar_scatter("scatter", base, indices, src, dim,
                                     [](const TensorImplPtr& b, const TensorImplPtr& i,
                                        const TensorImplPtr& s,
                                        int ax) { return scatter_op(b, ax, i, s); }))
        return out;

    auto ops = axis_scatter_operands("scatter", base, indices, src, dim);

    // A true overwrite.  This was a scatter-add of the delta
    // ``src - base[idx]``, which is not one: NaN or inf in ``base`` stayed
    // NaN (``x[isnan(x)] = 0`` did nothing), a value far smaller than the
    // one it replaced rounded away (``x[0] = 1e-30`` wrote 0), bool could
    // not become False, a half overflowed in the subtraction, and a
    // repeated index summed its deltas (CHA-158).
    auto out = dispatch_axis_scatter("scatter_set", &backend::IBackend::scatter_set, base, ops);
    const auto& [d, idx, values] = ops;

    if (!GradMode::is_enabled() || !(base->requires_grad() || values->requires_grad()))
        return out;

    auto bwd = std::make_shared<ScatterSetNode>();
    bwd->dim_ = d;
    bwd->saved_indices_ = idx;
    bwd->base_shape_ = base->shape();
    bwd->idx_shape_ = idx->shape();
    bwd->dtype_ = base->dtype();
    bwd->device_ = base->device();
    bwd->set_next_edges({Edge(detail::ensure_grad_fn(base), base->grad_output_nr()),
                         Edge(detail::ensure_grad_fn(values), values->grad_output_nr())});
    bwd->set_saved_versions({idx->version()});
    out->set_grad_fn(std::move(bwd));
    out->set_leaf(false);
    out->set_requires_grad(true);
    return out;
}

TensorImplPtr kthvalue_op(const TensorImplPtr& a, std::int64_t k, int dim, bool keepdim) {
    if (!a)
        ErrorBuilder("kthvalue").fail("null input");
    // Refused for bool as the reference does; it sorts underneath, and
    // sorting bool is allowed, so without this the answer depended on it.
    if (a->dtype() == Dtype::Bool)
        ErrorBuilder("kthvalue")
            .not_implemented("kthvalue does not accept bool — cast to an integer dtype");
    if (a->shape().empty())
        return on_scalar_axis(a, dim, "kthvalue", [&](const TensorImplPtr& v, int ax) {
            return kthvalue_op(v, k, ax, keepdim);
        });
    const int d = wrap_axis_of(a, dim, "kthvalue");
    const std::int64_t size = a->shape()[static_cast<std::size_t>(d)];
    // An empty axis has no k-th value whatever k is: an index error, as for
    // argmax and the reference, not a bad k.
    if (size == 0)
        ErrorBuilder("kthvalue")
            .index_error("Expected reduction dim " + std::to_string(d) + " to have non-zero size.");
    if (k < 1 || k > size)
        ErrorBuilder("kthvalue")
            .invalid_argument("selected number k out of range: k must be in "
                              "[1, " +
                              std::to_string(size) + "], got " + std::to_string(k));

    auto sorted = sort_op(a, d);

    // Build a same-rank index tensor whose entries are all ``k - 1`` along
    // ``d`` and 1 elsewhere — what ``gather_op`` needs to pluck a single
    // slice.  We allocate via the backend directly because ``full_like_op``
    // would carry a redundant graph dependency on ``a``.
    Shape idx_shape = a->shape();
    idx_shape[static_cast<std::size_t>(d)] = 1;
    Storage idx_storage = backend::Dispatcher::for_device(a->device()).zeros(idx_shape, Dtype::I64);

    std::size_t numel = 1;
    for (auto s : idx_shape)
        numel *= static_cast<std::size_t>(s);
    std::vector<std::int64_t> host(numel, k - 1);
    if (a->device() == Device::CPU) {
        auto& cs = std::get<CpuStorage>(idx_storage);
        std::memcpy(cs.ptr.get(), host.data(), numel * sizeof(std::int64_t));
    } else {
        // GPU path: build a CPU buffer then upload via the backend.
        CpuStorage stage;
        stage.dtype = Dtype::I64;
        stage.nbytes = numel * sizeof(std::int64_t);
        stage.ptr = allocate_aligned_bytes(stage.nbytes);
        std::memcpy(stage.ptr.get(), host.data(), stage.nbytes);
        idx_storage =
            backend::Dispatcher::for_device(Device::GPU).from_cpu(std::move(stage), idx_shape);
    }
    auto idx_tensor = std::make_shared<TensorImpl>(std::move(idx_storage), idx_shape, Dtype::I64,
                                                   a->device(), false);

    auto val = gather_op(sorted, idx_tensor, d);
    if (keepdim)
        return val;
    return squeeze_op(val, d);
}

}  // namespace lucid
