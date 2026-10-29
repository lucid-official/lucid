// lucid/_C/ops/utils/View.cpp
//
// Implements reshape, view, squeeze, and unsqueeze.  A dense input is
// reshaped by relabelling: the result is a view over the same buffer — on
// Metal, over its family's root — in the input's ViewFamily.  So is a strided
// view whose strides can step through the new shape, by the reference's
// chunk rule.  Anything else is copied by the backend dispatcher's reshape,
// which ``view`` refuses.
// Either way the central helper build_view_output wires the ViewBackward
// autograd node so that gradients flow back through the changed shape.

#include "View.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/Helpers.h"
#include "../../core/OpRegistry.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/_BinaryOp.h"
#include "../gfunc/Gfunc.h"
#include "Contiguous.h"

namespace lucid {

// OpSchema registration for ViewBackward.  The "true" at position 9 marks
// this op as an in-place-compatible view (no data copy is required for the
// forward pass).
const OpSchema ViewBackward::schema_v1{"view", 1, AmpPolicy::KeepInput, true, "", -1, 1, {}, true};

namespace {

// Resolve the target shape for a reshape call, handling the single -1
// wildcard dimension.  Returns the fully-specified shape with all dimensions
// filled in.
//
// Throws via ErrorBuilder when:
//   - more than one -1 appears in new_shape
//   - a dimension value other than -1 is negative
//   - the known product of non-wildcard dims is zero (so inference is impossible)
//   - the inferred element count does not equal in_shape's numel
Shape resolve_reshape_shape(const Shape& in_shape, const std::vector<std::int64_t>& new_shape) {
    const std::size_t in_numel = shape_numel(in_shape);
    Shape resolved;
    resolved.reserve(new_shape.size());
    int wildcard_pos = -1;
    std::int64_t known_product = 1;
    for (std::size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (wildcard_pos != -1) {
                ErrorBuilder("reshape").invalid_argument("only one -1 is allowed");
            }
            wildcard_pos = static_cast<int>(i);
            resolved.push_back(0);  // placeholder; filled in below
        } else if (new_shape[i] < 0) {
            ErrorContextGuard ctx{"reshape"};
            throw_negative_dim(new_shape);
        } else {
            known_product = mul_extent(known_product, new_shape[i]);
            resolved.push_back(new_shape[i]);
        }
    }
    if (wildcard_pos != -1) {
        if (known_product == 0) {
            ErrorBuilder("reshape").fail("cannot infer -1 from a product of zero");
        }
        if (in_numel % static_cast<std::size_t>(known_product) != 0) {
            throw ShapeMismatch(in_shape, resolved,
                                "reshape: -1 inference failed (numel mismatch)");
        }
        // Fill in the placeholder value computed from the remaining element count.
        resolved[wildcard_pos] = static_cast<std::int64_t>(in_numel) / known_product;
    }
    if (shape_numel(resolved) != in_numel) {
        throw ShapeMismatch(in_shape, resolved, "reshape: total numel mismatch");
    }
    return resolved;
}

// Whether ``out`` only drops or inserts size-1 axes of ``a``'s shape.  If it
// does, ``stride`` gets strides that leave every element of ``a`` where it
// was: the kept axes keep theirs, and a size-1 axis — which never moves the
// address — gets the one a packed layout would give it.
bool squeezes_only(const TensorImpl& a, const Shape& out, Stride& stride) {
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> kept;
    for (std::size_t d = 0; d < a.shape().size(); ++d)
        if (a.shape()[d] != 1) {
            sizes.push_back(a.shape()[d]);
            kept.push_back(a.stride()[d]);
        }
    std::vector<std::int64_t> want;
    for (const auto n : out)
        if (n != 1)
            want.push_back(n);
    if (want != sizes)
        return false;
    stride.assign(out.size(), 0);
    std::size_t k = kept.size();
    std::int64_t next = static_cast<std::int64_t>(dtype_size(a.dtype()));
    for (std::size_t d = out.size(); d-- > 0;) {
        stride[d] = out[d] != 1 ? kept[--k] : next;
        next = stride[d] * out[d];
    }
    return true;
}

// The strides that read ``a``'s elements in place as ``out`` — the
// reference's rule for view and reshape — or false when none do.
//
// ``a``'s axes fall into chunks: runs over which each axis's stride is the
// next one's times that axis's length, so the run steps through its elements
// in order by its innermost stride.  A size-1 axis moves no element, so its
// stride never ends a chunk.  ``out`` is viewable when its axes, innermost
// first, multiply out to exactly the chunks; each axis then steps through
// its chunk, and a size-1 axis takes the extent of the axes inside it.  A
// tensor with no elements is viewable as any shape: it keeps its strides if
// the shape does not change, and is packed otherwise.
bool fill_chunk_axes(const Shape& out,
                     std::int64_t in_numel,
                     std::int64_t chunk_stride,
                     std::ptrdiff_t& view_d,
                     Stride& stride) {
    std::int64_t view_numel = 1;
    while (view_d >= 0 && (view_numel < in_numel || out[static_cast<std::size_t>(view_d)] == 1)) {
        stride[static_cast<std::size_t>(view_d)] = view_numel * chunk_stride;
        view_numel *= out[static_cast<std::size_t>(view_d)];
        --view_d;
    }
    return view_numel == in_numel;
}

bool chunk_strides(const TensorImpl& a, const Shape& out, Stride& stride) {
    const Shape& in = a.shape();
    const Stride& in_stride = a.stride();
    const std::size_t elem = dtype_size(a.dtype());
    if (in.empty()) {
        stride.assign(out.size(), static_cast<std::int64_t>(elem));
        return true;
    }
    if (shape_numel(in) == 0) {
        stride = (in == out) ? in_stride : contiguous_stride(out, elem);
        return true;
    }
    stride.assign(out.size(), 0);
    auto view_d = static_cast<std::ptrdiff_t>(out.size()) - 1;
    std::int64_t chunk_stride = in_stride.back();
    std::int64_t in_numel = 1;
    for (auto d = static_cast<std::ptrdiff_t>(in.size()) - 1; d >= 0; --d) {
        const auto k = static_cast<std::size_t>(d);
        if (in[k] == 1 && k > 0 && in_numel == 1) {
            chunk_stride = in_stride[k - 1];
            continue;
        }
        in_numel *= in[k];
        if (k > 0 && (in[k - 1] == 1 || in_stride[k - 1] == in_numel * chunk_stride))
            continue;
        // The chunk ends at ``k``.  The view's axes that make up its
        // elements, and the size-1 axes just outside them, step through it.
        if (!fill_chunk_axes(out, in_numel, chunk_stride, view_d, stride))
            return false;
        if (k > 0) {
            chunk_stride = in_stride[k - 1];
            in_numel = 1;
        }
    }
    return view_d == -1;
}

// Which strides a view's size-1 axes take.  They move no element, so both
// read the same elements, but ``stride()`` shows them: squeeze and unsqueeze
// give an inserted axis the extent of its inner neighbour (one element when
// innermost), view and reshape the extent of the axes inside it in its chunk.
enum class UnitAxes { kChunk, kSqueeze };

// The strides under which ``a``'s elements, read as ``out_shape``, are a view
// of its buffer — or false when reading them so takes a copy.
//
// A dense tensor holds its elements in row-major order from the first byte
// of its buffer (on Metal, of its family's root), which is what any shape
// with the same element count reads there.  A strided CPU view, or a Metal
// family member, which reads its root through its geometry, is a view as any
// shape its strides can step through: a squeeze or unsqueeze always, a merge
// or split of its axes when chunk_strides finds strides for it.
bool view_strides(const TensorImpl& a, const Shape& out_shape, Stride& stride, UnitAxes unit) {
    if (a.is_dense()) {
        if (unit == UnitAxes::kSqueeze)
            return squeezes_only(a, out_shape, stride);
        stride = (shape_numel(a.shape()) == 0 && a.shape() == out_shape)
                     ? a.stride()
                     : contiguous_stride(out_shape, dtype_size(a.dtype()));
        return true;
    }
    // A phantom on Metal views as a family member would: by its strides.
    if (!a.views_by_stride() && a.family_key() == nullptr && !a.is_phantom())
        return false;
    return unit == UnitAxes::kSqueeze ? squeezes_only(a, out_shape, stride)
                                      : chunk_strides(a, out_shape, stride);
}

// Shared implementation used by reshape_op, view_op, squeeze_op,
// squeeze_all_op, and unsqueeze_op.  An input view_strides can view gives a
// view over the same buffer, in its ViewFamily — no copy, and an in-place
// write to either reaches the other, on Metal too; anything else is copied.
// Either way ViewBackward is attached so that autograd can reshape gradients
// back to the input's shape.
TensorImplPtr build_view_output(const TensorImplPtr& a,
                                Shape out_shape,
                                const char* op_name,
                                UnitAxes unit = UnitAxes::kChunk) {
    Validator::input(a, std::string(op_name) + ".a").non_null();
    OpScopeFull scope{op_name, a->device(), a->dtype(), out_shape};

    TensorImplPtr out;
    if (Stride stride; view_strides(*a, out_shape, stride, unit)) {
        out = TensorImpl::make_view(a, out_shape, std::move(stride));
    } else {
        // Any other reshape — of a view at an offset, or not contiguous —
        // copies: ``storage()`` hands the backend its elements packed.
        Storage out_storage = backend::Dispatcher::for_device(a->device())
                                  .reshape(a->storage(), a->shape(), out_shape, a->dtype());
        out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, a->dtype(),
                                           a->device(), false);
    }
    // Autograd wires to ``a`` itself either way.
    kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, out, false);
    return out;
}

}  // namespace

// Backward for all view ops: reshape the incoming gradient from out_shape_
// back to the original input shape recorded in input_shapes_[0].  Because
// the forward pass was a pure re-interpretation of a contiguous buffer, the
// backward is likewise a zero-copy reshape of the gradient buffer.
std::vector<Storage> ViewBackward::apply(Storage grad_out) {
    auto out = backend::Dispatcher::for_device(device_).reshape(grad_out, out_shape_,
                                                                input_shapes_[0], dtype_);
    return {std::move(out)};
}

// Graph-mode backward: reshape grad_out back to the input shape using reshape_op
// so the operation itself is recorded in the autograd graph.
std::vector<TensorImplPtr> ViewBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // Convert Shape (vector<size_t>) to vector<int64_t> for reshape_op.
    std::vector<std::int64_t> target_shape;
    target_shape.reserve(input_shapes_[0].size());
    for (auto s : input_shapes_[0])
        target_shape.push_back(static_cast<std::int64_t>(s));
    return {reshape_op(grad_out, target_shape)};
}

// Resolve the target shape (handling -1 wildcard inference), then delegate
// to build_view_output.  The resolved shape will have exactly the same numel
// as `a`; any mismatch throws ShapeMismatch before reaching the backend.
TensorImplPtr reshape_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape) {
    Validator::input(a, "reshape.a").non_null();
    Shape resolved = resolve_reshape_shape(a->shape(), new_shape);
    return build_view_output(a, std::move(resolved), "reshape");
}

// ``reshape``'s view without its copy: an input whose elements the new
// shape cannot read in place is refused rather than copied, so the result
// always shares ``a``'s storage.
TensorImplPtr view_op(const TensorImplPtr& a, const std::vector<std::int64_t>& new_shape) {
    Validator::input(a, "view.a").non_null();
    Shape resolved = resolve_reshape_shape(a->shape(), new_shape);
    if (Stride unused; !view_strides(*a, resolved, unused, UnitAxes::kChunk))
        ErrorBuilder("view").fail(
            "view size is not compatible with input tensor's size and stride (at least one "
            "dimension spans across two contiguous subspaces) — use .reshape(...) instead");
    // Traced as the reshape it is: compile and Core ML know it by that name.
    return build_view_output(a, std::move(resolved), "reshape");
}

// Remove the size-1 dimension at `dim` by building a new shape that omits
// the element at index `wrapped`.  Validates that the specified dimension
// exists and has size 1 before forwarding to build_view_output.
TensorImplPtr squeeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "squeeze.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());
    const int wrapped = wrap_dim(dim, ndim, "squeeze");
    // A 0-d tensor's notional axis has size 1: squeezing it is the identity.
    if (ndim == 0)
        return build_view_output(a, Shape{}, "squeeze", UnitAxes::kSqueeze);
    if (a->shape()[static_cast<std::size_t>(wrapped)] != 1) {
        ErrorBuilder("squeeze").fail("target dim must be size 1");
    }
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) - 1);
    for (int i = 0; i < ndim; ++i) {
        if (i != wrapped)
            new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    }
    return build_view_output(a, std::move(new_shape), "squeeze", UnitAxes::kSqueeze);
}

// Remove every size-1 dimension from `a`, preserving all dimensions whose
// size is greater than 1.  If no size-1 dimensions exist, the output is a
// view with the unchanged shape.
TensorImplPtr squeeze_all_op(const TensorImplPtr& a) {
    Validator::input(a, "squeeze.a").non_null();
    Shape new_shape;
    for (auto d : a->shape()) {
        if (d != 1)
            new_shape.push_back(d);
    }
    return build_view_output(a, std::move(new_shape), "squeeze", UnitAxes::kSqueeze);
}

// Insert a new size-1 axis at the position given by `wrapped` (already
// resolved from `dim`).  The loop iterates over the output positions [0,
// ndim], inserts a 1 at `wrapped`, and copies the original dimensions at all
// other positions.  Negative dim is resolved against (ndim + 1) so that -1
// refers to the position just before the last dimension of the output.
TensorImplPtr unsqueeze_op(const TensorImplPtr& a, int dim) {
    Validator::input(a, "unsqueeze.a").non_null();
    const int ndim = static_cast<int>(a->shape().size());

    // The output has one more axis, so ``dim`` names a slot of rank ndim + 1.
    const int wrapped = wrap_dim(dim, ndim + 1, "unsqueeze");
    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) + 1);
    for (int i = 0; i <= ndim; ++i) {
        if (i == wrapped) {
            new_shape.push_back(1);
        }
        if (i < ndim) {
            new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
        }
    }
    return build_view_output(a, std::move(new_shape), "unsqueeze", UnitAxes::kSqueeze);
}

// ── as_strided ───────────────────────────────────────────────────────────────

const OpSchema AsStridedBackward::schema_v1{"as_strided", 1, AmpPolicy::KeepInput, true};

std::vector<Storage> AsStridedBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    const Shape flat{static_cast<std::int64_t>(shape_numel(out_shape_))};
    Storage g = be.reshape(grad_out, out_shape_, flat, dtype_);
    Storage summed = be.scatter_add_axis(g, indices_, Shape{numel_}, flat, 0, dtype_);
    return {be.reshape(summed, Shape{numel_}, input_shapes_[0], dtype_)};
}

std::vector<TensorImplPtr> AsStridedBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto n_out = static_cast<std::int64_t>(shape_numel(out_shape_));
    auto flat = reshape_op(grad_out, {n_out});
    auto idx = std::make_shared<TensorImpl>(indices_, Shape{n_out}, Dtype::I32, device_, false);
    auto summed = scatter_add_op(zeros_op(Shape{numel_}, dtype_, device_), idx, flat, 0);
    return {reshape_op(
        summed, std::vector<std::int64_t>(input_shapes_[0].begin(), input_shapes_[0].end()))};
}

std::vector<std::int32_t> as_strided_positions(const Shape& size,
                                               const std::vector<std::int64_t>& stride,
                                               std::int64_t offset,
                                               std::int64_t own,
                                               std::int64_t capacity,
                                               std::int64_t n_out) {
    if (capacity > std::numeric_limits<std::int32_t>::max())
        ErrorBuilder("as_strided").not_implemented("a storage past 2^31 elements");
    std::vector<std::int32_t> positions(static_cast<std::size_t>(n_out));
    std::vector<std::int64_t> idx(size.size(), 0);
    for (std::int64_t i = 0; i < n_out; ++i) {
        std::int64_t at = offset;
        for (std::size_t d = 0; d < size.size(); ++d)
            at += idx[d] * stride[d];
        positions[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(at - own);
        for (std::size_t d = size.size(); d-- > 0;) {
            if (++idx[d] < size[d])
                break;
            idx[d] = 0;
        }
    }
    return positions;
}

void remap_as_strided_grad_positions(const TensorImpl& a,
                                     std::int64_t elem,
                                     std::int64_t own,
                                     std::int64_t capacity,
                                     std::int64_t n_in,
                                     std::vector<std::int32_t>& positions) {
    if (!a.is_contiguous()) {
        std::vector<std::int64_t> elem_stride;
        elem_stride.reserve(a.stride().size());
        for (const auto s : a.stride())
            elem_stride.push_back(s / elem);
        const auto own_pos = as_strided_positions(a.shape(), elem_stride, own, own, capacity, n_in);
        std::unordered_map<std::int32_t, std::int32_t> held;
        held.reserve(static_cast<std::size_t>(n_in));
        for (std::int64_t k = 0; k < n_in; ++k)
            held.try_emplace(own_pos[static_cast<std::size_t>(k)], static_cast<std::int32_t>(k));
        for (auto& p : positions) {
            const auto it = held.find(p);
            if (it == held.end())
                ErrorBuilder("as_strided")
                    .not_implemented("a gradient through a view that does not hold every "
                                     "element of the window — call contiguous() first and "
                                     "take the window of that");
            p = it->second;
        }
        return;
    }
    for (const auto p : positions) {
        if (p < 0 || p >= n_in)
            ErrorBuilder("as_strided")
                .not_implemented("a gradient through a view that does not hold every "
                                 "element of the window — call contiguous() first and "
                                 "take the window of that");
    }
}

void validate_as_strided_window(const Shape& size,
                                const std::vector<std::int64_t>& stride,
                                std::int64_t offset,
                                std::int64_t capacity) {
    if (size.size() != stride.size())
        ErrorBuilder("as_strided").fail("size and stride need the same length");
    // The furthest element, ``offset + sum((size - 1) * stride)``, is formed
    // with overflow checks: wrapped, a huge stride made it small enough to
    // pass, and the view read far outside the storage.
    std::int64_t last = offset;
    bool empty = false;
    bool past = false;
    for (std::size_t d = 0; d < size.size(); ++d) {
        if (stride[d] < 0)
            ErrorBuilder("as_strided").invalid_argument("size and stride must be non-negative");
        if (size[d] == 0) {
            empty = true;
            continue;
        }
        std::int64_t reach = 0;
        past |= __builtin_mul_overflow(size[d] - 1, stride[d], &reach);
        past |= __builtin_add_overflow(last, reach, &last);
    }
    if (!empty && (past || last >= capacity))
        ErrorBuilder("as_strided").fail("the view reaches past the end of the storage");
    // Byte strides and the byte offset are these times the element size, so
    // they are held to the element-count limit, which keeps them in 64 bits
    // at every dtype.  A stride over a length-1 axis moves nothing but is
    // still stored, and an empty view's offset is checked by nothing above.
    const auto max_elements = static_cast<std::int64_t>(kMaxNumel);
    if (offset > max_elements)
        ErrorBuilder("as_strided")
            .invalid_argument("storage_offset " + std::to_string(offset) +
                              " is past the 64-bit byte range");
    for (const auto st : stride)
        if (st > max_elements)
            ErrorBuilder("as_strided")
                .invalid_argument("stride " + std::to_string(st) +
                                  " is past the 64-bit byte range");
}

TensorImplPtr as_strided_op(const TensorImplPtr& a,
                            const Shape& size,
                            const std::vector<std::int64_t>& stride,
                            std::int64_t storage_offset) {
    Validator::input(a, "as_strided.a").non_null();
    const Dtype dt = a->dtype();
    const Device device = a->device();
    OpScopeFull scope{"as_strided", device, dt, size};
    const auto n_out = static_cast<std::int64_t>(shape_numel(size));
    const auto elem = static_cast<std::int64_t>(dtype_size(dt));
    const std::int64_t own = static_cast<std::int64_t>(a->allocation_offset()) / elem;
    const std::int64_t capacity = static_cast<std::int64_t>(a->allocation_nbytes()) / elem;
    const std::int64_t offset = storage_offset < 0 ? own : storage_offset;
    validate_as_strided_window(size, stride, offset, capacity);
    scope.set_attr("size", std::vector<std::int64_t>(size.begin(), size.end()));
    scope.set_attr("stride", stride);
    scope.set_attr("storage_offset", offset);

    const auto n_in = static_cast<std::int64_t>(shape_numel(a->shape()));
    const bool needs_grad = GradMode::is_enabled() && a->requires_grad();

    Stride byte_stride;
    byte_stride.reserve(stride.size());
    for (const auto s : stride)
        byte_stride.push_back(s * elem);
    TensorImplPtr out = TensorImpl::make_allocation_view(a, size, std::move(byte_stride),
                                                         static_cast<std::size_t>(offset * elem));

    auto bwd = std::make_shared<AsStridedBackward>();
    if (needs_grad) {
        auto positions = as_strided_positions(size, stride, offset, own, capacity, n_out);
        remap_as_strided_grad_positions(*a, elem, own, capacity, n_in, positions);
        auto& be = backend::Dispatcher::for_device(device);
        auto cpu = helpers::allocate_cpu(Shape{n_out}, Dtype::I32);
        if (n_out > 0)
            std::memcpy(cpu.ptr.get(), positions.data(), positions.size() * sizeof(std::int32_t));
        bwd->input_shapes_ = {a->shape()};
        bwd->out_shape_ = size;
        bwd->dtype_ = dt;
        bwd->device_ = device;
        bwd->input_tensors_ = {a};
        bwd->numel_ = n_in;
        bwd->indices_ = be.from_cpu(std::move(cpu), Shape{n_out});
    }
    kernel::NaryKernel<AsStridedBackward, 1>::wire_autograd(std::move(bwd), {a}, out, false);
    return out;
}

LUCID_REGISTER_OP(AsStridedBackward)
LUCID_REGISTER_OP(ViewBackward)

}  // namespace lucid
