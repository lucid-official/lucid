// lucid/_C/nn/Spatial.cpp
//
// Implementation of affine_grid and grid_sample.
//
// AffineGrid forward: IBackend::affine_grid_forward produces a (N, H, W, 2)
//   grid of normalized sample coordinates from a batch of 2x3 affine matrices.
//   Backward wiring is skipped when theta does not require a gradient.
//
// GridSample forward: IBackend::grid_sample_forward resamples the input image
//   using bilinear/nearest/bicubic interpolation at the grid positions.
//   Backward wiring is skipped when neither input nor grid requires a gradient.
//
// The backward for both ops is fully delegated to the backend; no activations
// are saved beyond the mode and shape parameters recorded in the backward node.

#include "Spatial.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Engine.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/composite/Indexing.h"
#include "../ops/composite/Logical.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Discrete.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/ScalarParam.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/utils/Concat.h"
#include "../ops/utils/Contiguous.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/View.h"

namespace lucid {

const OpSchema AffineGridBackward::schema_v1{"affine_grid", 1, AmpPolicy::Promote, true};

TensorImplPtr
AffineGridBackward::forward(const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    if (!theta)
        ErrorBuilder("affine_grid").fail("null theta");
    if (theta->shape().size() != 3 || theta->shape()[0] != N || theta->shape()[1] != 2 ||
        theta->shape()[2] != 3)
        throw ShapeMismatch(Shape{static_cast<std::int64_t>(N), 2, 3}, theta->shape(),
                            "affine_grid: theta must be (N, 2, 3)");
    // A negative extent sized the output as 2^64 bytes.
    if (H < 0 || W < 0)
        ErrorBuilder("affine_grid")
            .invalid_argument("H and W must be >= 0, got H=" + std::to_string(H) +
                              ", W=" + std::to_string(W));
    SchemaGuard sg{schema_v1, theta->dtype(), theta->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr theta_eff = astype_op(theta, eff_dt);

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(H),
                    static_cast<std::int64_t>(W), 2};
    OpScopeFull scope{schema_v1.name, theta_eff->device(), eff_dt, out_shape};
    scope.set_attr("H", static_cast<std::int64_t>(H));
    scope.set_attr("W", static_cast<std::int64_t>(W));
    scope.set_attr("align_corners", align_corners);

    auto& be = backend::Dispatcher::for_device(theta_eff->device());
    Storage out_storage =
        be.affine_grid_forward(theta_eff->storage(), N, H, W, align_corners, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, eff_dt,
                                            theta_eff->device(), false);
    // wire_autograd records on_op_io internally, but the early-return
    // below skips it when grad is off — record explicitly so the trace
    // captures inputs in both code paths.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({theta_eff}, out);
    }

    if (!GradMode::is_enabled() || !theta_eff->requires_grad())
        return out;

    auto bwd = std::make_shared<AffineGridBackward>();
    bwd->align_corners_ = align_corners;
    bwd->N_ = N;
    bwd->H_ = H;
    bwd->W_ = W;
    bwd->orig_theta_shape_ = theta_eff->shape();
    kernel::NaryKernel<AffineGridBackward, 1>::wire_autograd(std::move(bwd), {theta_eff}, out,
                                                             false);
    return out;
}

std::vector<Storage> AffineGridBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.affine_grid_backward(grad_out, N_, H_, W_, align_corners_, dtype_)};
}

std::vector<TensorImplPtr> AffineGridBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // grid[n, h, w] = theta[n] @ [x_w, y_h, 1], so dtheta[n] is the grid's
    // gradient contracted with those base coordinates.  The coordinates come
    // from this op's own forward on the identity transform — the one way to
    // be sure they follow the same align_corners rule.  Linear in theta: the
    // gradient does not depend on it, only on g.
    auto identity = reshape_op(eye_op(2, 3, 0, dtype_, device_), {1, 2, 3});
    TensorImplPtr xy;
    {
        NoGradGuard constant;
        xy = affine_grid_op(identity, 1, H_, W_, align_corners_);  // (1, H, W, 2)
    }
    auto ones = ones_op(Shape{1, H_, W_, 1}, dtype_, device_);
    const std::int64_t points = static_cast<std::int64_t>(H_) * W_;
    auto base = reshape_op(concatenate_op({xy, ones}, 3), {points, 3});
    auto g = permute_op(reshape_op(grad_out, {N_, points, 2}), {0, 2, 1});  // (N, 2, HW)
    auto dtheta = matmul_op(g, base);                                       // (N, 2, 3)
    return {reshape_op(
        dtheta, std::vector<std::int64_t>(orig_theta_shape_.begin(), orig_theta_shape_.end()))};
}

TensorImplPtr affine_grid_op(const TensorImplPtr& theta, int N, int H, int W, bool align_corners) {
    return AffineGridBackward::forward(theta, N, H, W, align_corners);
}
LUCID_REGISTER_OP(AffineGridBackward)

namespace {
void validate_grid_sample_inputs(const TensorImplPtr& input,
                                 const TensorImplPtr& grid,
                                 int mode,
                                 int padding_mode) {
    if (input->device() != grid->device())
        throw DeviceMismatch(std::string(device_name(input->device())),
                             std::string(device_name(grid->device())), "grid_sample: input/grid");
    if (input->shape().size() != 4)
        throw ShapeMismatch(input->shape(), Shape{},
                            "grid_sample: input must be (N, C, H_in, W_in)");
    if (grid->shape().size() != 4 || grid->shape()[3] != 2)
        throw ShapeMismatch(grid->shape(), Shape{},
                            "grid_sample: grid must be (N, H_out, W_out, 2)");
    if (input->dtype() != grid->dtype())
        throw DtypeMismatch(std::string(dtype_name(input->dtype())),
                            std::string(dtype_name(grid->dtype())), "grid_sample");
    if (input->shape()[0] != grid->shape()[0])
        throw ShapeMismatch(input->shape(), grid->shape(), "grid_sample: batch size mismatch");
    if (mode != 0 && mode != 1)
        ErrorBuilder("grid_sample")
            .invalid_argument("mode must be 0 (bilinear) or 1 (nearest), got " +
                              std::to_string(mode) +
                              " (bicubic and volumetric grids are lowered in Python)");
    if (padding_mode != 0 && padding_mode != 1)
        ErrorBuilder("grid_sample")
            .invalid_argument("padding_mode must be 0 (zeros) or 1 (border), got " +
                              std::to_string(padding_mode) +
                              " (reflection padding is folded into the grid in Python)");
}
}  // namespace

// A bool image has no sampling kernel in the reference (``NotImplementedError``).
// A bool *grid* beside a float image is a dtype ``RuntimeError`` there; this
// schema's guard runs on both operands and answers that one with the image's
// class.
const OpSchema GridSampleBackward::schema_v1 =
    OpSchema{"grid_sample", 1, AmpPolicy::Promote, true, "", true}.refusing_bool(
        BoolRefusal::Kernel, BoolCast::Floating);

TensorImplPtr GridSampleBackward::forward(const TensorImplPtr& input0,
                                          const TensorImplPtr& grid0,
                                          int mode,
                                          int padding_mode,
                                          bool align_corners) {
    if (!input0 || !grid0)
        ErrorBuilder("grid_sample").fail("null input");
    // Both operands, not just the primary, and together.  Resampling
    // interpolates *between* samples, so the answer is real whatever went
    // in — the schema says so and this forward is assembled by hand, so it
    // has to ask.  The grid carries fractional coordinates and is no more
    // integral than the image.  Under Metal autocast the row is WIDEST: a
    // float16 image with a float32 grid (deformable attention's sampling
    // locations) runs in float32, as the reference's does.
    const auto operands = promote_for_schema(schema_v1, {input0, grid0});
    const TensorImplPtr& input = operands[0];
    const TensorImplPtr& grid = operands[1];
    validate_grid_sample_inputs(input, grid, mode, padding_mode);

    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    const int H_out = static_cast<int>(grid->shape()[1]);
    const int W_out = static_cast<int>(grid->shape()[2]);
    (void)C;

    Shape out_shape{static_cast<std::int64_t>(N), static_cast<std::int64_t>(C),
                    static_cast<std::int64_t>(H_out), static_cast<std::int64_t>(W_out)};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    // None of these is recoverable from the shapes, and each one changes
    // the answer rather than the layout: ``mode`` picks the filter,
    // ``padding_mode`` decides what lies outside the image, and
    // ``align_corners`` moves every sample.  An emitter that assumed a
    // value would return a plausible image sampled on the wrong grid.
    scope.set_attr("mode", static_cast<std::int64_t>(mode));
    scope.set_attr("padding_mode", static_cast<std::int64_t>(padding_mode));
    scope.set_attr("align_corners", align_corners);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage =
        be.grid_sample_forward(input->storage(), grid->storage(), input->shape(), grid->shape(),
                               mode, padding_mode, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    // Same as ``affine_grid`` above: ``wire_autograd`` records on_op_io,
    // but the early return below skips it whenever no gradient is
    // wanted — which is every inference pass, and tracing is one.  The
    // node then reached the emitters with no operands at all, so both
    // backends listed this op as one the trace could not carry.
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({input, grid}, out);
    }

    if (!GradMode::is_enabled() || !(input->requires_grad() || grid->requires_grad()))
        return out;

    auto bwd = std::make_shared<GridSampleBackward>();
    bwd->mode_ = mode;
    bwd->padding_mode_ = padding_mode;
    bwd->align_corners_ = align_corners;
    bwd->input_shape_ = input->shape();
    bwd->grid_shape_ = grid->shape();
    kernel::NaryKernel<GridSampleBackward, 2>::wire_autograd(std::move(bwd), {input, grid}, out);
    return out;
}

std::vector<Storage> GridSampleBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return be.grid_sample_backward(grad_out, saved_inputs_[0], saved_inputs_[1], input_shape_,
                                   grid_shape_, mode_, padding_mode_, align_corners_, dtype_);
}

std::vector<TensorImplPtr> GridSampleBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The adjoint the kernel computes, step for step, in ops.  Sampling is
    // a weighted gather of the input at (ix, iy); its input gradient is the
    // matching weighted scatter-add of the incoming gradient, and its grid
    // gradient is the gradient's projection onto the weights' slope — the
    // differences between neighbouring corners.  The weights depend on the
    // grid and the corner values on the input, so both gradients are
    // differentiable in both.
    const auto& input = saved_impl_inputs_[0];
    const auto& grid = saved_impl_inputs_[1];
    if (!input || !grid)
        ErrorBuilder("grid_sample").fail("graph-mode backward is missing a saved input");
    if (input_shape_.size() != 4)
        ErrorBuilder("grid_sample").not_implemented("create_graph=True needs a 4-D input");
    const std::int64_t n = input_shape_[0], ch = input_shape_[1];
    const std::int64_t h = input_shape_[2], w = input_shape_[3];
    const std::int64_t points = grid_shape_[1] * grid_shape_[2];
    const auto constant = [](const TensorImplPtr& like, double v) {
        return full_like_op(like, v, false);
    };

    // Source coordinates as the kernel forms them: (g + 1) * (dim - 1) / 2
    // aligned, (g + 1) * dim / 2 - 0.5 otherwise.  The same order of
    // operations, so the corner each point picks is the kernel's.
    const auto source = [&](int axis, std::int64_t dim, double& slope) {
        auto g = reshape_op(narrow_op(grid, 3, axis, 1), {n, points});
        slope = align_corners_ ? (static_cast<double>(dim) - 1.0) / 2.0
                               : static_cast<double>(dim) / 2.0;
        auto coord = mul_op(add_op(g, constant(g, 1.0)), constant(g, slope));
        return align_corners_ ? coord : sub_op(coord, constant(coord, 0.5));
    };
    double sx = 0.0, sy = 0.0;
    auto ix = source(0, w, sx);
    auto iy = source(1, h, sy);

    // A border-clipped coordinate's gradient is zero where the clip bit.
    const bool clip = mode_ == 0 && padding_mode_ == 1;
    const auto inside = [&](const TensorImplPtr& c, std::int64_t dim) {
        return astype_op(
            logical_and_op(greater_equal_op(c, constant(c, 0.0)),
                           less_equal_op(c, constant(c, static_cast<double>(dim - 1)))),
            dtype_);
    };
    TensorImplPtr keep_x, keep_y;
    if (clip) {
        keep_x = inside(ix, w);
        keep_y = inside(iy, h);
        ix = clip_op(ix, 0.0, static_cast<double>(w - 1));
        iy = clip_op(iy, 0.0, static_cast<double>(h - 1));
    }

    auto input_flat = reshape_op(input, {n, ch, h * w});
    auto g_flat = reshape_op(grad_out, {n, ch, points});
    const auto spread = [&](const TensorImplPtr& t) {  // (N, P) -> (N, 1, P)
        return unsqueeze_op(t, 1);
    };
    // A corner's flat index, clamped as the kernel clamps its reads, and
    // whether zero padding keeps it.
    const auto corner = [&](const TensorImplPtr& xc, const TensorImplPtr& yc,
                            TensorImplPtr& valid) {
        valid =
            padding_mode_ == 0
                ? astype_op(logical_and_op(
                                logical_and_op(
                                    greater_equal_op(xc, constant(xc, 0.0)),
                                    less_equal_op(xc, constant(xc, static_cast<double>(w - 1)))),
                                logical_and_op(
                                    greater_equal_op(yc, constant(yc, 0.0)),
                                    less_equal_op(yc, constant(yc, static_cast<double>(h - 1))))),
                            dtype_)
                : TensorImplPtr{};
        auto xi = astype_op(clip_op(xc, 0.0, static_cast<double>(w - 1)), Dtype::I32);
        auto yi = astype_op(clip_op(yc, 0.0, static_cast<double>(h - 1)), Dtype::I32);
        auto flat = add_op(mul_op(yi, full_like_op(yi, static_cast<double>(w), false)), xi);
        return contiguous_op(broadcast_to_op(spread(flat), Shape{n, ch, points}));
    };
    const auto masked = [&](const TensorImplPtr& t, const TensorImplPtr& valid) {
        return valid ? mul_op(t, valid) : t;
    };

    auto dinput = zeros_op(Shape{n, ch, h * w}, dtype_, device_);
    if (mode_ == 1) {
        // Nearest: one source per point, a gradient only for the input.
        TensorImplPtr valid;
        auto idx = corner(round_op(ix), round_op(iy), valid);
        auto src = valid ? mul_op(g_flat, spread(valid)) : g_flat;
        dinput = scatter_add_op(dinput, idx, src, 2);
        return {zero_derivative_through(reshape_op(dinput, {n, ch, h, w}), next_edges()[0]),
                zeros_like_op(grid)};
    }

    auto x0 = floor_op(ix);
    auto y0 = floor_op(iy);
    auto x1 = add_op(x0, constant(x0, 1.0));
    auto y1 = add_op(y0, constant(y0, 1.0));
    auto fx = sub_op(ix, x0);
    auto fy = sub_op(iy, y0);
    auto gx = sub_op(constant(fx, 1.0), fx);  // weight toward x0
    auto gy = sub_op(constant(fy, 1.0), fy);  // weight toward y0

    TensorImplPtr va, vb, vc, vd;
    auto ia = corner(x0, y0, va);  // (y0, x0)
    auto ib = corner(x0, y1, vb);  // (y1, x0)
    auto ic = corner(x1, y0, vc);  // (y0, x1)
    auto id = corner(x1, y1, vd);  // (y1, x1)

    const auto scatter = [&](const TensorImplPtr& idx, const TensorImplPtr& weight,
                             const TensorImplPtr& valid) {
        dinput = scatter_add_op(dinput, idx, mul_op(g_flat, spread(masked(weight, valid))), 2);
    };
    scatter(ia, mul_op(gx, gy), va);
    scatter(ib, mul_op(gx, fy), vb);
    scatter(ic, mul_op(fx, gy), vc);
    scatter(id, mul_op(fx, fy), vd);

    const auto value = [&](const TensorImplPtr& idx, const TensorImplPtr& valid) {
        auto v = gather_op(input_flat, idx, 2);
        return valid ? mul_op(v, spread(valid)) : v;
    };
    auto a = value(ia, va), b = value(ib, vb), c = value(ic, vc), d = value(id, vd);
    auto dix = sum_op(
        mul_op(g_flat, add_op(mul_op(sub_op(c, a), spread(gy)), mul_op(sub_op(d, b), spread(fy)))),
        std::vector<int>{1}, false);
    auto diy = sum_op(
        mul_op(g_flat, add_op(mul_op(sub_op(b, a), spread(gx)), mul_op(sub_op(d, c), spread(fx)))),
        std::vector<int>{1}, false);
    if (clip) {
        dix = mul_op(dix, keep_x);
        diy = mul_op(diy, keep_y);
    }
    auto dgx = mul_op(dix, constant(dix, sx));
    auto dgy = mul_op(diy, constant(diy, sy));
    auto dgrid = reshape_op(stack_op({dgx, dgy}, 2), {n, grid_shape_[1], grid_shape_[2], 2});
    // With a constant grid the input gradient reads no tensor that requires
    // grad; it stays in the graph through the input with derivative zero.
    return {zero_derivative_through(reshape_op(dinput, {n, ch, h, w}), next_edges()[0]), dgrid};
}

TensorImplPtr grid_sample_op(const TensorImplPtr& input,
                             const TensorImplPtr& grid,
                             int mode,
                             int padding_mode,
                             bool align_corners) {
    return GridSampleBackward::forward(input, grid, mode, padding_mode, align_corners);
}
LUCID_REGISTER_OP(GridSampleBackward)

}  // namespace lucid
