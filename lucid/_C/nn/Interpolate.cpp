// lucid/_C/nn/Interpolate.cpp
//
// Bilinear, trilinear, and nearest-neighbor interpolation implementations.
//
// Bilinear / trilinear: forward delegates to IBackend::interpolate_*_forward;
//   the backward node saves only the original shape and the output dimensions —
//   the backend recomputes interpolation weights during backward.
//
// Nearest 2-D / 3-D: forward delegates to IBackend::interpolate_nearest_*_forward
//   and attaches a backward node that scatter-adds each output gradient onto its
//   unique source pixel/voxel (same floor coordinate map as the forward).

#include "Interpolate.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>
#include <string>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/Helpers.h"
#include "../autograd/Node.h"
#include "../backend/Dispatcher.h"
#include "../backend/SamplePositions.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/Helpers.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/View.h"

namespace lucid {

namespace {

// Every resampler reads neighbours of each output coordinate from the
// input's spatial extents, so neither side may be empty — an empty source
// segfaulted, a negative output size was an allocation of 2^64 bytes.
void check_resample(const std::string& op,
                    const Shape& in,
                    int n_spatial,
                    std::initializer_list<int> out) {
    require_rank(in, static_cast<std::size_t>(n_spatial) + 2, op + ": input",
                 channels_first_layout(n_spatial));
    for (int i = 0; i < n_spatial; ++i)
        require_at_least(in[2 + i], 1, op + ": input spatial extent");
    for (int o : out)
        require_at_least(o, 1, op + ": output extent");
    // Each extent fits ``int``; their product with the batch and channels
    // need not, and the backends size the output buffer by it.
    Shape out_shape{in[0], in[1]};
    out_shape.insert(out_shape.end(), out.begin(), out.end());
    check_concrete_shape(out_shape);
}

}  // namespace

const OpSchema InterpolateBilinearBackward::schema_v1{
    "interpolate_bilinear", 1, AmpPolicy::Promote, true, "", true};

TensorImplPtr InterpolateBilinearBackward::forward(const TensorImplPtr& input0,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners) {
    Validator::input(input0, "interpolate_bilinear.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    check_resample("interpolate_bilinear", input->shape(), 2, {H_out, W_out});
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));
    scope.set_attr("align_corners", align_corners);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_bilinear_forward(input->storage(), input->shape(), H_out,
                                                          W_out, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    // wire_autograd records on_op_io internally — no explicit call.
    {
        auto bwd = std::make_shared<InterpolateBilinearBackward>();
        bwd->H_in_ = static_cast<int>(input->shape()[2]);
        bwd->W_in_ = static_cast<int>(input->shape()[3]);
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->align_corners_ = align_corners;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateBilinearBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                          out, false);
    }
    return out;
}

namespace {

// The 1-D linear resampling map as a constant matrix: row ``o`` holds the
// two weights the forward kernel gives output ``o``, from the same taps
// (``backend/SamplePositions.h``), so the matrix is exactly the map the
// kernel applies along one axis.  Returned as ``(in, out)`` when
// ``transposed``, else ``(out, in)``.
template <typename R>
void fill_resample(double* m, int in_dim, int out_dim, bool align, bool transposed) {
    const auto taps = backend::resample::linear_axis<R>(in_dim, out_dim, align);
    for (int o = 0; o < out_dim; ++o) {
        const auto& t = taps[static_cast<std::size_t>(o)];
        const auto at = [&](std::int64_t i) -> double& {
            return transposed ? m[static_cast<std::size_t>(i) * out_dim + o]
                              : m[static_cast<std::size_t>(o) * in_dim + i];
        };
        at(t.lo) += static_cast<double>(R{1} - t.frac);
        at(t.hi) += static_cast<double>(t.frac);
    }
}

TensorImplPtr
resample_matrix(int in_dim, int out_dim, bool align, bool transposed, Dtype dt, Device device) {
    const Shape shape = transposed ? Shape{in_dim, out_dim} : Shape{out_dim, in_dim};
    CpuStorage cpu = helpers::allocate_cpu(shape, Dtype::F64);
    auto* m = reinterpret_cast<double*>(cpu.ptr.get());
    if (dt == Dtype::F64)
        fill_resample<double>(m, in_dim, out_dim, align, transposed);
    else
        fill_resample<float>(m, in_dim, out_dim, align, transposed);
    Storage host{std::move(cpu)};
    if (dt != Dtype::F64)
        host = backend::Dispatcher::for_device(Device::CPU).astype(host, shape, Dtype::F64, dt);
    Storage placed = backend::Dispatcher::for_device(device).from_cpu(
        std::get<CpuStorage>(std::move(host)), shape);
    return std::make_shared<TensorImpl>(std::move(placed), shape, dt, device, false);
}

}  // namespace

std::vector<Storage> InterpolateBilinearBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_bilinear_backward(grad_out, orig_shape_, H_out_, W_out_, align_corners_,
                                             dtype_)};
}

std::vector<TensorImplPtr>
InterpolateBilinearBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // Bilinear resampling is linear in its input and separable: each channel
    // becomes M_h x M_w^T, with M the 1-D resampling matrices its weights
    // come from.  The adjoint is M_h^T g M_w — two matmuls against
    // constants, differentiable in g and, rightly, not in x.
    auto rows =
        resample_matrix(H_in_, H_out_, align_corners_, /*transposed=*/true, dtype_, device_);
    auto cols =
        resample_matrix(W_in_, W_out_, align_corners_, /*transposed=*/false, dtype_, device_);
    return {matmul_op(matmul_op(rows, grad_out), cols)};
}

TensorImplPtr
interpolate_bilinear_op(const TensorImplPtr& input, int H_out, int W_out, bool align_corners) {
    return InterpolateBilinearBackward::forward(input, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateBilinearBackward)

const OpSchema InterpolateTrilinearBackward::schema_v1{
    "interpolate_trilinear", 1, AmpPolicy::Promote, true, "", true};

TensorImplPtr InterpolateTrilinearBackward::forward(
    const TensorImplPtr& input0, int D_out, int H_out, int W_out, bool align_corners) {
    Validator::input(input0, "interpolate_trilinear.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    check_resample("interpolate_trilinear", input->shape(), 3, {D_out, H_out, W_out});
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    // The compile / Core ML emitters need the same geometry the 2-D
    // resamplers report.  ``align_corners`` in particular is not
    // recoverable from the shapes: it changes which source coordinate
    // each output sample reads, so an emitter that assumes a value
    // produces a plausible image computed from the wrong grid.
    scope.set_attr("D_out", static_cast<std::int64_t>(D_out));
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));
    scope.set_attr("align_corners", align_corners);

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_trilinear_forward(
        input->storage(), input->shape(), D_out, H_out, W_out, align_corners, input->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateTrilinearBackward>();
        bwd->D_in_ = static_cast<int>(input->shape()[2]);
        bwd->H_in_ = static_cast<int>(input->shape()[3]);
        bwd->W_in_ = static_cast<int>(input->shape()[4]);
        bwd->D_out_ = D_out;
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->align_corners_ = align_corners;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateTrilinearBackward, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, false);
    }
    return out;
}

std::vector<Storage> InterpolateTrilinearBackward::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_trilinear_backward(grad_out, orig_shape_, D_out_, H_out_, W_out_,
                                              align_corners_, dtype_)};
}

std::vector<TensorImplPtr>
InterpolateTrilinearBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // The trilinear adjoint axis by axis, as the bilinear one: width and
    // height are matmuls on the trailing two axes, depth one more after
    // folding height and width together.
    auto cols =
        resample_matrix(W_in_, W_out_, align_corners_, /*transposed=*/false, dtype_, device_);
    auto rows =
        resample_matrix(H_in_, H_out_, align_corners_, /*transposed=*/true, dtype_, device_);
    auto deep =
        resample_matrix(D_in_, D_out_, align_corners_, /*transposed=*/true, dtype_, device_);
    auto g = matmul_op(rows, matmul_op(grad_out, cols));
    const std::int64_t n = orig_shape_[0];
    const std::int64_t ch = orig_shape_[1];
    g = reshape_op(g, {n, ch, D_out_, static_cast<std::int64_t>(H_in_) * W_in_});
    g = matmul_op(deep, g);
    return {reshape_op(g, {n, ch, D_in_, H_in_, W_in_})};
}

TensorImplPtr interpolate_trilinear_op(
    const TensorImplPtr& input, int D_out, int H_out, int W_out, bool align_corners) {
    return InterpolateTrilinearBackward::forward(input, D_out, H_out, W_out, align_corners);
}
LUCID_REGISTER_OP(InterpolateTrilinearBackward)

const OpSchema InterpolateNearestBackward2D::schema_v1{
    "interpolate_nearest_2d", 1, AmpPolicy::KeepInput, true, "", true};

TensorImplPtr
InterpolateNearestBackward2D::forward(const TensorImplPtr& input0, int H_out, int W_out) {
    Validator::input(input0, "interpolate_nearest.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    check_resample("interpolate_nearest", input->shape(), 2, {H_out, W_out});
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_2d_forward(input->storage(), input->shape(), H_out,
                                                            W_out, input->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateNearestBackward2D>();
        bwd->H_in_ = static_cast<int>(input->shape()[2]);
        bwd->W_in_ = static_cast<int>(input->shape()[3]);
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateNearestBackward2D, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, false);
    }
    return out;
}

std::vector<Storage> InterpolateNearestBackward2D::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {be.interpolate_nearest_2d_backward(grad_out, orig_shape_, H_out_, W_out_, dtype_)};
}

std::vector<TensorImplPtr>
InterpolateNearestBackward2D::apply_for_graph(const TensorImplPtr& grad_out) {
    // Nearest-neighbour upscaling by an integer factor repeats each input
    // pixel over a k x k block, so its adjoint sums each block back down.
    // That is the mirror of the expand this file's forward performs, and it
    // is expressible as reshape + sum.  A non-integer ratio maps blocks of
    // uneven size and is refused rather than approximated.
    if (H_out_ % H_in_ != 0 || W_out_ % W_in_ != 0)
        ErrorBuilder("interpolate")
            .not_implemented("create_graph=True needs an integer scale factor for nearest mode");
    const std::int64_t kh = H_out_ / H_in_;
    const std::int64_t kw = W_out_ / W_in_;

    const std::int64_t n = orig_shape_[0];
    const std::int64_t ch = orig_shape_[1];
    auto split = reshape_op(grad_out, {n, ch, H_in_, kh, W_in_, kw});
    auto summed = sum_op(split, std::vector<int>{3, 5}, /*keepdims=*/false);
    std::vector<std::int64_t> orig(orig_shape_.begin(), orig_shape_.end());
    return {reshape_op(summed, orig)};
}

TensorImplPtr interpolate_nearest_2d_op(const TensorImplPtr& input, int H_out, int W_out) {
    return InterpolateNearestBackward2D::forward(input, H_out, W_out);
}
LUCID_REGISTER_OP(InterpolateNearestBackward2D)

const OpSchema InterpolateNearestBackward3D::schema_v1{
    "interpolate_nearest_3d", 1, AmpPolicy::KeepInput, true, "", true};

TensorImplPtr InterpolateNearestBackward3D::forward(const TensorImplPtr& input0,
                                                    int D_out,
                                                    int H_out,
                                                    int W_out) {
    Validator::input(input0, "interpolate_nearest_3d.input").non_null();
    // Resampling asks for the schema dtype the kernel templates
    // would have applied; see promote_for_schema.
    const TensorImplPtr input = promote_for_schema(schema_v1, input0);
    check_resample("interpolate_nearest_3d", input->shape(), 3, {D_out, H_out, W_out});
    const int N = static_cast<int>(input->shape()[0]);
    const int C = static_cast<int>(input->shape()[1]);
    Shape out_shape{N, C, D_out, H_out, W_out};
    OpScopeFull scope{schema_v1.name, input->device(), input->dtype(), out_shape};
    // Reported for symmetry with the other resamplers; nearest has no
    // ``align_corners`` to carry.
    scope.set_attr("D_out", static_cast<std::int64_t>(D_out));
    scope.set_attr("H_out", static_cast<std::int64_t>(H_out));
    scope.set_attr("W_out", static_cast<std::int64_t>(W_out));

    auto& be = backend::Dispatcher::for_device(input->device());
    Storage out_storage = be.interpolate_nearest_3d_forward(input->storage(), input->shape(), D_out,
                                                            H_out, W_out, input->dtype());
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), out_shape, input->dtype(),
                                            input->device(), false);
    {
        auto bwd = std::make_shared<InterpolateNearestBackward3D>();
        bwd->D_in_ = static_cast<int>(input->shape()[2]);
        bwd->H_in_ = static_cast<int>(input->shape()[3]);
        bwd->W_in_ = static_cast<int>(input->shape()[4]);
        bwd->D_out_ = D_out;
        bwd->H_out_ = H_out;
        bwd->W_out_ = W_out;
        bwd->orig_shape_ = input->shape();
        kernel::NaryKernel<InterpolateNearestBackward3D, 1>::wire_autograd(std::move(bwd), {input},
                                                                           out, false);
    }
    return out;
}

std::vector<Storage> InterpolateNearestBackward3D::apply(Storage grad_out) {
    auto& be = backend::Dispatcher::for_device(device_);
    return {
        be.interpolate_nearest_3d_backward(grad_out, orig_shape_, D_out_, H_out_, W_out_, dtype_)};
}

std::vector<TensorImplPtr>
InterpolateNearestBackward3D::apply_for_graph(const TensorImplPtr& grad_out) {
    // The 2-D rule one axis deeper: each voxel was repeated over a
    // kd x kh x kw block, so the adjoint sums the blocks.
    if (D_out_ % D_in_ != 0 || H_out_ % H_in_ != 0 || W_out_ % W_in_ != 0)
        ErrorBuilder("interpolate")
            .not_implemented("create_graph=True needs an integer scale factor for nearest mode");
    const std::int64_t kd = D_out_ / D_in_;
    const std::int64_t kh = H_out_ / H_in_;
    const std::int64_t kw = W_out_ / W_in_;
    auto split =
        reshape_op(grad_out, {orig_shape_[0], orig_shape_[1], D_in_, kd, H_in_, kh, W_in_, kw});
    auto summed = sum_op(split, std::vector<int>{3, 5, 7}, /*keepdims=*/false);
    return {reshape_op(summed, orig_shape_)};
}

TensorImplPtr
interpolate_nearest_3d_op(const TensorImplPtr& input, int D_out, int H_out, int W_out) {
    return InterpolateNearestBackward3D::forward(input, D_out, H_out, W_out);
}
LUCID_REGISTER_OP(InterpolateNearestBackward3D)

}  // namespace lucid
