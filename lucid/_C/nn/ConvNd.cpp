// lucid/_C/nn/ConvNd.cpp
//
// N-dimensional convolution and Unfold implementation.
//
// Forward shape formula for each spatial axis i:
//   O[i] = (S[i] + 2*pad[i] - dilation[i]*(K[i]-1) - 1) / stride[i] + 1
//
// FLOP estimate (multiply-add pairs):
//   2 * B * C_out * prod(O) * C_in_per_group * prod(K)
//
// The forward delegates to IBackend::conv_nd_forward (im2col + GEMM on CPU,
// native MLX conv on GPU).  The backward delegates to
// IBackend::conv_nd_backward, which returns [dx, dW, db].
//
// Unfold (im2col) delegates to IBackend::unfold_forward; its backward
// (fold / col2im) delegates to IBackend::unfold_backward.

#include "ConvNd.h"

#include <string>
#include <vector>

#include "../autograd/Engine.h"
#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../compile/Tracer.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/composite/Indexing.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/ufunc/Transpose.h"
#include "../ops/utils/Promote.h"
#include "../ops/utils/View.h"
#include "ConvTransposeNd.h"

namespace lucid {

template <>
const OpSchema Conv1dBackward::schema_v1{"conv1d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema Conv2dBackward::schema_v1{"conv2d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema Conv3dBackward::schema_v1{"conv3d", 1, AmpPolicy::Promote, true};

namespace {

// Compute the output size for one spatial dimension.
// S – input size, K – kernel size, stride, pad, dilation are scalars.
inline int compute_out(int S, int K, int stride, int pad, int dilation) {
    const int eff = dilation * (K - 1) + 1;  // Effective (dilated) kernel size.
    return (S + 2 * pad - eff) / stride + 1;
}

}  // namespace

template <int N>
TensorImplPtr ConvNdBackward<N>::forward(const TensorImplPtr& x,
                                         const TensorImplPtr& W,
                                         const TensorImplPtr& b,
                                         const int (&stride)[N],
                                         const int (&pad)[N],
                                         const int (&dilation)[N],
                                         int groups) {
    if (!x || !W || !b)
        ErrorBuilder("conv").fail("null input");
    const std::string op(ConvNdBackward<N>::schema_v1.name);
    Validator::pair(x, W, op + ": x/weight").same_device();
    Validator::pair(x, b, op + ": x/bias").same_device();
    require_rank(x->shape(), N + 2, op + ": x", channels_first_layout(N, "N", "C_in"));
    require_rank(W->shape(), N + 2, op + ": weight",
                 channels_first_layout(N, "C_out", "C_in/groups"));
    require_at_least(groups, 1, op + ": groups");
    require_each_at_least(stride, 1, op + ": stride");
    require_each_at_least(pad, 0, op + ": padding");
    require_each_at_least(dilation, 1, op + ": dilation");

    // 3.3 AMP plumbing: schema_v1.amp_policy == Promote.  Under an
    // ``AutocastGuard(F16)`` scope, SchemaGuard resolves ``eff_dt`` to
    // the autocast target (F32 on CPU per the Accelerate-no-F16 carve-out,
    // F16 on GPU where ``mlx::core::conv_general`` accepts native F16
    // and Apple Silicon GPUs run the conv at half throughput → speedup).
    // Without this plumbing, ``with amp.autocast(F16):`` had no effect on
    // Conv2d — F32 was used regardless.  Under autocast, the surrounding
    // Conv has cast x to F16 while W / b on the Parameter slots are still
    // F32; after the cast all three share ``eff_dt``.
    //
    // Use ``astype_op`` (not ``maybe_cast_for_kernel``) so the cast
    // tensors carry an ``AstypeBackward`` grad_fn pointing back to the
    // original Parameter / activation.  Without this the cast tensors
    // would have requires_grad=false → ``NaryKernel::wire_autograd`` would
    // short-circuit with any_grad=false → the entire conv backward chain
    // would be dropped under AMP.  ``astype_op`` returns the input
    // unchanged when dtypes already match, so the F32 baseline incurs
    // zero overhead.
    const auto [x_eff, W_eff, b_eff] =
        promote_and_check_layer_operands(ConvNdBackward<N>::schema_v1, x, W, b);
    SchemaGuard sg{ConvNdBackward<N>::schema_v1, x_eff->dtype(), x_eff->device()};
    const Dtype eff_dt = x_eff->dtype();

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int Cin = static_cast<int>(x_eff->shape()[1]);
    const int Cout = static_cast<int>(W_eff->shape()[0]);
    const int Cw = static_cast<int>(W_eff->shape()[1]);

    if (Cin % groups != 0)
        throw ShapeMismatch(x_eff->shape(), W_eff->shape(),
                            "conv: C_in must be divisible by groups");
    if (Cout % groups != 0)
        throw ShapeMismatch(W_eff->shape(), x_eff->shape(),
                            "conv: C_out must be divisible by groups");
    const int Cin_g = Cin / groups;
    const int Cout_g = Cout / groups;
    if (Cw != Cin_g)
        throw ShapeMismatch(W_eff->shape(), x_eff->shape(),
                            "conv: W.shape[1] must equal C_in / groups");
    require_length(b_eff->shape(), Cout, op + ": bias");

    int S[N];
    int K[N];
    int O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        K[i] = static_cast<int>(W_eff->shape()[2 + i]);
        O[i] = compute_out(S[i], K[i], stride[i], pad[i], dilation[i]);
        if (O[i] <= 0)
            throw ShapeMismatch(x_eff->shape(), W_eff->shape(), "conv: output shape non-positive");
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(Cout));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    int O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    OpScopeFull scope{ConvNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt, out_shape};
    // 2 ops (mul+add) per element of the output per input-channel-kernel position.
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout * O_total * Cin_g * K_total);
    // 3.5 Phase 1.2: report conv params for the compile-path emitter.
    {
        std::vector<std::int64_t> Sv(stride, stride + N);
        std::vector<std::int64_t> Pv(pad, pad + N);
        std::vector<std::int64_t> Dv(dilation, dilation + N);
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
        scope.set_attr("groups", static_cast<std::int64_t>(groups));
    }

    backend::IBackend::ConvNdOpts opts{};
    opts.N = N;
    opts.groups = groups;
    for (int i = 0; i < N; ++i) {
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
        opts.dilation[i] = dilation[i];
    }
    auto& be = backend::Dispatcher::for_device(x_eff->device());
    Storage out_storage =
        be.conv_nd_forward(x_eff->storage(), W_eff->storage(), b_eff->storage(), B, Cin, Cout,
                           Cin_g, Cout_g, S, K, O, opts, out_shape, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            x_eff->device(), false);
    // wire_autograd (below) calls ``on_op_io`` internally — recording it
    // here too would double-feed the trace (6 inputs instead of 3) and
    // abort the compile.  Conv1d/3d still rely on the wire_autograd
    // path; the explicit hook is intentionally omitted.

    auto bwd = std::make_shared<ConvNdBackward<N>>();
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
        bwd->dilation_[i] = dilation[i];
    }
    bwd->groups_ = groups;
    // saved_inputs_[0..1] hold the *cast* (effective-dtype) {x, W} so
    // the backward conv_general dispatches at eff_dt to match the forward.
    // The bias is read by shape alone and is not saved.
    kernel::NaryKernel<ConvNdBackward<N>, 3>::wire_autograd(std::move(bwd), {x_eff, W_eff, b_eff},
                                                            out, {true, true, false});
    return out;
}

template <int N>
std::vector<Storage> ConvNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin = static_cast<int>(this->input_shapes_[0][1]);
    const int Cout = static_cast<int>(this->input_shapes_[1][0]);
    const int Cin_g = Cin / this->groups_;
    const int Cout_g = Cout / this->groups_;
    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
    }
    backend::IBackend::ConvNdOpts opts{};
    opts.N = N;
    opts.groups = this->groups_;
    for (int i = 0; i < N; ++i) {
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
        opts.dilation[i] = this->dilation_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Returns [dx, dW, db].
    return be.conv_nd_backward(grad_out, this->saved_or_zeros(0, grad_out),
                               this->saved_or_zeros(1, grad_out), B, Cin, Cout, Cin_g, Cout_g, S, K,
                               O, opts, this->dtype_);
}

template <int N>
TensorImplPtr conv_weight_grad(const TensorImplPtr& input,
                               const TensorImplPtr& grad,
                               const int (&stride)[N],
                               const int (&pad)[N],
                               const int (&dilation)[N],
                               int groups,
                               const Shape& weight_shape) {
    // dW[co, ci, k] = sum over batch b and output position o of
    // grad[b, co, o] * input[b, ci, o * stride + k * dilation - pad].  With
    // batch and channel swapped that is an ordinary convolution in which the
    // batch is the reduction axis and stride and dilation trade places.
    //
    // Groups ride along as groups.  Each group's input channels go into the
    // swapped convolution's batch and the groups stack along its channel
    // axis, so a grouped convolution pairs every group's input with that
    // group's gradient and nothing else.
    const Shape& is = input->shape();
    const std::int64_t batch = is[0];
    const std::int64_t per_group = is[1] / groups;
    const std::int64_t cout = grad->shape()[1];

    std::vector<std::int64_t> split{batch, groups, per_group};
    std::vector<int> to_front{2, 1, 0};
    std::vector<std::int64_t> stacked{per_group, groups * batch};
    std::vector<int> swap{1, 0};
    for (int i = 0; i < N; ++i) {
        split.push_back(is[2 + i]);
        to_front.push_back(3 + i);
        stacked.push_back(is[2 + i]);
        swap.push_back(2 + i);
    }
    auto lhs = reshape_op(permute_op(reshape_op(input, split), to_front), stacked);
    auto rhs = permute_op(grad, swap);

    auto no_bias = zeros_op(Shape{cout}, input->dtype(), input->device());
    auto dW = ConvNdBackward<N>::forward(lhs, rhs, no_bias, dilation, pad, stride, groups);
    // When the forward's last window stopped short of the padded input, the
    // swapped convolution runs on past the kernel's extent.  Those taps are
    // not weights.
    for (int i = 0; i < N; ++i) {
        const std::int64_t k = weight_shape[2 + i];
        if (dW->shape()[2 + i] != k)
            dW = narrow_op(dW, 2 + i, 0, k);
    }
    return permute_op(dW, swap);
}

template TensorImplPtr conv_weight_grad<1>(const TensorImplPtr&,
                                           const TensorImplPtr&,
                                           const int (&)[1],
                                           const int (&)[1],
                                           const int (&)[1],
                                           int,
                                           const Shape&);
template TensorImplPtr conv_weight_grad<2>(const TensorImplPtr&,
                                           const TensorImplPtr&,
                                           const int (&)[2],
                                           const int (&)[2],
                                           const int (&)[2],
                                           int,
                                           const Shape&);
template TensorImplPtr conv_weight_grad<3>(const TensorImplPtr&,
                                           const TensorImplPtr&,
                                           const int (&)[3],
                                           const int (&)[3],
                                           const int (&)[3],
                                           int,
                                           const Shape&);

template <int N>
std::vector<TensorImplPtr> ConvNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // A convolution's adjoint is two more convolutions: dx is the transposed
    // convolution of the gradient with the same weights, dW the correlation
    // of the input against the gradient (``conv_weight_grad``), and db the
    // gradient summed over everything but the output channel.
    const auto& x = this->saved_impl_inputs_[0];
    const auto& W = this->saved_impl_inputs_[1];
    if ((this->needs_input_grad(0) && !W) || (this->needs_input_grad(1) && !x))
        ErrorBuilder("conv").fail("graph-mode backward is missing a saved input");

    const Shape& xs = this->input_shapes_[0];
    const Shape& ws = this->input_shapes_[1];
    // A transposed convolution grows each axis to (O - 1) * s - 2p +
    // d * (K - 1) + 1; what the forward's floor division dropped is what
    // output_padding restores.
    int opad[N];
    for (int i = 0; i < N; ++i)
        opad[i] = static_cast<int>(xs[2 + i] -
                                   ((this->out_shape_[2 + i] - 1) * this->stride_[i] -
                                    2 * this->pad_[i] + this->dilation_[i] * (ws[2 + i] - 1) + 1));

    TensorImplPtr dx;
    if (this->needs_input_grad(0)) {
        auto no_bias = zeros_op(Shape{xs[1]}, this->dtype_, this->device_);
        dx = ConvTransposeNdBackward<N>::forward(grad_out, W, no_bias, this->stride_, this->pad_,
                                                 opad, this->dilation_, this->groups_);
        // ``dx`` reads the weights and the gradient only; with constant weights
        // it is kept in the graph through the input with derivative zero, as the
        // reference's convolution backward is an op of the input.
        dx = zero_derivative_through(dx, this->next_edges()[0]);
    }
    TensorImplPtr dW = this->needs_input_grad(1)
                           ? conv_weight_grad<N>(x, grad_out, this->stride_, this->pad_,
                                                 this->dilation_, this->groups_, ws)
                           : nullptr;
    TensorImplPtr db;
    if (this->needs_input_grad(2)) {
        std::vector<int> not_channel{0};
        for (int i = 0; i < N; ++i)
            not_channel.push_back(2 + i);
        db = sum_op(grad_out, not_channel, /*keepdims=*/false);
    }
    return {dx, dW, db};
}

template class ConvNdBackward<1>;
template class ConvNdBackward<2>;
template class ConvNdBackward<3>;

// Entry points flatten individual scalar parameters into fixed-size arrays.
TensorImplPtr conv1d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sl,
                        int pl,
                        int dl,
                        int groups) {
    int stride[1]{sl};
    int pad[1]{pl};
    int dilation[1]{dl};
    return Conv1dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv2d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sh,
                        int sw,
                        int ph,
                        int pw,
                        int dh,
                        int dw,
                        int groups) {
    int stride[2]{sh, sw};
    int pad[2]{ph, pw};
    int dilation[2]{dh, dw};
    return Conv2dBackward::forward(x, W, b, stride, pad, dilation, groups);
}
TensorImplPtr conv3d_op(const TensorImplPtr& x,
                        const TensorImplPtr& W,
                        const TensorImplPtr& b,
                        int sd,
                        int sh,
                        int sw,
                        int pd,
                        int ph,
                        int pw,
                        int dd,
                        int dh,
                        int dw,
                        int groups) {
    int stride[3]{sd, sh, sw};
    int pad[3]{pd, ph, pw};
    int dilation[3]{dd, dh, dw};
    return Conv3dBackward::forward(x, W, b, stride, pad, dilation, groups);
}

LUCID_REGISTER_OP(Conv1dBackward)
LUCID_REGISTER_OP(Conv2dBackward)
LUCID_REGISTER_OP(Conv3dBackward)

const OpSchema UnfoldBackward::schema_v1{"unfold", 1, AmpPolicy::KeepInput, true};

TensorImplPtr UnfoldBackward::forward(const TensorImplPtr& x,
                                      const std::vector<int>& kernel,
                                      const std::vector<int>& stride,
                                      const std::vector<int>& pad,
                                      const std::vector<int>& dilation) {
    Validator::input(x, "unfold.x").non_null();

    const int N = static_cast<int>(kernel.size());
    if (N < 1 || N > 3)
        ErrorBuilder("unfold").invalid_argument("only 1D / 2D / 3D supported");
    if (static_cast<int>(stride.size()) != N || static_cast<int>(pad.size()) != N ||
        static_cast<int>(dilation.size()) != N)
        ErrorBuilder("unfold").invalid_argument("stride/pad/dilation length must match kernel");
    require_rank(x->shape(), static_cast<std::size_t>(N) + 2, "unfold: x",
                 channels_first_layout(N));
    require_each_at_least(kernel, 1, "unfold: kernel_size");
    require_each_at_least(stride, 1, "unfold: stride");
    require_each_at_least(pad, 0, "unfold: padding");
    require_each_at_least(dilation, 1, "unfold: dilation");

    const int B = static_cast<int>(x->shape()[0]);
    const int C = static_cast<int>(x->shape()[1]);
    std::vector<int> S(N), O(N);
    const std::vector<int>& K = kernel;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        const int eff = dilation[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad[i] - eff) / stride[i] + 1;
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), Shape{}, "unfold: non-positive output dim");
    }
    int O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    // Output: (B, C * prod(K), prod(O)) — the im2col column matrix.
    Shape out_shape{static_cast<std::int64_t>(B), static_cast<std::int64_t>(C * K_total),
                    static_cast<std::int64_t>(O_total)};

    OpScopeFull scope{schema_v1.name, x->device(), x->dtype(), out_shape};
    {
        std::vector<std::int64_t> Kv(kernel.begin(), kernel.end());
        std::vector<std::int64_t> Sv(stride.begin(), stride.end());
        std::vector<std::int64_t> Pv(pad.begin(), pad.end());
        std::vector<std::int64_t> Dv(dilation.begin(), dilation.end());
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
    }

    auto& be = backend::Dispatcher::for_device(x->device());
    Storage out_storage = be.unfold_forward(x->storage(), B, C, S, K, O, stride, pad, dilation,
                                            out_shape, x->dtype());

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape),
                                            x->dtype(), x->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x}, out);
    }

    {
        auto bwd = std::make_shared<UnfoldBackward>();
        bwd->kernel_ = kernel;
        bwd->stride_ = stride;
        bwd->pad_ = pad;
        bwd->dilation_ = dilation;
        // The adjoint (fold) reads the input's shape alone.
        kernel::NaryKernel<UnfoldBackward, 1>::wire_autograd(std::move(bwd), {x}, out, false);
    }
    return out;
}

std::vector<Storage> UnfoldBackward::apply(Storage grad_out) {
    const int N = static_cast<int>(kernel_.size());
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int C = static_cast<int>(this->input_shapes_[0][1]);
    std::vector<int> S(N), K = kernel_, O(N);
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        const int eff = dilation_[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad_[i] - eff) / stride_[i] + 1;
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // fold / col2im to recover the original input gradient.
    return {be.unfold_backward(grad_out, B, C, S, K, O, stride_, pad_, dilation_, this->dtype_)};
}

std::vector<TensorImplPtr> UnfoldBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // Unfold copies windows out; its adjoint adds them back where they came
    // from, which is fold onto the input's spatial extent.
    const int n = static_cast<int>(kernel_.size());
    std::vector<int> extent(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        extent[static_cast<std::size_t>(i)] = static_cast<int>(this->input_shapes_[0][2 + i]);
    return {fold_op(grad_out, extent, kernel_, stride_, pad_, dilation_)};
}

TensorImplPtr unfold_op(const TensorImplPtr& x,
                        const std::vector<int>& kernel,
                        const std::vector<int>& stride,
                        const std::vector<int>& pad,
                        const std::vector<int>& dilation) {
    return UnfoldBackward::forward(x, kernel, stride, pad, dilation);
}

LUCID_REGISTER_OP(UnfoldBackward)

// ── Fold (col2im) ─────────────────────────────────────────────────────────────

TensorImplPtr fold_op(const TensorImplPtr& x,
                      const std::vector<int>& output_size,
                      const std::vector<int>& kernel_size,
                      const std::vector<int>& stride,
                      const std::vector<int>& padding,
                      const std::vector<int>& dilation) {
    Validator::input(x, "fold.x").non_null();
    require_rank(x->shape(), 3, "fold: x", "(N, C*kH*kW, L)");
    for (const auto* v : {&output_size, &kernel_size, &stride, &padding, &dilation})
        if (v->size() != 2)
            ErrorBuilder("fold").invalid_argument(
                "output_size, kernel_size, stride, padding and dilation must each have 2 "
                "entries (only 2-D folding is supported)");
    require_each_at_least(output_size, 1, "fold: output_size");
    require_each_at_least(kernel_size, 1, "fold: kernel_size");
    require_each_at_least(stride, 1, "fold: stride");
    require_each_at_least(padding, 0, "fold: padding");
    require_each_at_least(dilation, 1, "fold: dilation");

    const int kH = kernel_size[0], kW = kernel_size[1];
    const int outH = output_size[0], outW = output_size[1];
    const int N = static_cast<int>(x->shape()[0]);
    if (x->shape()[1] % (static_cast<std::int64_t>(kH) * kW) != 0)
        ErrorBuilder("fold").invalid_argument(
            "x.shape[1]=" + std::to_string(x->shape()[1]) +
            " must be a multiple of kH*kW=" + std::to_string(kH * kW));
    const int C = static_cast<int>(x->shape()[1]) / (kH * kW);

    // The block count must match the column axis exactly; otherwise the
    // backends walk past the end of the column buffer.
    const int effH = dilation[0] * (kH - 1) + 1;
    const int effW = dilation[1] * (kW - 1) + 1;
    const int blocks_h = (outH + 2 * padding[0] - effH) / stride[0] + 1;
    const int blocks_w = (outW + 2 * padding[1] - effW) / stride[1] + 1;
    if (blocks_h <= 0 || blocks_w <= 0)
        ErrorBuilder("fold").fail("output_size too small for the given kernel/stride/dilation");
    if (static_cast<std::int64_t>(blocks_h) * blocks_w != x->shape()[2])
        ErrorBuilder("fold").shape_mismatch(Shape{static_cast<std::int64_t>(blocks_h) * blocks_w},
                                            Shape{x->shape()[2]});

    Shape out_shape = {static_cast<std::int64_t>(N), static_cast<std::int64_t>(C),
                       static_cast<std::int64_t>(outH), static_cast<std::int64_t>(outW)};
    OpScopeFull scope{"fold", x->device(), x->dtype(), out_shape};
    {
        std::vector<std::int64_t> Sv(output_size.begin(), output_size.end());
        std::vector<std::int64_t> Kv(kernel_size.begin(), kernel_size.end());
        std::vector<std::int64_t> Tv(stride.begin(), stride.end());
        std::vector<std::int64_t> Pv(padding.begin(), padding.end());
        std::vector<std::int64_t> Dv(dilation.begin(), dilation.end());
        scope.set_attr("output_size", std::move(Sv));
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Tv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("dilation", std::move(Dv));
    }

    auto& be = backend::Dispatcher::for_device(x->device());
    Storage out = be.nn_fold(x->storage(), x->shape(), out_shape, kernel_size, stride, padding,
                             dilation, x->dtype());
    auto result =
        std::make_shared<TensorImpl>(std::move(out), out_shape, x->dtype(), x->device(), false);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x}, result);
    }

    {
        auto bwd = std::make_shared<FoldBackward>();
        bwd->output_size_ = output_size;
        bwd->kernel_ = kernel_size;
        bwd->stride_ = stride;
        bwd->pad_ = padding;
        bwd->dilation_ = dilation;
        // The adjoint (unfold) reads the input's shape alone.
        kernel::NaryKernel<FoldBackward, 1>::wire_autograd(std::move(bwd), {x}, result, false);
    }
    return result;
}

std::vector<Storage> FoldBackward::apply(Storage grad_out) {
    // fold is the adjoint of unfold: d/dx = unfold(grad_out) with the same
    // hyper-parameters, recovering the (B, C*prod(K), prod(O)) column matrix.
    const int N = static_cast<int>(output_size_.size());
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    int kprod = 1;
    for (int i = 0; i < N; ++i)
        kprod *= kernel_[i];
    const int C = static_cast<int>(this->input_shapes_[0][1]) / kprod;

    std::vector<int> S = output_size_, K = kernel_, O(N);
    for (int i = 0; i < N; ++i) {
        const int eff = dilation_[i] * (K[i] - 1) + 1;
        O[i] = (S[i] + 2 * pad_[i] - eff) / stride_[i] + 1;
    }
    Shape out_shape = this->input_shapes_[0];
    auto& be = backend::Dispatcher::for_device(this->device_);
    return {be.unfold_forward(grad_out, B, C, S, K, O, stride_, pad_, dilation_, out_shape,
                              this->dtype_)};
}

std::vector<TensorImplPtr> FoldBackward::apply_for_graph(const TensorImplPtr& grad_out) {
    // Fold adds windows into place; its adjoint reads them back out.
    return {unfold_op(grad_out, kernel_, stride_, pad_, dilation_)};
}

const OpSchema FoldBackward::schema_v1{"fold", 1, AmpPolicy::KeepInput, true};

LUCID_REGISTER_OP(FoldBackward)

}  // namespace lucid
