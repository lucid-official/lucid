// lucid/_C/nn/ConvTransposeNd.cpp
//
// Transposed convolution (deconvolution) forward and backward.
//
// Output shape per spatial axis i:
//   O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + K[i] + opad[i]
//
// FLOP estimate: 2 * B * C_out * prod(O) * C_in * prod(K)
//
// The forward delegates to IBackend::conv_transpose_nd_forward (col2im + GEMM
// on CPU, MLX conv-transpose on GPU).  The backward delegates to
// IBackend::conv_transpose_nd_backward, which returns [dx, dW, db].

#include "ConvTransposeNd.h"

#include <algorithm>
#include <string>
#include <vector>

#include "../autograd/Engine.h"
#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../core/Error.h"
#include "../core/OpRegistry.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/composite/Indexing.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Reductions.h"
#include "ConvNd.h"

namespace lucid {

template <>
const OpSchema ConvTranspose1dBackward::schema_v1{"conv_transpose1d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose2dBackward::schema_v1{"conv_transpose2d", 1, AmpPolicy::Promote, true};
template <>
const OpSchema ConvTranspose3dBackward::schema_v1{"conv_transpose3d", 1, AmpPolicy::Promote, true};

namespace {

template <int N>
backend::IBackend::ConvTransposeNdOpts make_conv_transpose_opts(int groups,
                                                                const int (&stride)[N],
                                                                const int (&pad)[N],
                                                                const int (&dilation)[N],
                                                                const int (&opad)[N]) {
    backend::IBackend::ConvTransposeNdOpts opts{};
    opts.N = N;
    opts.groups = groups;
    for (int i = 0; i < N; ++i) {
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
        opts.dilation[i] = dilation[i];
        opts.opad[i] = opad[i];
    }
    return opts;
}

template <int N>
void validate_conv_transpose_params(const std::string& op,
                                    int groups,
                                    const int (&stride)[N],
                                    const int (&pad)[N],
                                    const int (&opad)[N],
                                    const int (&dilation)[N]) {
    require_at_least(groups, 1, op + ": groups");
    require_each_at_least(stride, 1, op + ": stride");
    require_each_at_least(pad, 0, op + ": padding");
    require_each_at_least(opad, 0, op + ": output_padding");
    require_each_at_least(dilation, 1, op + ": dilation");
    for (int i = 0; i < N; ++i) {
        if (opad[i] >= std::max(stride[i], dilation[i])) {
            ErrorBuilder(op).invalid_argument("output_padding must be smaller than stride or "
                                              "dilation, got " +
                                              std::to_string(opad[i]) + " on axis " +
                                              std::to_string(i));
        }
    }
}

std::array<TensorImplPtr, 3> cast_conv_transpose_operands(const TensorImplPtr& x,
                                                          const TensorImplPtr& W,
                                                          const TensorImplPtr& b,
                                                          Dtype eff_dt) {
    return {astype_op(x, eff_dt), astype_op(W, eff_dt), astype_op(b, eff_dt)};
}

template <int N>
TensorImplPtr finish_conv_transpose_forward(Storage out_storage,
                                            Shape out_shape,
                                            Dtype eff_dt,
                                            const std::array<TensorImplPtr, 3>& ins,
                                            int groups,
                                            const int (&stride)[N],
                                            const int (&pad)[N],
                                            const int (&opad)[N],
                                            const int (&dilation)[N]) {
    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            ins[0]->device(), false);
    auto bwd = std::make_shared<ConvTransposeNdBackward<N>>();
    bwd->groups_ = groups;
    for (int i = 0; i < N; ++i) {
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
        bwd->opad_[i] = opad[i];
        bwd->dilation_[i] = dilation[i];
    }
    kernel::NaryKernel<ConvTransposeNdBackward<N>, 3>::wire_autograd(
        std::move(bwd), {ins[0], ins[1], ins[2]}, out, {true, true, false});
    return out;
}

}  // namespace

template <int N>
TensorImplPtr ConvTransposeNdBackward<N>::forward(const TensorImplPtr& x,
                                                  const TensorImplPtr& W,
                                                  const TensorImplPtr& b,
                                                  const int (&stride)[N],
                                                  const int (&pad)[N],
                                                  const int (&opad)[N],
                                                  const int (&dilation)[N],
                                                  int groups) {
    if (!x || !W || !b)
        ErrorBuilder("conv_transpose").fail("null input");
    const std::string op(ConvTransposeNdBackward<N>::schema_v1.name);
    validate_conv_transpose_params(op, groups, stride, pad, opad, dilation);
    Validator::pair(x, W, op + ": x/weight").same_device();
    Validator::pair(x, b, op + ": x/bias").same_device();
    require_rank(x->shape(), N + 2, op + ": x", channels_first_layout(N, "N", "C_in"));
    require_rank(W->shape(), N + 2, op + ": weight",
                 channels_first_layout(N, "C_in", "C_out/groups"));

    SchemaGuard sg{ConvTransposeNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const auto ins = cast_conv_transpose_operands(x, W, b, eff_dt);

    const int B = static_cast<int>(x->shape()[0]);
    const int Cin = static_cast<int>(x->shape()[1]);
    const int Cw = static_cast<int>(W->shape()[0]);
    // Unlike a forward convolution, the transposed weight's *second* axis
    // is the divided one, so C_out is recovered by multiplying it back up.
    const int Cout_g = static_cast<int>(W->shape()[1]);
    const int Cout = Cout_g * groups;
    if (Cw != Cin)
        throw ShapeMismatch(W->shape(), x->shape(), "conv_transpose: C_in mismatch");
    if (Cin % groups != 0)
        throw ShapeMismatch(x->shape(), W->shape(),
                            "conv_transpose: C_in must be divisible by groups");
    require_length(b->shape(), Cout, op + ": bias");

    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x->shape()[2 + i]);
        K[i] = static_cast<int>(W->shape()[2 + i]);
        // Transposed-conv output formula (the inverse of the forward
        // convolution's, with ``opad`` picking one of the ``stride``
        // input extents that map to this output extent).
        O[i] = (S[i] - 1) * stride[i] - 2 * pad[i] + dilation[i] * (K[i] - 1) + 1 + opad[i];
        if (O[i] <= 0)
            throw ShapeMismatch(x->shape(), W->shape(),
                                "conv_transpose: output shape non-positive");
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(Cout));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    int O_total = 1, S_total = 1, K_total = 1;
    for (int i = 0; i < N; ++i) {
        O_total *= O[i];
        S_total *= S[i];
        K_total *= K[i];
    }

    OpScopeFull scope{ConvTransposeNdBackward<N>::schema_v1.name, x->device(), eff_dt, out_shape};
    scope.set_flops(static_cast<std::int64_t>(2) * B * Cout_g * O_total * Cin * K_total);
    // Reported for the compile / Core ML emitters, which have no other way
    // to recover the geometry from the traced shapes alone.
    scope.set_attr("stride", std::vector<std::int64_t>(stride, stride + N));
    scope.set_attr("padding", std::vector<std::int64_t>(pad, pad + N));
    scope.set_attr("output_padding", std::vector<std::int64_t>(opad, opad + N));
    scope.set_attr("dilation", std::vector<std::int64_t>(dilation, dilation + N));
    scope.set_attr("groups", static_cast<std::int64_t>(groups));

    const auto opts = make_conv_transpose_opts(groups, stride, pad, dilation, opad);
    Storage out_storage =
        backend::Dispatcher::for_device(x->device())
            .conv_transpose_nd_forward(ins[0]->storage(), ins[1]->storage(), ins[2]->storage(), B,
                                       Cin, Cout, S, K, O, opts, out_shape, eff_dt);
    return finish_conv_transpose_forward(std::move(out_storage), std::move(out_shape), eff_dt, ins,
                                         groups, stride, pad, opad, dilation);
}

template <int N>
std::vector<Storage> ConvTransposeNdBackward<N>::apply(Storage grad_out) {
    const int B = static_cast<int>(this->input_shapes_[0][0]);
    const int Cin = static_cast<int>(this->input_shapes_[0][1]);
    // input_shapes_[1][1] is C_out / groups — the weight's divided axis.
    const int Cout = static_cast<int>(this->input_shapes_[1][1]) * this->groups_;
    int S[N], K[N], O[N];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(this->input_shapes_[0][2 + i]);
        K[i] = static_cast<int>(this->input_shapes_[1][2 + i]);
        O[i] = static_cast<int>(this->out_shape_[2 + i]);
    }

    const auto opts = make_conv_transpose_opts(this->groups_, this->stride_, this->pad_,
                                               this->dilation_, this->opad_);

    // Returns [dx, dW, db].
    return backend::Dispatcher::for_device(this->device_)
        .conv_transpose_nd_backward(grad_out, this->saved_or_zeros(0, grad_out),
                                    this->saved_or_zeros(1, grad_out), B, Cin, Cout, S, K, O, opts,
                                    this->dtype_);
}

template <int N>
std::vector<TensorImplPtr>
ConvTransposeNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // A transposed convolution's adjoint is the convolution it transposes:
    // dx is that convolution of the gradient with the same weights, dW the
    // weight gradient of that convolution with the gradient as its input
    // and x as its output gradient, db the gradient summed over everything
    // but the output channel.
    const auto& x = this->saved_impl_inputs_[0];
    const auto& W = this->saved_impl_inputs_[1];
    if ((this->needs_input_grad(0) && !W) || (this->needs_input_grad(1) && !x))
        ErrorBuilder("conv_transpose").fail("graph-mode backward is missing a saved input");

    const Shape& xs = this->input_shapes_[0];
    const Shape& ws = this->input_shapes_[1];
    TensorImplPtr dx;
    if (this->needs_input_grad(0)) {
        auto no_bias = zeros_op(Shape{xs[1]}, this->dtype_, this->device_);
        dx = ConvNdBackward<N>::forward(grad_out, W, no_bias, this->stride_, this->pad_,
                                        this->dilation_, this->groups_);
        // An output_padding of a stride or more — allowed while the dilation is
        // larger — lets the convolution fit a window past the input's end.
        for (int i = 0; i < N; ++i)
            if (dx->shape()[2 + i] != xs[2 + i])
                dx = narrow_op(dx, 2 + i, 0, xs[2 + i]);
        // ``dx`` reads the weights and the gradient only; with constant weights
        // it is kept in the graph through the input with derivative zero, as the
        // reference's convolution backward is an op of the input.
        dx = zero_derivative_through(dx, this->next_edges()[0]);
    }
    TensorImplPtr dW = this->needs_input_grad(1)
                           ? conv_weight_grad<N>(grad_out, x, this->stride_, this->pad_,
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

template class ConvTransposeNdBackward<1>;
template class ConvTransposeNdBackward<2>;
template class ConvTransposeNdBackward<3>;

// Entry points pack scalar parameters into fixed-size arrays.
TensorImplPtr conv_transpose1d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sl,
                                  int pl,
                                  int opl,
                                  int dl,
                                  int groups) {
    int s[1]{sl};
    int p[1]{pl};
    int op[1]{opl};
    int d[1]{dl};
    return ConvTranspose1dBackward::forward(x, W, b, s, p, op, d, groups);
}
TensorImplPtr conv_transpose2d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sh,
                                  int sw,
                                  int ph,
                                  int pw,
                                  int oph,
                                  int opw,
                                  int dh,
                                  int dw,
                                  int groups) {
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    int op[2]{oph, opw};
    int d[2]{dh, dw};
    return ConvTranspose2dBackward::forward(x, W, b, s, p, op, d, groups);
}
TensorImplPtr conv_transpose3d_op(const TensorImplPtr& x,
                                  const TensorImplPtr& W,
                                  const TensorImplPtr& b,
                                  int sd,
                                  int sh,
                                  int sw,
                                  int pd,
                                  int ph,
                                  int pw,
                                  int opd,
                                  int oph,
                                  int opw,
                                  int dd,
                                  int dh,
                                  int dw,
                                  int groups) {
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    int op[3]{opd, oph, opw};
    int d[3]{dd, dh, dw};
    return ConvTranspose3dBackward::forward(x, W, b, s, p, op, d, groups);
}

LUCID_REGISTER_OP(ConvTranspose1dBackward)
LUCID_REGISTER_OP(ConvTranspose2dBackward)
LUCID_REGISTER_OP(ConvTranspose3dBackward)

}  // namespace lucid
