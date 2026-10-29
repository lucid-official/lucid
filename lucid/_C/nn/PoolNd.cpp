// lucid/_C/nn/PoolNd.cpp
//
// N-dimensional MaxPool and AvgPool implementations.
//
// Output size formula (same as conv, no dilation):
//   O[i] = (S[i] + 2*pad[i] - K[i]) / stride[i] + 1
//
// MaxPool forward: IBackend::max_pool_nd_forward returns [out, argmax].
//   Backward: IBackend::max_pool_nd_backward scatters grad_out via argmax.
//
// AvgPool forward: IBackend::avg_pool_nd_forward returns out.
//   Backward: IBackend::avg_pool_nd_backward distributes grad uniformly.
//
// Both forward paths skip wiring the autograd node when grad is not needed
// (GradMode disabled or x.requires_grad == false).

#include "PoolNd.h"

#include <cstring>
#include <limits>
#include <string>
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
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Div.h"
#include "../ops/bfunc/Floordiv.h"
#include "../ops/bfunc/Maximum.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/View.h"

namespace lucid {

template <>
const OpSchema MaxPool1dBackward::schema_v1{"max_pool1d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema MaxPool2dBackward::schema_v1{"max_pool2d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema MaxPool3dBackward::schema_v1{"max_pool3d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool1dBackward::schema_v1{"avg_pool1d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool2dBackward::schema_v1{"avg_pool2d", 1, AmpPolicy::KeepInput, true};
template <>
const OpSchema AvgPool3dBackward::schema_v1{"avg_pool3d", 1, AmpPolicy::KeepInput, true};

namespace {

// Division rounding toward negative infinity.  Plain C++ ``/`` truncates
// toward zero, which differs for a negative numerator — reachable when the
// window is larger than the padded input.
inline int div_floor(int a, int b) {
    int q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

// Pooling output-size formula (no dilation).
//
// ``ceil_mode`` replaces the floor with a ceiling, which yields one extra
// output element whenever the last window overhangs the padded input.  The
// trailing correction is the standard guard: a window that starts entirely
// inside the *right* padding would pool over nothing real, so it is dropped.
inline int compute_out(int S, int K, int stride, int pad, bool ceil_mode) {
    int out = div_floor(S + 2 * pad - K + (ceil_mode ? stride - 1 : 0), stride) + 1;
    if (ceil_mode && (out - 1) * stride >= S + pad)
        --out;
    return out;
}

// Validate x's rank and the window geometry: a zero kernel turns the zero
// stride sentinel into a division by zero, and a negative padding a read
// before the first element.
template <int N>
void validate_input(const TensorImplPtr& x,
                    std::string_view op_name,
                    const int (&K)[N],
                    const int (&stride_in)[N],
                    const int (&pad)[N]) {
    const std::string op(op_name);
    Validator::input(x, op + ".x").non_null();
    require_rank(x->shape(), N + 2, op + ": x", channels_first_layout(N));
    require_each_at_least(K, 1, op + ": kernel_size");
    require_each_at_least(stride_in, 0, op + ": stride (0 means kernel_size)");
    require_each_at_least(pad, 0, op + ": padding");
}

}  // namespace

template <int N>
TensorImplPtr MaxPoolNdBackward<N>::forward(const TensorImplPtr& x,
                                            const int (&K)[N],
                                            const int (&stride_in)[N],
                                            const int (&pad)[N],
                                            bool ceil_mode) {
    validate_input<N>(x, MaxPoolNdBackward<N>::schema_v1.name, K, stride_in, pad);
    SchemaGuard sg{MaxPoolNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    int stride[N];
    // stride == 0 is a sentinel meaning "use kernel size" (non-overlapping).
    for (int i = 0; i < N; ++i)
        stride[i] = (stride_in[i] == 0) ? K[i] : stride_in[i];

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    int S[N], O[N];
    int O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        O[i] = compute_out(S[i], K[i], stride[i], pad[i], ceil_mode);
        if (O[i] <= 0)
            throw ShapeMismatch(x_eff->shape(), Shape{}, "max_pool: output non-positive");
        O_total *= O[i];
        S_total *= S[i];
    }
    int K_total = 1;
    for (int i = 0; i < N; ++i)
        K_total *= K[i];

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(C));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    OpScopeFull scope{MaxPoolNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt, out_shape};
    // 3.5 Phase 1.2: report pool params so the compile-path emitter can
    // rebuild the descriptor.
    {
        std::vector<std::int64_t> Kv(K, K + N), Sv(stride, stride + N), Pv(pad, pad + N);
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("ceil_mode", std::vector<std::int64_t>{ceil_mode ? 1 : 0});
    }

    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    opts.ceil_mode = ceil_mode;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = K[i];
        opts.stride[i] = stride[i];
        opts.pad[i] = pad[i];
    }
    auto& be = backend::Dispatcher::for_device(x_eff->device());
    // max_pool_nd_forward returns [output, argmax].
    auto pool_out =
        be.max_pool_nd_forward(x_eff->storage(), x_eff->shape(), out_shape, opts, eff_dt);
    Storage out_storage = std::move(pool_out[0]);
    Storage saved_argmax = std::move(pool_out[1]);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            x_eff->device(), false);
    // wire_autograd (below) records ``on_op_io`` internally — calling it
    // again here would double-feed the trace (2 inputs instead of 1) and
    // abort the compile.

    // wire_autograd is always invoked so the 3.5 compile-path trace
    // hook fires regardless of GradMode (autograd is gated inside).
    auto bwd = std::make_shared<MaxPoolNdBackward<N>>();
    bwd->saved_argmax_ = std::move(saved_argmax);
    bwd->ceil_mode_ = ceil_mode;
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<MaxPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x_eff}, out, false);
    return out;
}

template <int N>
std::vector<Storage> MaxPoolNdBackward<N>::apply(Storage grad_out) {
    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    opts.ceil_mode = this->ceil_mode_;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = this->K_[i];
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Scatter grad to input positions using saved argmax indices.
    return {be.max_pool_nd_backward(grad_out, this->saved_argmax_, this->input_shapes_[0],
                                    this->out_shape_, opts, this->dtype_)};
}

template <int N>
TensorImplPtr AvgPoolNdBackward<N>::forward(const TensorImplPtr& x,
                                            const int (&K)[N],
                                            const int (&stride_in)[N],
                                            const int (&pad)[N],
                                            bool ceil_mode,
                                            bool count_include_pad) {
    validate_input<N>(x, AvgPoolNdBackward<N>::schema_v1.name, K, stride_in, pad);
    SchemaGuard sg{AvgPoolNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    int stride[N];
    for (int i = 0; i < N; ++i)
        stride[i] = (stride_in[i] == 0) ? K[i] : stride_in[i];

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    int S[N], O[N];
    int O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        O[i] = compute_out(S[i], K[i], stride[i], pad[i], ceil_mode);
        if (O[i] <= 0)
            throw ShapeMismatch(x_eff->shape(), Shape{}, "avg_pool: output non-positive");
        O_total *= O[i];
        S_total *= S[i];
    }

    Shape out_shape;
    out_shape.reserve(N + 2);
    out_shape.push_back(static_cast<std::int64_t>(B));
    out_shape.push_back(static_cast<std::int64_t>(C));
    for (int i = 0; i < N; ++i)
        out_shape.push_back(static_cast<std::int64_t>(O[i]));

    OpScopeFull scope{AvgPoolNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt, out_shape};
    // 3.5 Phase 1.2: report pool params for the compile-path emitter.
    {
        std::vector<std::int64_t> Kv(K, K + N), Sv(stride, stride + N), Pv(pad, pad + N);
        scope.set_attr("kernel_size", std::move(Kv));
        scope.set_attr("stride", std::move(Sv));
        scope.set_attr("padding", std::move(Pv));
        scope.set_attr("ceil_mode", std::vector<std::int64_t>{ceil_mode ? 1 : 0});
        scope.set_attr("count_include_pad", std::vector<std::int64_t>{count_include_pad ? 1 : 0});
    }

    backend::IBackend::PoolOpts avg_opts{};
    avg_opts.N = N;
    avg_opts.ceil_mode = ceil_mode;
    avg_opts.count_include_pad = count_include_pad;
    for (int i = 0; i < N; ++i) {
        avg_opts.K[i] = K[i];
        avg_opts.stride[i] = stride[i];
        avg_opts.pad[i] = pad[i];
    }
    auto& avg_be = backend::Dispatcher::for_device(x_eff->device());
    Storage out_storage =
        avg_be.avg_pool_nd_forward(x_eff->storage(), x_eff->shape(), out_shape, avg_opts, eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(out_storage), std::move(out_shape), eff_dt,
                                            x_eff->device(), false);
    // wire_autograd (below) records ``on_op_io`` internally — calling it
    // again here would double-feed the trace.

    // wire_autograd always — trace-hook visibility under no-grad.
    auto bwd = std::make_shared<AvgPoolNdBackward<N>>();
    bwd->ceil_mode_ = ceil_mode;
    bwd->count_include_pad_ = count_include_pad;
    for (int i = 0; i < N; ++i) {
        bwd->K_[i] = K[i];
        bwd->stride_[i] = stride[i];
        bwd->pad_[i] = pad[i];
    }
    kernel::NaryKernel<AvgPoolNdBackward<N>, 1>::wire_autograd(std::move(bwd), {x_eff}, out, false);
    return out;
}

template <int N>
std::vector<Storage> AvgPoolNdBackward<N>::apply(Storage grad_out) {
    backend::IBackend::PoolOpts opts{};
    opts.N = N;
    opts.ceil_mode = this->ceil_mode_;
    opts.count_include_pad = this->count_include_pad_;
    for (int i = 0; i < N; ++i) {
        opts.K[i] = this->K_[i];
        opts.stride[i] = this->stride_[i];
        opts.pad[i] = this->pad_[i];
    }
    auto& be = backend::Dispatcher::for_device(this->device_);
    // Distribute grad evenly across each pooling window.
    return {be.avg_pool_nd_backward(grad_out, this->input_shapes_[0], this->out_shape_, opts,
                                    this->dtype_)};
}

template <int N>
std::vector<TensorImplPtr> MaxPoolNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Max pooling sends each output's gradient back to the one input it came
    // from, so the adjoint is a scatter at the saved winners.  Those are
    // integers with no derivative, which is why reusing them costs nothing —
    // unlike a saved statistic, which depends on the input and has to be
    // recomputed.
    const Shape& xs = this->input_shapes_[0];
    std::int64_t planes = xs[0] * xs[1];
    std::int64_t in_plane = 1, out_plane = 1;
    for (std::size_t i = 2; i < xs.size(); ++i)
        in_plane *= xs[i];
    for (std::size_t i = 2; i < this->out_shape_.size(); ++i)
        out_plane *= this->out_shape_[i];

    const std::vector<std::int64_t> flat_out{planes, out_plane};

    // The index stays I32: scatter_add's backend reads it as int32_t and
    // never sees a dtype, so widening it here would be read back at half
    // width — the gradient lands on the wrong elements and nothing raises.
    auto idx = std::make_shared<TensorImpl>(this->saved_argmax_, this->out_shape_, Dtype::I32,
                                            this->device_, false);
    if (this->device_ == Device::GPU) {
        // On the GPU backend, saved_argmax_ stores the within-window index
        // k in [0, K_0 * ... * K_{N-1}) rather than the flat in-plane index.
        // Convert k and the output window coordinate into the flat in-plane
        // coordinate in [0, in_plane), marking out-of-bounds padded winners
        // with -1 so the mask below drops them identically to the CPU path.
        int K_suffix[4];
        K_suffix[N] = 1;
        for (int i = N - 1; i >= 0; --i)
            K_suffix[i] = K_suffix[i + 1] * this->K_[i];
        auto make_i32 = [&](std::int64_t v) {
            return full_op(Shape{1}, static_cast<double>(v), Dtype::I32, this->device_);
        };
        auto zero_i32 = make_i32(0);
        auto one_i32 = make_i32(1);
        TensorImplPtr flat_idx = nullptr;
        TensorImplPtr valid_mask = nullptr;
        for (int i = 0; i < N; ++i) {
            auto ki =
                remainder_op(floordiv_op(idx, make_i32(K_suffix[i + 1])), make_i32(this->K_[i]));
            std::vector<std::int64_t> rs(N + 2, 1);
            rs[2 + i] = this->out_shape_[2 + i];
            auto o_r = reshape_op(arange_op(0.0, static_cast<double>(this->out_shape_[2 + i]), 1.0,
                                            Dtype::I32, this->device_),
                                  rs);
            auto coord = sub_op(add_op(mul_op(o_r, make_i32(this->stride_[i])), ki),
                                make_i32(this->pad_[i]));
            auto s_i = make_i32(xs[2 + i]);
            auto in_bounds = mul_op(astype_op(greater_equal_op(coord, zero_i32), Dtype::I32),
                                    astype_op(less_op(coord, s_i), Dtype::I32));
            if (i == 0) {
                flat_idx = coord;
                valid_mask = in_bounds;
            } else {
                flat_idx = add_op(mul_op(flat_idx, s_i), coord);
                valid_mask = mul_op(valid_mask, in_bounds);
            }
        }
        idx = add_op(mul_op(flat_idx, valid_mask), sub_op(valid_mask, one_i32));
    }
    idx = reshape_op(idx, flat_out);
    auto g = reshape_op(grad_out, flat_out);

    // A window that saw only padding has no winner.  Clamping the index and
    // masking the gradient says that in dtypes both streams support; the CPU
    // backend has no integer ``where``.
    auto zeros_idx = zeros_like_op(idx);
    auto mask = astype_op(greater_equal_op(idx, zeros_idx), this->dtype_);
    auto safe_idx = maximum_op(idx, zeros_idx);
    auto src = mul_op(g, mask);

    std::vector<std::int64_t> in_shape(xs.begin(), xs.end());
    auto base = zeros_op(Shape{planes, in_plane}, this->dtype_, this->device_);
    auto dx = scatter_add_op(base, safe_idx, src, /*dim=*/1);
    // The winners are a function of the input: the gradient stays in the
    // graph with derivative zero, as the reference's pooling backward does.
    return {zero_derivative_through(reshape_op(dx, in_shape), this->next_edges()[0])};
}

template class MaxPoolNdBackward<1>;
template class MaxPoolNdBackward<2>;
template class MaxPoolNdBackward<3>;
template <int N>
std::vector<TensorImplPtr> AvgPoolNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    // Average pooling spreads each output's gradient evenly over its window,
    // so with non-overlapping unpadded windows the adjoint is exactly a
    // repeat: expand each output element across its K positions and divide
    // by the window size.  Overlapping or padded windows are a
    // conv-transpose against a constant kernel instead, and conv-transpose
    // has no graph-mode formula yet — so those are refused by name rather
    // than answered with the wrong tensor.
    const Shape& xs = this->input_shapes_[0];
    for (int i = 0; i < N; ++i) {
        const bool tiles = this->stride_[i] == this->K_[i] && this->pad_[i] == 0 &&
                           xs[static_cast<std::size_t>(2 + i)] ==
                               this->out_shape_[static_cast<std::size_t>(2 + i)] * this->K_[i];
        if (!tiles)
            ErrorBuilder("avg_pool")
                .not_implemented(
                    "create_graph=True needs non-overlapping, unpadded windows that tile "
                    "the input exactly (stride == kernel, padding == 0)");
    }

    // (B, C, O0, O1, ...) -> (B, C, O0, 1, O1, 1, ...) -> broadcast the 1s to
    // K -> (B, C, O0*K0, O1*K1, ...).
    std::vector<std::int64_t> split{xs[0], xs[1]};
    std::vector<std::int64_t> blown{xs[0], xs[1]};
    for (int i = 0; i < N; ++i) {
        split.push_back(this->out_shape_[static_cast<std::size_t>(2 + i)]);
        split.push_back(1);
        blown.push_back(this->out_shape_[static_cast<std::size_t>(2 + i)]);
        blown.push_back(this->K_[i]);
    }
    std::int64_t window = 1;
    for (int i = 0; i < N; ++i)
        window *= this->K_[i];

    auto spread = broadcast_to_op(reshape_op(grad_out, split), Shape(blown.begin(), blown.end()));
    std::vector<std::int64_t> in_shape(xs.begin(), xs.end());
    auto dx = reshape_op(spread, in_shape);
    // Kept in the graph with derivative zero, as max pooling's is: the
    // reference's pooling backward is an op of the input.
    return {zero_derivative_through(div_op(dx, full_like_op(dx, static_cast<double>(window))),
                                    this->next_edges()[0])};
}

template class AvgPoolNdBackward<1>;
template class AvgPoolNdBackward<2>;
template class AvgPoolNdBackward<3>;

// Entry points pack scalar parameters into fixed-size arrays.
TensorImplPtr max_pool1d_op(const TensorImplPtr& x, int KL, int sl, int pl, bool ceil_mode) {
    int K[1]{KL};
    int s[1]{sl};
    int p[1]{pl};
    return MaxPool1dBackward::forward(x, K, s, p, ceil_mode);
}
TensorImplPtr max_pool2d_op(
    const TensorImplPtr& x, int KH, int KW, int sh, int sw, int ph, int pw, bool ceil_mode) {
    int K[2]{KH, KW};
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    return MaxPool2dBackward::forward(x, K, s, p, ceil_mode);
}
TensorImplPtr max_pool3d_op(const TensorImplPtr& x,
                            int KD,
                            int KH,
                            int KW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw,
                            bool ceil_mode) {
    int K[3]{KD, KH, KW};
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    return MaxPool3dBackward::forward(x, K, s, p, ceil_mode);
}
TensorImplPtr avg_pool1d_op(
    const TensorImplPtr& x, int KL, int sl, int pl, bool ceil_mode, bool count_include_pad) {
    int K[1]{KL};
    int s[1]{sl};
    int p[1]{pl};
    return AvgPool1dBackward::forward(x, K, s, p, ceil_mode, count_include_pad);
}
TensorImplPtr avg_pool2d_op(const TensorImplPtr& x,
                            int KH,
                            int KW,
                            int sh,
                            int sw,
                            int ph,
                            int pw,
                            bool ceil_mode,
                            bool count_include_pad) {
    int K[2]{KH, KW};
    int s[2]{sh, sw};
    int p[2]{ph, pw};
    return AvgPool2dBackward::forward(x, K, s, p, ceil_mode, count_include_pad);
}
TensorImplPtr avg_pool3d_op(const TensorImplPtr& x,
                            int KD,
                            int KH,
                            int KW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw,
                            bool ceil_mode,
                            bool count_include_pad) {
    int K[3]{KD, KH, KW};
    int s[3]{sd, sh, sw};
    int p[3]{pd, ph, pw};
    return AvgPool3dBackward::forward(x, K, s, p, ceil_mode, count_include_pad);
}

LUCID_REGISTER_OP(MaxPool1dBackward)
LUCID_REGISTER_OP(MaxPool2dBackward)
LUCID_REGISTER_OP(MaxPool3dBackward)
LUCID_REGISTER_OP(AvgPool1dBackward)
LUCID_REGISTER_OP(AvgPool2dBackward)
LUCID_REGISTER_OP(AvgPool3dBackward)

}  // namespace lucid
