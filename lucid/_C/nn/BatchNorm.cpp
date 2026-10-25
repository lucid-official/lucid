// lucid/_C/nn/BatchNorm.cpp
//
// Training-mode Batch Normalization for 1-D, 2-D, and 3-D inputs.
//
// For each channel c, statistics are computed over the (B, S_total) axes:
//   mean_c  = mean over all (b, spatial) pairs
//   rstd_c  = 1 / sqrt(var_c + eps)
//   y_{b,c,...} = (x_{b,c,...} - mean_c) * rstd_c * gamma_c + beta_c
//
// The forward delegates to IBackend::batch_norm_forward, which returns
// [y, mean, rstd].  The backward delegates to IBackend::batch_norm_backward,
// which returns [dx, d_gamma, d_beta].  Running-stats inference is handled by
// BatchNormEvalBackward in NormExt.h.

#include "BatchNorm.h"

#include <cstring>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include <mlx/array.h>
#include <mlx/transforms.h>

#include "../autograd/Helpers.h"
#include "../backend/Dispatcher.h"
#include "../backend/IBackend.h"
#include "../compile/Tracer.h"
#include "../core/AmpPolicy.h"
#include "../core/ElementWrite.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "../core/OpRegistry.h"
#include "../core/Profiler.h"
#include "../core/SchemaGuard.h"
#include "../core/Scope.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "../core/Validate.h"
#include "../kernel/BinaryKernel.h"
#include "../kernel/NaryKernel.h"
#include "../ops/bfunc/Add.h"
#include "../ops/bfunc/Mul.h"
#include "../ops/bfunc/Sub.h"
#include "../ops/bfunc/_BinaryOp.h"
#include "../ops/gfunc/Gfunc.h"
#include "../ops/ufunc/Astype.h"
#include "../ops/ufunc/Exponential.h"
#include "../ops/ufunc/Reductions.h"
#include "../ops/utils/View.h"

namespace lucid {

namespace {

// Puts a running statistic's new values into ``stat``, whose buffer is its
// elements (write_elements).  A buffer other tensors also read — its live
// views, or a larger buffer it is the start of — takes them into its own
// bytes, so the others see the update, and every member's version moves;
// any other has its storage slot replaced, as it always has, and its
// version stays, as the reference's does.  A plain copy rather than
// write_through: the op's own input storage (``into_stat``) still holds the
// buffer here, and no node saves the running statistics — they are not
// inputs of the graph.
void put_running_stat(TensorImpl& stat, Storage fresh) {
    Storage& slot = stat.mutable_storage();
    auto* dst = std::get_if<CpuStorage>(&slot);
    const auto* src = std::get_if<CpuStorage>(&fresh);
    if (dst && src && (stat.is_aliased() || !stat.is_dense())) {
        if (stat.nbytes() > 0)
            std::memcpy(dst->ptr.get(), src->ptr.get(), stat.nbytes());
        stat.bump_version();
        return;
    }
    slot = std::move(fresh);
}

// Takes a running statistic's new values into its own elements, wherever
// they sit in its buffer: a column of a larger tensor, a transpose, a Metal
// view of part of one.  The update took them as the first C of the buffer
// and wrote there (LCD-491).  A real statistic refuses a phantom update
// there too, before its slot could take it (write_elements).
void take_running_stat(TensorImplPtr stat, Storage fresh) {
    write_elements(stat, fresh, "batch_norm running statistics",
                   [&](TensorImplPtr& elems) { put_running_stat(*elems, std::move(fresh)); });
}

// Takes both running statistics' new values, or neither.  Only a write-back
// into a statistic whose buffer is not its elements can refuse (one whose
// elements overlap), so the mean's values are kept aside, and put back if
// the variance is refused, only when the variance takes that path.
void take_running_stats(const TensorImplPtr& mean,
                        const TensorImplPtr& var,
                        Storage new_mean,
                        Storage new_var) {
    const TensorImplPtr mean_before = buffer_is_elements(*var) ? nullptr : dense_elements(*mean);
    take_running_stat(mean, std::move(new_mean));
    try {
        take_running_stat(var, std::move(new_var));
    } catch (...) {
        if (mean_before)
            mean->copy_from(*mean_before);
        throw;
    }
}

// Schedules the new running statistics on Metal, so the next step's update
// does not carry this one's lazy graph.  Their buffers' arrays: a Metal
// view's family root, which the write replaced.
void flush_running_stats(const TensorImplPtr& mean, const TensorImplPtr& var) {
    std::vector<mlx::core::array> arrs;
    arrs.reserve(2);
    for (const auto* stat : {&mean, &var})
        if (const auto* gs = std::get_if<GpuStorage>(&(*stat)->raw_storage()); gs && gs->arr)
            arrs.push_back(*gs->arr);
    if (!arrs.empty())
        mlx::core::async_eval(arrs);
}

// Running statistics are a pair of (C,) buffers on x's device sharing one
// dtype: the update reads the variance at the mean's dtype and writes C
// entries into each.
void check_running_stats(const std::string& op,
                         const TensorImplPtr& x,
                         const TensorImplPtr& mean,
                         const TensorImplPtr& var,
                         std::int64_t C) {
    if (!mean && !var)
        return;
    if (!mean || !var)
        ErrorBuilder(op).invalid_argument("running_mean and running_var must be given together");
    Validator::pair(x, mean, op + ": x/running_mean").same_device();
    Validator::pair(mean, var, op + ": running_mean/running_var").same_device().same_dtype();
    require_length(mean->shape(), C, op + ": running_mean");
    require_length(var->shape(), C, op + ": running_var");
}

// Every operand of a training-mode batch norm over ``n_spatial`` spatial
// axes, checked before the kernel reads ``x->shape()[2 + i]`` or C entries
// of any per-channel operand.
void check_batch_norm_operands(const std::string& op,
                               int n_spatial,
                               const TensorImplPtr& x,
                               const TensorImplPtr& gamma,
                               const TensorImplPtr& beta,
                               const TensorImplPtr& running_mean,
                               const TensorImplPtr& running_var) {
    Validator::pair(x, gamma, op + ": x/weight").same_device();
    Validator::pair(x, beta, op + ": x/bias").same_device();
    require_rank(x->shape(), static_cast<std::size_t>(n_spatial) + 2, op + ": x",
                 channels_first_layout(n_spatial));
    const std::int64_t C = x->shape()[1];
    require_length(gamma->shape(), C, op + ": weight");
    require_length(beta->shape(), C, op + ": bias");
    check_running_stats(op, x, running_mean, running_var, C);
}

// The number of elements each channel's statistics reduce over.  The
// reduction must leave exactly the (C,) channel axis the buffers hold, so
// ``axes`` names every other axis, each once.
int reduced_count(const std::string& op, const Shape& x_shape, const std::vector<int>& axes) {
    const std::string refusal =
        "reduce_axes must name every axis but the channel axis 1, each once";
    std::vector<bool> reduced(x_shape.size(), false);
    int n = 1;
    for (int d : axes) {
        if (d < 0 || d >= static_cast<int>(x_shape.size()))
            ErrorBuilder(op).index_error("reduce axis out of range");
        if (d == 1 || reduced[static_cast<std::size_t>(d)])
            ErrorBuilder(op).invalid_argument(refusal);
        reduced[static_cast<std::size_t>(d)] = true;
        n *= static_cast<int>(x_shape[static_cast<std::size_t>(d)]);
    }
    if (axes.size() + 1 != x_shape.size())
        ErrorBuilder(op).invalid_argument(refusal);
    return n;
}

}  // namespace

template <>
const OpSchema BatchNorm1dBackward::schema_v1{"batch_norm1d", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm2dBackward::schema_v1{"batch_norm", 1, AmpPolicy::ForceFP32, true};
template <>
const OpSchema BatchNorm3dBackward::schema_v1{"batch_norm3d", 1, AmpPolicy::ForceFP32, true};

template <int N>
TensorImplPtr BatchNormNdBackward<N>::forward(const TensorImplPtr& x,
                                              const TensorImplPtr& gamma,
                                              const TensorImplPtr& beta,
                                              double eps,
                                              const TensorImplPtr& running_mean,
                                              const TensorImplPtr& running_var,
                                              double momentum) {
    if (!x || !gamma || !beta)
        ErrorBuilder("batch_norm").fail("null input");
    check_batch_norm_operands(std::string(BatchNormNdBackward<N>::schema_v1.name), N, x, gamma,
                              beta, running_mean, running_var);

    // 3.3 AMP plumbing: schema_v1.amp_policy == ForceFP32.  Under
    // ``AutocastGuard(F16)`` SchemaGuard returns ``Dtype::F32`` regardless
    // of input dtype — BN's batch statistics are numerically sensitive
    // and running them in F16 risks catastrophic cancellation of the
    // mean/variance reductions.  Under autocast, the surrounding Conv has
    // cast x to F16 while gamma / beta on the ``nn.BatchNorm`` Parameter
    // slots are still F32; after the cast all three operands share ``eff_dt``.
    // Outside an autocast scope this is a no-op (``astype_op`` returns the
    // input unchanged when dtypes already match).
    //
    // ``astype_op`` (not ``maybe_cast_for_kernel``) is used so the cast
    // tensors carry an ``AstypeBackward`` grad_fn.  Without this the
    // F32-cast x_eff has requires_grad=false and ``wire_autograd`` would
    // drop the entire BN backward chain under AMP.
    SchemaGuard sg{BatchNormNdBackward<N>::schema_v1, x->dtype(), x->device()};
    const Dtype eff_dt = sg.effective_dtype();
    const TensorImplPtr x_eff = astype_op(x, eff_dt);
    const TensorImplPtr gamma_eff = astype_op(gamma, eff_dt);
    const TensorImplPtr beta_eff = astype_op(beta, eff_dt);

    const int B = static_cast<int>(x_eff->shape()[0]);
    const int C = static_cast<int>(x_eff->shape()[1]);
    int S[N > 0 ? N : 1];
    int spatial_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_eff->shape()[2 + i]);
        spatial_total *= S[i];
    }

    OpScopeFull scope{BatchNormNdBackward<N>::schema_v1.name, x_eff->device(), eff_dt,
                      x_eff->shape()};
    // 3.5 Phase 1.2: report eps for the compile-path BN-train emitter.
    scope.set_attr("eps", eps);
    // 3.5: report momentum so the compile-path BN-train emitter can build the
    // running-stats EMA (new_rm / new_rv) as graph outputs and write them back
    // each compiled run — the eager in-place EMA below is skipped under trace.
    scope.set_attr("momentum", momentum);

    // batch_norm_forward returns [y, mean, rstd].
    auto forward =
        backend::Dispatcher::for_device(x_eff->device())
            .batch_norm_forward(x_eff->storage(), gamma_eff->storage(), beta_eff->storage(), B, C,
                                spatial_total, N, eps, x_eff->shape(), eff_dt);

    auto out = std::make_shared<TensorImpl>(std::move(forward[0]), x_eff->shape(), eff_dt,
                                            x_eff->device(), false);
    auto bwd = std::make_shared<BatchNormNdBackward<N>>();
    bwd->saved_mean_ = std::move(forward[1]);
    bwd->saved_rstd_ = std::move(forward[2]);
    // 3.4+ Phase A.4: forward slot 3 holds xnorm = (x - mean) * rstd (MLX
    // path only — MPSGraph dispatch + CPU return an empty Storage here and
    // the backend backward then recomputes inline).  Saving the lazy MLX
    // intermediate is zero forward cost — it's already in the graph for
    // ``y = xnorm * γ + β`` — and lets backward skip the 2 element-wise
    // ops (subtract + multiply) that recompute it.
    if (forward.size() >= 4) {
        bwd->saved_xnorm_ = std::move(forward[3]);
    }
    bwd->B_ = B;
    bwd->C_ = C;
    for (int i = 0; i < N; ++i)
        bwd->S_[i] = S[i];
    bwd->eps_ = eps;

    // Running-stats EMA update — reuses the SAME mean+rstd the backend
    // just computed (saved on bwd) so we avoid a second mean+var
    // reduction over x.  Profile (Mac Studio M4 Max, ResNet-18 BS=32):
    // the prior separate ``_update_running_stats`` recomputed mean+var
    // for each of 16 BNs (~1.5 ms / step of duplicated GPU work);
    // folding it in here drops that to zero.  Variance is recovered
    // from rstd via ``var = 1/rstd² - eps`` — a few cheap elementwise
    // ops on a (C,) tensor.
    // 3.5 compile: under a Tracer the running-stats EMA must become graph
    // outputs (so the cached MPSGraph executable writes the buffers every run)
    // instead of the eager in-place mutation below — otherwise the buffers
    // freeze at their trace-time value and eval() reads stale stats.  These
    // hold the cloned new-stat impls so the post-wire_autograd on_op_io calls
    // can register them as the BN op's 2 extra trace outputs.
    const bool tracing = (::lucid::compile::current_tracer() != nullptr);
    TensorImplPtr new_rm_t;
    TensorImplPtr new_rv_t;
    if (running_mean) {
        int n_total = B;
        for (int i = 0; i < N; ++i)
            n_total *= S[i];
        const double m = momentum;
        const double unbiased_factor = (n_total > 1) ? double(n_total) / double(n_total - 1) : 1.0;

        const Dtype buf_dt = running_mean->dtype();
        backend::IBackend& be = backend::Dispatcher::for_device(x_eff->device());

        // The update runs in float32 (float64 for a float64 forward), not at
        // the buffers' dtype nor at a half forward's.
        //
        // Two things were wrong with using ``buf_dt`` throughout.
        // ``saved_rstd_C`` is at ``eff_dt``, and passing it to ``square``
        // labelled as ``buf_dt`` reads one dtype's bytes as another's
        // whenever the two differ — the relabel this codebase has had to
        // fix in three other places.  And a running mean and variance are
        // real numbers: handed integer buffers, every op here ran at that
        // integer width and ``reciprocal`` refused on the CPU while Metal
        // answered, so the same call worked on one device and not the
        // other.
        //
        // And a half forward's ``rstd`` cannot be squared in half: for a
        // batch variance near 0 it is about 1/sqrt(eps), whose square
        // overflows float16, so ``1/rstd^2 - eps`` came out as ``-eps`` — a
        // negative running variance, and NaN from eval() after it.
        //
        // ``stat_dt`` is that float.  The buffers and the saved statistics
        // are read into it and the result is cast back on the way out, so
        // the caller's storage keeps its own dtype and nothing is
        // reinterpreted in between.
        const Dtype stat_dt = (eff_dt == Dtype::F64) ? Dtype::F64 : Dtype::F32;
        auto into_stat = [&](const Storage& v, Dtype from) {
            return from == stat_dt ? v
                                   : be.cast(v, Shape{static_cast<std::int64_t>(C)}, from, stat_dt);
        };

        // saved_mean/rstd are at shape (1, C, 1, ..., 1); reshape to (C,)
        // for the elementwise update.  Backend reshape is metadata-only.
        Shape kept_shape;
        kept_shape.reserve(2 + N);
        kept_shape.push_back(1);
        kept_shape.push_back(C);
        for (int i = 0; i < N; ++i)
            kept_shape.push_back(1);
        Shape stat_shape{static_cast<std::int64_t>(C)};

        Storage saved_mean_C =
            into_stat(be.reshape(bwd->saved_mean_, kept_shape, stat_shape, eff_dt), eff_dt);
        Storage saved_rstd_C =
            into_stat(be.reshape(bwd->saved_rstd_, kept_shape, stat_shape, eff_dt), eff_dt);

        // var = max(rstd^-2 - eps, 0): the subtraction cancels when the
        // variance is small beside eps, and rounding must not leave it
        // below zero.  One pow, not a square and a reciprocal, so the clamp
        // costs no buffer more than the step had.
        Storage inv_rstd_sq = be.pow_scalar(saved_rstd_C, stat_shape, stat_dt, -2.0);
        Storage var =
            be.relu(be.add_scalar(inv_rstd_sq, stat_shape, stat_dt, -eps), stat_shape, stat_dt);

        auto* tracer = ::lucid::compile::current_tracer();
        const auto rm_value = tracer ? tracer->buffer_value(running_mean) : running_mean;
        const auto rv_value = tracer ? tracer->buffer_value(running_var) : running_var;
        Storage rm_in = into_stat(rm_value->storage(), buf_dt);
        Storage rv_in = into_stat(rv_value->storage(), buf_dt);

        // new_rm = (1-m) * running_mean + m * mean
        Storage rm_scaled = be.mul_scalar(rm_in, stat_shape, stat_dt, 1.0 - m);
        Storage bm_scaled = be.mul_scalar(saved_mean_C, stat_shape, stat_dt, m);
        Storage new_rm = be.add(rm_scaled, bm_scaled, stat_shape, stat_dt);

        // new_rv = (1-m) * running_var + (m * n/(n-1)) * var
        Storage rv_scaled = be.mul_scalar(rv_in, stat_shape, stat_dt, 1.0 - m);
        Storage bv_scaled = be.mul_scalar(var, stat_shape, stat_dt, m * unbiased_factor);
        Storage new_rv = be.add(rv_scaled, bv_scaled, stat_shape, stat_dt);

        // Back to the buffers' own dtype for the write.
        if (stat_dt != buf_dt) {
            new_rm = be.cast(new_rm, stat_shape, stat_dt, buf_dt);
            new_rv = be.cast(new_rv, stat_shape, stat_dt, buf_dt);
        }

        if (tracing) {
            // Compile trace: leave the live buffers UNTOUCHED — the compiled
            // executable owns the per-run write-back (edits in _fused_step.py /
            // _step.py register new_rm/new_rv as output-feed targets).  Clone
            // the computed new stats into distinct requires_grad=false impls so
            // the on_op_io calls after wire_autograd can register them as the
            // op's 2 extra outputs.  The cloned VALUES are unused by the graph
            // (the emitter recomputes the EMA in-graph); only their identity /
            // shape / dtype / device matter.
            Storage rm_copy = be.clone(new_rm, stat_shape, buf_dt);
            Storage rv_copy = be.clone(new_rv, stat_shape, buf_dt);
            new_rm_t = std::make_shared<TensorImpl>(std::move(rm_copy), stat_shape, buf_dt,
                                                    x_eff->device(), false);
            new_rv_t = std::make_shared<TensorImpl>(std::move(rv_copy), stat_shape, buf_dt,
                                                    x_eff->device(), false);
        } else {
            take_running_stats(running_mean, running_var, std::move(new_rm), std::move(new_rv));
            if (x_eff->device() == Device::GPU)
                flush_running_stats(running_mean, running_var);
        }
    }

    // saved_inputs_[0..1] hold {x, gamma} at eff_dt; beta is read by shape
    // alone and is not saved.
    kernel::NaryKernel<BatchNormNdBackward<N>, 3>::wire_autograd(
        std::move(bwd), {x_eff, gamma_eff, beta_eff}, out, {true, true, false});

    // 3.5 compile: promote the traced BN node from 3-in/1-out to 5-in/3-out so
    // the running-stats EMA rides the compiled executable.  wire_autograd has
    // already recorded inputs={x,gamma,beta} + outputs[0]=y; these calls extend
    // it.  Each on_op_io MUST carry the FULL 5-input list because Tracer clears
    // node.inputs on every call (Tracer.cpp:89) — the running_mean/running_var
    // impls become external feeds (graph inputs), and new_rm_t/new_rv_t each get
    // a fresh appended output id (Tracer multi-output append path).
    if (tracing && new_rm_t && new_rv_t) {
        if (auto* trc = ::lucid::compile::current_tracer()) {
            const std::vector<TensorImplPtr> bn_inputs{x_eff, gamma_eff, beta_eff, running_mean,
                                                       running_var};
            trc->on_op_io(bn_inputs, out);       // alias: refresh outputs[0]=y, rebuild 5 inputs
            trc->on_op_io(bn_inputs, new_rm_t);  // append outputs[1] = new_rm
            trc->on_op_io(bn_inputs, new_rv_t);  // append outputs[2] = new_rv
            // A shared layer can execute again in this trace. Its next EMA
            // must consume these outputs, not the unchanged external buffers.
            trc->on_buffer_update(running_mean, new_rm_t);
            trc->on_buffer_update(running_var, new_rv_t);
        }
    }
    return out;
}

template <int N>
std::vector<Storage> BatchNormNdBackward<N>::apply(Storage grad_out) {
    int spatial_total = 1;
    for (int i = 0; i < N; ++i)
        spatial_total *= this->S_[i];

    // Returns [dx, d_gamma, d_beta].
    return backend::Dispatcher::for_device(this->device_)
        .batch_norm_backward(this->saved_inputs_[0], this->saved_inputs_[1], this->saved_mean_,
                             this->saved_rstd_, this->saved_xnorm_, grad_out, this->B_, this->C_,
                             spatial_total, N, this->input_shapes_[0], this->dtype_, this->eps_);
}

template <int N>
std::vector<TensorImplPtr> BatchNormNdBackward<N>::apply_for_graph(const TensorImplPtr& grad_out) {
    const auto& x = this->saved_impl_inputs_[0];
    const auto& gamma = this->saved_impl_inputs_[1];
    if (!x || !gamma)
        ErrorBuilder("batch_norm").fail("graph-mode backward is missing a saved input");

    // Batch normalisation reduces over everything except the channel axis,
    // which is the only thing separating it from layer normalisation.
    const Shape& xs = this->input_shapes_[0];
    std::vector<int> dims;
    for (std::size_t i = 0; i < xs.size(); ++i)
        if (i != 1)
            dims.push_back(static_cast<int>(i));

    auto mu = mean_op(x, dims, /*keepdims=*/true);
    auto xc = sub_op(x, mu);
    auto var = mean_op(mul_op(xc, xc), dims, /*keepdims=*/true);
    auto rstd = rsqrt_op(add_op(var, full_like_op(var, this->eps_)));
    auto xhat = mul_op(xc, rstd);

    std::vector<std::int64_t> chan(xs.size(), 1);
    chan[1] = static_cast<std::int64_t>(this->C_);
    auto gamma_b = reshape_op(gamma, chan);

    auto gg = mul_op(grad_out, gamma_b);
    auto term = sub_op(gg, mean_op(gg, dims, /*keepdims=*/true));
    term = sub_op(term, mul_op(xhat, mean_op(mul_op(gg, xhat), dims, /*keepdims=*/true)));
    auto dx = mul_op(rstd, term);

    auto d_gamma = sum_op(mul_op(grad_out, xhat), dims, /*keepdims=*/false);
    auto d_beta = sum_op(grad_out, dims, /*keepdims=*/false);
    return {dx, d_gamma, d_beta};
}

template class BatchNormNdBackward<1>;
template class BatchNormNdBackward<2>;
template class BatchNormNdBackward<3>;

TensorImplPtr batch_norm1d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps,
                              const TensorImplPtr& running_mean,
                              const TensorImplPtr& running_var,
                              double momentum) {
    return BatchNorm1dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}
TensorImplPtr batch_norm_op(const TensorImplPtr& x,
                            const TensorImplPtr& gamma,
                            const TensorImplPtr& beta,
                            double eps,
                            const TensorImplPtr& running_mean,
                            const TensorImplPtr& running_var,
                            double momentum) {
    return BatchNorm2dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}
TensorImplPtr batch_norm3d_op(const TensorImplPtr& x,
                              const TensorImplPtr& gamma,
                              const TensorImplPtr& beta,
                              double eps,
                              const TensorImplPtr& running_mean,
                              const TensorImplPtr& running_var,
                              double momentum) {
    return BatchNorm3dBackward::forward(x, gamma, beta, eps, running_mean, running_var, momentum);
}

LUCID_REGISTER_OP(BatchNorm1dBackward)
LUCID_REGISTER_OP(BatchNorm2dBackward)
LUCID_REGISTER_OP(BatchNorm3dBackward)

// ── Running-stats update (BatchNorm + InstanceNorm hot path) ──────────────
//
// Collapses the Python composition
//
//     batch_mean = x.mean(reduce_dims)
//     batch_var  = x.var(reduce_dims, correction=0)
//     new_rm = (1-eff)*running_mean + eff*batch_mean
//     new_rv = (1-eff)*running_var  + (eff*n/(n-1))*batch_var
//     buffers["running_mean"] = new_rm
//     buffers["running_var"]  = new_rv
//     buffers["num_batches_tracked"] += 1
//     eval_tensors_async([running_mean, running_var, num_batches_tracked])
//
// into a single C++ entry.  Profile (Mac Studio M4 Max, ResNet-18 BS=32) shows
// the Python composition takes ~8.8 ms / forward via ~160 pybind11 crossings
// (16 BN × ~10 Python ops); a single C++ call eliminates the per-arithmetic-op
// crossing cost.  Behaviour matches Lucid's existing Python `_update_running_stats`
// 1:1.
//
// Constraints:
//   - momentum must be a finite double (cumulative mode — Python's
//     momentum=None — is rejected so callers can keep that on the Python
//     path; cumulative requires reading num_batches_tracked.item() which
//     forces a GPU sync, defeating the optimisation).
//   - running_mean / running_var must share the same buffer dtype and the
//     shape (C,) where C == x->shape()[1].
//   - reduce_axes is the list of dims to reduce over: e.g. [0, 2, 3] for
//     BatchNorm2d, [2, 3] for InstanceNorm2d.
void batch_norm_update_running_stats(std::shared_ptr<TensorImpl> running_mean,
                                     std::shared_ptr<TensorImpl> running_var,
                                     const std::shared_ptr<TensorImpl>& x,
                                     std::vector<int> reduce_axes,
                                     double momentum,
                                     bool unbiased_var) {
    if (!x || !running_mean || !running_var) {
        ErrorBuilder("batch_norm_update_running_stats").fail("null tensor input");
    }

    if (!std::isfinite(momentum)) {
        ErrorBuilder("batch_norm_update_running_stats")
            .fail("momentum must be a finite double; cumulative mode (None) "
                  "is not supported in the C++ fast path");
    }

    const std::string op = "batch_norm_update_running_stats";
    require_min_rank(x->shape(), 2, op + ": x", "(N, C, *)");
    check_running_stats(op, x, running_mean, running_var, x->shape()[1]);

    Device device = x->device();
    backend::IBackend& be = backend::Dispatcher::for_device(device);
    const Shape& x_shape = x->shape();

    // Product of reduced-dim sizes — needed for the (n/(n-1)) Bessel factor.
    const int n_total = reduced_count(op, x_shape, reduce_axes);

    Dtype buf_dt = running_mean->dtype();

    // Cast x → buf_dt if needed.  ``astype`` is a no-op when dtypes match.
    Storage x_eff = x->storage();
    Dtype x_dt = x->dtype();
    if (x_dt != buf_dt) {
        x_eff = be.astype(x_eff, x_shape, x_dt, buf_dt);
    }

    backend::ReduceOpts opts;
    opts.axes = reduce_axes;
    opts.keepdims = false;
    Shape stat_shape{static_cast<std::int64_t>(x_shape[1])};

    Storage batch_mean = be.reduce_mean(x_eff, x_shape, opts, buf_dt);
    Storage batch_var = be.variance(x_eff, x_shape, opts, buf_dt);

    const double eff = momentum;
    const double unbiased_factor =
        (unbiased_var && n_total > 1) ? double(n_total) / double(n_total - 1) : 1.0;

    // new_rm = (1-eff) * running_mean + eff * batch_mean
    Storage rm_scaled =
        be.mul_scalar(running_mean->storage(), running_mean->shape(), buf_dt, 1.0 - eff);
    Storage bm_scaled = be.mul_scalar(batch_mean, stat_shape, buf_dt, eff);
    Storage new_rm = be.add(rm_scaled, bm_scaled, stat_shape, buf_dt);

    // new_rv = (1-eff) * running_var + (eff * unbiased_factor) * batch_var
    Storage rv_scaled =
        be.mul_scalar(running_var->storage(), running_var->shape(), buf_dt, 1.0 - eff);
    Storage bv_scaled = be.mul_scalar(batch_var, stat_shape, buf_dt, eff * unbiased_factor);
    Storage new_rv = be.add(rv_scaled, bv_scaled, stat_shape, buf_dt);

    // Into the buffers' own elements: the TensorImpl identity is preserved
    // and every view of the buffers sees the update (take_running_stat).
    take_running_stats(running_mean, running_var, std::move(new_rm), std::move(new_rv));

    // GPU: break the MLX lazy expression chain so subsequent training steps
    // don't carry an ever-growing parent graph through running_mean's lineage.
    // num_batches_tracked is left to the Python caller (its dtype is I64
    // which not all backends support via add_scalar; the increment is one
    // cheap op in Python).
    if (device == Device::GPU)
        flush_running_stats(running_mean, running_var);
}

}  // namespace lucid
