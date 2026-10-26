// lucid/_C/ops/ufunc/Var.cpp
//
// Biased variance reduction (denominator = N, not N-1) along arbitrary axes.
//
// Forward: delegates to IBackend::variance, which uses Accelerate on CPU and
// MLX on GPU.
//
// Backward formula (biased variance):
//   dL/dx_i = (2 / N) * (x_i - mean(x)) * broadcast(dL/dy)
// where N is the product of the reduced dimension sizes and mean(x) is
// precomputed during the forward pass and saved on the backward node to avoid
// a second mean reduction at backward time.
//
// VarBackward is defined in an anonymous namespace because its interface is not
// part of the public API; var_op is the only external entry point.

#include "Var.h"

#include <vector>

#include "../../autograd/AccumulateGrad.h"
#include "../../autograd/AutogradNode.h"
#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../backend/Dispatcher.h"
#include "../../compile/Tracer.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/OpSchema.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/NaryKernel.h"
#include "../bfunc/Mul.h"
#include "../bfunc/Sub.h"
#include "../bfunc/_BinaryOp.h"
#include "../gfunc/Gfunc.h"
#include "../utils/Layout.h"
#include "../utils/Promote.h"
#include "../utils/View.h"
#include "Reductions.h"
#include "_Detail.h"

namespace lucid {

namespace {

using ufunc_detail::fresh;

// Private backward node for variance.
//
// Saved state:
//   input_shape_ — original input shape, needed to broadcast the gradient.
//   axes_        — normalised reduction axes.
//   keepdims_    — whether collapsed axes were kept as size-1 dims.
//   count_       — number of elements reduced (N); guards against empty axes.
//   saved_mean_  — mean(x) already broadcast to the input shape, so apply()
//                  can subtract it directly without an extra broadcast call.
class VarBackward : public AutogradNode<VarBackward, 1> {
public:
    static constexpr bool kKeepsOwnValues = true;
    static const OpSchema schema_v1;

    Shape input_shape_;
    std::vector<int> axes_;
    bool keepdims_;
    std::int64_t count_;
    Storage saved_mean_;

    // dL/dx = (2/N) * (x - mean) * broadcast_back(dL/dy).
    std::vector<Storage> apply(Storage grad_out) override {
        const std::size_t n = shape_numel(input_shape_);
        if (is_half_float(dtype_)) {
            auto& be = backend::Dispatcher::for_device(device_);
            grad_out = be.astype(grad_out, out_shape_, Dtype::F32, dtype_);
        }
        // (x - mean): mean is already broadcast to input shape from forward.
        Storage centered = subtract_storages(saved_inputs_[0], saved_mean_, n, dtype_, device_);
        // Expand the reduced gradient back to the full input shape.
        Storage g_b = broadcast_back_for_reduce(grad_out, out_shape_, input_shape_, axes_,
                                                keepdims_, dtype_, device_);
        Storage scaled = mul_scalar_storage(g_b, 2.0 / (double)count_, n, dtype_, device_);
        Storage dx = multiply_storages(centered, scaled, n, dtype_, device_);
        return {std::move(dx)};
    }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        // dVar/dx = (2 / count) * (x - mean) * g, with the mean recomputed
        // from the input so it keeps its own dependence on x — the saved
        // mean is data only, and reusing it would drop that term from the
        // second derivative without anything looking wrong.
        const auto& x = saved_impl_inputs_[0];
        if (!x)
            ErrorBuilder("var").fail("graph-mode backward is missing its saved input");

        auto mean = mean_op(x, axes_, /*keepdims=*/true);
        auto centered = sub_op(x, mean);

        // Put the reduced gradient back onto the input's shape.
        TensorImplPtr g =
            grad_out->dtype() == x->dtype() ? grad_out : astype_op(grad_out, x->dtype());
        if (!keepdims_ && !axes_.empty()) {
            std::vector<std::int64_t> keep(input_shape_.begin(), input_shape_.end());
            for (int ax : axes_)
                keep[static_cast<std::size_t>(ax)] = 1;
            g = reshape_op(g, keep);
        }
        g = broadcast_to_op(g, input_shape_);

        auto scale = full_like_op(x, 2.0 / static_cast<double>(count_));
        return {mul_op(centered, mul_op(g, scale))};
    }

    void release_own_saved() { clear_saved(saved_mean_); }
};

// KeepInput: variance is computed in the input's own dtype under an
// autocast scope.  ``real_valued`` is separate and says the answer is
// a real number whatever went in — the variance of integers is not an
// integer, and computing it as one returned 0.
const OpSchema VarBackward::schema_v1{"var", 1, AmpPolicy::KeepInput, true, "", true};

}  // namespace

// Compute the biased variance, then build VarBackward when autograd is active.
// The mean is computed with keepdims=true so that it can be passed to
// IBackend::broadcast to expand it to the input shape in one call.
TensorImplPtr var_op(const TensorImplPtr& a, const std::vector<int>& axes_user, bool keepdims) {
    Validator::input(a, "var.a").non_null();
    // Builds its own forward, so it asks for the schema dtype the kernel
    // templates would have applied.  See promote_for_schema: without it
    // ``var`` of an integer tensor computed in that integer type and
    // answered 0 on Metal while the CPU raised.
    const TensorImplPtr x = promote_for_schema(VarBackward::schema_v1, a);
    const Dtype dt = x->dtype();
    const Dtype out_dt = is_half_float(dt) ? Dtype::F32 : dt;
    const Device device = x->device();
    const auto axes = normalize_axes(axes_user, static_cast<int>(x->shape().size()), "var");
    const Shape out_shape = reduce_output_shape(x->shape(), axes, keepdims);
    OpScopeFull scope{"var", device, out_dt, out_shape};
    {
        std::vector<std::int64_t> dims_attr(axes.begin(), axes.end());
        scope.set_attr("dims", dims_attr);
        scope.set_attr("keepdim", keepdims);
        scope.set_attr("unbiased", false);  // var_op default is biased
    }
    backend::ReduceOpts reduce_opts{axes, keepdims};
    auto& be = backend::Dispatcher::for_device(device);

    Storage out_storage = be.variance(x->storage(), x->shape(), reduce_opts, dt);
    TensorImplPtr out = fresh(std::move(out_storage), out_shape, out_dt, device);
    if (auto* trc = ::lucid::compile::current_tracer()) {
        trc->on_op_io({x}, out);
    }

    // Count the number of elements collapsed; clamp to 1 to avoid division by
    // zero for empty-axis edge cases.
    std::size_t reduced = 1;
    for (auto ax : axes)
        reduced *= static_cast<std::size_t>(x->shape()[ax]);
    if (reduced == 0)
        reduced = 1;

    if (GradMode::is_enabled() && x->requires_grad()) {
        // Compute mean with keepdims=true, then broadcast to full input shape
        // so that VarBackward::apply can compute (x - mean) without a second
        // broadcast.
        Shape mean_keepdims_shape = reduce_output_shape(x->shape(), axes, true);
        Storage mean_keepdims =
            be.reduce_mean(x->storage(), x->shape(), backend::ReduceOpts{axes, true}, dt);
        Storage mean_storage = be.broadcast(mean_keepdims, mean_keepdims_shape, x->shape(), dt);

        auto bwd = std::make_shared<VarBackward>();
        bwd->input_shape_ = x->shape();
        bwd->axes_ = axes;
        bwd->keepdims_ = keepdims;
        bwd->count_ = static_cast<std::int64_t>(reduced);
        bwd->saved_mean_ = std::move(mean_storage);
        // The input is saved by the wiring, into the slot its version guards.
        kernel::NaryKernel<VarBackward, 1>::wire_autograd(std::move(bwd), {x}, out);
    }
    return out;
}

}  // namespace lucid
