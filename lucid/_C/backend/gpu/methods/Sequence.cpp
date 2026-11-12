// lucid/_C/backend/gpu/methods/Sequence.cpp
// Out-of-line GpuBackend method definitions for the Sequence family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

bool GpuBackend::lstm_metal_supported(const LstmOpts& opts,
                                      const std::vector<Storage>& weights,
                                      Dtype dt) const {
    const int min_w = (opts.proj_size > 0) ? 5 : 4;
    return (dt == Dtype::F32 || is_half_float(dt)) && !opts.bidirectional && opts.num_layers == 1 &&
           static_cast<int>(weights.size()) >= min_w;
}

std::vector<Storage> GpuBackend::lstm_forward_train(const Storage& input,
                                                    const Storage& h0,
                                                    const Storage& c0,
                                                    const std::vector<Storage>& weights,
                                                    const LstmOpts& opts,
                                                    Dtype dt) {
    if (!lstm_metal_supported(opts, weights, dt))
        return IBackend::lstm_forward_train(input, h0, c0, weights, opts, dt);
    if (!is_half_float(dt))
        return gpu::lstm::forward_train(input, h0, c0, weights, opts, dt);
    using gpu::lstm::cast;
    auto res =
        gpu::lstm::forward_train(cast(input, Dtype::F32), cast(h0, Dtype::F32),
                                 cast(c0, Dtype::F32), cast(weights, Dtype::F32), opts, Dtype::F32);
    for (std::size_t i = 0; i < 3; ++i)
        res[i] = cast(res[i], dt);
    return res;
}

std::vector<Storage> GpuBackend::lstm_forward(const Storage& input,
                                              const Storage& h0,
                                              const Storage& c0,
                                              const std::vector<Storage>& weights,
                                              const LstmOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
    if (!lstm_metal_supported(opts, weights, dt))
        return IBackend::lstm_forward(input, h0, c0, weights, opts, out_shape, dt);
    // Inference path delegates to the train path and discards the saved
    // gate / cell trajectories.  The MLX graph is lazy so the unused
    // ``gates_all`` / ``cells_all`` are pruned before evaluation.
    auto v = lstm_forward_train(input, h0, c0, weights, opts, dt);
    return {std::move(v[0]), std::move(v[1]), std::move(v[2])};
}

std::vector<Storage> GpuBackend::lstm_backward(const Storage& grad_output,
                                               const Storage& grad_hn,
                                               const Storage& grad_cn,
                                               const Storage& input,
                                               const Storage& h0,
                                               const std::vector<Storage>& weights,
                                               const Storage& gates_all,
                                               const Storage& cells_all,
                                               const LstmOpts& opts,
                                               Dtype dt) {
    if (!lstm_metal_supported(opts, weights, dt))
        return IBackend::lstm_backward(grad_output, grad_hn, grad_cn, input, h0, weights, gates_all,
                                       cells_all, opts, dt);
    if (!is_half_float(dt))
        return gpu::lstm::backward(grad_output, grad_hn, grad_cn, input, h0, weights, gates_all,
                                   cells_all, opts, dt);
    using gpu::lstm::cast;
    return cast(gpu::lstm::backward(cast(grad_output, Dtype::F32), cast(grad_hn, Dtype::F32),
                                    cast(grad_cn, Dtype::F32), cast(input, Dtype::F32),
                                    cast(h0, Dtype::F32), cast(weights, Dtype::F32), gates_all,
                                    cells_all, opts, Dtype::F32),
                dt);
}

Storage GpuBackend::embedding_bag_forward(const Storage& weight,
                                          const Storage& indices,
                                          const Storage& offsets,
                                          const Shape& weight_shape,
                                          const Shape& indices_shape,
                                          int mode,
                                          int padding_idx,
                                          bool include_last_offset,
                                          Dtype dt) {
    const auto& gw = std::get<GpuStorage>(weight);
    const auto& gi = std::get<GpuStorage>(indices);
    const auto& go = std::get<GpuStorage>(offsets);

    // Segment-membership approach (the earlier scatter version misused MLX's
    // update-shape rule and threw).  Build a one-hot (B, n_idx) of which bag
    // each gathered row belongs to, then reduce: matmul for sum/mean, a
    // masked max for max.  Stays on the GPU; correct for all modes.
    namespace mx = ::mlx::core;
    // Under include_last_offset the final offset is a sentinel that ends
    // the last bag: one bag fewer than offsets, and an index at or past
    // the sentinel belongs to no bag.
    const int n_offsets = static_cast<int>(go.arr->shape(0));
    const int B = n_offsets - (include_last_offset ? 1 : 0);
    const int D = static_cast<int>(weight_shape[1]);
    const int n_idx = static_cast<int>(indices_shape[0]);
    const auto mdt = gpu::to_mlx_dtype(dt);
    if (B <= 0 || n_idx == 0)
        return Storage{gpu::wrap_mlx_array(mx::zeros({std::max(B, 0), D}, mdt), dt)};

    // (n_idx, D).  An index outside the table reads a row of NaN, so its
    // bag answers NaN (policy B, ``gpu_axis_index``).  The index is
    // checked at its own width: it was narrowed to int32 first, and an
    // index of 2^32 + 1 read row 1.
    auto emb = gpu_take(*gw.arr, *gi.arr, 0, IndexSite::row("embedding_bag"));

    // seg[k] = (#offsets <= k) - 1  — the bag position k falls in.  The
    // sentinel counts here, so a position past it lands on bag B.
    auto off_i = mx::astype(*go.arr, mx::int32);
    auto kcol = mx::reshape(mx::arange(0, n_idx, 1, mx::int32), {n_idx, 1});
    auto orow = mx::reshape(off_i, {1, n_offsets});
    auto ge = mx::astype(mx::greater_equal(kcol, orow), mx::int32);  // (n_idx, n_offsets)
    auto seg = mx::subtract(mx::sum(ge, std::vector<int>{1}, false),
                            mx::array(static_cast<std::int32_t>(1), mx::int32));

    // valid[k] (float): 1 unless the index equals padding_idx or k lies
    // outside every bag: past the sentinel, or before the first offset
    // (seg -1), which the CPU kernel also leaves out.  MLX would read
    // seg -1 as the last bag.
    auto in_bag =
        mx::logical_and(mx::greater_equal(seg, mx::array(static_cast<std::int32_t>(0), mx::int32)),
                        mx::less(seg, mx::array(static_cast<std::int32_t>(B), mx::int32)));
    mx::array valid = mx::astype(in_bag, mdt);
    if (padding_idx >= 0) {
        valid = mx::multiply(
            valid, mx::astype(mx::not_equal(mx::astype(*gi.arr, mx::int64),
                                            mx::array(std::int64_t{padding_idx}, mx::int64)),
                              mdt));
    }
    // Rows outside every bag are zeroed through ``valid`` below; clamp
    // their segment so the scatter stays in bounds.
    seg = mx::clip(seg, mx::array(static_cast<std::int32_t>(0), mx::int32),
                   mx::array(static_cast<std::int32_t>(B - 1), mx::int32));

    // one-hot membership (B, n_idx): oh[b,k] = (seg[k]==b) * valid[k]
    auto brow = mx::reshape(mx::arange(0, B, 1, mx::int32), {B, 1});
    auto srow = mx::reshape(seg, {1, n_idx});
    auto oh = mx::multiply(mx::astype(mx::equal(brow, srow), mdt),
                           mx::reshape(valid, {1, n_idx}));  // (B, n_idx)

    mx::array out = mx::zeros({B, D}, mdt);
    if (mode == 2) {  // max
        auto keep = mx::broadcast_to(
            mx::reshape(mx::greater(oh, mx::array(0.0f, mdt)), {B, n_idx, 1}), {B, n_idx, D});
        auto emb3 = mx::broadcast_to(mx::reshape(emb, {1, n_idx, D}), {B, n_idx, D});
        auto neg = mx::full({B, n_idx, D}, -std::numeric_limits<float>::infinity(), mdt);
        out = mx::max(mx::where(keep, emb3, neg), std::vector<int>{1}, false);
        // empty bag → -inf → 0
        auto sizes = mx::reshape(mx::sum(oh, std::vector<int>{1}, false), {B, 1});
        auto empty = mx::broadcast_to(mx::less_equal(sizes, mx::array(0.0f, mdt)), {B, D});
        out = mx::where(empty, mx::zeros({B, D}, mdt), out);
    } else {  // sum / mean
        // Scattered, not multiplied.
        //
        // The sum used to be ``matmul(oh, emb)`` — a one-hot matrix
        // against the gathered rows — which is a masked reduction
        // written as arithmetic, and the mask is zero for every row
        // outside the bag.  ``0 * inf`` is NaN, so one infinite entry
        // anywhere in the table poisoned every bag that did not contain
        // it: the CPU and the reference both answered ``[nan, inf]``
        // for a row summing NaN and infinity, and Metal answered
        // ``[nan, nan]``.
        //
        // A scatter-add adds each row exactly once into its own bag and
        // never touches the others, so a value outside the bag cannot
        // reach the result at all.  Rows excluded by ``padding_idx`` are
        // *selected* to zero first, for the same reason.
        out = embedding_bag_sum_or_mean(emb, seg, valid, oh, B, D, n_idx, mode);
    }
    return Storage{gpu::wrap_mlx_array(mx::contiguous(out), dt)};
}

Storage GpuBackend::embedding_bag_backward(const Storage& grad_out,
                                           const Storage& weight,
                                           const Storage& indices,
                                           const Storage& offsets,
                                           const Shape& weight_shape,
                                           const Shape& indices_shape,
                                           int mode,
                                           int padding_idx,
                                           bool include_last_offset,
                                           Dtype dt) {
    // Through the CPU kernel, which is the same choice ``scatter``
    // makes a few hundred lines up.  The ``max`` mode has to recompute
    // an argmax per column and then scatter one row per column, and
    // expressing that in MLX is more machinery than the op is worth —
    // an EmbeddingBag backward is bounded by the bag contents, not by
    // the vocabulary.  The device axis compares the two paths, so if
    // this ever diverges the audit says so.
    const std::int64_t n_off = offsets_count(offsets);
    const std::int64_t n_bags = n_off - (include_last_offset ? 1 : 0);
    auto dl = [](const Storage& s, const Shape& sh) {
        return Storage{gpu::download_gpu_to_cpu(std::get<GpuStorage>(s), sh)};
    };
    Storage grad_cpu = dl(grad_out, {n_bags, weight_shape[1]});
    Storage weight_cpu = mode == 2 ? dl(weight, weight_shape) : Storage{};
    Storage out_cpu = backend::Dispatcher::for_device(Device::CPU)
                          .embedding_bag_backward(grad_cpu, weight_cpu, dl(indices, indices_shape),
                                                  dl(offsets, {n_off}), weight_shape, indices_shape,
                                                  mode, padding_idx, include_last_offset, dt);
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(out_cpu), weight_shape)};
}

std::int64_t GpuBackend::offsets_count(const Storage& offsets) {
    return static_cast<std::int64_t>(std::get<GpuStorage>(offsets).arr->shape(0));
}

std::vector<Storage> GpuBackend::sdpa_forward(const Storage& q,
                                              const Storage& k,
                                              const Storage& v,
                                              const Storage* attn_mask,
                                              const Shape& q_shape,
                                              const Shape& k_shape,
                                              const Shape& v_shape,
                                              Dtype mask_dtype,
                                              std::size_t mask_numel,
                                              double scale,
                                              bool is_causal,
                                              bool need_weights,
                                              Dtype dt) {
    (void)mask_numel;

    const auto& gQ = std::get<GpuStorage>(q);
    const auto& gK = std::get<GpuStorage>(k);
    const auto& gV = std::get<GpuStorage>(v);

    const bool has_custom_mask = (attn_mask != nullptr);
    const bool custom_mask_is_additive = has_custom_mask && (mask_dtype != Dtype::Bool);
    const float scale_f = static_cast<float>(scale);

    if (!need_weights) {
        // Memory-efficient fused path.  Unlike the manual softmax below, the
        // fused kernel correctly applies a query-broadcast additive mask
        // (query dim == 1, e.g. a (B,1,1,Lk) padding mask).  Precedence
        // matches the reference eager semantics (and the backward): an
        // additive float mask wins over is_causal; else causal; else a bool
        // keep-mask.
        std::string mask_mode;
        std::optional<::mlx::core::array> mlx_mask;
        if (custom_mask_is_additive) {
            mlx_mask = *std::get<GpuStorage>(*attn_mask).arr;
        } else if (is_causal) {
            mask_mode = "causal";
        } else if (has_custom_mask) {  // bool keep-mask
            mlx_mask = *std::get<GpuStorage>(*attn_mask).arr;
        }

        auto out = gpu::sdpa_fused(*gQ.arr, *gK.arr, *gV.arr, scale_f, mask_mode, mlx_mask);

        auto dummy_w = ::mlx::core::zeros({1}, gpu::to_mlx_dtype(dt));
        return {Storage{gpu::wrap_mlx_array(std::move(dummy_w), dt)},
                Storage{gpu::wrap_mlx_array(std::move(out), dt)}};
    }

    // ── Weights-materializing path (``scaled_dot_product_attention_with_weights``):
    // the fused kernel exposes no dense W, so build W = softmax(s·QKᵀ + M) by
    // hand.  Inherently O(T²); used only when the caller asks for the weights.
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    auto scale_arr = ::mlx::core::astype(::mlx::core::array(scale_f), mlx_dt);
    auto k_t = ::mlx::core::swapaxes(*gK.arr, -2, -1);
    auto scores = ::mlx::core::multiply(::mlx::core::matmul(*gQ.arr, k_t), scale_arr);

    auto neg_inf =
        ::mlx::core::astype(::mlx::core::array(-std::numeric_limits<float>::infinity()), mlx_dt);
    if (custom_mask_is_additive) {
        scores = ::mlx::core::add(scores, *std::get<GpuStorage>(*attn_mask).arr);
    } else if (is_causal) {
        // Bottom-right aligned causal mask (lower-triangular when Lq == Lk):
        // keep key j for query i iff j <= i + (Lk - Lq).
        const int Lq = static_cast<int>(q_shape[q_shape.size() - 2]);
        const int Lk = static_cast<int>(k_shape[k_shape.size() - 2]);
        auto rows =
            ::mlx::core::reshape(::mlx::core::arange(static_cast<double>(Lq), mlx_dt), {Lq, 1});
        auto cols =
            ::mlx::core::reshape(::mlx::core::arange(static_cast<double>(Lk), mlx_dt), {1, Lk});
        auto offset = ::mlx::core::astype(::mlx::core::array(static_cast<double>(Lk - Lq)), mlx_dt);
        auto keep = ::mlx::core::less_equal(cols, ::mlx::core::add(rows, offset));
        scores = ::mlx::core::where(keep, scores, neg_inf);
    } else if (has_custom_mask) {  // bool keep-mask (true = attend)
        scores = ::mlx::core::where(*std::get<GpuStorage>(*attn_mask).arr, scores, neg_inf);
    }

    auto weights = ::mlx::core::softmax(scores, std::vector<int>{-1}, true);
    auto output = ::mlx::core::matmul(weights, *gV.arr);

    return {Storage{gpu::wrap_mlx_array(std::move(weights), dt)},
            Storage{gpu::wrap_mlx_array(std::move(output), dt)}};
}

std::vector<Storage> GpuBackend::sdpa_backward(const Storage& grad_out,
                                               const Storage& q,
                                               const Storage& k,
                                               const Storage& v,
                                               const Storage& saved_weights,
                                               const Storage* attn_mask,
                                               const Shape&,
                                               const Shape&,
                                               const Shape&,
                                               Dtype mask_dtype,
                                               double scale,
                                               bool is_causal,
                                               Dtype dt) {
    const auto& gQ = std::get<GpuStorage>(q);
    const auto& gK = std::get<GpuStorage>(k);
    const auto& gV = std::get<GpuStorage>(v);
    const auto& gG = std::get<GpuStorage>(grad_out);
    const auto& gW = std::get<GpuStorage>(saved_weights);
    const float scale_f = static_cast<float>(scale);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);

    if (gW.arr->ndim() > 1) {
        // ── Materialized forward saved the real (masked) weight matrix W.
        // The backward is then mask-agnostic — W already encodes the mask —
        // so use the closed-form softmax-attention VJP directly on W.  This
        // avoids the fused ``mask_arr`` kernel entirely (which miscomputes on
        // lazy inputs), yet stays correct for additive/bool masks.
        auto scale_arr = ::mlx::core::astype(::mlx::core::array(scale_f), mlx_dt);
        ::mlx::core::array W = *gW.arr;

        auto W_t = ::mlx::core::swapaxes(W, -2, -1);
        auto dV = ::mlx::core::matmul(W_t, *gG.arr);

        auto V_t = ::mlx::core::swapaxes(*gV.arr, -2, -1);
        auto dW = ::mlx::core::matmul(*gG.arr, V_t);
        auto sum_wdw = ::mlx::core::sum(::mlx::core::multiply(W, dW), std::vector<int>{-1}, true);
        auto dscores = ::mlx::core::multiply(W, ::mlx::core::subtract(dW, sum_wdw));

        auto dQ = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores, *gK.arr));
        auto dscores_t = ::mlx::core::swapaxes(dscores, -2, -1);
        auto dK = ::mlx::core::multiply(scale_arr, ::mlx::core::matmul(dscores_t, *gQ.arr));

        // S = QK^T·s + M ⇒ dL/dM == dL/dS, which ``dscores`` already is.
        // The slot is always emitted so the vector matches the node's edge
        // count; the engine drops it when the mask edge is null.
        auto dM = (attn_mask != nullptr && mask_dtype != Dtype::Bool)
                      ? std::move(dscores)
                      : ::mlx::core::zeros({1}, mlx_dt);
        return {Storage{gpu::wrap_mlx_array(std::move(dQ), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dK), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dV), dt)},
                Storage{gpu::wrap_mlx_array(std::move(dM), dt)}};
    }

    // ── Fused forward saved a {1} placeholder (no-mask / causal / masked).
    // Differentiate the fused kernel via VJP, replaying the same effective
    // mask routing the forward used: an additive float mask wins over
    // is_causal; else causal; else a bool keep-mask.  The saved mask is the
    // query-expanded one recorded at forward, so it applies correctly.
    const bool has_custom_mask = (attn_mask != nullptr);
    const bool custom_mask_is_additive = has_custom_mask && (mask_dtype != Dtype::Bool);
    std::string mask_mode;
    std::optional<::mlx::core::array> mlx_mask;
    if (custom_mask_is_additive) {
        mlx_mask = *std::get<GpuStorage>(*attn_mask).arr;
    } else if (is_causal) {
        mask_mode = "causal";
    } else if (has_custom_mask) {
        mlx_mask = *std::get<GpuStorage>(*attn_mask).arr;
    }
    // An additive mask is differentiable, so it has to enter the VJP as an
    // *input* rather than a closure capture — captured, MLX has nothing to
    // return a cotangent for and the bias table silently never trains.
    if (custom_mask_is_additive) {
        auto fwd_m =
            [scale_f](
                const std::vector<::mlx::core::array>& in) -> std::vector<::mlx::core::array> {
            return {::mlx::core::fast::scaled_dot_product_attention(in[0], in[1], in[2], scale_f,
                                                                    std::string{}, in[3])};
        };
        auto vjp_m = ::mlx::core::vjp(fwd_m, {*gQ.arr, *gK.arr, *gV.arr, *mlx_mask}, {*gG.arr});
        std::vector<::mlx::core::array>& gm = vjp_m.second;  // {dQ, dK, dV, dM}

        return {Storage{gpu::wrap_mlx_array(std::move(gm[0]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(gm[1]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(gm[2]), dt)},
                Storage{gpu::wrap_mlx_array(std::move(gm[3]), dt)}};
    }

    auto fwd = [scale_f, mask_mode, mlx_mask](
                   const std::vector<::mlx::core::array>& in) -> std::vector<::mlx::core::array> {
        return {::mlx::core::fast::scaled_dot_product_attention(in[0], in[1], in[2], scale_f,
                                                                mask_mode, mlx_mask)};
    };

    auto vjp_res = ::mlx::core::vjp(fwd, {*gQ.arr, *gK.arr, *gV.arr}, {*gG.arr});
    std::vector<::mlx::core::array>& grads = vjp_res.second;  // {dQ, dK, dV}

    return {Storage{gpu::wrap_mlx_array(std::move(grads[0]), dt)},
            Storage{gpu::wrap_mlx_array(std::move(grads[1]), dt)},
            Storage{gpu::wrap_mlx_array(std::move(grads[2]), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::zeros({1}, mlx_dt), dt)}};
}

Storage GpuBackend::embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) {
    (void)weight_shape;
    (void)indices_shape;
    (void)out_shape;
    const auto& gw = std::get<GpuStorage>(weight);
    const auto& gi = std::get<GpuStorage>(indices);
    // padding_idx does not mask the lookup.  The convention it comes
    // from zeroes that row at initialisation and stops its gradient,
    // so the row holds whatever training left there — and a loaded
    // checkpoint carries a real vector in it.  BERT's [PAD] is the
    // usual case: masking here returned zeros where the reference
    // returns the trained embedding, and bert_base drifted 4.14 from
    // the implementation its weights came from.  The gradient block
    // lives in embedding_backward, which is the whole of the
    // contract.
    (void)padding_idx;
    // An index outside the table, negative ones included, reads a row of
    // NaN (policy B, ``gpu_axis_index``); the CPU raises IndexError.
    auto out = gpu_take(*gw.arr, *gi.arr, 0, IndexSite::row("embedding"));
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) {
    // dW[idx[i], :] += grad_out[i, :], summed across duplicates.
    // The natural primitive is reference-framework-style scatter-add
    // along axis 0.  Lucid 3.3.0 used an onehot-matmul trick that
    // allocates a `(M_total × N)` boolean tensor — 820 MB for GPT-2
    // vocab — and a full matmul.  Replacing with
    // `mlx::core::scatter_add_axis` is asymptotically optimal and
    // matches the engine's existing
    // [[engine-mlx-scatter-axis-vs-multiaxis]] convention (reference
    // semantics → `scatter_add_axis`, not multi-axis `scatter_add`).
    const std::int64_t N = weight_shape[0];
    const std::int64_t D = weight_shape[1];
    std::size_t M_total = 1;
    for (auto d : indices_shape)
        M_total *= static_cast<std::size_t>(d);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    const auto& gg = std::get<GpuStorage>(grad_out);
    const auto& gi = std::get<GpuStorage>(indices);

    auto base = ::mlx::core::zeros({static_cast<int>(N), static_cast<int>(D)}, mlx_dt);
    if (N == 0)
        return Storage{gpu::wrap_mlx_array(std::move(base), dt)};
    auto grad_flat =
        ::mlx::core::reshape(*gg.arr, {static_cast<int>(M_total), static_cast<int>(D)});
    // A row contributes when its index names a row of the table and is
    // not the padding slot.  An index outside the table read NaN in the
    // forward and sends nothing back (policy B, ``gpu_axis_index``).
    // The rest are *selected* away, not multiplied by 0: ``0 * inf`` is
    // NaN, and an infinite gradient at a padding position reached the
    // padding row as NaN.
    auto ix = gpu_axis_index(::mlx::core::reshape(*gi.arr, {static_cast<int>(M_total)}), N,
                             IndexSite::row("embedding_backward"));
    auto keep = ix.in_range;
    if (padding_idx >= 0)
        keep = ::mlx::core::logical_and(
            keep,
            ::mlx::core::not_equal(ix.safe, ::mlx::core::array(padding_idx, ix.safe.dtype())));
    grad_flat = ::mlx::core::where(::mlx::core::reshape(keep, {static_cast<int>(M_total), 1}),
                                   grad_flat, scatter_identity(ScatterReduce::Add, mlx_dt));
    // scatter_add_axis requires indices and updates to share shape.
    // Broadcast the index column across D so each row of grad_flat
    // lands on the correct vocab row of dW.
    auto idx_bcast =
        ::mlx::core::broadcast_to(::mlx::core::reshape(ix.safe, {static_cast<int>(M_total), 1}),
                                  {static_cast<int>(M_total), static_cast<int>(D)});
    auto dW = scatter_add_axis_widened(base, idx_bcast, grad_flat, /*axis=*/0);
    return Storage{gpu::wrap_mlx_array(std::move(dW), dt)};
}

Storage
GpuBackend::sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) {
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    const int L = static_cast<int>(seq_len);
    const int D = static_cast<int>(embed_dim);
    const int Dh = D / 2;
    Shape out_shape{seq_len, embed_dim};
    if (L * D == 0) {
        auto z = ::mlx::core::zeros(gpu::to_mlx_shape(out_shape), mlx_dt);
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(z), dt)};
    }
    auto pos =
        ::mlx::core::reshape(::mlx::core::astype(::mlx::core::arange(0, L, 1), mlx_dt), {L, 1});
    auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), mlx_dt);
    const float inv_d = static_cast<float>(-std::log(10000.0) / static_cast<double>(D));
    auto two_inv_d = ::mlx::core::astype(::mlx::core::array(2.0f * inv_d), mlx_dt);
    auto theta = ::mlx::core::exp(::mlx::core::multiply(two_inv_d, k_arr));
    auto theta_row = ::mlx::core::reshape(theta, {1, Dh});
    auto angle = ::mlx::core::matmul(pos, theta_row);
    auto sin_t = ::mlx::core::sin(angle);
    auto cos_t = ::mlx::core::cos(angle);
    auto sin_e = ::mlx::core::expand_dims(sin_t, -1);
    auto cos_e = ::mlx::core::expand_dims(cos_t, -1);
    auto stacked = ::mlx::core::concatenate(std::vector<::mlx::core::array>{sin_e, cos_e}, -1);
    auto out = ::mlx::core::reshape(stacked, {L, 2 * Dh});
    if (D % 2 != 0) {
        auto pad_col = ::mlx::core::zeros({L, 1}, mlx_dt);
        out = ::mlx::core::concatenate(std::vector<::mlx::core::array>{out, pad_col}, -1);
    }
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

std::vector<Storage> GpuBackend::rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) {
    (void)pos_dtype;
    const auto& gx = std::get<GpuStorage>(x);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    const std::size_t ndim = x_shape.size();
    const int L = static_cast<int>(x_shape[ndim - 2]);
    const int D = static_cast<int>(x_shape.back());
    const int Dh = D / 2;
    auto in_shape = gpu::to_mlx_shape(x_shape);
    ::mlx::core::array pos_arr{0};
    if (position_ids) {
        const auto& gp = std::get<GpuStorage>(*position_ids);
        pos_arr = ::mlx::core::astype(*gp.arr, mlx_dt);
    } else {
        pos_arr = ::mlx::core::astype(::mlx::core::arange(0, L, 1), mlx_dt);
    }
    pos_arr = ::mlx::core::reshape(pos_arr, {L, 1});
    const float coef = static_cast<float>(-2.0 * std::log(10000.0) / static_cast<double>(D));
    auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), mlx_dt);
    auto theta = ::mlx::core::exp(
        ::mlx::core::multiply(::mlx::core::astype(::mlx::core::array(coef), mlx_dt), k_arr));
    auto angle = ::mlx::core::matmul(pos_arr, ::mlx::core::reshape(theta, {1, Dh}));
    auto cos_t = ::mlx::core::cos(angle);
    auto sin_t = ::mlx::core::sin(angle);
    ::mlx::core::Shape cos_b_shape;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        cos_b_shape.push_back(1);
    cos_b_shape.push_back(L);
    cos_b_shape.push_back(Dh);
    auto cos_b = ::mlx::core::reshape(cos_t, cos_b_shape);
    auto sin_b = ::mlx::core::reshape(sin_t, cos_b_shape);
    ::mlx::core::array out_arr{0};
    if (interleaved) {
        ::mlx::core::Shape x_shape_p = in_shape;
        x_shape_p.pop_back();
        x_shape_p.push_back(Dh);
        x_shape_p.push_back(2);
        auto x_re = ::mlx::core::reshape(*gx.arr, x_shape_p);
        auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
        auto xe = ::mlx::core::take(x_re, idx0, -1);
        auto xo = ::mlx::core::take(x_re, idx1, -1);
        auto out_e = ::mlx::core::subtract(::mlx::core::multiply(xe, cos_b),
                                           ::mlx::core::multiply(xo, sin_b));
        auto out_o =
            ::mlx::core::add(::mlx::core::multiply(xo, cos_b), ::mlx::core::multiply(xe, sin_b));
        auto oe = ::mlx::core::expand_dims(out_e, -1);
        auto oo = ::mlx::core::expand_dims(out_o, -1);
        out_arr = ::mlx::core::reshape(
            ::mlx::core::concatenate(std::vector<::mlx::core::array>{oe, oo}, -1), in_shape);
    } else {
        auto i1 = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), ::mlx::core::int64);
        auto i2 = ::mlx::core::astype(::mlx::core::arange(Dh, D, 1), ::mlx::core::int64);
        auto xa = ::mlx::core::take(*gx.arr, i1, -1);
        auto xb = ::mlx::core::take(*gx.arr, i2, -1);
        auto oa = ::mlx::core::subtract(::mlx::core::multiply(xa, cos_b),
                                        ::mlx::core::multiply(xb, sin_b));
        auto ob =
            ::mlx::core::add(::mlx::core::multiply(xb, cos_b), ::mlx::core::multiply(xa, sin_b));
        out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{oa, ob}, -1);
    }
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out_arr), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(cos_t), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(sin_t), dt)}};
}

Storage GpuBackend::rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad_out);
    const auto& cs = std::get<GpuStorage>(saved_cos);
    const auto& ss = std::get<GpuStorage>(saved_sin);
    const std::size_t ndim = x_shape.size();
    const int L = static_cast<int>(x_shape[ndim - 2]);
    const int D = static_cast<int>(x_shape.back());
    const int Dh = D / 2;
    auto in_shape = gpu::to_mlx_shape(x_shape);
    ::mlx::core::Shape cos_b_shape;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        cos_b_shape.push_back(1);
    cos_b_shape.push_back(L);
    cos_b_shape.push_back(Dh);
    auto cos_b = ::mlx::core::reshape(*cs.arr, cos_b_shape);
    auto sin_b = ::mlx::core::reshape(*ss.arr, cos_b_shape);
    ::mlx::core::array out_arr{0};
    if (interleaved) {
        ::mlx::core::Shape g_shape = in_shape;
        g_shape.pop_back();
        g_shape.push_back(Dh);
        g_shape.push_back(2);
        auto g_re = ::mlx::core::reshape(*gg.arr, g_shape);
        auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
        auto ge = ::mlx::core::take(g_re, idx0, -1);
        auto go = ::mlx::core::take(g_re, idx1, -1);
        auto dxe =
            ::mlx::core::add(::mlx::core::multiply(cos_b, ge), ::mlx::core::multiply(sin_b, go));
        auto dxo = ::mlx::core::subtract(::mlx::core::multiply(cos_b, go),
                                         ::mlx::core::multiply(sin_b, ge));
        auto de = ::mlx::core::expand_dims(dxe, -1);
        auto do_ = ::mlx::core::expand_dims(dxo, -1);
        out_arr = ::mlx::core::reshape(
            ::mlx::core::concatenate(std::vector<::mlx::core::array>{de, do_}, -1), in_shape);
    } else {
        auto i1 = ::mlx::core::astype(::mlx::core::arange(0, Dh, 1), ::mlx::core::int64);
        auto i2 = ::mlx::core::astype(::mlx::core::arange(Dh, D, 1), ::mlx::core::int64);
        auto ga = ::mlx::core::take(*gg.arr, i1, -1);
        auto gb = ::mlx::core::take(*gg.arr, i2, -1);
        auto dxa =
            ::mlx::core::add(::mlx::core::multiply(cos_b, ga), ::mlx::core::multiply(sin_b, gb));
        auto dxb = ::mlx::core::subtract(::mlx::core::multiply(cos_b, gb),
                                         ::mlx::core::multiply(sin_b, ga));
        out_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{dxa, dxb}, -1);
    }
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out_arr), dt)};
}

}  // namespace backend
}  // namespace lucid
