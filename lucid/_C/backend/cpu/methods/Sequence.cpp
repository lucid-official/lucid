// lucid/_C/backend/cpu/methods/Sequence.cpp
// Out-of-line CpuBackend method definitions for the Sequence family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::embedding_bag_forward(const Storage& weight,
                                          const Storage& indices,
                                          const Storage& offsets,
                                          const Shape& weight_shape,
                                          const Shape& indices_shape,
                                          int mode,
                                          int padding_idx,
                                          bool include_last_offset,
                                          Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(embedding_bag_forward(detail::as_f32(weight), indices, offsets,
                                                         weight_shape, indices_shape, mode,
                                                         padding_idx, include_last_offset,
                                                         Dtype::F32),
                                   dt);
    const auto& cw = std::get<CpuStorage>(weight);
    const auto& ci = std::get<CpuStorage>(indices);
    const auto& co = std::get<CpuStorage>(offsets);

    const std::int64_t num_emb = weight_shape[0];
    const int D = static_cast<int>(weight_shape[1]);
    constexpr const char* kOp = "embedding_bag";
    // Indices and offsets are each read at their own width.  This path
    // cast an int64 index to int, so an index of 2^32 + 1 read row 1.
    const auto read_idx = cpu::index_reader(kOp, ci);
    const auto bags = cpu::embedding_bags(kOp, co, shape_numel(indices_shape), include_last_offset);
    const int B = static_cast<int>(bags.start.size());

    std::size_t out_nb = static_cast<std::size_t>(B) * D * dtype_size(dt);
    auto out_ptr = allocate_aligned_bytes(out_nb, Device::CPU);
    std::memset(out_ptr.get(), 0, out_nb);

    const std::size_t esz = dtype_size(dt);
    const std::uint8_t* wp =
        static_cast<const std::uint8_t*>(static_cast<const void*>(cw.ptr.get()));

    auto run = [&](auto* op) {
        using T = std::remove_pointer_t<decltype(op)>;
        for (int b = 0; b < B; ++b) {
            const std::size_t s = bags.start[static_cast<std::size_t>(b)];
            const std::size_t e = bags.end[static_cast<std::size_t>(b)];
            T* row = op + b * D;
            if (mode == 2) {  // max: seed with the lowest value so an
                // all-negative bag isn't masked by the 0 memset init.
                for (int d = 0; d < D; ++d)
                    row[d] = std::numeric_limits<T>::lowest();
            }
            int count = 0;
            for (std::size_t k = s; k < e; ++k) {
                const std::int64_t emb = read_idx(k);
                if (padding_idx >= 0 && emb == padding_idx)
                    continue;
                // Refused, as the reference refuses it.  It used to be
                // skipped, so a bad index silently left its bag short.
                // Metal reads a row of NaN instead (policy B).
                cpu::checked_row(kOp, emb, num_emb);
                const T* src = reinterpret_cast<const T*>(
                    wp + static_cast<std::size_t>(emb) * static_cast<std::size_t>(D) * esz);
                if (mode == 2) {  // max
                    for (int d = 0; d < D; ++d)
                        if (src[d] > row[d])
                            row[d] = src[d];
                } else {  // sum / mean
                    for (int d = 0; d < D; ++d)
                        row[d] += src[d];
                }
                ++count;
            }
            if (mode == 2 && count == 0) {  // empty bag → 0 (match reference)
                for (int d = 0; d < D; ++d)
                    row[d] = T(0);
            }
            if (mode == 1 && count > 0) {  // mean
                for (int d = 0; d < D; ++d)
                    row[d] /= static_cast<T>(count);
            }
        }
    };

    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out_ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out_ptr.get()));
    else
        ErrorBuilder("embedding_bag").not_implemented("only F32/F64");
    return Storage{CpuStorage{out_ptr, out_nb, dt}};
}

Storage CpuBackend::embedding_bag_backward(const Storage& grad_out,
                                           const Storage& weight,
                                           const Storage& indices,
                                           const Storage& offsets,
                                           const Shape& weight_shape,
                                           const Shape& indices_shape,
                                           int mode,
                                           int padding_idx,
                                           bool include_last_offset,
                                           Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            embedding_bag_backward(detail::as_f32(grad_out), detail::as_f32(weight), indices,
                                   offsets, weight_shape, indices_shape, mode, padding_idx,
                                   include_last_offset, Dtype::F32),
            dt);

    const auto& cg = std::get<CpuStorage>(grad_out);
    const auto& cw = std::get<CpuStorage>(weight);
    const auto& ci = std::get<CpuStorage>(indices);
    const auto& co = std::get<CpuStorage>(offsets);

    const std::int64_t num_emb = weight_shape[0];
    const int D = static_cast<int>(weight_shape[1]);
    constexpr const char* kOp = "embedding_bag_backward";
    const auto read_idx = cpu::index_reader(kOp, ci);
    // The same bag boundaries the forward computed, derived by the same
    // helper so the two cannot disagree about which rows a bag owns.
    const auto bags = cpu::embedding_bags(kOp, co, shape_numel(indices_shape), include_last_offset);
    const int B = static_cast<int>(bags.start.size());
    // An index outside the table sends nothing back.  The CPU forward
    // refuses one, so on the CPU only an index changed in place since
    // then is skipped here.  Metal's backward runs through this kernel,
    // and there the skip is policy B's drop: the forward answered NaN
    // for that row, and nothing was read from the table.
    const auto usable = [&](std::int64_t emb) {
        return emb != padding_idx && emb >= 0 && emb < num_emb;
    };

    const std::size_t gw_nb = static_cast<std::size_t>(num_emb) * D * dtype_size(dt);
    auto gw_ptr = allocate_aligned_bytes(gw_nb, Device::CPU);
    std::memset(gw_ptr.get(), 0, gw_nb);

    auto run = [&](auto* tag) {
        using T = std::remove_pointer_t<decltype(tag)>;
        const T* gp = reinterpret_cast<const T*>(cg.ptr.get());
        const T* wp = reinterpret_cast<const T*>(cw.ptr.get());
        T* gwp = reinterpret_cast<T*>(gw_ptr.get());
        for (int b = 0; b < B; ++b) {
            const std::size_t s = bags.start[static_cast<std::size_t>(b)];
            const std::size_t e = bags.end[static_cast<std::size_t>(b)];
            const T* grow = gp + static_cast<std::size_t>(b) * D;

            int count = 0;
            for (std::size_t k = s; k < e; ++k)
                count += usable(read_idx(k)) ? 1 : 0;
            if (count == 0)
                continue;

            if (mode == 2) {
                // max: only the row that won each column receives
                // anything.  Recomputed rather than saved, because the
                // forward records no argmax; ties go to the first,
                // matching the forward's strict ``>``.
                for (int d = 0; d < D; ++d) {
                    std::int64_t best = -1;
                    T best_val = std::numeric_limits<T>::lowest();
                    for (std::size_t k = s; k < e; ++k) {
                        const std::int64_t emb = read_idx(k);
                        if (!usable(emb))
                            continue;
                        const T v = wp[static_cast<std::size_t>(emb) * D + d];
                        if (best < 0 || v > best_val) {
                            best = emb;
                            best_val = v;
                        }
                    }
                    if (best >= 0)
                        gwp[static_cast<std::size_t>(best) * D + d] += grow[d];
                }
                continue;
            }

            const T scale = mode == 1 ? T(1) / static_cast<T>(count) : T(1);
            for (std::size_t k = s; k < e; ++k) {
                const std::int64_t emb = read_idx(k);
                if (!usable(emb))
                    continue;
                T* dst = gwp + static_cast<std::size_t>(emb) * D;
                for (int d = 0; d < D; ++d)
                    dst[d] += grow[d] * scale;
            }
        }
    };

    if (dt == Dtype::F32)
        run(static_cast<float*>(nullptr));
    else if (dt == Dtype::F64)
        run(static_cast<double*>(nullptr));
    else
        ErrorBuilder("embedding_bag_backward").not_implemented("only F32/F64");
    return Storage{CpuStorage{gw_ptr, gw_nb, dt}};
}

std::vector<Storage> CpuBackend::sdpa_forward(const Storage& q,
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
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so a bool mask
    // travels through untouched and an additive one widens with q/k/v.
    if (detail::is_half_like(dt)) {
        std::optional<Storage> mask32;
        return detail::back_to_f16(
            sdpa_forward(detail::as_f32(q), detail::as_f32(k), detail::as_f32(v),
                         detail::as_f32(attn_mask, mask32), q_shape, k_shape, v_shape, mask_dtype,
                         mask_numel, scale, is_causal, need_weights, Dtype::F32),
            dt);
    }
    // The Accelerate reference always rolls the full softmax weight matrix
    // (its backward reuses it), so ``need_weights`` is a no-op here.
    (void)need_weights;
    // The mask's dtype and extent come from its storage: trusting these
    // two is how a half mask was read as float.  See cpu::SdpaMask.
    (void)mask_dtype;
    (void)mask_numel;
    std::size_t B = 1;
    for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
        B *= static_cast<std::size_t>(q_shape[i]);
    const std::size_t Lq = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
    const std::size_t Dk = static_cast<std::size_t>(q_shape.back());
    const std::size_t Lk = static_cast<std::size_t>(k_shape[k_shape.size() - 2]);
    const std::size_t Dv = static_cast<std::size_t>(v_shape.back());

    const std::size_t weights_numel = B * Lq * Lk;
    const std::size_t output_numel = B * Lq * Dv;

    CpuStorage weights_cpu = alloc_cpu(weights_numel, dt);
    CpuStorage output_cpu = alloc_cpu(output_numel, dt);

    const auto& qs = std::get<CpuStorage>(q);
    const auto& ks = std::get<CpuStorage>(k);
    const auto& vs = std::get<CpuStorage>(v);
    const CpuStorage* ms = attn_mask ? &std::get<CpuStorage>(*attn_mask) : nullptr;
    const cpu::SdpaMask mask = ms ? cpu::sdpa_mask_layout(*ms, dt, Lq * Lk, B) : cpu::SdpaMask{};

    // The masks for slice ``b`` alone — what one parallel task may touch.
    // A keep-mask's ``true`` attends: this kernel once read it as "mask
    // out" while MLX read it as "keep", inverting between devices.
    auto apply_masks_to = [&](auto* Wb, std::size_t b) {
        using T = std::remove_pointer_t<decltype(Wb)>;
        const T neg_inf = -std::numeric_limits<T>::infinity();
        const std::size_t pb = Lq * Lk;
        if (ms != nullptr) {
            const std::size_t offset = b * mask.slice_step;
            if (mask.keep) {
                const auto* mp = reinterpret_cast<const std::uint8_t*>(ms->ptr.get()) + offset;
                for (std::size_t i = 0; i < pb; ++i)
                    if (!mp[i])
                        Wb[i] = neg_inf;
            } else {
                const auto* mp = reinterpret_cast<const T*>(ms->ptr.get()) + offset;
                for (std::size_t i = 0; i < pb; ++i)
                    Wb[i] += mp[i];
            }
        }
        if (is_causal) {
            for (std::size_t i = 0; i < Lq; ++i) {
                T* row = Wb + i * Lk;
                for (std::size_t j = i + 1; j < Lk; ++j)
                    row[j] = neg_inf;
            }
        }
    };

    if (dt == Dtype::F32) {
        const float* Qp = reinterpret_cast<const float*>(qs.ptr.get());
        const float* Kp = reinterpret_cast<const float*>(ks.ptr.get());
        const float* Vp = reinterpret_cast<const float*>(vs.ptr.get());
        float* Wp = reinterpret_cast<float*>(weights_cpu.ptr.get());
        float* Op = reinterpret_cast<float*>(output_cpu.ptr.get());
        const float sc = static_cast<float>(scale);

        // One task per (batch x head) slice, which owns its rows of W and
        // O outright: a small transformer's 128 heads ran one after
        // another, each GEMM too small for Accelerate to thread, 3 ms a
        // call.  Each slice does exactly what the serial loop did, so the
        // result is the same bits.
        cpu::parallel_for(B, 1, [&](std::size_t lo, std::size_t hi) {
            for (std::size_t b = lo; b < hi; ++b) {
                float* Wb = Wp + b * Lq * Lk;
                cpu::sgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                           static_cast<int>(Dk), sc, Qp + b * Lq * Dk, static_cast<int>(Dk),
                           Kp + b * Lk * Dk, static_cast<int>(Dk), 0.0f, Wb, static_cast<int>(Lk));
                apply_masks_to(Wb, b);
                for (std::size_t r = 0; r < Lq; ++r) {
                    float* row = Wb + r * Lk;
                    const float m = cpu::vmaxval_f32(row, Lk);
                    if (!std::isfinite(m)) {
                        std::memset(row, 0, Lk * sizeof(float));
                        continue;
                    }
                    cpu::vsadd_f32(row, -m, row, Lk);
                    cpu::vexp_f32(row, row, Lk);
                    const float s = cpu::vsum_f32(row, Lk);
                    cpu::vsmul_f32(row, s > 0.f ? 1.f / s : 0.f, row, Lk);
                }
                cpu::sgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dv),
                           static_cast<int>(Lk), 1.0f, Wb, static_cast<int>(Lk), Vp + b * Lk * Dv,
                           static_cast<int>(Dv), 0.0f, Op + b * Lq * Dv, static_cast<int>(Dv));
            }
        });
    } else if (dt == Dtype::F64) {
        const double* Qp = reinterpret_cast<const double*>(qs.ptr.get());
        const double* Kp = reinterpret_cast<const double*>(ks.ptr.get());
        const double* Vp = reinterpret_cast<const double*>(vs.ptr.get());
        double* Wp = reinterpret_cast<double*>(weights_cpu.ptr.get());
        double* Op = reinterpret_cast<double*>(output_cpu.ptr.get());
        const double sc = scale;
        for (std::size_t b = 0; b < B; ++b)
            cpu::dgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                       static_cast<int>(Dk), sc, Qp + b * Lq * Dk, static_cast<int>(Dk),
                       Kp + b * Lk * Dk, static_cast<int>(Dk), 0.0, Wp + b * Lq * Lk,
                       static_cast<int>(Lk));
        for (std::size_t b = 0; b < B; ++b)
            apply_masks_to(Wp + b * Lq * Lk, b);

        // With no keys there are no weights to normalise — and no
        // ``row[0]`` to seed the max with; the product below then writes
        // the zero output.
        for (std::size_t r = 0; r < B * Lq && Lk > 0; ++r) {
            double* row = Wp + r * Lk;
            const double m = cpu::vmaxval_f64(row, Lk);
            if (!std::isfinite(m)) {
                std::memset(row, 0, Lk * sizeof(double));
                continue;
            }
            cpu::vsadd_f64(row, -m, row, Lk);
            cpu::vexp_f64(row, row, Lk);
            const double s = cpu::vsum_f64(row, Lk);
            cpu::vsmul_f64(row, s > 0.0 ? 1.0 / s : 0.0, row, Lk);
        }
        for (std::size_t b = 0; b < B; ++b)
            cpu::dgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dv),
                       static_cast<int>(Lk), 1.0, Wp + b * Lq * Lk, static_cast<int>(Lk),
                       Vp + b * Lk * Dv, static_cast<int>(Dv), 0.0, Op + b * Lq * Dv,
                       static_cast<int>(Dv));
    } else {
        ErrorBuilder("cpu_backend::sdpa_forward")
            .not_implemented("dtype not supported (F32/F64 only)");
    }

    return {Storage{std::move(weights_cpu)}, Storage{std::move(output_cpu)}};
}

std::vector<Storage> CpuBackend::sdpa_backward(const Storage& grad_out,
                                               const Storage& q,
                                               const Storage& k,
                                               const Storage& v,
                                               const Storage& saved_weights,
                                               const Storage* attn_mask,
                                               const Shape& q_shape,
                                               const Shape& k_shape,
                                               const Shape& v_shape,
                                               Dtype mask_dtype,
                                               double scale,
                                               bool is_causal,
                                               Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so a bool mask
    // travels through untouched and an additive one widens with q/k/v.
    if (detail::is_half_like(dt)) {
        std::optional<Storage> mask32;
        return detail::back_to_f16(
            sdpa_backward(detail::as_f32(grad_out), detail::as_f32(q), detail::as_f32(k),
                          detail::as_f32(v), detail::as_f32(saved_weights),
                          detail::as_f32(attn_mask, mask32), q_shape, k_shape, v_shape, mask_dtype,
                          scale, is_causal, Dtype::F32),
            dt);
    }
    // The Accelerate forward saves the genuine (already masked/causal)
    // weight matrix W, so the backward is fully determined by ``saved_weights``
    // and needs no mask/causal replay — the extra args exist only to match
    // the interface the MLX VJP path requires.  Whether the mask is
    // additive is read off its storage, as the forward does.
    (void)is_causal;
    (void)mask_dtype;
    std::size_t B = 1;
    for (std::size_t i = 0; i + 2 < q_shape.size(); ++i)
        B *= static_cast<std::size_t>(q_shape[i]);
    const std::size_t Lq = static_cast<std::size_t>(q_shape[q_shape.size() - 2]);
    const std::size_t Dk = static_cast<std::size_t>(q_shape.back());
    const std::size_t Lk = static_cast<std::size_t>(k_shape[k_shape.size() - 2]);
    const std::size_t Dv = static_cast<std::size_t>(v_shape.back());

    CpuStorage dQ_cpu = alloc_cpu(B * Lq * Dk, dt);
    CpuStorage dK_cpu = alloc_cpu(B * Lk * Dk, dt);
    CpuStorage dV_cpu = alloc_cpu(B * Lk * Dv, dt);

    // dL/dM == dL/dS, since S = QK^T·s + M.  ``dscores`` below is already
    // that quantity, so an additive mask costs one extra buffer, not extra
    // arithmetic.  Emitted at full score shape; the node reduces it back
    // through whatever broadcast the caller's mask used.
    // Always emitted so the returned vector matches the node's edge count
    // (the same contract ``conv_nd_backward`` keeps for a null bias); the
    // engine drops this slot when the mask edge is null.
    const bool want_mask_grad =
        attn_mask != nullptr &&
        !cpu::sdpa_mask_layout(std::get<CpuStorage>(*attn_mask), dt, Lq * Lk, B).keep;
    CpuStorage dM_cpu = alloc_cpu(want_mask_grad ? B * Lq * Lk : 1, dt);

    const auto& qs = std::get<CpuStorage>(q);
    const auto& ks = std::get<CpuStorage>(k);
    const auto& vs = std::get<CpuStorage>(v);
    const auto& ws = std::get<CpuStorage>(saved_weights);
    const auto& gs = std::get<CpuStorage>(grad_out);

    auto do_backward = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T sc = static_cast<T>(scale);
        const T* Qp = reinterpret_cast<const T*>(qs.ptr.get());
        const T* Kp = reinterpret_cast<const T*>(ks.ptr.get());
        const T* Vp = reinterpret_cast<const T*>(vs.ptr.get());
        const T* Wp = reinterpret_cast<const T*>(ws.ptr.get());
        const T* Gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* dQp = reinterpret_cast<T*>(dQ_cpu.ptr.get());
        T* dKp = reinterpret_cast<T*>(dK_cpu.ptr.get());
        T* dVp = reinterpret_cast<T*>(dV_cpu.ptr.get());
        T* dMp = want_mask_grad ? reinterpret_cast<T*>(dM_cpu.ptr.get()) : nullptr;

        // One task per chunk of (batch x head) slices, as in the forward;
        // the two scratch matrices are the chunk's own.
        cpu::parallel_for(B, 1, [&](std::size_t lo, std::size_t hi) {
            std::vector<T> dweights(Lq * Lk), dscores(Lq * Lk);
            for (std::size_t b = lo; b < hi; ++b) {
                const T* Qb = Qp + b * Lq * Dk;
                const T* Kb = Kp + b * Lk * Dk;
                const T* Vb = Vp + b * Lk * Dv;
                const T* Wb = Wp + b * Lq * Lk;
                const T* Gb = Gp + b * Lq * Dv;
                T* dQb = dQp + b * Lq * Dk;
                T* dKb = dKp + b * Lk * Dk;
                T* dVb = dVp + b * Lk * Dv;

                if constexpr (std::is_same_v<T, float>) {
                    cpu::sgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dv),
                               static_cast<int>(Lq), 1.0f, Wb, static_cast<int>(Lk), Gb,
                               static_cast<int>(Dv), 0.0f, dVb, static_cast<int>(Dv));

                    cpu::sgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                               static_cast<int>(Dv), 1.0f, Gb, static_cast<int>(Dv), Vb,
                               static_cast<int>(Dv), 0.0f, dweights.data(), static_cast<int>(Lk));

                    for (std::size_t r = 0; r < Lq; ++r) {
                        const float* wr = Wb + r * Lk;
                        const float* dwr = dweights.data() + r * Lk;
                        float sum = 0.f;
                        for (std::size_t j = 0; j < Lk; ++j)
                            sum += wr[j] * dwr[j];
                        float* dr = dscores.data() + r * Lk;
                        for (std::size_t j = 0; j < Lk; ++j)
                            dr[j] = wr[j] * (dwr[j] - sum);
                    }

                    cpu::sgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dk),
                               static_cast<int>(Lk), sc, dscores.data(), static_cast<int>(Lk), Kb,
                               static_cast<int>(Dk), 0.0f, dQb, static_cast<int>(Dk));

                    cpu::sgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dk),
                               static_cast<int>(Lq), sc, dscores.data(), static_cast<int>(Lk), Qb,
                               static_cast<int>(Dk), 0.0f, dKb, static_cast<int>(Dk));
                } else {
                    cpu::dgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dv),
                               static_cast<int>(Lq), 1.0, Wb, static_cast<int>(Lk), Gb,
                               static_cast<int>(Dv), 0.0, dVb, static_cast<int>(Dv));
                    cpu::dgemm(false, true, static_cast<int>(Lq), static_cast<int>(Lk),
                               static_cast<int>(Dv), 1.0, Gb, static_cast<int>(Dv), Vb,
                               static_cast<int>(Dv), 0.0, dweights.data(), static_cast<int>(Lk));
                    for (std::size_t r = 0; r < Lq; ++r) {
                        const double* wr = Wb + r * Lk;
                        const double* dwr = dweights.data() + r * Lk;
                        double sum = 0.0;
                        for (std::size_t j = 0; j < Lk; ++j)
                            sum += wr[j] * dwr[j];
                        double* dr = dscores.data() + r * Lk;
                        for (std::size_t j = 0; j < Lk; ++j)
                            dr[j] = wr[j] * (dwr[j] - sum);
                    }
                    cpu::dgemm(false, false, static_cast<int>(Lq), static_cast<int>(Dk),
                               static_cast<int>(Lk), sc, dscores.data(), static_cast<int>(Lk), Kb,
                               static_cast<int>(Dk), 0.0, dQb, static_cast<int>(Dk));
                    cpu::dgemm(true, false, static_cast<int>(Lk), static_cast<int>(Dk),
                               static_cast<int>(Lq), sc, dscores.data(), static_cast<int>(Lk), Qb,
                               static_cast<int>(Dk), 0.0, dKb, static_cast<int>(Dk));
                }

                if (dMp != nullptr)
                    std::copy(dscores.begin(), dscores.end(), dMp + b * Lq * Lk);
            }
        });
    };

    if (dt == Dtype::F32)
        do_backward(float{});
    else if (dt == Dtype::F64)
        do_backward(double{});
    else
        ErrorBuilder("cpu_backend::sdpa_backward").not_implemented("dtype not supported");

    return {Storage{std::move(dQ_cpu)}, Storage{std::move(dK_cpu)}, Storage{std::move(dV_cpu)},
            Storage{std::move(dM_cpu)}};
}

Storage CpuBackend::embedding_forward(const Storage& weight,
                                      const Storage& indices,
                                      const Shape& weight_shape,
                                      const Shape& indices_shape,
                                      const Shape& out_shape,
                                      int padding_idx,
                                      Dtype dt) {
    const std::int64_t N = weight_shape[0];
    const std::int64_t D = weight_shape[1];
    std::size_t M = 1;
    for (auto d : indices_shape)
        M *= static_cast<std::size_t>(d);
    (void)out_shape;
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& is = std::get<CpuStorage>(indices);
    auto out_cpu = alloc_cpu(M * static_cast<std::size_t>(D), dt);
    const std::size_t row_bytes = static_cast<std::size_t>(D) * dtype_size(dt);
    const auto read_idx = cpu::index_reader("embedding", is);
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
    for (std::size_t i = 0; i < M; ++i) {
        // Refuse rather than copy a row from outside the table.  Nothing
        // checked this: the Python wrapper's host-side range check was
        // the only guard, and the engine op read past the allocation.
        const std::int64_t id = cpu::checked_row("embedding", read_idx(i), N);
        std::byte* dst = out_cpu.ptr.get() + i * row_bytes;
        const std::byte* src = ws.ptr.get() + static_cast<std::size_t>(id) * row_bytes;
        std::memcpy(dst, src, row_bytes);
    }
    return Storage{std::move(out_cpu)};
}

Storage CpuBackend::embedding_backward(const Storage& grad_out,
                                       const Storage& indices,
                                       const Shape& weight_shape,
                                       const Shape& indices_shape,
                                       int padding_idx,
                                       Dtype dt) {
    // Accelerate has no half kernels, so half widens at the door, sums in
    // float32 and rounds once on the way out.  The dispatch below used to
    // name F32 and hand every other dtype to a bare ``else`` that walked
    // the buffer in ``double`` lanes: a half gradient was summed four
    // elements to a lane, into rows no index named and, from the last
    // row, past the end of the allocation.  It now refuses what it does
    // not name.  tools/check_dtype_dispatch.py keeps it that way.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(embedding_backward(detail::as_f32(grad_out), indices,
                                                      weight_shape, indices_shape, padding_idx,
                                                      Dtype::F32),
                                   dt);
    const std::int64_t N = weight_shape[0];
    const std::int64_t D = weight_shape[1];
    std::size_t M = 1;
    for (auto d : indices_shape)
        M *= static_cast<std::size_t>(d);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const auto& is = std::get<CpuStorage>(indices);
    // The index buffer is read at its own width, so any other integer
    // width would be misread the same way.
    const auto read_idx = cpu::index_reader("embedding_backward", is);
    auto dW = alloc_cpu(static_cast<std::size_t>(N) * static_cast<std::size_t>(D), dt);
    std::memset(dW.ptr.get(), 0, dW.nbytes);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        T* wp = reinterpret_cast<T*>(dW.ptr.get());
        const std::size_t width = static_cast<std::size_t>(D);
        for (std::size_t i = 0; i < M; ++i) {
            // Refuse rather than add into memory past the gradient.  The
            // forward refuses the same index, so only indices changed in
            // place since then can reach here.
            const std::int64_t id = cpu::checked_row("embedding_backward", read_idx(i), N);
            if (padding_idx >= 0 && id == static_cast<std::int64_t>(padding_idx))
                continue;
            const T* sp = gp + i * width;
            T* dp = wp + static_cast<std::size_t>(id) * width;
            for (std::size_t d = 0; d < width; ++d)
                dp[d] += sp[d];
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::embedding_backward").not_implemented("dtype not supported");
    return Storage{std::move(dW)};
}

Storage
CpuBackend::sinusoidal_pos_embedding(std::int64_t seq_len, std::int64_t embed_dim, Dtype dt) {
    const std::size_t L = static_cast<std::size_t>(seq_len);
    const std::size_t D = static_cast<std::size_t>(embed_dim);
    const std::size_t Dh = D / 2;
    auto out = alloc_cpu(L * D, dt);
    if (L * D == 0)
        return Storage{std::move(out)};
    constexpr double kLog10000 = 9.2103403719761836;
    const double inv_d = -kLog10000 / static_cast<double>(D);
    std::vector<double> freq(Dh);
    for (std::size_t k = 0; k < Dh; ++k)
        freq[k] = 2.0 * static_cast<double>(k) * inv_d;
    cpu::vexp_f64(freq.data(), freq.data(), Dh);
    const std::size_t LDh = L * Dh;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        std::vector<T> angles(LDh), sines(LDh), cosines(LDh);
        for (std::size_t i = 0; i < L; ++i)
            for (std::size_t k = 0; k < Dh; ++k)
                angles[i * Dh + k] = static_cast<T>(static_cast<double>(i) * freq[k]);
        cpu::vsin(angles.data(), sines.data(), LDh);
        cpu::vcos(angles.data(), cosines.data(), LDh);
        T* op = reinterpret_cast<T*>(out.ptr.get());
        for (std::size_t i = 0; i < L; ++i)
            for (std::size_t k = 0; k < Dh; ++k) {
                op[i * D + 2 * k] = sines[i * Dh + k];
                if (2 * k + 1 < D)
                    op[i * D + 2 * k + 1] = cosines[i * Dh + k];
            }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("sinusoidal_pos_embedding").not_implemented("dtype");
    return Storage{std::move(out)};
}

std::vector<Storage> CpuBackend::rope_forward(const Storage& x,
                                              const Storage* position_ids,
                                              const Shape& x_shape,
                                              bool interleaved,
                                              Dtype pos_dtype,
                                              Dtype dt) {
    (void)pos_dtype;
    const std::size_t ndim = x_shape.size();
    const std::size_t L = static_cast<std::size_t>(x_shape[ndim - 2]);
    const std::size_t D = static_cast<std::size_t>(x_shape.back());
    const std::size_t Dh = D / 2;
    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        batch *= static_cast<std::size_t>(x_shape[i]);
    const auto& xs = std::get<CpuStorage>(x);
    auto out = alloc_cpu(batch * L * D, dt);
    auto cos_t = alloc_cpu(L * Dh, dt);
    auto sin_t = alloc_cpu(L * Dh, dt);
    std::vector<double> pos(L);
    if (position_ids) {
        const auto& ps = std::get<CpuStorage>(*position_ids);
        for (std::size_t i = 0; i < L; ++i)
            pos[i] = static_cast<double>(reinterpret_cast<const std::int64_t*>(ps.ptr.get())[i]);
    } else {
        for (std::size_t i = 0; i < L; ++i)
            pos[i] = static_cast<double>(i);
    }
    std::vector<double> theta(Dh);
    constexpr double kLog10000 = 9.2103403719761836;
    for (std::size_t k = 0; k < Dh; ++k)
        theta[k] = -2.0 * static_cast<double>(k) * kLog10000 / static_cast<double>(D);
    cpu::vexp_f64(theta.data(), theta.data(), Dh);
    const std::size_t LDh = L * Dh;
    auto run = [&](auto tag) {
        using T = decltype(tag);
        T* cp = reinterpret_cast<T*>(cos_t.ptr.get());
        T* sp = reinterpret_cast<T*>(sin_t.ptr.get());
        for (std::size_t i = 0; i < L; ++i)
            for (std::size_t k = 0; k < Dh; ++k)
                sp[i * Dh + k] = static_cast<T>(pos[i] * theta[k]);
        cpu::vcos(sp, cp, LDh);
        cpu::vsin(sp, sp, LDh);
        const T* xp = reinterpret_cast<const T*>(xs.ptr.get());
        T* op = reinterpret_cast<T*>(out.ptr.get());
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t i = 0; i < L; ++i) {
                const T* xrow = xp + (b * L + i) * D;
                T* orow = op + (b * L + i) * D;
                const T* crow = cp + i * Dh;
                const T* srow = sp + i * Dh;
                if (interleaved) {
                    for (std::size_t k = 0; k < Dh; ++k) {
                        const T xe = xrow[2 * k], xo = xrow[2 * k + 1];
                        orow[2 * k] = xe * crow[k] - xo * srow[k];
                        orow[2 * k + 1] = xo * crow[k] + xe * srow[k];
                    }
                } else {
                    for (std::size_t k = 0; k < Dh; ++k) {
                        orow[k] = xrow[k] * crow[k] - xrow[k + Dh] * srow[k];
                        orow[k + Dh] = xrow[k + Dh] * crow[k] + xrow[k] * srow[k];
                    }
                }
            }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("rope_forward").not_implemented("dtype");
    return {Storage{std::move(out)}, Storage{std::move(cos_t)}, Storage{std::move(sin_t)}};
}

Storage CpuBackend::rope_backward(const Storage& grad_out,
                                  const Storage& saved_cos,
                                  const Storage& saved_sin,
                                  const Shape& x_shape,
                                  bool interleaved,
                                  Dtype dt) {
    const std::size_t ndim = x_shape.size();
    const std::size_t L = static_cast<std::size_t>(x_shape[ndim - 2]);
    const std::size_t D = static_cast<std::size_t>(x_shape.back());
    const std::size_t Dh = D / 2;
    std::size_t batch = 1;
    for (std::size_t i = 0; i + 2 < ndim; ++i)
        batch *= static_cast<std::size_t>(x_shape[i]);
    const auto& gs = std::get<CpuStorage>(grad_out);
    const auto& cs = std::get<CpuStorage>(saved_cos);
    const auto& ss = std::get<CpuStorage>(saved_sin);
    auto dx = alloc_cpu(batch * L * D, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* gp = reinterpret_cast<const T*>(gs.ptr.get());
        const T* cp = reinterpret_cast<const T*>(cs.ptr.get());
        const T* sp = reinterpret_cast<const T*>(ss.ptr.get());
        T* dxp = reinterpret_cast<T*>(dx.ptr.get());
        for (std::size_t b = 0; b < batch; ++b)
            for (std::size_t i = 0; i < L; ++i) {
                const T* grow = gp + (b * L + i) * D;
                T* drow = dxp + (b * L + i) * D;
                const T* crow = cp + i * Dh;
                const T* srow = sp + i * Dh;
                if (interleaved) {
                    for (std::size_t k = 0; k < Dh; ++k) {
                        const T ge = grow[2 * k], go = grow[2 * k + 1];
                        drow[2 * k] = ge * crow[k] + go * srow[k];
                        drow[2 * k + 1] = go * crow[k] - ge * srow[k];
                    }
                } else {
                    for (std::size_t k = 0; k < Dh; ++k) {
                        drow[k] = grow[k] * crow[k] + grow[k + Dh] * srow[k];
                        drow[k + Dh] = grow[k + Dh] * crow[k] - grow[k] * srow[k];
                    }
                }
            }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("rope_backward").not_implemented("dtype");
    return Storage{std::move(dx)};
}

std::vector<Storage> CpuBackend::lstm_forward(const Storage& input,
                                              const Storage& h0,
                                              const Storage& c0,
                                              const std::vector<Storage>& weights,
                                              const LstmOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
    auto res = lstm_forward_train(input, h0, c0, weights, opts, dt);
    res.resize(3);
    return res;
}

std::vector<Storage> CpuBackend::lstm_forward_train(const Storage& input,
                                                    const Storage& h0,
                                                    const Storage& c0,
                                                    const std::vector<Storage>& weights,
                                                    const LstmOpts& opts,
                                                    Dtype dt) {
    if (is_half_float(dt)) {
        auto res = lstm_forward_train(detail::as_f32(input), detail::as_f32(h0), detail::as_f32(c0),
                                      lstm_widen(weights), opts, Dtype::F32);
        for (std::size_t i = 0; i < 3; ++i)
            res[i] = detail::back_to_f16(res[i], dt);
        return res;
    }
    if (dt == Dtype::F64)
        return cpu::lstm::forward_train<double>(input, h0, c0, weights, opts, dt);
    if (dt == Dtype::F32)
        return cpu::lstm::forward_train<float>(input, h0, c0, weights, opts, dt);
    return IBackend::lstm_forward_train(input, h0, c0, weights, opts, dt);
}

std::vector<Storage> CpuBackend::lstm_backward(const Storage& grad_output,
                                               const Storage& grad_hn,
                                               const Storage& grad_cn,
                                               const Storage& input,
                                               const Storage& h0,
                                               const std::vector<Storage>& weights,
                                               const Storage& gates_all,
                                               const Storage& cells_all,
                                               const LstmOpts& opts,
                                               Dtype dt) {
    if (is_half_float(dt))
        return detail::back_to_f16(
            lstm_backward(detail::as_f32(grad_output), detail::as_f32(grad_hn),
                          detail::as_f32(grad_cn), detail::as_f32(input), detail::as_f32(h0),
                          lstm_widen(weights), gates_all, cells_all, opts, Dtype::F32),
            dt);
    if (dt == Dtype::F64)
        return cpu::lstm::backward<double>(grad_output, grad_hn, grad_cn, input, h0, weights,
                                           gates_all, cells_all, opts, dt);
    if (dt == Dtype::F32)
        return cpu::lstm::backward<float>(grad_output, grad_hn, grad_cn, input, h0, weights,
                                          gates_all, cells_all, opts, dt);
    return IBackend::lstm_backward(grad_output, grad_hn, grad_cn, input, h0, weights, gates_all,
                                   cells_all, opts, dt);
}

std::vector<Storage> CpuBackend::lstm_widen(const std::vector<Storage>& weights) {
    std::vector<Storage> wide;
    wide.reserve(weights.size());
    for (const auto& w : weights)
        wide.push_back(detail::as_f32(w));
    return wide;
}

}  // namespace backend
}  // namespace lucid
