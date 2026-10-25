// lucid/_C/backend/gpu/methods/Spatial.cpp
// Out-of-line GpuBackend method definitions for the Spatial family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::drop_block_mask(const Storage& seed,
                                    double drop_prob,
                                    std::int64_t block_size,
                                    const Shape& x_shape,
                                    Dtype dt) {
    const auto& sg = std::get<GpuStorage>(seed);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);
    const int K = static_cast<int>(block_size);
    const int pad_sz = K / 2;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    const int H = static_cast<int>(x_shape[2]);
    const int W = static_cast<int>(x_shape[3]);
    auto seed_4d = ::mlx::core::reshape(*sg.arr, ::mlx::core::Shape{B, C, H, W});
    auto seed_pad = ::mlx::core::pad(
        seed_4d,
        std::vector<std::pair<int, int>>{{0, 0}, {0, 0}, {pad_sz, pad_sz}, {pad_sz, pad_sz}},
        ::mlx::core::array(0.0f, mlx_dt));
    ::mlx::core::array dilated = ::mlx::core::zeros({B, C, H, W}, mlx_dt);
    for (int dy = 0; dy < K; ++dy) {
        for (int dx = 0; dx < K; ++dx) {
            auto s = ::mlx::core::slice(seed_pad, {0, 0, dy, dx}, {B, C, dy + H, dx + W});
            dilated = (dy == 0 && dx == 0) ? s : ::mlx::core::maximum(dilated, s);
        }
    }
    auto scale_v = ::mlx::core::array(static_cast<float>(1.0 / (1.0 - drop_prob)), mlx_dt);
    auto keep = ::mlx::core::multiply(
        scale_v, ::mlx::core::subtract(::mlx::core::array(1.0f, mlx_dt), dilated));
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(keep), dt)};
}

std::vector<Storage> GpuBackend::max_pool_nd_forward(const Storage& x,
                                                     const Shape& x_shape,
                                                     const Shape& out_shape,
                                                     const PoolOpts& opts,
                                                     Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3], K[3], stride[3], Sp[3], right_extra[3]{};
    int K_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        K[i] = opts.K[i];
        stride[i] = opts.stride[i];
        // Ceil mode lets the last window overhang the padded input; give
        // it room on the right so the window view stays in bounds.  The
        // fill is -inf, so the overhang can never win a max.
        const int extra = gpu_ceil_overhang(S[i], K[i], stride[i], opts.pad[i], O[i]);
        Sp[i] = S[i] + 2 * opts.pad[i] + extra;
        right_extra[i] = extra;
        K_total *= K[i];
    }
    // The pad must never win a max: -inf for a float, the type's lowest
    // value for an integer.  Converting -inf to an integer is undefined —
    // the UBSan build halted on every integer max-pool here.
    const auto pad_dt = gpu::to_mlx_dtype(dt);
    ::mlx::core::array neg_inf = [&]() -> ::mlx::core::array {
        switch (dt) {
        case Dtype::Bool:
            return ::mlx::core::array(false, pad_dt);
        case Dtype::I8:
            return ::mlx::core::array(std::numeric_limits<std::int8_t>::lowest(), pad_dt);
        case Dtype::I16:
            return ::mlx::core::array(std::numeric_limits<std::int16_t>::lowest(), pad_dt);
        case Dtype::I32:
            return ::mlx::core::array(std::numeric_limits<std::int32_t>::lowest(), pad_dt);
        case Dtype::I64:
            return ::mlx::core::array(std::numeric_limits<std::int64_t>::lowest(), pad_dt);
        default:
            return ::mlx::core::array(-std::numeric_limits<double>::infinity(), pad_dt);
        }
    }();
    std::vector<std::pair<int, int>> pad_widths;
    pad_widths.emplace_back(0, 0);
    pad_widths.emplace_back(0, 0);
    for (int i = 0; i < N; ++i)
        pad_widths.emplace_back(opts.pad[i], opts.pad[i] + right_extra[i]);
    auto x_pad = ::mlx::core::pad(*gx.arr, pad_widths, neg_inf);
    auto wins = gpu_build_window_view(x_pad, B, C, Sp, O, K, stride, N);
    std::vector<int> kernel_axes;
    for (int i = 0; i < N; ++i)
        kernel_axes.push_back(2 + N + i);
    auto y = wins.size() == 0 ? empty_extreme(wins, kernel_axes, false, wins.dtype())
                              : ::mlx::core::max(wins, kernel_axes, false);
    ::mlx::core::Shape flat_win;
    flat_win.push_back(B);
    flat_win.push_back(C);
    for (int i = 0; i < N; ++i)
        flat_win.push_back(O[i]);
    flat_win.push_back(K_total);
    auto wins_flat = ::mlx::core::reshape(wins, flat_win);
    // Window-argmax (saved for the backward), chosen by window size K_total.
    // ``mlx::core::argmax`` is pathological for a TINY reduction axis with
    // MANY groups (regular pools: K_total=4/9, millions of windows — measured
    // 168 vs 1.8 ms for max @64ch@112 BS8), but FAST for a LARGE axis with
    // FEW groups (adaptive/global pools: K_total=3136, ~3.4 ms). The manual
    // max+strict-greater select is the inverse: O(K_total) ops, great for
    // small windows (bit-exact first-max tie-break) but ~924 ms at K_total=
    // 3136. So pick per K_total. (Both give argmax's first-max index.)
    const int amax_axis = 2 + N;
    ::mlx::core::array argmax = ::mlx::core::array(0);
    if (K_total <= 16) {
        ::mlx::core::Shape am_shape;
        am_shape.push_back(B);
        am_shape.push_back(C);
        for (int i = 0; i < N; ++i)
            am_shape.push_back(O[i]);
        auto take_k = [&](int k) -> ::mlx::core::array {
            ::mlx::core::Shape lo(2 + N + 1, 0), hi = wins_flat.shape();
            lo[amax_axis] = k;
            hi[amax_axis] = k + 1;
            return ::mlx::core::reshape(::mlx::core::slice(wins_flat, lo, hi), am_shape);
        };
        auto best = take_k(0);
        argmax = ::mlx::core::zeros(am_shape, ::mlx::core::int32);
        for (int k = 1; k < K_total; ++k) {
            auto ek = take_k(k);
            auto cond = ::mlx::core::greater(ek, best);
            argmax = ::mlx::core::where(
                cond, ::mlx::core::array(static_cast<std::int32_t>(k), ::mlx::core::int32), argmax);
            best = ::mlx::core::maximum(best, ek);
        }
    } else {
        argmax = wins_flat.size() == 0
                     ? empty_extreme(wins_flat, {amax_axis}, false, ::mlx::core::int32)
                     : ::mlx::core::astype(::mlx::core::argmax(wins_flat, amax_axis, false),
                                           ::mlx::core::int32);
    }
    return {Storage{gpu::wrap_mlx_array(std::move(y), dt)},
            Storage{gpu::wrap_mlx_array(std::move(argmax), Dtype::I32)}};
}

Storage GpuBackend::max_pool_nd_backward(const Storage& grad_out,
                                         const Storage& saved_argmax,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) {
    const auto& gG = std::get<GpuStorage>(grad_out);
    const auto& gA = std::get<GpuStorage>(saved_argmax);
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3], Sp[3];
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        Sp[i] = S[i] + 2 * opts.pad[i];
    }

    // 3.5+ FAST PATH: non-overlapping 2-D pool (stride==K, pad==0) — the
    // VGG/CNN common case.  The general ``scatter_add_axis`` path below is
    // 38-75x slower than the reference framework because MLX's scatter uses
    // collision atomics; with non-overlapping windows there are NO
    // collisions, so reconstruct dx via onehot(argmax)*grad +
    // interleave-reshape (pure elementwise, no scatter).  Measured 72x
    // faster (730->10ms @64ch@224, reference-class) with exact parity.
    // Overlapping / padded pools fall through to the scatter path.
    bool nonoverlap_2d = (N == 2);
    for (int i = 0; i < N && nonoverlap_2d; ++i)
        // The interleave-reshape reconstruction assumes the windows tile
        // the input exactly.  A ceil-mode overhang breaks that (O*K > S),
        // so such pools must take the general scatter path.
        if (opts.stride[i] != opts.K[i] || opts.pad[i] != 0 || O[i] * opts.K[i] != S[i])
            nonoverlap_2d = false;
    if (nonoverlap_2d) {
        const int Oh = O[0], Ow = O[1], Kh = opts.K[0], Kw = opts.K[1];
        const auto idt = ::mlx::core::int32;
        const auto mdt = gpu::to_mlx_dtype(dt);
        using SE2 = ::mlx::core::ShapeElem;
        auto Kw_a = ::mlx::core::array(static_cast<std::int32_t>(Kw), idt);
        // argmax (B,C,Oh,Ow) -> within-window (kh, kw)
        auto kh = ::mlx::core::reshape(::mlx::core::floor_divide(*gA.arr, Kw_a),
                                       ::mlx::core::Shape{B, C, Oh, Ow, 1});
        auto kw = ::mlx::core::reshape(::mlx::core::remainder(*gA.arr, Kw_a),
                                       ::mlx::core::Shape{B, C, Oh, Ow, 1});
        auto arh = ::mlx::core::reshape(::mlx::core::arange(0, Kh, 1, idt),
                                        ::mlx::core::Shape{1, 1, 1, 1, Kh});
        auto arw = ::mlx::core::reshape(::mlx::core::arange(0, Kw, 1, idt),
                                        ::mlx::core::Shape{1, 1, 1, 1, Kw});
        // onehot_h (B,C,Oh,Ow,Kh,1) ; onehot_w (B,C,Oh,Ow,1,Kw)
        auto oh_h = ::mlx::core::reshape(::mlx::core::astype(::mlx::core::equal(kh, arh), mdt),
                                         ::mlx::core::Shape{B, C, Oh, Ow, Kh, 1});
        auto oh_w = ::mlx::core::reshape(::mlx::core::astype(::mlx::core::equal(kw, arw), mdt),
                                         ::mlx::core::Shape{B, C, Oh, Ow, 1, Kw});
        auto g6 = ::mlx::core::reshape(*gG.arr, ::mlx::core::Shape{B, C, Oh, Ow, 1, 1});
        // dxw (B,C,Oh,Ow,Kh,Kw) = grad * onehot_h * onehot_w
        auto dxw = ::mlx::core::multiply(::mlx::core::multiply(g6, oh_h), oh_w);
        // interleave (Oh,Kh,Ow,Kw) -> reshape to (B,C,H,W)
        auto dxw_t = ::mlx::core::transpose(dxw, {0, 1, 2, 4, 3, 5});
        auto dx = ::mlx::core::reshape(
            dxw_t, ::mlx::core::Shape{B, C, static_cast<SE2>(Oh * Kh), static_cast<SE2>(Ow * Kw)});
        // When the input is not an exact multiple of the kernel (stride==K,
        // pad==0), the pooled region (Oh*Kh, Ow*Kw) is SMALLER than the true
        // input (S[0], S[1]) — the trailing remainder rows/cols are dropped by
        // pooling and never belong to a window, so their gradient is zero.
        // Pad dx up to the input shape; without this the (Oh*Kh,Ow*Kw) buffer
        // is silently reinterpreted at the wrong row stride and gradients land
        // on wrong positions.  No-op / no perf cost for the divisible case.
        const int ph = S[0] - Oh * Kh;
        const int pw = S[1] - Ow * Kw;
        if (ph > 0 || pw > 0) {
            const std::vector<std::pair<int, int>> pad_w = {{0, 0}, {0, 0}, {0, ph}, {0, pw}};
            ::mlx::core::array zero(static_cast<float>(0.0), mdt);
            dx = ::mlx::core::pad(dx, pad_w, zero);
        }
        return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
    }

    int K_suffix[4];
    K_suffix[N] = 1;
    for (int i = N - 1; i >= 0; --i)
        K_suffix[i] = K_suffix[i + 1] * opts.K[i];
    using SE = ::mlx::core::ShapeElem;
    const auto idt = ::mlx::core::int32;
    auto compute_ih = [&](int i) -> ::mlx::core::array {
        auto div_a = ::mlx::core::array(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
        auto mod_a = ::mlx::core::array(static_cast<std::int32_t>(opts.K[i]), idt);
        auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(*gA.arr, div_a), mod_a);
        ::mlx::core::Shape rs(N + 2, 1);
        rs[2 + i] = static_cast<SE>(O[i]);
        auto o_r = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), rs);
        return ::mlx::core::add(
            ::mlx::core::multiply(
                o_r, ::mlx::core::array(static_cast<std::int32_t>(opts.stride[i]), idt)),
            ki);
    };
    ::mlx::core::array flat_idx = compute_ih(0);
    for (int i = 1; i < N; ++i)
        flat_idx = ::mlx::core::add(
            ::mlx::core::multiply(flat_idx,
                                  ::mlx::core::array(static_cast<std::int32_t>(Sp[i]), idt)),
            compute_ih(i));
    ::mlx::core::Shape full_O;
    full_O.push_back(B);
    full_O.push_back(C);
    for (int i = 0; i < N; ++i)
        full_O.push_back(O[i]);
    flat_idx = ::mlx::core::broadcast_to(flat_idx, full_O);
    const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
    std::int64_t Sp_total = 1;
    for (int i = 0; i < N; ++i)
        Sp_total *= Sp[i];
    // The reshape pair previously here (idx_2d / g_2d using x_shape[2]
    // / out_shape[2]) was leftover dead code from a 1-D-only iteration
    // of this impl.  For 2-D and 3-D pools O_total != out_shape[2] and
    // the reshape would attempt a size mismatch, raising "[reshape]
    // Cannot reshape array of size N into shape …".  MLX evaluates
    // even unused graph nodes when the parent computation runs, so
    // marking the result with ``(void)`` did not avoid the failure.
    // The actual scatter uses idx_2d_flat / g_2d_flat below.
    auto zero_pad = ::mlx::core::zeros({BC, static_cast<SE>(Sp_total)}, gpu::to_mlx_dtype(dt));

    std::int64_t O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    auto idx_2d_flat = ::mlx::core::reshape(flat_idx, {BC, static_cast<SE>(O_total)});
    auto g_2d_flat = ::mlx::core::reshape(*gG.arr, {BC, static_cast<SE>(O_total)});
    auto dx_pad_2d = scatter_add_axis_widened(zero_pad, idx_2d_flat, g_2d_flat, 1);
    ::mlx::core::Shape full_Sp;
    full_Sp.push_back(B);
    full_Sp.push_back(C);
    for (int i = 0; i < N; ++i)
        full_Sp.push_back(Sp[i]);
    auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);
    ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
    crop_hi.push_back(B);
    crop_hi.push_back(C);
    for (int i = 0; i < N; ++i) {
        crop_lo[2 + i] = static_cast<SE>(opts.pad[i]);
        crop_hi.push_back(static_cast<SE>(opts.pad[i] + S[i]));
    }
    auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
    return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
}

Storage GpuBackend::avg_pool_nd_forward(const Storage& x,
                                        const Shape& x_shape,
                                        const Shape& out_shape,
                                        const PoolOpts& opts,
                                        Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3], K[3], stride[3], Sp[3], right_extra[3]{};
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        K[i] = opts.K[i];
        stride[i] = opts.stride[i];
        // Ceil-mode overhang needs room on the right; the fill is zero, so
        // it adds nothing to the sum, and the divisor below excludes it.
        right_extra[i] = gpu_ceil_overhang(S[i], K[i], stride[i], opts.pad[i], O[i]);
        Sp[i] = S[i] + 2 * opts.pad[i] + right_extra[i];
    }
    // The window sums and their division accumulate in float32 for a
    // half input (see ``widen_half``): a float16 global average over a
    // 56 x 56 map of 30s summed to inf.  The average is rounded back.
    const auto xw = widen_half(*gx.arr);
    const Dtype acc_dt = is_half(gx.arr->dtype()) ? Dtype::F32 : dt;
    ::mlx::core::array zero(0.0, xw.dtype());
    std::vector<std::pair<int, int>> pad_widths;
    pad_widths.emplace_back(0, 0);
    pad_widths.emplace_back(0, 0);
    for (int i = 0; i < N; ++i)
        pad_widths.emplace_back(opts.pad[i], opts.pad[i] + right_extra[i]);
    auto x_pad = ::mlx::core::pad(xw, pad_widths, zero);
    auto wins = gpu_build_window_view(x_pad, B, C, Sp, O, K, stride, N);
    std::vector<int> kernel_axes;
    for (int i = 0; i < N; ++i)
        kernel_axes.push_back(2 + N + i);
    // ``mean`` would divide by the full window everywhere.  That is only
    // right when no window is clipped — border windows under padding, and
    // every window under a ceil-mode overhang, divide by less.
    auto y = ::mlx::core::sum(wins, kernel_axes, false);
    auto avg = gpu_apply_pool_divisor(y, B, C, S, O, K, stride, opts, N, acc_dt);
    return Storage{gpu::wrap_mlx_array(narrow_half(avg, gx.arr->dtype()), dt)};
}

Storage GpuBackend::avg_pool_nd_backward(const Storage& grad_out,
                                         const Shape& x_shape,
                                         const Shape& out_shape,
                                         const PoolOpts& opts,
                                         Dtype dt) {
    const auto& gG = std::get<GpuStorage>(grad_out);
    const int N = opts.N;
    const int B = static_cast<int>(x_shape[0]);
    const int C = static_cast<int>(x_shape[1]);
    int S[3], O[3], Sp[3];
    int K_total = 1;
    for (int i = 0; i < N; ++i) {
        S[i] = static_cast<int>(x_shape[2 + i]);
        O[i] = static_cast<int>(out_shape[2 + i]);
        Sp[i] = S[i] + 2 * opts.pad[i] +
                gpu_ceil_overhang(S[i], opts.K[i], opts.stride[i], opts.pad[i], O[i]);
        K_total *= opts.K[i];
    }
    using SE = ::mlx::core::ShapeElem;
    const auto idt = ::mlx::core::int32;
    ::mlx::core::Shape g_with_k;
    g_with_k.push_back(B);
    g_with_k.push_back(C);
    for (int i = 0; i < N; ++i)
        g_with_k.push_back(O[i]);
    g_with_k.push_back(1);
    // Each output's gradient is spread over the elements that fed it, so
    // it carries that window's own divisor — not a uniform 1/prod(K).
    auto g_scaled = gpu_apply_pool_divisor(*gG.arr, B, C, S, O, opts.K, opts.stride, opts, N, dt);
    auto g_exp = ::mlx::core::reshape(g_scaled, g_with_k);
    ::mlx::core::Shape full_k = g_with_k;
    full_k[N + 2] = static_cast<SE>(K_total);
    auto updates = ::mlx::core::broadcast_to(g_exp, full_k);
    ::mlx::core::Shape kr_shape(N + 3, 1);
    kr_shape[N + 2] = static_cast<SE>(K_total);
    auto k_range = ::mlx::core::reshape(::mlx::core::arange(0, K_total, 1, idt), kr_shape);
    int K_suffix[4];
    K_suffix[N] = 1;
    for (int i = N - 1; i >= 0; --i)
        K_suffix[i] = K_suffix[i + 1] * opts.K[i];
    auto compute_ih = [&](int i) -> ::mlx::core::array {
        auto div_a = ::mlx::core::array(static_cast<std::int32_t>(K_suffix[i + 1]), idt);
        auto mod_a = ::mlx::core::array(static_cast<std::int32_t>(opts.K[i]), idt);
        auto ki = ::mlx::core::remainder(::mlx::core::floor_divide(k_range, div_a), mod_a);
        ::mlx::core::Shape rs(N + 3, 1);
        rs[2 + i] = static_cast<SE>(O[i]);
        auto o_r = ::mlx::core::reshape(::mlx::core::arange(0, O[i], 1, idt), rs);
        return ::mlx::core::add(
            ::mlx::core::multiply(
                o_r, ::mlx::core::array(static_cast<std::int32_t>(opts.stride[i]), idt)),
            ki);
    };
    ::mlx::core::array flat_idx = compute_ih(0);
    for (int i = 1; i < N; ++i)
        flat_idx = ::mlx::core::add(
            ::mlx::core::multiply(flat_idx,
                                  ::mlx::core::array(static_cast<std::int32_t>(Sp[i]), idt)),
            compute_ih(i));
    flat_idx = ::mlx::core::broadcast_to(flat_idx, full_k);
    const SE BC = static_cast<SE>(B) * static_cast<SE>(C);
    std::int64_t Sp_total = 1;
    for (int i = 0; i < N; ++i)
        Sp_total *= Sp[i];
    std::int64_t O_total = 1;
    for (int i = 0; i < N; ++i)
        O_total *= O[i];
    auto idx_2d = ::mlx::core::reshape(flat_idx, {BC, static_cast<SE>(O_total * K_total)});
    auto upd_2d = ::mlx::core::reshape(updates, {BC, static_cast<SE>(O_total * K_total)});
    auto zero_pad = ::mlx::core::zeros({BC, static_cast<SE>(Sp_total)}, gpu::to_mlx_dtype(dt));
    auto dx_pad_2d = scatter_add_axis_widened(zero_pad, idx_2d, upd_2d, 1);
    ::mlx::core::Shape full_Sp;
    full_Sp.push_back(B);
    full_Sp.push_back(C);
    for (int i = 0; i < N; ++i)
        full_Sp.push_back(Sp[i]);
    auto dx_pad = ::mlx::core::reshape(dx_pad_2d, full_Sp);
    ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
    crop_hi.push_back(B);
    crop_hi.push_back(C);
    for (int i = 0; i < N; ++i) {
        crop_lo[2 + i] = static_cast<SE>(opts.pad[i]);
        crop_hi.push_back(static_cast<SE>(opts.pad[i] + S[i]));
    }
    auto dx = ::mlx::core::slice(dx_pad, crop_lo, crop_hi);
    return Storage{gpu::wrap_mlx_array(std::move(dx), dt)};
}

::mlx::core::array GpuBackend::gpu_apply_pool_divisor(::mlx::core::array y,
                                                      int B,
                                                      int C,
                                                      const int* S,
                                                      const int* O,
                                                      const int* K,
                                                      const int* stride,
                                                      const PoolOpts& opts,
                                                      int N,
                                                      Dtype dt) {
    (void)B;
    (void)C;
    using SE = ::mlx::core::ShapeElem;
    const auto mdt = gpu::to_mlx_dtype(dt);
    double uniform_div = 1.0;
    std::optional<::mlx::core::array> non_uniform_div;
    for (int i = 0; i < N; ++i) {
        std::vector<float> spans(static_cast<std::size_t>(O[i]));
        bool uniform = true;
        for (int o = 0; o < O[i]; ++o) {
            spans[static_cast<std::size_t>(o)] = static_cast<float>(
                gpu_avg_pool_span(o, S[i], K[i], stride[i], opts.pad[i], opts.count_include_pad));
            if (spans[static_cast<std::size_t>(o)] != static_cast<float>(K[i]))
                uniform = false;
        }
        if (uniform) {
            uniform_div *= static_cast<double>(K[i]);
            continue;
        }
        ::mlx::core::Shape bshape(static_cast<std::size_t>(N) + 2, 1);
        bshape[static_cast<std::size_t>(2 + i)] = static_cast<SE>(O[i]);
        ::mlx::core::array vec = ::mlx::core::reshape(
            ::mlx::core::array(spans.data(), {static_cast<SE>(O[i])}, ::mlx::core::float32),
            bshape);
        non_uniform_div =
            non_uniform_div ? ::mlx::core::multiply(*non_uniform_div, vec) : std::move(vec);
    }
    if (!non_uniform_div)
        return ::mlx::core::divide(y, ::mlx::core::array(uniform_div, mdt));
    if (uniform_div != 1.0) {
        non_uniform_div = ::mlx::core::multiply(
            *non_uniform_div, ::mlx::core::array(static_cast<float>(uniform_div)));
    }
    return ::mlx::core::divide(y, ::mlx::core::astype(*non_uniform_div, mdt));
}

int GpuBackend::gpu_avg_pool_span(int o, int S, int K, int stride, int pad, bool count_pad) {
    const int start = o * stride - pad;
    const int end = std::min(start + K, S + pad);
    if (count_pad)
        return end - start;
    const int lo = std::max(start, 0);
    const int hi = std::min(end, S);
    return hi > lo ? hi - lo : 0;
}

int GpuBackend::gpu_ceil_overhang(int S, int K, int stride, int pad, int O) {
    const int reach = (O - 1) * stride + K;
    const int have = S + 2 * pad;
    return reach > have ? reach - have : 0;
}

::mlx::core::array GpuBackend::gpu_build_window_view(const ::mlx::core::array& padded,
                                                     int B,
                                                     int C,
                                                     const int* Sp,
                                                     const int* O,
                                                     const int* K,
                                                     const int* stride,
                                                     int N) {
    using SE = ::mlx::core::ShapeElem;
    using SS = std::int64_t;
    ::mlx::core::Shape windowed;
    windowed.reserve(2 + 2 * N);
    windowed.push_back(B);
    windowed.push_back(C);
    for (int i = 0; i < N; ++i)
        windowed.push_back(static_cast<SE>(O[i]));
    for (int i = 0; i < N; ++i)
        windowed.push_back(static_cast<SE>(K[i]));
    SS suffix[4];
    suffix[N] = 1;
    for (int i = N - 1; i >= 0; --i)
        suffix[i] = suffix[i + 1] * static_cast<SS>(Sp[i]);
    ::mlx::core::Strides strides_v;
    strides_v.reserve(2 + 2 * N);
    strides_v.push_back(static_cast<SS>(C) * suffix[0]);
    strides_v.push_back(suffix[0]);
    for (int i = 0; i < N; ++i)
        strides_v.push_back(static_cast<SS>(stride[i]) * suffix[i + 1]);
    for (int i = 0; i < N; ++i)
        strides_v.push_back(suffix[i + 1]);
    return ::mlx::core::as_strided(padded, windowed, strides_v, 0);
}

Storage GpuBackend::interpolate_nearest_2d_forward(
    const Storage& input, const Shape& in_shape, int H_out, int W_out, Dtype dt) {
    return gpu::interp::resample(input, in_shape, {H_out, W_out}, gpu::interp::Filter::Nearest,
                                 false, dt, false);
}

Storage GpuBackend::interpolate_nearest_3d_forward(
    const Storage& input, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) {
    return gpu::interp::resample(input, in_shape, {D_out, H_out, W_out},
                                 gpu::interp::Filter::Nearest, false, dt, false);
}

Storage GpuBackend::interpolate_nearest_2d_backward(
    const Storage& grad_out, const Shape& in_shape, int H_out, int W_out, Dtype dt) {
    return gpu::interp::resample(grad_out, in_shape, {H_out, W_out}, gpu::interp::Filter::Nearest,
                                 false, dt, true);
}

Storage GpuBackend::interpolate_nearest_3d_backward(
    const Storage& grad_out, const Shape& in_shape, int D_out, int H_out, int W_out, Dtype dt) {
    return gpu::interp::resample(grad_out, in_shape, {D_out, H_out, W_out},
                                 gpu::interp::Filter::Nearest, false, dt, true);
}

Storage GpuBackend::interpolate_bilinear_forward(const Storage& input,
                                                 const Shape& in_shape,
                                                 int H_out,
                                                 int W_out,
                                                 bool align_corners,
                                                 Dtype dt) {
    return gpu::interp::resample(input, in_shape, {H_out, W_out}, gpu::interp::Filter::Linear,
                                 align_corners, dt, false);
}

Storage GpuBackend::interpolate_bilinear_backward(const Storage& grad_out,
                                                  const Shape& in_shape,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) {
    return gpu::interp::resample(grad_out, in_shape, {H_out, W_out}, gpu::interp::Filter::Linear,
                                 align_corners, dt, true);
}

Storage GpuBackend::interpolate_trilinear_forward(const Storage& input,
                                                  const Shape& in_shape,
                                                  int D_out,
                                                  int H_out,
                                                  int W_out,
                                                  bool align_corners,
                                                  Dtype dt) {
    return gpu::interp::resample(input, in_shape, {D_out, H_out, W_out},
                                 gpu::interp::Filter::Linear, align_corners, dt, false);
}

Storage GpuBackend::interpolate_trilinear_backward(const Storage& grad_out,
                                                   const Shape& in_shape,
                                                   int D_out,
                                                   int H_out,
                                                   int W_out,
                                                   bool align_corners,
                                                   Dtype dt) {
    return gpu::interp::resample(grad_out, in_shape, {D_out, H_out, W_out},
                                 gpu::interp::Filter::Linear, align_corners, dt, true);
}

Storage GpuBackend::affine_grid_forward(
    const Storage& theta, int N, int H, int W, bool align_corners, Dtype dt) {
    const auto& gt = std::get<GpuStorage>(theta);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);

    auto build_norm = [&](int dim) {
        if (align_corners) {
            if (dim <= 1)
                return ::mlx::core::zeros({1}, mlx_dt);
            auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
            auto step =
                ::mlx::core::astype(::mlx::core::array(2.0f / static_cast<float>(dim - 1)), mlx_dt);
            auto offset = ::mlx::core::astype(::mlx::core::array(-1.0f), mlx_dt);
            return ::mlx::core::add(::mlx::core::multiply(a, step), offset);
        }
        auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
        auto two = ::mlx::core::astype(::mlx::core::array(2.0f), mlx_dt);
        auto inv_dim =
            ::mlx::core::astype(::mlx::core::array(1.0f / static_cast<float>(dim)), mlx_dt);
        auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
        auto two_w_p1 = ::mlx::core::add(::mlx::core::multiply(two, a), one);
        return ::mlx::core::subtract(::mlx::core::multiply(two_w_p1, inv_dim), one);
    };
    auto xs = build_norm(W);
    auto ys = build_norm(H);
    auto xs_b = ::mlx::core::reshape(xs, {1, W, 1});
    auto ys_b = ::mlx::core::reshape(ys, {H, 1, 1});
    auto ones_b = ::mlx::core::ones({H, W, 1}, mlx_dt);
    auto xs_broad = ::mlx::core::broadcast_to(xs_b, {H, W, 1});
    auto ys_broad = ::mlx::core::broadcast_to(ys_b, {H, W, 1});
    auto grid_xy =
        ::mlx::core::concatenate(std::vector<::mlx::core::array>{xs_broad, ys_broad, ones_b}, -1);
    auto grid_flat = ::mlx::core::reshape(grid_xy, {1, H * W, 3});
    auto grid_b = ::mlx::core::broadcast_to(grid_flat, {N, H * W, 3});
    auto theta_T = ::mlx::core::transpose(*gt.arr, {0, 2, 1});
    auto out_flat = ::mlx::core::matmul(grid_b, theta_T);
    auto out_arr = ::mlx::core::reshape(out_flat, {N, H, W, 2});
    return Storage{gpu::wrap_mlx_array(std::move(out_arr), dt)};
}

Storage GpuBackend::affine_grid_backward(
    const Storage& grad_grid, int N, int H, int W, bool align_corners, Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad_grid);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);

    auto build_norm = [&](int dim) {
        if (align_corners) {
            if (dim <= 1)
                return ::mlx::core::zeros({1}, mlx_dt);
            auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
            auto step =
                ::mlx::core::astype(::mlx::core::array(2.0f / static_cast<float>(dim - 1)), mlx_dt);
            auto offset = ::mlx::core::astype(::mlx::core::array(-1.0f), mlx_dt);
            return ::mlx::core::add(::mlx::core::multiply(a, step), offset);
        }
        auto a = ::mlx::core::astype(::mlx::core::arange(0, dim, 1), mlx_dt);
        auto two = ::mlx::core::astype(::mlx::core::array(2.0f), mlx_dt);
        auto inv_dim =
            ::mlx::core::astype(::mlx::core::array(1.0f / static_cast<float>(dim)), mlx_dt);
        auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
        auto two_w_p1 = ::mlx::core::add(::mlx::core::multiply(two, a), one);
        return ::mlx::core::subtract(::mlx::core::multiply(two_w_p1, inv_dim), one);
    };
    auto xs = build_norm(W);
    auto ys = build_norm(H);
    auto xs_b = ::mlx::core::reshape(xs, {1, W, 1});
    auto ys_b = ::mlx::core::reshape(ys, {H, 1, 1});
    auto ones_b = ::mlx::core::ones({H, W, 1}, mlx_dt);
    auto xs_broad = ::mlx::core::broadcast_to(xs_b, {H, W, 1});
    auto ys_broad = ::mlx::core::broadcast_to(ys_b, {H, W, 1});
    auto grid_xy =
        ::mlx::core::concatenate(std::vector<::mlx::core::array>{xs_broad, ys_broad, ones_b}, -1);
    auto grid_flat = ::mlx::core::reshape(grid_xy, {H * W, 3});
    auto dg = ::mlx::core::reshape(*gg.arr, {N, H * W, 2});
    auto grid_T = ::mlx::core::transpose(grid_flat);
    auto grid_T_b =
        ::mlx::core::broadcast_to(::mlx::core::reshape(grid_T, {1, 3, H * W}), {N, 3, H * W});
    auto dtheta_T = ::mlx::core::matmul(grid_T_b, dg);
    auto dtheta = ::mlx::core::contiguous(::mlx::core::transpose(dtheta_T, {0, 2, 1}));
    return Storage{gpu::wrap_mlx_array(std::move(dtheta), dt)};
}

Storage GpuBackend::grid_sample_forward(const Storage& input,
                                        const Storage& grid,
                                        const Shape& in_shape,
                                        const Shape& grid_shape,
                                        int mode,
                                        int padding_mode,
                                        bool align_corners,
                                        Dtype dt) {
    const int N = static_cast<int>(in_shape[0]);
    const int C = static_cast<int>(in_shape[1]);
    const int H_in = static_cast<int>(in_shape[2]);
    const int W_in = static_cast<int>(in_shape[3]);
    const int H_out = static_cast<int>(grid_shape[1]);
    const int W_out = static_cast<int>(grid_shape[2]);
    const auto& gx = std::get<GpuStorage>(input);
    const auto& gg = std::get<GpuStorage>(grid);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);

    auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
    auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
    auto gx_n = ::mlx::core::take(*gg.arr, idx0, -1);
    auto gy_n = ::mlx::core::take(*gg.arr, idx1, -1);

    auto denorm_arr = [&](const ::mlx::core::array& g, int dim) {
        auto half = ::mlx::core::astype(::mlx::core::array(0.5f), mlx_dt);
        auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
        auto slope =
            gpu::mlx_scalar(resample::unnormalize_slope<double>(dim, align_corners), mlx_dt);
        auto scaled = ::mlx::core::multiply(::mlx::core::add(g, one), slope);
        return align_corners ? scaled : ::mlx::core::subtract(scaled, half);
    };
    auto ix = denorm_arr(gx_n, W_in);
    auto iy = denorm_arr(gy_n, H_in);

    auto gather_corner = [&](const ::mlx::core::array& y_arr, const ::mlx::core::array& x_arr,
                             bool zero_oob) {
        auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto Hm1 = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
        auto Wm1 = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
        auto in_y = ::mlx::core::logical_and(::mlx::core::greater_equal(y_arr, zero_i),
                                             ::mlx::core::less_equal(y_arr, Hm1));
        auto in_x = ::mlx::core::logical_and(::mlx::core::greater_equal(x_arr, zero_i),
                                             ::mlx::core::less_equal(x_arr, Wm1));
        auto in_bounds = ::mlx::core::logical_and(in_y, in_x);
        auto y_cl = ::mlx::core::clip(y_arr, std::optional<::mlx::core::array>(zero_i),
                                      std::optional<::mlx::core::array>(Hm1));
        auto x_cl = ::mlx::core::clip(x_arr, std::optional<::mlx::core::array>(zero_i),
                                      std::optional<::mlx::core::array>(Wm1));
        auto n_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64), {N, 1, 1, 1});
        auto c_idx = ::mlx::core::reshape(
            ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64), {1, C, 1, 1});
        auto y_b = ::mlx::core::reshape(y_cl, {N, 1, H_out, W_out});
        auto x_b = ::mlx::core::reshape(x_cl, {N, 1, H_out, W_out});
        auto stride_n =
            ::mlx::core::astype(::mlx::core::array(C * H_in * W_in), ::mlx::core::int64);
        auto stride_c = ::mlx::core::astype(::mlx::core::array(H_in * W_in), ::mlx::core::int64);
        auto stride_y = ::mlx::core::astype(::mlx::core::array(W_in), ::mlx::core::int64);
        auto flat_idx =
            ::mlx::core::add(::mlx::core::add(::mlx::core::multiply(n_idx, stride_n),
                                              ::mlx::core::multiply(c_idx, stride_c)),
                             ::mlx::core::add(::mlx::core::multiply(y_b, stride_y), x_b));
        auto flat_idx_b = ::mlx::core::broadcast_to(flat_idx, {N, C, H_out, W_out});
        auto in_flat = ::mlx::core::reshape(*gx.arr, {N * C * H_in * W_in});
        auto vals = ::mlx::core::take(in_flat, flat_idx_b);
        if (zero_oob) {
            // Select, not multiply.  The gather above clamps the index,
            // so an out-of-bounds tap still fetches a real pixel, and
            // scaling it by a 0.0 mask leaves ``value * 0`` — which is
            // 0 for finite pixels and NaN for an infinite one.  The CPU
            // path substitutes a hard zero for these taps, so the two
            // devices disagreed on any image containing an infinity:
            // a single-pixel ``inf`` sampled exactly on itself came back
            // ``inf`` on CPU and ``NaN`` on Metal, because its
            // zero-weight neighbours poisoned the sum.
            auto in_b_b = ::mlx::core::broadcast_to(
                ::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}), {N, C, H_out, W_out});
            auto zero_v = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
            vals = ::mlx::core::where(in_b_b, vals, zero_v);
        }
        return vals;
    };

    // Positions become indices, and step to a neighbour, by the
    // saturating rule in ``gpu/Conversion.h`` (LCD-420).
    auto round_int = [&](const ::mlx::core::array& a) {
        return gpu_index_of(::mlx::core::round(a, 0));
    };

    ::mlx::core::array y_out{0};
    if (mode == 1) {
        ::mlx::core::array ix_use = ix;
        ::mlx::core::array iy_use = iy;
        if (padding_mode == 1) {
            auto zf = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
            auto Hm1f =
                ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
            auto Wm1f =
                ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);
            ix_use = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zf),
                                       std::optional<::mlx::core::array>(Wm1f));
            iy_use = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zf),
                                       std::optional<::mlx::core::array>(Hm1f));
        }
        auto ix_r = round_int(ix_use);
        auto iy_r = round_int(iy_use);
        y_out = gather_corner(iy_r, ix_r, padding_mode == 0);
    } else {
        ::mlx::core::array ix_use = ix;
        ::mlx::core::array iy_use = iy;
        if (padding_mode == 1) {
            auto zf = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
            auto Hm1f =
                ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
            auto Wm1f =
                ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);
            ix_use = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zf),
                                       std::optional<::mlx::core::array>(Wm1f));
            iy_use = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zf),
                                       std::optional<::mlx::core::array>(Hm1f));
        }
        // Weights from the floored positions, not the saturated
        // indices cast back: two far axes made 0 * inf = NaN.
        auto x0_f = ::mlx::core::floor(ix_use), y0_f = ::mlx::core::floor(iy_use);
        auto x1_f = ::mlx::core::add(x0_f, ::mlx::core::array(1.0f, mlx_dt));
        auto y1_f = ::mlx::core::add(y0_f, ::mlx::core::array(1.0f, mlx_dt));
        auto x0 = gpu_index_of(x0_f), y0 = gpu_index_of(y0_f);
        auto x1 = gpu_next_index(x0), y1 = gpu_next_index(y0);
        auto wa = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix_use),
                                        ::mlx::core::subtract(y1_f, iy_use));
        auto wb = ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix_use),
                                        ::mlx::core::subtract(iy_use, y0_f));
        auto wc = ::mlx::core::multiply(::mlx::core::subtract(ix_use, x0_f),
                                        ::mlx::core::subtract(y1_f, iy_use));
        auto wd = ::mlx::core::multiply(::mlx::core::subtract(ix_use, x0_f),
                                        ::mlx::core::subtract(iy_use, y0_f));
        auto wa_b = ::mlx::core::reshape(wa, {N, 1, H_out, W_out});
        auto wb_b = ::mlx::core::reshape(wb, {N, 1, H_out, W_out});
        auto wc_b = ::mlx::core::reshape(wc, {N, 1, H_out, W_out});
        auto wd_b = ::mlx::core::reshape(wd, {N, 1, H_out, W_out});
        auto Ia = gather_corner(y0, x0, padding_mode == 0);
        auto Ib = gather_corner(y1, x0, padding_mode == 0);
        auto Ic = gather_corner(y0, x1, padding_mode == 0);
        auto Id = gather_corner(y1, x1, padding_mode == 0);
        y_out = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(Ia, wa_b), ::mlx::core::multiply(Ib, wb_b)),
            ::mlx::core::add(::mlx::core::multiply(Ic, wc_b), ::mlx::core::multiply(Id, wd_b)));
    }
    return Storage{gpu::wrap_mlx_array(std::move(y_out), dt)};
}

std::vector<Storage> GpuBackend::grid_sample_backward(const Storage& grad_out,
                                                      const Storage& input,
                                                      const Storage& grid,
                                                      const Shape& in_shape,
                                                      const Shape& grid_shape,
                                                      int mode,
                                                      int padding_mode,
                                                      bool align_corners,
                                                      Dtype dt) {
    const int N = static_cast<int>(in_shape[0]);
    const int C = static_cast<int>(in_shape[1]);
    const int H_in = static_cast<int>(in_shape[2]);
    const int W_in = static_cast<int>(in_shape[3]);
    const int H_out = static_cast<int>(grid_shape[1]);
    const int W_out = static_cast<int>(grid_shape[2]);
    const auto& gx_arr = std::get<GpuStorage>(input);
    const auto& gg_arr = std::get<GpuStorage>(grid);
    const auto& go_arr = std::get<GpuStorage>(grad_out);
    const auto mlx_dt = gpu::to_mlx_dtype(dt);

    auto idx0 = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
    auto idx1 = ::mlx::core::astype(::mlx::core::array(1), ::mlx::core::int64);
    auto gx_n = ::mlx::core::take(*gg_arr.arr, idx0, -1);
    auto gy_n = ::mlx::core::take(*gg_arr.arr, idx1, -1);

    auto half = ::mlx::core::astype(::mlx::core::array(0.5f), mlx_dt);
    auto one = ::mlx::core::astype(::mlx::core::array(1.0f), mlx_dt);
    auto zero = ::mlx::core::astype(::mlx::core::array(0.0f), mlx_dt);
    auto Hm1f = ::mlx::core::astype(::mlx::core::array(static_cast<float>(H_in - 1)), mlx_dt);
    auto Wm1f = ::mlx::core::astype(::mlx::core::array(static_cast<float>(W_in - 1)), mlx_dt);

    auto denorm_arr = [&](const ::mlx::core::array& g, int dim) {
        auto slope =
            gpu::mlx_scalar(resample::unnormalize_slope<double>(dim, align_corners), mlx_dt);
        auto scaled = ::mlx::core::multiply(::mlx::core::add(g, one), slope);
        return align_corners ? scaled : ::mlx::core::subtract(scaled, half);
    };
    auto ix = denorm_arr(gx_n, W_in);
    auto iy = denorm_arr(gy_n, H_in);

    ::mlx::core::array clipped_x =
        ::mlx::core::astype(::mlx::core::array(false), ::mlx::core::bool_);
    ::mlx::core::array clipped_y = clipped_x;
    clipped_x = ::mlx::core::broadcast_to(clipped_x, gx_n.shape());
    clipped_y = ::mlx::core::broadcast_to(clipped_y, gy_n.shape());
    if (mode == 0 && padding_mode == 1) {
        // A position on the edge counts as clipped, as in the reference.
        clipped_x = ::mlx::core::logical_or(::mlx::core::less_equal(ix, zero),
                                            ::mlx::core::greater_equal(ix, Wm1f));
        clipped_y = ::mlx::core::logical_or(::mlx::core::less_equal(iy, zero),
                                            ::mlx::core::greater_equal(iy, Hm1f));
        ix = ::mlx::core::clip(ix, std::optional<::mlx::core::array>(zero),
                               std::optional<::mlx::core::array>(Wm1f));
        iy = ::mlx::core::clip(iy, std::optional<::mlx::core::array>(zero),
                               std::optional<::mlx::core::array>(Hm1f));
    }

    const std::int64_t total_in = static_cast<std::int64_t>(N) * C * H_in * W_in;
    auto in_flat = ::mlx::core::reshape(*gx_arr.arr, {static_cast<int>(total_in)});

    auto sn = ::mlx::core::astype(::mlx::core::array(C * H_in * W_in), ::mlx::core::int64);
    auto sc = ::mlx::core::astype(::mlx::core::array(H_in * W_in), ::mlx::core::int64);
    auto sy = ::mlx::core::astype(::mlx::core::array(W_in), ::mlx::core::int64);
    auto n_idx = ::mlx::core::reshape(
        ::mlx::core::astype(::mlx::core::arange(0, N, 1), ::mlx::core::int64), {N, 1, 1, 1});
    auto c_idx = ::mlx::core::reshape(
        ::mlx::core::astype(::mlx::core::arange(0, C, 1), ::mlx::core::int64), {1, C, 1, 1});

    auto build_flat_idx = [&](const ::mlx::core::array& yi, const ::mlx::core::array& xi) {
        auto y_b = ::mlx::core::reshape(yi, {N, 1, H_out, W_out});
        auto x_b = ::mlx::core::reshape(xi, {N, 1, H_out, W_out});
        auto flat = ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(n_idx, sn), ::mlx::core::multiply(c_idx, sc)),
            ::mlx::core::add(::mlx::core::multiply(y_b, sy), x_b));
        return ::mlx::core::broadcast_to(flat, {N, C, H_out, W_out});
    };
    auto build_in_bounds_mask = [&](const ::mlx::core::array& yi_pre_clip,
                                    const ::mlx::core::array& xi_pre_clip) {
        auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
        auto in_y = ::mlx::core::logical_and(::mlx::core::greater_equal(yi_pre_clip, zero_i),
                                             ::mlx::core::less_equal(yi_pre_clip, Hm1_i));
        auto in_x = ::mlx::core::logical_and(::mlx::core::greater_equal(xi_pre_clip, zero_i),
                                             ::mlx::core::less_equal(xi_pre_clip, Wm1_i));
        return ::mlx::core::logical_and(in_y, in_x);
    };

    auto dinput_flat = ::mlx::core::zeros({static_cast<int>(total_in)}, accumulation_dtype(mlx_dt));
    ::mlx::core::array dgrid_x = zero;
    ::mlx::core::array dgrid_y = zero;
    dgrid_x = ::mlx::core::broadcast_to(dgrid_x, gx_n.shape());
    dgrid_y = ::mlx::core::broadcast_to(dgrid_y, gy_n.shape());

    if (mode == 1) {
        auto ixr_f = ::mlx::core::round(ix, 0);
        auto iyr_f = ::mlx::core::round(iy, 0);
        auto ixr = gpu_index_of(ixr_f);
        auto iyr = gpu_index_of(iyr_f);
        auto in_bounds = build_in_bounds_mask(iyr, ixr);
        auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
        ixr = ::mlx::core::clip(ixr, std::optional<::mlx::core::array>(zero_i),
                                std::optional<::mlx::core::array>(Wm1_i));
        iyr = ::mlx::core::clip(iyr, std::optional<::mlx::core::array>(zero_i),
                                std::optional<::mlx::core::array>(Hm1_i));
        auto flat_idx = build_flat_idx(iyr, ixr);
        auto contrib = *go_arr.arr;
        if (padding_mode == 0)
            contrib = ::mlx::core::where(::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}),
                                         contrib, zero);
        auto flat_idx_1d = ::mlx::core::reshape(flat_idx, {N * C * H_out * W_out});
        auto contrib_2d = ::mlx::core::reshape(contrib, {N * C * H_out * W_out, 1});
        dinput_flat = gpu_scatter_add(dinput_flat, flat_idx_1d, widen_half(contrib_2d), 0);
    } else {
        // As in the forward: weights from the floored positions.
        auto x0_f = ::mlx::core::floor(ix), y0_f = ::mlx::core::floor(iy);
        auto x1_f = ::mlx::core::add(x0_f, one), y1_f = ::mlx::core::add(y0_f, one);
        auto x0 = gpu_index_of(x0_f), y0 = gpu_index_of(y0_f);
        auto x1 = gpu_next_index(x0), y1 = gpu_next_index(y0);
        auto wa =
            ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix), ::mlx::core::subtract(y1_f, iy));
        auto wb =
            ::mlx::core::multiply(::mlx::core::subtract(x1_f, ix), ::mlx::core::subtract(iy, y0_f));
        auto wc =
            ::mlx::core::multiply(::mlx::core::subtract(ix, x0_f), ::mlx::core::subtract(y1_f, iy));
        auto wd =
            ::mlx::core::multiply(::mlx::core::subtract(ix, x0_f), ::mlx::core::subtract(iy, y0_f));

        auto zero_i = ::mlx::core::astype(::mlx::core::array(0), ::mlx::core::int64);
        auto Hm1_i = ::mlx::core::astype(::mlx::core::array(H_in - 1), ::mlx::core::int64);
        auto Wm1_i = ::mlx::core::astype(::mlx::core::array(W_in - 1), ::mlx::core::int64);
        auto clip_y = [&](const ::mlx::core::array& v) {
            return ::mlx::core::clip(v, std::optional<::mlx::core::array>(zero_i),
                                     std::optional<::mlx::core::array>(Hm1_i));
        };
        auto clip_x = [&](const ::mlx::core::array& v) {
            return ::mlx::core::clip(v, std::optional<::mlx::core::array>(zero_i),
                                     std::optional<::mlx::core::array>(Wm1_i));
        };
        auto y0c = clip_y(y0), y1c = clip_y(y1);
        auto x0c = clip_x(x0), x1c = clip_x(x1);
        auto in00 = build_in_bounds_mask(y0, x0);
        auto in01 = build_in_bounds_mask(y0, x1);
        auto in10 = build_in_bounds_mask(y1, x0);
        auto in11 = build_in_bounds_mask(y1, x1);

        auto scatter_one_corner = [&](const ::mlx::core::array& yi, const ::mlx::core::array& xi,
                                      const ::mlx::core::array& w,
                                      const ::mlx::core::array& in_bounds) {
            auto flat_idx = build_flat_idx(yi, xi);
            auto w_b = ::mlx::core::broadcast_to(::mlx::core::reshape(w, {N, 1, H_out, W_out}),
                                                 {N, C, H_out, W_out});
            ::mlx::core::array contrib = ::mlx::core::multiply(*go_arr.arr, w_b);
            // Select, not multiply: a NaN weight times a 0 mask is NaN.
            if (padding_mode == 0)
                contrib = ::mlx::core::where(::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}),
                                             contrib, zero);
            auto flat_idx_1d = ::mlx::core::reshape(flat_idx, {N * C * H_out * W_out});
            auto contrib_2d = ::mlx::core::reshape(contrib, {N * C * H_out * W_out, 1});
            dinput_flat = gpu_scatter_add(dinput_flat, flat_idx_1d, widen_half(contrib_2d), 0);
        };
        scatter_one_corner(y0c, x0c, wa, in00);
        scatter_one_corner(y1c, x0c, wb, in10);
        scatter_one_corner(y0c, x1c, wc, in01);
        scatter_one_corner(y1c, x1c, wd, in11);

        auto fetch_corner = [&](const ::mlx::core::array& yi, const ::mlx::core::array& xi,
                                const ::mlx::core::array& in_bounds) {
            auto flat_idx = build_flat_idx(yi, xi);
            auto vals = ::mlx::core::take(in_flat, flat_idx);
            if (padding_mode == 0)
                vals = ::mlx::core::where(::mlx::core::reshape(in_bounds, {N, 1, H_out, W_out}),
                                          vals, zero);
            return vals;
        };
        auto Ia = fetch_corner(y0c, x0c, in00);
        auto Ib = fetch_corner(y1c, x0c, in10);
        auto Ic = fetch_corner(y0c, x1c, in01);
        auto Id = fetch_corner(y1c, x1c, in11);

        auto dy_t1 = ::mlx::core::subtract(y1_f, iy);
        auto dy_t2 = ::mlx::core::subtract(iy, y0_f);
        auto dx_t1 = ::mlx::core::subtract(x1_f, ix);
        auto dx_t2 = ::mlx::core::subtract(ix, x0_f);
        auto dy_t1_b = ::mlx::core::broadcast_to(::mlx::core::reshape(dy_t1, {N, 1, H_out, W_out}),
                                                 {N, C, H_out, W_out});
        auto dy_t2_b = ::mlx::core::broadcast_to(::mlx::core::reshape(dy_t2, {N, 1, H_out, W_out}),
                                                 {N, C, H_out, W_out});
        auto dx_t1_b = ::mlx::core::broadcast_to(::mlx::core::reshape(dx_t1, {N, 1, H_out, W_out}),
                                                 {N, C, H_out, W_out});
        auto dx_t2_b = ::mlx::core::broadcast_to(::mlx::core::reshape(dx_t2, {N, 1, H_out, W_out}),
                                                 {N, C, H_out, W_out});
        auto dix_per = ::mlx::core::multiply(
            *go_arr.arr,
            ::mlx::core::add(::mlx::core::multiply(::mlx::core::subtract(Ic, Ia), dy_t1_b),
                             ::mlx::core::multiply(::mlx::core::subtract(Id, Ib), dy_t2_b)));
        auto diy_per = ::mlx::core::multiply(
            *go_arr.arr,
            ::mlx::core::add(::mlx::core::multiply(::mlx::core::subtract(Ib, Ia), dx_t1_b),
                             ::mlx::core::multiply(::mlx::core::subtract(Id, Ic), dx_t2_b)));
        auto dix_sum = ::mlx::core::sum(dix_per, std::vector<int>{1}, false);
        auto diy_sum = ::mlx::core::sum(diy_per, std::vector<int>{1}, false);
        auto sx = gpu::mlx_scalar(resample::unnormalize_slope<double>(W_in, align_corners), mlx_dt);
        auto sy_arr =
            gpu::mlx_scalar(resample::unnormalize_slope<double>(H_in, align_corners), mlx_dt);
        dgrid_x = ::mlx::core::multiply(
            dix_sum, padding_mode == 1 ? ::mlx::core::where(clipped_x, zero, sx) : sx);
        dgrid_y = ::mlx::core::multiply(
            diy_sum, padding_mode == 1 ? ::mlx::core::where(clipped_y, zero, sy_arr) : sy_arr);
    }

    auto dx_arr =
        ::mlx::core::reshape(narrow_half(dinput_flat, mlx_dt), gpu::to_mlx_shape(in_shape));
    auto dgx_e = ::mlx::core::expand_dims(dgrid_x, -1);
    auto dgy_e = ::mlx::core::expand_dims(dgrid_y, -1);
    auto dg_arr = ::mlx::core::concatenate(std::vector<::mlx::core::array>{dgx_e, dgy_e}, -1);
    return {Storage{gpu::wrap_mlx_array(std::move(dx_arr), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dg_arr), dt)}};
}

Storage GpuBackend::rotate_forward(const Storage& input,
                                   const Shape& shape,
                                   double angle_rad_neg,
                                   double cx,
                                   double cy,
                                   Dtype dt) {
    const auto& gx = std::get<GpuStorage>(input);
    const int N = static_cast<int>(shape[0]);
    const int C = static_cast<int>(shape[1]);
    const int H = static_cast<int>(shape[2]);
    const int W = static_cast<int>(shape[3]);
    auto mlx_dt = gpu::to_mlx_dtype(dt);
    const float cf = static_cast<float>(std::cos(angle_rad_neg));
    const float sf = static_cast<float>(std::sin(angle_rad_neg));
    const float cxf = static_cast<float>(cx);
    const float cyf = static_cast<float>(cy);

    auto y_arr = ::mlx::core::astype(::mlx::core::arange(0, H, 1), mlx_dt);
    auto x_arr = ::mlx::core::astype(::mlx::core::arange(0, W, 1), mlx_dt);
    auto y_2d = ::mlx::core::reshape(y_arr, {H, 1});
    auto x_2d = ::mlx::core::reshape(x_arr, {1, W});

    auto xs_d = ::mlx::core::add(
        ::mlx::core::add(
            ::mlx::core::multiply(::mlx::core::array(cf, mlx_dt),
                                  ::mlx::core::subtract(x_2d, ::mlx::core::array(cxf, mlx_dt))),
            ::mlx::core::multiply(::mlx::core::array(-sf, mlx_dt),
                                  ::mlx::core::subtract(y_2d, ::mlx::core::array(cyf, mlx_dt)))),
        ::mlx::core::array(cxf, mlx_dt));
    auto ys_d = ::mlx::core::add(
        ::mlx::core::add(
            ::mlx::core::multiply(::mlx::core::array(sf, mlx_dt),
                                  ::mlx::core::subtract(x_2d, ::mlx::core::array(cxf, mlx_dt))),
            ::mlx::core::multiply(::mlx::core::array(cf, mlx_dt),
                                  ::mlx::core::subtract(y_2d, ::mlx::core::array(cyf, mlx_dt)))),
        ::mlx::core::array(cyf, mlx_dt));
    const auto idt = ::mlx::core::int32;
    auto x_s = ::mlx::core::astype(
        ::mlx::core::floor(::mlx::core::add(xs_d, ::mlx::core::array(0.5f, mlx_dt))), idt);
    auto y_s = ::mlx::core::astype(
        ::mlx::core::floor(::mlx::core::add(ys_d, ::mlx::core::array(0.5f, mlx_dt))), idt);
    auto in_range = ::mlx::core::logical_and(
        ::mlx::core::logical_and(::mlx::core::greater_equal(x_s, ::mlx::core::array(0, idt)),
                                 ::mlx::core::less(x_s, ::mlx::core::array(W, idt))),
        ::mlx::core::logical_and(::mlx::core::greater_equal(y_s, ::mlx::core::array(0, idt)),
                                 ::mlx::core::less(y_s, ::mlx::core::array(H, idt))));
    auto x_safe =
        ::mlx::core::clip(x_s, ::mlx::core::array(0, idt), ::mlx::core::array(W - 1, idt));
    auto y_safe =
        ::mlx::core::clip(y_s, ::mlx::core::array(0, idt), ::mlx::core::array(H - 1, idt));

    auto flat = ::mlx::core::add(::mlx::core::multiply(y_safe, ::mlx::core::array(W, idt)), x_safe);

    auto sN = ::mlx::core::array(C * H * W, idt);
    auto sC = ::mlx::core::array(H * W, idt);
    auto sH = ::mlx::core::array(W, idt);
    auto n_idx =
        ::mlx::core::astype(::mlx::core::reshape(::mlx::core::arange(0, N, 1), {N, 1, 1, 1}), idt);
    auto c_idx =
        ::mlx::core::astype(::mlx::core::reshape(::mlx::core::arange(0, C, 1), {1, C, 1, 1}), idt);
    auto flat_hw = ::mlx::core::reshape(flat, {1, 1, H, W});
    auto flat_b = ::mlx::core::broadcast_to(
        ::mlx::core::add(
            ::mlx::core::add(::mlx::core::multiply(n_idx, sN), ::mlx::core::multiply(c_idx, sC)),
            flat_hw),
        ::mlx::core::Shape{N, C, H, W});
    auto in_flat = ::mlx::core::reshape(*gx.arr, {N * C * H * W});
    auto sampled = ::mlx::core::take(in_flat, flat_b);
    auto in_range_4d = ::mlx::core::astype(
        ::mlx::core::broadcast_to(::mlx::core::reshape(in_range, {1, 1, H, W}), {N, C, H, W}),
        mlx_dt);
    auto out = ::mlx::core::multiply(sampled, in_range_4d);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

}  // namespace backend
}  // namespace lucid
