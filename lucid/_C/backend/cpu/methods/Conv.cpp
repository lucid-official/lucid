// lucid/_C/backend/cpu/methods/Conv.cpp
// Out-of-line CpuBackend method definitions for the Conv family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::nn_fold(const Storage& x,
                            const Shape& x_shape,
                            const Shape& out_shape,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& stride,
                            const std::vector<int>& padding,
                            const std::vector<int>& dilation,
                            Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(nn_fold(detail::as_f32(x), x_shape, out_shape, kernel_size,
                                           stride, padding, dilation, Dtype::F32),
                                   dt);
    // Integers take the same road, through int64.  Selection, movement
    // and addition are all exact there, so this is the answer a native
    // kernel would have produced rather than an approximation of it.
    if (detail::is_narrow_int(dt))
        return detail::back_to_int(nn_fold(detail::as_i64(x), x_shape, out_shape, kernel_size,
                                           stride, padding, dilation, Dtype::I64),
                                   dt);
    const auto& cx = std::get<CpuStorage>(x);
    const int N = static_cast<int>(x_shape[0]);
    const int CKK = static_cast<int>(x_shape[1]);
    const int L = static_cast<int>(x_shape[2]);
    const int kH = kernel_size[0], kW = kernel_size[1];
    const int sH = stride[0], sW = stride[1];
    const int pH = padding[0], pW = padding[1];
    const int dH = dilation[0], dW = dilation[1];
    const int C = CKK / (kH * kW);
    const int outH = static_cast<int>(out_shape[2]);
    const int outW = static_cast<int>(out_shape[3]);

    // Number of output positions
    const int H_pad = outH + 2 * pH;
    const int W_pad = outW + 2 * pW;
    const std::size_t total_out = static_cast<std::size_t>(N) * C * outH * outW;
    auto out_ptr = allocate_aligned_bytes(total_out * dtype_size(dt), Device::CPU);
    std::memset(out_ptr.get(), 0, total_out * dtype_size(dt));

    // Block counts must use the *effective* (dilated) kernel extent.  Using
    // the raw kernel size overcounts blocks whenever dilation > 1, which
    // both mis-maps l_idx and reads past the end of the column buffer.
    const int effH = dH * (kH - 1) + 1;
    const int effW = dW * (kW - 1) + 1;
    auto run = [&](auto* op, const auto* xp) {
        int l_idx = 0;
        for (int oh = 0; oh < (H_pad - effH) / sH + 1; ++oh) {
            for (int ow = 0; ow < (W_pad - effW) / sW + 1; ++ow) {
                for (int n = 0; n < N; ++n) {
                    for (int c = 0; c < C; ++c) {
                        for (int ki = 0; ki < kH; ++ki) {
                            for (int kj = 0; kj < kW; ++kj) {
                                const int ri = oh * sH + ki * dH - pH;
                                const int ci = ow * sW + kj * dW - pW;
                                if (ri < 0 || ri >= outH || ci < 0 || ci >= outW)
                                    continue;
                                const int src_c = c * kH * kW + ki * kW + kj;
                                const int src_flat = n * CKK * L + src_c * L + l_idx;
                                const int dst_flat =
                                    n * C * outH * outW + c * outH * outW + ri * outW + ci;
                                op[dst_flat] += xp[src_flat];
                            }
                        }
                    }
                }
                ++l_idx;
            }
        }
    };

    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(out_ptr.get()), reinterpret_cast<const float*>(cx.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(out_ptr.get()),
            reinterpret_cast<const double*>(cx.ptr.get()));
    else if (dt == Dtype::I64)
        run(reinterpret_cast<std::int64_t*>(out_ptr.get()),
            reinterpret_cast<const std::int64_t*>(cx.ptr.get()));
    else
        ErrorBuilder("nn_fold").not_implemented("only F32/F64");
    return Storage{CpuStorage{out_ptr, total_out * dtype_size(dt), dt}};
}

Storage CpuBackend::conv_transpose_nd_forward(const Storage& x,
                                              const Storage& W,
                                              const Storage& b,
                                              int B,
                                              int Cin,
                                              int Cout,
                                              const int* S,
                                              const int* K,
                                              const int* O,
                                              const IBackend::ConvTransposeNdOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(conv_transpose_nd_forward(detail::as_f32(x), detail::as_f32(W),
                                                             detail::as_f32(b), B, Cin, Cout, S, K,
                                                             O, opts, out_shape, Dtype::F32),
                                   dt);
    const int N = opts.N;
    const int G = opts.groups;
    int S_total = 1, K_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S_total *= S[i];
        K_total *= K[i];
        O_total *= O[i];
    }
    // Grouping splits both channel axes.  The transposed weight is
    // ``(Cin, Cout / G, K...)``, so its *rows* carry the group (row
    // ``ci`` belongs to group ``ci / Cin_g``) and its row length is
    // already the per-group column height — unlike a forward conv,
    // where the divided axis is the second one.
    const int Cin_g = Cin / G;
    const int Cout_g = Cout / G;
    const int K_flat_g = Cout_g * K_total;

    CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cout * O_total, dt);
    CpuStorage cols_cpu = alloc_cpu(static_cast<std::size_t>(K_flat_g) * S_total, dt);
    if (out_cpu.nbytes)
        std::memset(out_cpu.ptr.get(), 0, out_cpu.nbytes);

    const auto& x_cpu = std::get<CpuStorage>(x);
    const auto& W_cpu = std::get<CpuStorage>(W);
    const auto& b_cpu = std::get<CpuStorage>(b);

    // One body for F32 and F64: the two used to be copy-pasted, which is
    // how a fix to one silently missed the other.
    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* xp0 = reinterpret_cast<const T*>(x_cpu.ptr.get());
        const T* Wp0 = reinterpret_cast<const T*>(W_cpu.ptr.get());
        const T* bp = reinterpret_cast<const T*>(b_cpu.ptr.get());
        T* cp = reinterpret_cast<T*>(cols_cpu.ptr.get());
        T* yp0 = reinterpret_cast<T*>(out_cpu.ptr.get());

        for (int bi = 0; bi < B; ++bi) {
            const T* xp = xp0 + static_cast<std::size_t>(bi) * Cin * S_total;
            T* yp = yp0 + static_cast<std::size_t>(bi) * Cout * O_total;
            for (int g = 0; g < G; ++g) {
                const T* Wg = Wp0 + static_cast<std::size_t>(g) * Cin_g * K_flat_g;
                const T* xg = xp + static_cast<std::size_t>(g) * Cin_g * S_total;
                T* yg = yp + static_cast<std::size_t>(g) * Cout_g * O_total;
                ctnd_gemm(true, false, K_flat_g, S_total, Cin_g, T(1), Wg, K_flat_g, xg, S_total,
                          T(0), cp, S_total);
                ctnd_col2im(cp, yg, Cout_g, O, K, S, opts.stride, opts.pad, opts.dilation, N);
            }
            for (int c = 0; c < Cout; ++c) {
                T* row = yp + static_cast<std::size_t>(c) * O_total;
                const T bv = bp[c];
                for (int i = 0; i < O_total; ++i)
                    row[i] += bv;
            }
        }
    };

    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::conv_transpose_nd_forward")
            .not_implemented("dtype not supported (F32/F64)");

    return Storage{std::move(out_cpu)};
}

std::vector<Storage>
CpuBackend::conv_transpose_nd_backward(const Storage& grad_out,
                                       const Storage& x,
                                       const Storage& W,
                                       int B,
                                       int Cin,
                                       int Cout,
                                       const int* S,
                                       const int* K,
                                       const int* O,
                                       const IBackend::ConvTransposeNdOpts& opts,
                                       Dtype dt,
                                       GradMask3 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            conv_transpose_nd_backward(detail::as_f32(grad_out),
                                       detail::as_f32_if(grad_mask.want(1), x),
                                       detail::as_f32_if(grad_mask.want(0), W), B, Cin, Cout, S, K,
                                       O, opts, Dtype::F32, grad_mask),
            dt);
    const int N = opts.N;
    const int G = opts.groups;
    int S_total = 1, K_total = 1, O_total = 1;
    for (int i = 0; i < N; ++i) {
        S_total *= S[i];
        K_total *= K[i];
        O_total *= O[i];
    }
    const int Cin_g = Cin / G;
    const int Cout_g = Cout / G;
    const int K_flat_g = Cout_g * K_total;

    cpu::GradBuffers<3> grads(grad_mask, dt,
                              {static_cast<std::size_t>(B) * Cin * S_total,
                               static_cast<std::size_t>(Cin) * K_flat_g,
                               static_cast<std::size_t>(Cout)});
    grads.zero();
    CpuStorage cols_cpu = grad_mask.want_any(0, 1)
                              ? alloc_cpu(static_cast<std::size_t>(K_flat_g) * S_total, dt)
                              : CpuStorage{};
    const CpuStorage* x_cpu = cpu::cpu_operand(grad_mask.want(1), x);
    const CpuStorage* W_cpu = cpu::cpu_operand(grad_mask.want(0), W);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* xp0 = cpu::operand_data<T>(x_cpu);
        const T* Wp0 = cpu::operand_data<T>(W_cpu);
        const T* gp0 = reinterpret_cast<const T*>(g_cpu.ptr.get());
        T* cp = reinterpret_cast<T*>(cols_cpu.ptr.get());
        for (int bi = 0; bi < B; ++bi) {
            const std::size_t x_off = static_cast<std::size_t>(bi) * Cin * S_total;
            const T* xp = offset_or_null(xp0, x_off);
            const T* gp = gp0 + static_cast<std::size_t>(bi) * Cout * O_total;
            T* dxp = offset_or_null(grads.data<T>(0), x_off);
            if (grad_mask.want_any(0, 1)) {
                for (int g = 0; g < G; ++g) {
                    const T* gg = gp + static_cast<std::size_t>(g) * Cout_g * O_total;
                    ctnd_im2col(gg, cp, Cout_g, O, K, S, opts.stride, opts.pad, opts.dilation, N);
                    if (grad_mask.want(0)) {
                        const T* Wg = Wp0 + static_cast<std::size_t>(g) * Cin_g * K_flat_g;
                        T* dxg = dxp + static_cast<std::size_t>(g) * Cin_g * S_total;
                        ctnd_gemm(false, false, Cin_g, S_total, K_flat_g, T(1), Wg, K_flat_g, cp,
                                  S_total, T(0), dxg, S_total);
                    }
                    if (grad_mask.want(1)) {
                        const T* xg = xp + static_cast<std::size_t>(g) * Cin_g * S_total;
                        T* dWg = grads.data<T>(1) + static_cast<std::size_t>(g) * Cin_g * K_flat_g;
                        ctnd_gemm(false, true, Cin_g, K_flat_g, S_total, T(1), xg, S_total, cp,
                                  S_total, T(1), dWg, K_flat_g);
                    }
                }
            }
            if (grad_mask.want(2))
                cpu::add_row_sums(gp, grads.data<T>(2), Cout, O_total);
        }
    };

    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::conv_transpose_nd_backward")
            .not_implemented("dtype not supported (F32/F64)");

    return grads.take_all();
}

Storage CpuBackend::conv_nd_forward(const Storage& x,
                                    const Storage& W,
                                    const Storage& b,
                                    int B,
                                    int Cin,
                                    int Cout,
                                    int Cin_g,
                                    int Cout_g,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const IBackend::ConvNdOpts& opts,
                                    const Shape& out_shape,
                                    Dtype dt) {
    // Half is not threaded through im2col and GEMM — there is no
    // half BLAS to call — so it widens at the door and rounds once on
    // the way out.  Metal accepted F16 convolutions and the CPU did
    // not, which made the same model run on one device and raise on
    // the other.  It widened with a loop of its own that read every
    // 16-bit float as IEEE half and rounded back to it, so a bfloat16
    // convolution came back as float16 bits; as_f32 / back_to_f16 read
    // each storage as what it is.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(conv_nd_forward(detail::as_f32(x), detail::as_f32(W),
                                                   detail::as_f32(b), B, Cin, Cout, Cin_g, Cout_g,
                                                   S, K, O, opts, out_shape, Dtype::F32),
                                   dt);

    const int N = opts.N;
    const int O_total = checked_extent(O, N, "output positions");
    const int K_total = checked_extent(K, N, "kernel taps");
    const int S_total = checked_extent(S, N, "input positions");

    const int K_flat = Cin_g * K_total;
    const int M_out = O_total;
    const int W_per_group = Cout_g * K_flat;

    CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * Cout * O_total, dt);
    const int slab = column_slab(O, N, K_flat);
    CpuStorage cols_cpu = alloc_cpu(column_slab_elements(O, N, K_flat, slab), dt);
    const auto& x_cpu = std::get<CpuStorage>(x);
    const auto& W_cpu = std::get<CpuStorage>(W);
    const auto& b_cpu = std::get<CpuStorage>(b);

    for (int bi = 0; bi < B; ++bi) {
        for (int g = 0; g < opts.groups; ++g) {
            if (dt == Dtype::F32) {
                const float* xp =
                    reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                    (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                        S_total;
                float* cp = reinterpret_cast<float*>(cols_cpu.ptr.get());
                const float* wp = reinterpret_cast<const float*>(W_cpu.ptr.get()) +
                                  static_cast<std::size_t>(g) * W_per_group;
                float* yp =
                    reinterpret_cast<float*>(out_cpu.ptr.get()) +
                    (static_cast<std::size_t>(bi) * Cout + static_cast<std::size_t>(g) * Cout_g) *
                        O_total;
                for_each_column_slab(
                    O, opts, N, slab,
                    [&](const int* Os, const int* pads, std::size_t begin, int ms) {
                        conv_nd_im2col_f32(xp, cp, Cin_g, S, K, Os, opts.stride, pads,
                                           opts.dilation, N);
                        cpu::sgemm(false, false, Cout_g, ms, K_flat, 1.0f, wp, K_flat, cp, ms, 0.0f,
                                   yp + begin, M_out);
                    });
            } else if (dt == Dtype::F64) {
                const double* xp =
                    reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                    (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                        S_total;
                double* cp = reinterpret_cast<double*>(cols_cpu.ptr.get());
                const double* wp = reinterpret_cast<const double*>(W_cpu.ptr.get()) +
                                   static_cast<std::size_t>(g) * W_per_group;
                double* yp =
                    reinterpret_cast<double*>(out_cpu.ptr.get()) +
                    (static_cast<std::size_t>(bi) * Cout + static_cast<std::size_t>(g) * Cout_g) *
                        O_total;
                for_each_column_slab(
                    O, opts, N, slab,
                    [&](const int* Os, const int* pads, std::size_t begin, int ms) {
                        conv_nd_im2col_f64(xp, cp, Cin_g, S, K, Os, opts.stride, pads,
                                           opts.dilation, N);
                        cpu::dgemm(false, false, Cout_g, ms, K_flat, 1.0, wp, K_flat, cp, ms, 0.0,
                                   yp + begin, M_out);
                    });
            } else {
                ErrorBuilder("cpu_backend::conv_nd_forward")
                    .not_implemented("dtype not supported (F32/F64)");
            }
        }

        if (dt == Dtype::F32) {
            float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                        static_cast<std::size_t>(bi) * Cout * O_total;
            const float* bp = reinterpret_cast<const float*>(b_cpu.ptr.get());
            for (int c = 0; c < Cout; ++c) {
                const float bv = bp[c];
                float* row = yp + static_cast<std::size_t>(c) * O_total;
                for (int i = 0; i < O_total; ++i)
                    row[i] += bv;
            }
        } else if (dt == Dtype::F64) {
            double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                         static_cast<std::size_t>(bi) * Cout * O_total;
            const double* bp = reinterpret_cast<const double*>(b_cpu.ptr.get());
            for (int c = 0; c < Cout; ++c) {
                const double bv = bp[c];
                double* row = yp + static_cast<std::size_t>(c) * O_total;
                for (int i = 0; i < O_total; ++i)
                    row[i] += bv;
            }
        }
    }
    return Storage{std::move(out_cpu)};
}

std::vector<Storage> CpuBackend::conv_nd_backward(const Storage& grad_out,
                                                  const Storage& x,
                                                  const Storage& W,
                                                  int B,
                                                  int Cin,
                                                  int Cout,
                                                  int Cin_g,
                                                  int Cout_g,
                                                  const int* S,
                                                  const int* K,
                                                  const int* O,
                                                  const IBackend::ConvNdOpts& opts,
                                                  Dtype dt,
                                                  GradMask3 grad_mask) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            conv_nd_backward(detail::as_f32(grad_out), detail::as_f32_if(grad_mask.want(1), x),
                             detail::as_f32_if(grad_mask.want(0), W), B, Cin, Cout, Cin_g, Cout_g,
                             S, K, O, opts, Dtype::F32, grad_mask),
            dt);
    const int N = opts.N;
    const int G = opts.groups;
    const int O_total = checked_extent(O, N, "output positions");
    const int K_total = checked_extent(K, N, "kernel taps");
    const int S_total = checked_extent(S, N, "input positions");

    const int K_flat = Cin_g * K_total;
    const int M_out = O_total;
    const int W_per_group = Cout_g * K_flat;

    cpu::GradBuffers<3> grads(grad_mask, dt,
                              {static_cast<std::size_t>(B) * Cin * S_total,
                               static_cast<std::size_t>(Cout) * K_flat,
                               static_cast<std::size_t>(Cout)});
    grads.zero();
    const int slab = column_slab(O, N, K_flat);
    const auto columns = [&] { return alloc_cpu(column_slab_elements(O, N, K_flat, slab), dt); };
    CpuStorage cols_cpu = grad_mask.make(1, columns);
    CpuStorage col_grad_cpu = grad_mask.make(0, columns);

    const CpuStorage* x_cpu = cpu::cpu_operand(grad_mask.want(1), x);
    const CpuStorage* W_cpu = cpu::cpu_operand(grad_mask.want(0), W);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    const auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* xp0 = cpu::operand_data<T>(x_cpu);
        const T* wp0 = cpu::operand_data<T>(W_cpu);
        const T* gp0 = reinterpret_cast<const T*>(g_cpu.ptr.get());
        T* cp = reinterpret_cast<T*>(cols_cpu.ptr.get());
        T* cgp = reinterpret_cast<T*>(col_grad_cpu.ptr.get());
        for (int bi = 0; bi < B; ++bi) {
            for (int g = 0; g < G; ++g) {
                const std::size_t x_off =
                    (static_cast<std::size_t>(bi) * Cin + static_cast<std::size_t>(g) * Cin_g) *
                    S_total;
                const std::size_t w_off = static_cast<std::size_t>(g) * W_per_group;
                const T* xp = offset_or_null(xp0, x_off);
                const T* gp = gp0 + (static_cast<std::size_t>(bi) * Cout +
                                     static_cast<std::size_t>(g) * Cout_g) *
                                        O_total;
                T* dxp = offset_or_null(grads.data<T>(0), x_off);
                const T* wp = offset_or_null(wp0, w_off);
                T* dwp = offset_or_null(grads.data<T>(1), w_off);
                if (grad_mask.want_any(0, 1)) {
                    for_each_column_slab(
                        O, opts, N, slab,
                        [&](const int* Os, const int* pads, std::size_t begin, int ms) {
                            if (grad_mask.want(1)) {
                                if constexpr (std::is_same_v<T, float>)
                                    conv_nd_im2col_f32(xp, cp, Cin_g, S, K, Os, opts.stride, pads,
                                                       opts.dilation, N);
                                else
                                    conv_nd_im2col_f64(xp, cp, Cin_g, S, K, Os, opts.stride, pads,
                                                       opts.dilation, N);
                                ctnd_gemm(false, true, Cout_g, K_flat, ms, T(1), gp + begin, M_out,
                                          cp, ms, T(1), dwp, K_flat);
                            }
                            if (grad_mask.want(0)) {
                                ctnd_gemm(true, false, K_flat, ms, Cout_g, T(1), wp, K_flat,
                                          gp + begin, M_out, T(0), cgp, ms);
                                if constexpr (std::is_same_v<T, float>)
                                    conv_nd_col2im_f32(cgp, dxp, Cin_g, S, K, Os, opts.stride, pads,
                                                       opts.dilation, N);
                                else
                                    conv_nd_col2im_f64(cgp, dxp, Cin_g, S, K, Os, opts.stride, pads,
                                                       opts.dilation, N);
                            }
                        });
                }
                if (grad_mask.want(2))
                    cpu::add_row_sums(gp, grads.data<T>(2) + static_cast<std::size_t>(g) * Cout_g,
                                      Cout_g, O_total);
            }
        }
    };

    // The dtype is rejected where a tile would be computed; with no batch or
    // no group there is no tile, and nothing to reject.
    if (B == 0 || G == 0)
        return grads.take_all();
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::conv_nd_backward").not_implemented("dtype not supported");
    return grads.take_all();
}

Storage CpuBackend::unfold_forward(const Storage& x,
                                   int B,
                                   int C,
                                   const std::vector<int>& S,
                                   const std::vector<int>& K,
                                   const std::vector<int>& O,
                                   const std::vector<int>& stride,
                                   const std::vector<int>& pad,
                                   const std::vector<int>& dilation,
                                   const Shape& p9_,
                                   Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(unfold_forward(detail::as_f32(x), B, C, S, K, O, stride, pad,
                                                  dilation, p9_, Dtype::F32),
                                   dt);
    // Integers ride the double path.  ``im2col`` is written out for
    // float and double with no template behind it, and unfold only
    // *copies* — so a double holds every value exactly, for every
    // integer dtype except an int64 magnitude above 2^53.
    if (detail::is_int_like(dt))
        return detail::f64_back_to_int(unfold_forward(detail::as_f64(x), B, C, S, K, O, stride, pad,
                                                      dilation, p9_, Dtype::F64),
                                       dt);
    const int N = static_cast<int>(K.size());
    int K_total = 1, O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        K_total *= K[i];
        O_total *= O[i];
        S_total *= S[i];
    }

    CpuStorage out_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * K_total * O_total, dt);
    const auto& x_cpu = std::get<CpuStorage>(x);

    for (int bi = 0; bi < B; ++bi) {
        if (dt == Dtype::F32) {
            const float* xp = reinterpret_cast<const float*>(x_cpu.ptr.get()) +
                              static_cast<std::size_t>(bi) * C * S_total;
            float* yp = reinterpret_cast<float*>(out_cpu.ptr.get()) +
                        static_cast<std::size_t>(bi) * C * K_total * O_total;
            unfold_im2col_f32(xp, yp, C, S.data(), K.data(), O.data(), stride.data(), pad.data(),
                              dilation.data(), N);
        } else if (dt == Dtype::F64) {
            const double* xp = reinterpret_cast<const double*>(x_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * C * S_total;
            double* yp = reinterpret_cast<double*>(out_cpu.ptr.get()) +
                         static_cast<std::size_t>(bi) * C * K_total * O_total;
            unfold_im2col_f64(xp, yp, C, S.data(), K.data(), O.data(), stride.data(), pad.data(),
                              dilation.data(), N);
        } else {
            ErrorBuilder("cpu_backend::unfold_forward").not_implemented("dtype not supported");
        }
    }
    return Storage{std::move(out_cpu)};
}

Storage CpuBackend::unfold_backward(const Storage& grad_out,
                                    int B,
                                    int C,
                                    const std::vector<int>& S,
                                    const std::vector<int>& K,
                                    const std::vector<int>& O,
                                    const std::vector<int>& stride,
                                    const std::vector<int>& pad,
                                    const std::vector<int>& dilation,
                                    Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(unfold_backward(detail::as_f32(grad_out), B, C, S, K, O, stride,
                                                   pad, dilation, Dtype::F32),
                                   dt);
    const int N = static_cast<int>(K.size());
    int K_total = 1, O_total = 1, S_total = 1;
    for (int i = 0; i < N; ++i) {
        K_total *= K[i];
        O_total *= O[i];
        S_total *= S[i];
    }

    CpuStorage dx_cpu = alloc_cpu(static_cast<std::size_t>(B) * C * S_total, dt);
    if (dx_cpu.nbytes)
        std::memset(dx_cpu.ptr.get(), 0, dx_cpu.nbytes);
    const auto& g_cpu = std::get<CpuStorage>(grad_out);

    for (int bi = 0; bi < B; ++bi) {
        if (dt == Dtype::F32) {
            const float* gp = reinterpret_cast<const float*>(g_cpu.ptr.get()) +
                              static_cast<std::size_t>(bi) * C * K_total * O_total;
            float* dxp = reinterpret_cast<float*>(dx_cpu.ptr.get()) +
                         static_cast<std::size_t>(bi) * C * S_total;
            unfold_col2im_f32(gp, dxp, C, S.data(), K.data(), O.data(), stride.data(), pad.data(),
                              dilation.data(), N);
        } else if (dt == Dtype::F64) {
            const double* gp = reinterpret_cast<const double*>(g_cpu.ptr.get()) +
                               static_cast<std::size_t>(bi) * C * K_total * O_total;
            double* dxp = reinterpret_cast<double*>(dx_cpu.ptr.get()) +
                          static_cast<std::size_t>(bi) * C * S_total;
            unfold_col2im_f64(gp, dxp, C, S.data(), K.data(), O.data(), stride.data(), pad.data(),
                              dilation.data(), N);
        } else {
            ErrorBuilder("cpu_backend::unfold_backward").not_implemented("dtype not supported");
        }
    }
    return Storage{std::move(dx_cpu)};
}

std::int64_t CpuBackend::column_budget() {
    static const std::int64_t budget = [] {
        const char* env = std::getenv("LUCID_CONV_COLUMN_BUDGET");
        const long long v = env ? std::atoll(env) : 0;
        return v > 0 ? static_cast<std::int64_t>(v) : kColumnBudget;
    }();
    return budget;
}

int CpuBackend::checked_extent(const int* dims, int n, const char* what) {
    std::int64_t total = 1;
    for (int i = 0; i < n; ++i)
        total *= dims[i];
    if (total > std::numeric_limits<int>::max())
        ErrorBuilder("cpu_backend::conv_nd")
            .fail(std::string("too many ") + what +
                  " for one convolution: " + std::to_string(total));
    return static_cast<int>(total);
}

std::int64_t CpuBackend::column_inner(const int* O, int N) {
    std::int64_t inner = 1;
    for (int i = 1; i < N; ++i)
        inner *= O[i];
    return inner;
}

int CpuBackend::column_slab(const int* O, int N, int K_flat) {
    const std::int64_t per_step =
        std::max<std::int64_t>(1, std::int64_t{K_flat} * column_inner(O, N));
    const std::int64_t steps = std::max<std::int64_t>(1, column_budget() / per_step);
    return static_cast<int>(std::min<std::int64_t>(steps, std::max(1, O[0])));
}

std::size_t CpuBackend::column_slab_elements(const int* O, int N, int K_flat, int slab) {
    return static_cast<std::size_t>(K_flat) * static_cast<std::size_t>(slab) *
           static_cast<std::size_t>(column_inner(O, N));
}

template <typename Fn>
void CpuBackend::for_each_column_slab(
    const int* O, const IBackend::ConvNdOpts& opts, int N, int slab, Fn&& fn) {
    const std::int64_t inner = column_inner(O, N);
    for (int o0 = 0; o0 < O[0]; o0 += slab) {
        const int count = std::min(slab, O[0] - o0);
        int Os[3] = {0, 0, 0};
        int pads[3] = {0, 0, 0};
        for (int i = 0; i < N; ++i) {
            Os[i] = O[i];
            pads[i] = opts.pad[i];
        }
        Os[0] = count;
        pads[0] = opts.pad[0] - o0 * opts.stride[0];
        fn(static_cast<const int*>(Os), static_cast<const int*>(pads),
           static_cast<std::size_t>(o0) * static_cast<std::size_t>(inner),
           static_cast<int>(count * inner));
    }
}

void CpuBackend::conv_nd_im2col_f32(const float* x,
                                    float* cols,
                                    int C,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const int* stride,
                                    const int* pad,
                                    const int* dilation,
                                    int N) {
    if (N == 1)
        cpu::im2col_1d_f32(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
    else if (N == 2)
        cpu::im2col_f32(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                        pad[0], pad[1], dilation[0], dilation[1]);
    else
        cpu::im2col_3d_f32(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                           dilation[1], dilation[2]);
}

void CpuBackend::conv_nd_im2col_f64(const double* x,
                                    double* cols,
                                    int C,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const int* stride,
                                    const int* pad,
                                    const int* dilation,
                                    int N) {
    if (N == 1)
        cpu::im2col_1d_f64(x, cols, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
    else if (N == 2)
        cpu::im2col_f64(x, cols, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                        pad[0], pad[1], dilation[0], dilation[1]);
    else
        cpu::im2col_3d_f64(x, cols, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                           dilation[1], dilation[2]);
}

void CpuBackend::conv_nd_col2im_f32(const float* cols,
                                    float* dx,
                                    int C,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const int* stride,
                                    const int* pad,
                                    const int* dilation,
                                    int N) {
    if (N == 1)
        cpu::col2im_1d_f32(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
    else if (N == 2)
        cpu::col2im_f32(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                        pad[0], pad[1], dilation[0], dilation[1]);
    else
        cpu::col2im_3d_f32(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                           dilation[1], dilation[2]);
}

void CpuBackend::conv_nd_col2im_f64(const double* cols,
                                    double* dx,
                                    int C,
                                    const int* S,
                                    const int* K,
                                    const int* O,
                                    const int* stride,
                                    const int* pad,
                                    const int* dilation,
                                    int N) {
    if (N == 1)
        cpu::col2im_1d_f64(cols, dx, C, S[0], K[0], O[0], stride[0], pad[0], dilation[0]);
    else if (N == 2)
        cpu::col2im_f64(cols, dx, C, S[0], S[1], K[0], K[1], O[0], O[1], stride[0], stride[1],
                        pad[0], pad[1], dilation[0], dilation[1]);
    else
        cpu::col2im_3d_f64(cols, dx, C, S[0], S[1], S[2], K[0], K[1], K[2], O[0], O[1], O[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dilation[0],
                           dilation[1], dilation[2]);
}

void CpuBackend::unfold_im2col_f32(const float* x,
                                   float* y,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
    conv_nd_im2col_f32(x, y, C, S, K, O, stride, pad, dilation, N);
}

void CpuBackend::unfold_im2col_f64(const double* x,
                                   double* y,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
    conv_nd_im2col_f64(x, y, C, S, K, O, stride, pad, dilation, N);
}

void CpuBackend::unfold_col2im_f32(const float* cols,
                                   float* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
    conv_nd_col2im_f32(cols, dx, C, S, K, O, stride, pad, dilation, N);
}

void CpuBackend::unfold_col2im_f64(const double* cols,
                                   double* dx,
                                   int C,
                                   const int* S,
                                   const int* K,
                                   const int* O,
                                   const int* stride,
                                   const int* pad,
                                   const int* dilation,
                                   int N) {
    conv_nd_col2im_f64(cols, dx, C, S, K, O, stride, pad, dilation, N);
}

void CpuBackend::ctnd_im2col(const float* src,
                             float* cols,
                             int C,
                             const int* O,
                             const int* K,
                             const int* S,
                             const int* stride,
                             const int* pad,
                             const int* dil,
                             int N) {
    if (N == 1)
        cpu::im2col_1d_f32(src, cols, C, O[0], K[0], S[0], stride[0], pad[0], dil[0]);
    else if (N == 2)
        cpu::im2col_f32(src, cols, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                        pad[0], pad[1], dil[0], dil[1]);
    else
        cpu::im2col_3d_f32(src, cols, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1],
                           dil[2]);
}

void CpuBackend::ctnd_im2col(const double* src,
                             double* cols,
                             int C,
                             const int* O,
                             const int* K,
                             const int* S,
                             const int* stride,
                             const int* pad,
                             const int* dil,
                             int N) {
    if (N == 1)
        cpu::im2col_1d_f64(src, cols, C, O[0], K[0], S[0], stride[0], pad[0], dil[0]);
    else if (N == 2)
        cpu::im2col_f64(src, cols, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                        pad[0], pad[1], dil[0], dil[1]);
    else
        cpu::im2col_3d_f64(src, cols, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1],
                           dil[2]);
}

void CpuBackend::ctnd_col2im(const float* cols,
                             float* dst,
                             int C,
                             const int* O,
                             const int* K,
                             const int* S,
                             const int* stride,
                             const int* pad,
                             const int* dil,
                             int N) {
    if (N == 1)
        cpu::col2im_1d_f32(cols, dst, C, O[0], K[0], S[0], stride[0], pad[0], dil[0]);
    else if (N == 2)
        cpu::col2im_f32(cols, dst, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                        pad[0], pad[1], dil[0], dil[1]);
    else
        cpu::col2im_3d_f32(cols, dst, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1],
                           dil[2]);
}

void CpuBackend::ctnd_col2im(const double* cols,
                             double* dst,
                             int C,
                             const int* O,
                             const int* K,
                             const int* S,
                             const int* stride,
                             const int* pad,
                             const int* dil,
                             int N) {
    if (N == 1)
        cpu::col2im_1d_f64(cols, dst, C, O[0], K[0], S[0], stride[0], pad[0], dil[0]);
    else if (N == 2)
        cpu::col2im_f64(cols, dst, C, O[0], O[1], K[0], K[1], S[0], S[1], stride[0], stride[1],
                        pad[0], pad[1], dil[0], dil[1]);
    else
        cpu::col2im_3d_f64(cols, dst, C, O[0], O[1], O[2], K[0], K[1], K[2], S[0], S[1], S[2],
                           stride[0], stride[1], stride[2], pad[0], pad[1], pad[2], dil[0], dil[1],
                           dil[2]);
}

void CpuBackend::ctnd_gemm(bool ta,
                           bool tb,
                           int M,
                           int Nn,
                           int Kk,
                           float alpha,
                           const float* A,
                           int lda,
                           const float* Bm,
                           int ldb,
                           float beta,
                           float* C,
                           int ldc) {
    cpu::sgemm(ta, tb, M, Nn, Kk, alpha, A, lda, Bm, ldb, beta, C, ldc);
}

void CpuBackend::ctnd_gemm(bool ta,
                           bool tb,
                           int M,
                           int Nn,
                           int Kk,
                           double alpha,
                           const double* A,
                           int lda,
                           const double* Bm,
                           int ldb,
                           double beta,
                           double* C,
                           int ldc) {
    cpu::dgemm(ta, tb, M, Nn, Kk, alpha, A, lda, Bm, ldb, beta, C, ldc);
}

}  // namespace backend
}  // namespace lucid
