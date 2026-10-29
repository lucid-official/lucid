// lucid/_C/backend/gpu/methods/Conv.cpp
// Out-of-line GpuBackend method definitions for the Conv family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::nn_fold(const Storage& x,
                            const Shape& x_shape,
                            const Shape& out_shape,
                            const std::vector<int>& kernel_size,
                            const std::vector<int>& stride,
                            const std::vector<int>& padding,
                            const std::vector<int>& dilation,
                            Dtype dt) {
    // MLX's Metal scatter has no int64 kernel, and the accumulation
    // below is a scatter.  Same route the scatter ops themselves take
    // — see scatter_via_cpu: there is no exact way around it
    // on-device, and refusing a dtype the CPU folds is the
    // device-dependent behaviour this backend pair exists to avoid.
    if (dt == Dtype::I64) {
        auto& cpu = backend::Dispatcher::for_device(Device::CPU);
        Storage x_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(x), x_shape)};
        Storage out_cpu =
            cpu.nn_fold(x_cpu, x_shape, out_shape, kernel_size, stride, padding, dilation, dt);
        return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(out_cpu), out_shape)};
    }
    const auto& gx = std::get<GpuStorage>(x);

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
    const int W_pad = outW + 2 * pW;
    // Must use the *effective* (dilated) kernel extent — the raw kernel
    // size overcounts blocks when dilation > 1 and mis-maps l → (oh, ow).
    const int effW = dW * (kW - 1) + 1;
    const int outW_blocks = (W_pad - effW) / sW + 1;
    const int K = kH * kW;
    const int M = K * L;  // total source positions per (n, c)

    // Build the (K * L,) flat index + mask tables on CPU.  Invalid
    // positions land at index 0 with mask 0 so the scatter_add is a
    // no-op there.
    std::vector<std::int32_t> idx_host(static_cast<std::size_t>(M));
    std::vector<float> mask_host(static_cast<std::size_t>(M), 0.0f);
    for (int k = 0; k < K; ++k) {
        const int ki = k / kW;
        const int kj = k % kW;
        for (int l = 0; l < L; ++l) {
            const int oh_blk = l / outW_blocks;
            const int ow_blk = l % outW_blocks;
            const int oh = oh_blk * sH + ki * dH - pH;
            const int ow = ow_blk * sW + kj * dW - pW;
            const std::size_t pos = static_cast<std::size_t>(k * L + l);
            if (oh < 0 || oh >= outH || ow < 0 || ow >= outW) {
                idx_host[pos] = 0;
                mask_host[pos] = 0.0f;
            } else {
                idx_host[pos] = oh * outW + ow;
                mask_host[pos] = 1.0f;
            }
        }
    }

    namespace mx = ::mlx::core;
    // Upload index + mask as small (M,) MLX arrays.
    mx::array idx_mlx(idx_host.data(), mx::Shape{static_cast<int>(M)}, mx::int32);
    mx::array mask_mlx(mask_host.data(), mx::Shape{static_cast<int>(M)}, mx::float32);

    // x_reshaped: (N, CKK, L) → (N, C, K, L) → (N, C, M).  C is the
    // outer of CKK so this is a pure reshape with no transpose.
    auto xr = mx::reshape(*gx.arr, mx::Shape{N, C, M});

    // Mask invalid positions by *selecting*, not multiplying.  A
    // padded column must contribute nothing, and ``0 * NaN`` is not
    // nothing — the same defect ``unfold`` had one function over.
    auto keep = mx::broadcast_to(mx::reshape(mask_mlx, mx::Shape{1, 1, M}), mx::Shape{N, C, M});
    auto x_masked = mx::where(mx::astype(keep, mx::bool_), xr, mx::zeros_like(xr));

    // out_flat: (N, C, outH * outW), whose last extent MLX holds in int32.
    // Formed in int it wrapped: a 2^20 x 2^20 output was a flat axis of 0.
    auto base =
        mx::zeros(gpu::to_mlx_shape(Shape{N, C, mul_extent(outH, outW)}), gpu::to_mlx_dtype(dt));

    // scatter_add_axis with axis=2: indices is broadcast (1, 1, M) →
    // every (n, c) pair scatters into the same flat layout.
    auto idx_bc = mx::reshape(idx_mlx, mx::Shape{1, 1, M});
    idx_bc = mx::broadcast_to(idx_bc, mx::Shape{N, C, M});

    auto folded = scatter_add_axis_widened(base, idx_bc, x_masked, /*axis=*/2);

    // Reshape (N, C, outH * outW) → (N, C, outH, outW).
    auto out = mx::reshape(folded, mx::Shape{N, C, outH, outW});
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage GpuBackend::conv_transpose_nd_forward(const Storage& x,
                                              const Storage& W,
                                              const Storage& b,
                                              int,
                                              int Cin,
                                              int Cout,
                                              const int*,
                                              const int*,
                                              const int*,
                                              const IBackend::ConvTransposeNdOpts& opts,
                                              const Shape& out_shape,
                                              Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gW = std::get<GpuStorage>(W);
    const auto& gb = std::get<GpuStorage>(b);
    const int N = opts.N;

    if (gpu_conv_has_empty_operand(*gx.arr, *gW.arr, shape_numel(out_shape)))
        return Storage{gpu::wrap_mlx_array(gpu_conv_bias_only(*gb.arr, out_shape), dt)};

    // PERF: contiguous before conv_transpose — same rationale as forward
    // conv path. See conv_nd_forward for the microbench data.
    auto x_nhwc =
        ::mlx::core::contiguous(::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N)));
    auto W_nhwc = gpu_w_to_mlx_conv_transpose(*gW.arr, Cin, Cout, opts.groups, N);
    auto y_nhwc = gpu_mlx_conv_transpose(x_nhwc, W_nhwc, opts.stride, opts.pad, opts.dilation,
                                         opts.opad, opts.groups, N);

    ::mlx::core::Shape b_brd(N + 2, 1);
    b_brd[N + 1] = static_cast<::mlx::core::ShapeElem>(Cout);
    auto b_view = ::mlx::core::reshape(*gb.arr, b_brd);
    y_nhwc = ::mlx::core::add(y_nhwc, b_view);
    // PERF: drop the explicit ::mlx::core::contiguous(...) — MLX ops handle
    // strided views transparently, and deferring materialization lets
    // downstream ops (relu, pool, conv) fuse with the transpose.
    // Measured +15.7 % on Conv2d microbench (M1 Pro, BS=64 LeNet conv shapes).
    auto y = ::mlx::core::transpose(y_nhwc, gpu_nhwc_to_nchw_perm(N));
    return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
}

std::vector<Storage>
GpuBackend::conv_transpose_nd_backward(const Storage& grad_out,
                                       const Storage& x,
                                       const Storage& W,
                                       int B,
                                       int Cin,
                                       int Cout,
                                       const int* S,
                                       const int* K,
                                       const int* O,
                                       const IBackend::ConvTransposeNdOpts& opts,
                                       Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gW = std::get<GpuStorage>(W);
    const auto& gG = std::get<GpuStorage>(grad_out);
    const int N = opts.N;
    const int G = opts.groups;
    const int Cin_g = Cin / G;
    const int Cout_g = Cout / G;
    std::vector<int> sv(opts.stride, opts.stride + N);
    std::vector<int> pv(opts.pad, opts.pad + N);
    std::vector<int> dv(opts.dilation, opts.dilation + N);
    std::vector<int> ones_n(N, 1);

    std::vector<int> db_axes;
    db_axes.reserve(N + 1);
    db_axes.push_back(0);
    for (int i = 0; i < N; ++i)
        db_axes.push_back(2 + i);
    auto db = sum_widened(*gG.arr, db_axes, false);

    if (gpu_conv_has_empty_operand(*gx.arr, *gW.arr, gG.arr->size()))
        return gpu_conv_empty_grads(*gx.arr, *gW.arr, std::move(db), dt);

    // dx — the adjoint of a transposed convolution is the plain forward
    // convolution it is the data gradient of, with the same geometry.
    // No channel rearrangement is needed for grouping either: MLX wants
    // ``(O, K..., I / G)`` and permuting Lucid's ``(Cin, Cout/G, K...)``
    // gives exactly ``(Cin, K..., Cout/G)``, with group ``j`` of the
    // output already paired with group ``j`` of the input.
    std::vector<int> w_dx_perm;
    w_dx_perm.push_back(0);
    for (int i = 0; i < N; ++i)
        w_dx_perm.push_back(2 + i);
    w_dx_perm.push_back(1);
    // PERF: contiguous before conv kernel — see conv_nd_forward microbench.
    auto grad_nhwc =
        ::mlx::core::contiguous(::mlx::core::transpose(*gG.arr, gpu_nchw_to_nhwc_perm(N)));
    auto W_dx_nhwc = ::mlx::core::contiguous(::mlx::core::transpose(*gW.arr, w_dx_perm));
    auto dx_nhwc = gpu_mlx_conv_grouped(grad_nhwc, W_dx_nhwc, sv, pv, dv, G, N);
    // An output_padding of a stride or more — allowed while the dilation
    // is larger — lets that convolution fit a window past the input's
    // end.  Left in, the extra row was read back as x's shape and every
    // row after the first came out shifted by one.
    {
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape lo(N + 2, 0);
        ::mlx::core::Shape hi = dx_nhwc.shape();
        bool longer = false;
        for (int i = 0; i < N; ++i) {
            if (hi[1 + i] != static_cast<SE>(S[i])) {
                hi[1 + i] = static_cast<SE>(S[i]);
                longer = true;
            }
        }
        if (longer)
            dx_nhwc = ::mlx::core::slice(dx_nhwc, lo, hi);
    }
    // PERF: see conv_transpose_nd_forward — strided view returned, MLX
    // ops handle strides natively. Materialization is deferred to the
    // final mx.eval() or to ops that genuinely need contiguous memory.
    auto dx = ::mlx::core::transpose(dx_nhwc, gpu_nhwc_to_nchw_perm(N));

    std::vector<int> perm_axes;
    perm_axes.push_back(1);
    for (int i = 0; i < N; ++i)
        perm_axes.push_back(2 + i);
    perm_axes.push_back(0);

    // dW — expressed as a convolution of grad_out by x with the batch
    // axis moved into the channel slot.  That rearrangement puts the
    // group structure on the axes MLX calls batch and out-channels,
    // which its ``groups`` argument cannot reach, so grouping is a
    // slice-and-concatenate loop here (as in conv_nd_backward).
    //
    // dW[k] = sum_s grad[s * stride + k * dilation - pad] * x[s], so the
    // OUTPUT index k steps by the transposed conv's dilation (conv
    // stride = dv) while consecutive x taps sit stride apart in grad
    // (kernel_dilation = sv).  The conv stride was previously pinned to
    // 1, which coincides with the correct mapping only at dilation == 1.
    const auto compute_dW = [&](const ::mlx::core::array& x_arr, const ::mlx::core::array& g_arr,
                                int local_Cin_g, int local_Cout_g) -> ::mlx::core::array {
        // PERF: contiguous before dW conv_general (see conv_nd_forward bench).
        auto g_perm = ::mlx::core::contiguous(::mlx::core::transpose(g_arr, perm_axes));
        auto x_perm = ::mlx::core::contiguous(::mlx::core::transpose(x_arr, perm_axes));
        auto raw = gpu_conv_general(g_perm, x_perm, dv, pv, pv, sv, ones_n,
                                    /*groups=*/1, /*flip=*/false);
        using SE = ::mlx::core::ShapeElem;
        ::mlx::core::Shape crop_lo(N + 2, 0);
        ::mlx::core::Shape crop_hi;
        crop_hi.push_back(static_cast<SE>(local_Cout_g));
        for (int i = 0; i < N; ++i)
            crop_hi.push_back(static_cast<SE>(K[i]));
        crop_hi.push_back(static_cast<SE>(local_Cin_g));
        return ::mlx::core::slice(raw, crop_lo, crop_hi);
    };

    // (Cout_g, K..., Cin_g) -> (Cin_g, Cout_g, K...), Lucid's layout.
    std::vector<int> dW_back;
    dW_back.push_back(N + 1);
    dW_back.push_back(0);
    for (int i = 0; i < N; ++i)
        dW_back.push_back(1 + i);

    ::mlx::core::array dW = ::mlx::core::array(0);
    if (G == 1) {
        dW = ::mlx::core::transpose(compute_dW(*gx.arr, *gG.arr, Cin_g, Cout_g), dW_back);
    } else {
        using SE = ::mlx::core::ShapeElem;
        std::vector<::mlx::core::array> dW_groups;
        dW_groups.reserve(G);
        for (int g = 0; g < G; ++g) {
            ::mlx::core::Shape x_lo(2 + N, 0), x_hi;
            x_hi.push_back(static_cast<SE>(B));
            x_hi.push_back(static_cast<SE>((g + 1) * Cin_g));
            for (int i = 0; i < N; ++i)
                x_hi.push_back(static_cast<SE>(S[i]));
            x_lo[1] = static_cast<SE>(g * Cin_g);
            auto x_g = ::mlx::core::slice(*gx.arr, x_lo, x_hi);

            ::mlx::core::Shape gr_lo(2 + N, 0), gr_hi;
            gr_hi.push_back(static_cast<SE>(B));
            gr_hi.push_back(static_cast<SE>((g + 1) * Cout_g));
            for (int i = 0; i < N; ++i)
                gr_hi.push_back(static_cast<SE>(O[i]));
            gr_lo[1] = static_cast<SE>(g * Cout_g);
            auto grad_g = ::mlx::core::slice(*gG.arr, gr_lo, gr_hi);

            dW_groups.push_back(
                ::mlx::core::transpose(compute_dW(x_g, grad_g, Cin_g, Cout_g), dW_back));
        }
        // Group j owns Cin rows [j * Cin_g, ...) of the weight, so the
        // per-group blocks stack along the leading axis.
        dW = ::mlx::core::concatenate(std::move(dW_groups), /*axis=*/0);
    }

    // 3.4+ Step 3.2 perf sweep: see conv_nd_backward — drop trailing
    // contiguous on dW so the optimizer's update can fuse with it.
    return {Storage{gpu::wrap_mlx_array(std::move(dx), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
            Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
}

Storage GpuBackend::conv_nd_forward(const Storage& x,
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
                                    const ConvNdOpts& opts,
                                    const Shape& out_shape,
                                    Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gW = std::get<GpuStorage>(W);
    const auto& gb = std::get<GpuStorage>(b);
    (void)Cin;
    (void)Cout;
    (void)Cin_g;
    (void)Cout_g;
    (void)S;
    (void)K;
    (void)O;
    (void)B;
    const int N = opts.N;
    std::vector<int> sv(opts.stride, opts.stride + N);
    std::vector<int> pv(opts.pad, opts.pad + N);
    std::vector<int> dv(opts.dilation, opts.dilation + N);

    if (gpu_conv_has_empty_operand(*gx.arr, *gW.arr, shape_numel(out_shape)))
        return Storage{gpu::wrap_mlx_array(gpu_conv_bias_only(*gb.arr, out_shape), dt)};

    // 1x1 POINTWISE fast-path: a stride-1, pad-0, dilation-1, groups-1, K=1
    // conv is exactly a per-position channel matmul. MLX conv_general is
    // slower for this degenerate kernel (esp. its weight-gradient at large
    // spatial); routing through mlx::core::matmul is up to 1.34x faster and
    // never slower (validated, exact parity), and stays in the MLX stream
    // (no cross-queue cost). Backward mirrors this in conv_nd_backward.
    {
        bool pointwise = (opts.groups == 1);
        for (int i = 0; i < N && pointwise; ++i)
            if (K[i] != 1 || opts.stride[i] != 1 || opts.pad[i] != 0 || opts.dilation[i] != 1)
                pointwise = false;
        if (pointwise) {
            const int Cout_pw = static_cast<int>(out_shape[1]);
            auto x_nhwc_pw =
                ::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N));  // (B,*S,Cin)
            auto W_cico = ::mlx::core::transpose(::mlx::core::reshape(*gW.arr, {Cout, Cin}),
                                                 {1, 0});        // (Cin,Cout)
            auto y_pw = ::mlx::core::matmul(x_nhwc_pw, W_cico);  // (B,*S,Cout)
            ::mlx::core::Shape b_brd(N + 2, 1);
            b_brd[N + 1] = Cout_pw;
            y_pw = ::mlx::core::add(y_pw, ::mlx::core::reshape(*gb.arr, b_brd));
            auto y = ::mlx::core::transpose(y_pw, gpu_nhwc_to_nchw_perm(N));
            return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
        }
    }

    // PERF: force contiguous NHWC before conv_general.  MLX's lazy graph
    // would otherwise pass strided transpose-views in, and conv_general
    // dispatches a slower stride-aware kernel path.  Forcing contiguous
    // up front lets MLX pick the fastest conv kernel.  Measured on a
    // training-pattern microbench (W changes every iter, simulating
    // bump_version after optimizer.step()): -7.8 % vs view-only.
    auto x_nhwc =
        ::mlx::core::contiguous(::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N)));
    auto W_nhwc =
        ::mlx::core::contiguous(::mlx::core::transpose(*gW.arr, gpu_nchw_to_nhwc_perm(N)));
    // Grouped rank-5 goes through the per-group fallback: MLX refuses
    // groups above two spatial dimensions, so a grouped conv3d used to
    // run on CPU and raise on Metal for the same model.
    auto y_nhwc = gpu_mlx_conv_grouped(x_nhwc, W_nhwc, sv, pv, dv, opts.groups, N);
    int Cout_local = static_cast<int>(out_shape[1]);
    ::mlx::core::Shape b_brd(N + 2, 1);
    b_brd[N + 1] = Cout_local;
    y_nhwc = ::mlx::core::add(y_nhwc, ::mlx::core::reshape(*gb.arr, b_brd));
    // PERF: keep output transpose as a lazy view — downstream relu / pool
    // / next conv handles strides natively and the trailing contiguous
    // would force a memcpy that the next kernel can fuse.
    auto y = ::mlx::core::transpose(y_nhwc, gpu_nhwc_to_nchw_perm(N));
    return Storage{gpu::wrap_mlx_array(std::move(y), dt)};
}

std::vector<Storage> GpuBackend::conv_nd_backward(const Storage& grad_out,
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
                                                  const ConvNdOpts& opts,
                                                  Dtype dt) {
    const auto& gG = std::get<GpuStorage>(grad_out);
    const auto& gx = std::get<GpuStorage>(x);
    const auto& gW = std::get<GpuStorage>(W);
    (void)Cin;
    const int N = opts.N;
    const int G = opts.groups;
    std::vector<int> sv(opts.stride, opts.stride + N);
    std::vector<int> pv(opts.pad, opts.pad + N);
    std::vector<int> dv(opts.dilation, opts.dilation + N);

    std::vector<int> db_axes;
    db_axes.push_back(0);
    for (int i = 0; i < N; ++i)
        db_axes.push_back(2 + i);
    auto db = sum_widened(*gG.arr, db_axes, false);

    if (gpu_conv_has_empty_operand(*gx.arr, *gW.arr, gG.arr->size()))
        return gpu_conv_empty_grads(*gx.arr, *gW.arr, std::move(db), dt);

    // 1x1 POINTWISE fast-path (mirrors conv_nd_forward): dx = grad @ W and
    // dW = grad^T @ x are pure matmuls — faster than conv_general at large
    // spatial, exact parity, MLX-stream-native. db is the reduction above.
    {
        bool pointwise = (G == 1);
        for (int i = 0; i < N && pointwise; ++i)
            if (K[i] != 1 || sv[i] != 1 || pv[i] != 0 || dv[i] != 1)
                pointwise = false;
        if (pointwise) {
            using SEpw = ::mlx::core::ShapeElem;
            auto g_nhwc = ::mlx::core::transpose(*gG.arr, gpu_nchw_to_nhwc_perm(N));  // (B,*S,Cout)
            auto x_nhwc = ::mlx::core::transpose(*gx.arr, gpu_nchw_to_nhwc_perm(N));  // (B,*S,Cin)
            auto W_2d = ::mlx::core::reshape(*gW.arr, {Cout, Cin});                   // (Cout,Cin)
            // dx = grad @ W : (B,*S,Cout) @ (Cout,Cin) -> (B,*S,Cin)
            auto dx_pw =
                ::mlx::core::transpose(::mlx::core::matmul(g_nhwc, W_2d), gpu_nhwc_to_nchw_perm(N));
            // dW = grad_mat^T @ x_mat over flattened (B,*S)
            SEpw BS = static_cast<SEpw>(B);
            for (int i = 0; i < N; ++i)
                BS *= static_cast<SEpw>(S[i]);
            auto g_mat = ::mlx::core::reshape(g_nhwc, {BS, static_cast<SEpw>(Cout)});
            auto x_mat = ::mlx::core::reshape(x_nhwc, {BS, static_cast<SEpw>(Cin)});
            auto dW_pw = ::mlx::core::reshape(
                ::mlx::core::matmul(::mlx::core::transpose(g_mat, {1, 0}), x_mat), gW.arr->shape());
            return {Storage{gpu::wrap_mlx_array(std::move(dx_pw), dt)},
                    Storage{gpu::wrap_mlx_array(std::move(dW_pw), dt)},
                    Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
        }
    }

    // Correct symmetric padding for dx backward:
    //   pad_lo[i] = dv[i]*(K[i]-1) - pv[i]
    //   pad_hi[i] = S[i] - O[i]*sv[i] + sv[i] - 1 + pv[i]
    //
    // Using asymmetric padding (pad_lo=pv, pad_hi=opad) gives the correct
    // OUTPUT SIZE but shifts the correlation window, producing wrong gradient
    // VALUES whenever pv ≠ dv*(K[i]-1)/2 (e.g. conv with padding=0).
    std::vector<int> pad_lo_dx(N), pad_hi_dx(N);
    for (int i = 0; i < N; ++i) {
        pad_lo_dx[i] = dv[i] * (K[i] - 1) - pv[i];
        pad_hi_dx[i] = S[i] - O[i] * sv[i] + sv[i] - 1 + pv[i];
    }
    std::vector<int> ones_n(N, 1);
    std::vector<int> W_t_perm = gpu_w_to_transpose_perm(N);
    std::vector<int> perm;
    perm.push_back(1);
    for (int i = 0; i < N; ++i)
        perm.push_back(2 + i);
    perm.push_back(0);
    // The dx-as-flipped-conv and dW-as-channel-permute tricks both
    // rearrange channels in a way that does NOT compose with MLX's
    // grouped conv (the rearranged "input Cin" is the original batch B
    // for dW, or the original Cout for dx — neither has a defined
    // relationship to opts.groups).  We therefore call the inner
    // conv_general with groups=1, and handle the group structure
    // outside by slicing the Cin / Cout axes of x / W / grad_out per
    // group and concatenating the per-group results.  This mirrors the
    // per-group loop in CpuBackend::conv_nd_backward.
    // 3.4+ Step 3.2 experiment: drop INPUT contigs to conv_general.  The
    // forward path's -7.8% measurement was for a stand-alone microbench;
    // in the backward chain context the upstream BN-bwd / relu-bwd outputs
    // are already strided lazy views, and forcing them contiguous here
    // breaks the conv-BN-relu backward fusion chain.  If this regresses
    // numerical parity or wall time, restore.
    auto compute_dx = [&](const ::mlx::core::array& grad_arr,
                          const ::mlx::core::array& W_arr) -> ::mlx::core::array {
        auto grad_nhwc = ::mlx::core::transpose(grad_arr, gpu_nchw_to_nhwc_perm(N));
        auto W_t_nhwc = ::mlx::core::transpose(W_arr, W_t_perm);
        auto dx_nhwc = gpu_conv_general(grad_nhwc, W_t_nhwc, ones_n, pad_lo_dx, pad_hi_dx, dv, sv,
                                        /*groups=*/1, /*flip=*/true);
        return ::mlx::core::transpose(dx_nhwc, gpu_nhwc_to_nchw_perm(N));
    };

    auto compute_dW = [&](const ::mlx::core::array& x_arr, const ::mlx::core::array& grad_arr,
                          int local_Cin_g, int local_Cout) -> ::mlx::core::array {
        auto x_perm = ::mlx::core::transpose(x_arr, perm);
        auto g_perm = ::mlx::core::transpose(grad_arr, perm);
        // dW[k] = sum_o x[k*dilation + o*stride] * grad[o].  As a convolution
        // of x by grad that means the OUTPUT index k steps by the original
        // dilation (conv stride = dv) while consecutive grad taps sit stride
        // apart in the input (kernel_dilation = sv); the input itself is not
        // dilated.  This was previously stride=1 / input_dilation=dv, which
        // coincides with the correct mapping ONLY when dilation == 1 — every
        // dilated conv silently produced wrong weight gradients (the forward
        // and dx were correct, so it never surfaced).
        auto dW_raw = gpu_conv_general(x_perm, g_perm, dv, pv, pv, sv, ones_n,
                                       /*groups=*/1, /*flip=*/false);
        ::mlx::core::Shape crop_lo(N + 2, 0), crop_hi;
        crop_hi.push_back(local_Cin_g);
        for (int i = 0; i < N; ++i)
            crop_hi.push_back(K[i]);
        crop_hi.push_back(local_Cout);
        return ::mlx::core::slice(dW_raw, crop_lo, crop_hi);
    };

    ::mlx::core::array dx_out = ::mlx::core::array(0);
    ::mlx::core::array dW_perm = ::mlx::core::array(0);
    if (G == 1) {
        dx_out = compute_dx(*gG.arr, *gW.arr);
        dW_perm = compute_dW(*gx.arr, *gG.arr, Cin_g, Cout);
    } else {
        std::vector<::mlx::core::array> dx_groups, dW_groups;
        dx_groups.reserve(G);
        dW_groups.reserve(G);
        using SE = ::mlx::core::ShapeElem;
        for (int g = 0; g < G; ++g) {
            ::mlx::core::Shape x_lo(2 + N, 0), x_hi;
            x_hi.push_back(static_cast<SE>(B));
            x_hi.push_back(static_cast<SE>((g + 1) * Cin_g));
            for (int i = 0; i < N; ++i)
                x_hi.push_back(static_cast<SE>(S[i]));
            x_lo[1] = static_cast<SE>(g * Cin_g);
            auto x_g = ::mlx::core::slice(*gx.arr, x_lo, x_hi);

            ::mlx::core::Shape W_lo(2 + N, 0), W_hi;
            W_hi.push_back(static_cast<SE>((g + 1) * Cout_g));
            W_hi.push_back(static_cast<SE>(Cin_g));
            for (int i = 0; i < N; ++i)
                W_hi.push_back(static_cast<SE>(K[i]));
            W_lo[0] = static_cast<SE>(g * Cout_g);
            auto W_g = ::mlx::core::slice(*gW.arr, W_lo, W_hi);

            ::mlx::core::Shape gr_lo(2 + N, 0), gr_hi;
            gr_hi.push_back(static_cast<SE>(B));
            gr_hi.push_back(static_cast<SE>((g + 1) * Cout_g));
            for (int i = 0; i < N; ++i)
                gr_hi.push_back(static_cast<SE>(O[i]));
            gr_lo[1] = static_cast<SE>(g * Cout_g);
            auto grad_g = ::mlx::core::slice(*gG.arr, gr_lo, gr_hi);

            dx_groups.push_back(compute_dx(grad_g, W_g));
            dW_groups.push_back(compute_dW(x_g, grad_g, Cin_g, Cout_g));
        }
        dx_out = ::mlx::core::concatenate(std::move(dx_groups), /*axis=*/1);
        dW_perm = ::mlx::core::concatenate(std::move(dW_groups), /*axis=*/N + 1);
    }

    std::vector<int> dW_back;
    dW_back.push_back(N + 1);
    dW_back.push_back(0);
    for (int i = 0; i < N; ++i)
        dW_back.push_back(1 + i);
    // 3.4+ Step 3.2 perf sweep: drop the trailing contiguous on dW.  The
    // dW feeds into the optimizer's parameter update (Adam: lazy mul / add
    // on the parameter tensor), which handles strided inputs natively.
    // Forcing contiguous here would materialize dW immediately and break
    // the lazy chain that Adam.step would otherwise fuse with.  Mirrors
    // the 3.1.0 forward-output sweep and 3.4.0 norm-backward sweep.
    auto dW = ::mlx::core::transpose(dW_perm, dW_back);
    return {Storage{gpu::wrap_mlx_array(std::move(dx_out), dt)},
            Storage{gpu::wrap_mlx_array(std::move(dW), dt)},
            Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
}

Storage GpuBackend::unfold_forward(const Storage& x,
                                   int B,
                                   int C,
                                   const std::vector<int>& S,
                                   const std::vector<int>& K,
                                   const std::vector<int>& O,
                                   const std::vector<int>& stride,
                                   const std::vector<int>& pad,
                                   const std::vector<int>& dilation,
                                   const Shape& out_shape,
                                   Dtype dt) {
    const auto& gx = std::get<GpuStorage>(x);
    const int N = static_cast<int>(S.size());
    int S_total = 1;
    for (int d : S)
        S_total *= d;
    ::mlx::core::Shape composite;
    composite.push_back(B);
    composite.push_back(C);
    for (int i = 0; i < N; ++i)
        composite.push_back(K[i]);
    for (int i = 0; i < N; ++i)
        composite.push_back(O[i]);
    const auto i32 = ::mlx::core::int32;
    ::mlx::core::Shape b_shape(composite.size(), 1);
    b_shape[0] = B;
    auto b_idx =
        ::mlx::core::reshape(::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);
    ::mlx::core::Shape c_shape(composite.size(), 1);
    c_shape[1] = C;
    auto c_idx =
        ::mlx::core::reshape(::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);
    std::optional<::mlx::core::array> valid_opt;
    std::vector<::mlx::core::array> in_d_clipped;
    for (int d = 0; d < N; ++d) {
        auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32);
        auto o_arr = ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32);
        ::mlx::core::Shape ks(composite.size(), 1);
        ks[2 + d] = K[d];
        ::mlx::core::Shape os(composite.size(), 1);
        os[2 + N + d] = O[d];
        k_arr = ::mlx::core::reshape(k_arr, ks);
        o_arr = ::mlx::core::reshape(o_arr, os);
        auto sd = ::mlx::core::array(stride[d], i32);
        auto dd = ::mlx::core::array(dilation[d], i32);
        auto pd = ::mlx::core::array(pad[d], i32);
        auto in_d = ::mlx::core::subtract(
            ::mlx::core::add(::mlx::core::multiply(sd, o_arr), ::mlx::core::multiply(dd, k_arr)),
            pd);
        auto zero_i = ::mlx::core::array(0, i32);
        auto cap_i = ::mlx::core::array(S[d] - 1, i32);
        auto v = ::mlx::core::logical_and(::mlx::core::greater_equal(in_d, zero_i),
                                          ::mlx::core::less_equal(in_d, cap_i));
        valid_opt = valid_opt.has_value() ? ::mlx::core::logical_and(*valid_opt, v) : v;
        in_d_clipped.push_back(::mlx::core::clip(in_d, std::optional<::mlx::core::array>(zero_i),
                                                 std::optional<::mlx::core::array>(cap_i)));
    }
    auto valid = *valid_opt;
    auto flat = ::mlx::core::add(::mlx::core::multiply(b_idx, ::mlx::core::array(C * S_total, i32)),
                                 ::mlx::core::multiply(c_idx, ::mlx::core::array(S_total, i32)));
    for (int d = 0; d < N; ++d) {
        int trailing = 1;
        for (int e = d + 1; e < N; ++e)
            trailing *= S[e];
        flat = ::mlx::core::add(
            flat, ::mlx::core::multiply(in_d_clipped[d], ::mlx::core::array(trailing, i32)));
    }
    flat = ::mlx::core::broadcast_to(flat, composite);
    auto x_flat = ::mlx::core::reshape(*gx.arr, {B * C * S_total});
    auto sampled = ::mlx::core::take(x_flat, flat);
    int K_total = 1;
    for (int k : K)
        K_total *= k;
    int O_total = 1;
    for (int o : O)
        O_total *= o;
    // Select, do not multiply.
    //
    // ``valid`` marks the positions that came from inside the image
    // rather than from the padding, and the index feeding ``take`` was
    // *clipped* — so a padded cell still sampled a real element and was
    // meant to be zeroed afterwards.  Multiplying by the mask does that
    // for every ordinary value and fails for exactly one: ``0 * NaN`` is
    // NaN.  A single NaN anywhere in the operand turned every padded
    // position NaN, 36 of 36 against the CPU's and the reference's 16,
    // and the two devices disagreed about an op that only moves data.
    auto zeros_like_sampled = ::mlx::core::zeros_like(sampled);
    auto masked = ::mlx::core::where(::mlx::core::broadcast_to(valid, composite), sampled,
                                     zeros_like_sampled);
    auto reshaped = ::mlx::core::reshape(masked, {B, C * K_total, O_total});
    return Storage{gpu::wrap_mlx_array(std::move(reshaped), dt)};
    (void)out_shape;
}

Storage GpuBackend::unfold_backward(const Storage& grad_out,
                                    int B,
                                    int C,
                                    const std::vector<int>& S,
                                    const std::vector<int>& K,
                                    const std::vector<int>& O,
                                    const std::vector<int>& stride,
                                    const std::vector<int>& pad,
                                    const std::vector<int>& dilation,
                                    Dtype dt) {
    const auto& gg = std::get<GpuStorage>(grad_out);
    const int N = static_cast<int>(S.size());
    int S_total = 1;
    for (int d : S)
        S_total *= d;
    int K_total = 1;
    for (int k : K)
        K_total *= k;
    int O_total = 1;
    for (int o : O)
        O_total *= o;
    ::mlx::core::Shape composite;
    composite.push_back(B);
    composite.push_back(C);
    for (int i = 0; i < N; ++i)
        composite.push_back(K[i]);
    for (int i = 0; i < N; ++i)
        composite.push_back(O[i]);
    const auto i32 = ::mlx::core::int32;
    ::mlx::core::Shape b_shape(composite.size(), 1);
    b_shape[0] = B;
    auto b_idx =
        ::mlx::core::reshape(::mlx::core::astype(::mlx::core::arange(0, B, 1), i32), b_shape);
    ::mlx::core::Shape c_shape(composite.size(), 1);
    c_shape[1] = C;
    auto c_idx =
        ::mlx::core::reshape(::mlx::core::astype(::mlx::core::arange(0, C, 1), i32), c_shape);
    std::optional<::mlx::core::array> valid_opt;
    std::vector<::mlx::core::array> in_d_clipped;
    for (int d = 0; d < N; ++d) {
        auto k_arr = ::mlx::core::astype(::mlx::core::arange(0, K[d], 1), i32);
        auto o_arr = ::mlx::core::astype(::mlx::core::arange(0, O[d], 1), i32);
        ::mlx::core::Shape ks(composite.size(), 1);
        ks[2 + d] = K[d];
        ::mlx::core::Shape os(composite.size(), 1);
        os[2 + N + d] = O[d];
        k_arr = ::mlx::core::reshape(k_arr, ks);
        o_arr = ::mlx::core::reshape(o_arr, os);
        auto sd = ::mlx::core::array(stride[d], i32);
        auto dd = ::mlx::core::array(dilation[d], i32);
        auto pd = ::mlx::core::array(pad[d], i32);
        auto in_d = ::mlx::core::subtract(
            ::mlx::core::add(::mlx::core::multiply(sd, o_arr), ::mlx::core::multiply(dd, k_arr)),
            pd);
        auto zero_i = ::mlx::core::array(0, i32);
        auto cap_i = ::mlx::core::array(S[d] - 1, i32);
        auto v = ::mlx::core::logical_and(::mlx::core::greater_equal(in_d, zero_i),
                                          ::mlx::core::less_equal(in_d, cap_i));
        valid_opt = valid_opt.has_value() ? ::mlx::core::logical_and(*valid_opt, v) : v;
        in_d_clipped.push_back(::mlx::core::clip(in_d, std::optional<::mlx::core::array>(zero_i),
                                                 std::optional<::mlx::core::array>(cap_i)));
    }
    auto valid = *valid_opt;
    auto flat = ::mlx::core::add(::mlx::core::multiply(b_idx, ::mlx::core::array(C * S_total, i32)),
                                 ::mlx::core::multiply(c_idx, ::mlx::core::array(S_total, i32)));
    for (int d = 0; d < N; ++d) {
        int trailing = 1;
        for (int e = d + 1; e < N; ++e)
            trailing *= S[e];
        flat = ::mlx::core::add(
            flat, ::mlx::core::multiply(in_d_clipped[d], ::mlx::core::array(trailing, i32)));
    }
    flat = ::mlx::core::broadcast_to(flat, composite);

    // Select rather than multiply, for the reason given in the forward:
    // a padded position must contribute nothing, and ``0 * NaN`` is not
    // nothing.
    auto g_comp = ::mlx::core::reshape(*gg.arr, composite);
    g_comp = ::mlx::core::where(::mlx::core::broadcast_to(valid, composite), g_comp,
                                ::mlx::core::zeros_like(g_comp));

    auto flat_1d = ::mlx::core::reshape(flat, {B * C * K_total * O_total});
    auto g_flat = ::mlx::core::reshape(g_comp, {B * C * K_total * O_total});
    auto dx_flat = ::mlx::core::zeros({B * C * S_total}, gpu::to_mlx_dtype(dt));
    auto flat_1d_i64 = ::mlx::core::astype(flat_1d, ::mlx::core::int64);
    dx_flat = scatter_add_axis_widened(dx_flat, flat_1d_i64, g_flat, 0);
    auto dx = ::mlx::core::reshape(dx_flat, {B, C, S_total});

    ::mlx::core::Shape s_shape;
    s_shape.push_back(B);
    s_shape.push_back(C);
    for (int d : S)
        s_shape.push_back(d);
    dx = ::mlx::core::reshape(dx_flat, s_shape);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(dx), dt)};
}

std::vector<int> GpuBackend::gpu_nchw_to_nhwc_perm(int N) {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    for (int i = 0; i < N; ++i)
        p.push_back(2 + i);
    p.push_back(1);
    return p;
}

std::vector<int> GpuBackend::gpu_nhwc_to_nchw_perm(int N) {
    std::vector<int> p;
    p.reserve(N + 2);
    p.push_back(0);
    p.push_back(N + 1);
    for (int i = 0; i < N; ++i)
        p.push_back(1 + i);
    return p;
}

std::vector<int> GpuBackend::gpu_w_to_transpose_perm(int N) {
    std::vector<int> p;
    p.push_back(1);
    for (int i = 0; i < N; ++i)
        p.push_back(2 + i);
    p.push_back(0);
    return p;
}

::mlx::core::array GpuBackend::gpu_w_to_mlx_conv_transpose(
    const ::mlx::core::array& W, int Cin, int Cout, int G, int N) {
    using SE = ::mlx::core::ShapeElem;
    ::mlx::core::Shape split;
    split.push_back(static_cast<SE>(G));
    split.push_back(static_cast<SE>(Cin / G));
    split.push_back(static_cast<SE>(Cout / G));
    for (int i = 0; i < N; ++i)
        split.push_back(W.shape(2 + i));

    std::vector<int> perm;
    perm.reserve(N + 3);
    perm.push_back(0);
    perm.push_back(2);
    for (int i = 0; i < N; ++i)
        perm.push_back(3 + i);
    perm.push_back(1);

    ::mlx::core::Shape merged;
    merged.push_back(static_cast<SE>(Cout));
    for (int i = 0; i < N; ++i)
        merged.push_back(W.shape(2 + i));
    merged.push_back(static_cast<SE>(Cin / G));

    // PERF: contiguous before the conv kernel — see conv_nd_forward.
    return ::mlx::core::reshape(
        ::mlx::core::contiguous(::mlx::core::transpose(::mlx::core::reshape(W, split), perm)),
        merged);
}

::mlx::core::array
GpuBackend::gpu_slice_axis(const ::mlx::core::array& a, int axis, int lo, int hi) {
    using SE = ::mlx::core::ShapeElem;
    ::mlx::core::Shape start(a.ndim(), 0);
    ::mlx::core::Shape stop = a.shape();
    start[axis] = static_cast<SE>(lo);
    stop[axis] = static_cast<SE>(hi);
    return ::mlx::core::slice(a, start, stop);
}

GpuBackend::AlignedConv GpuBackend::gpu_align_conv_channels(
    ::mlx::core::array x, ::mlx::core::array w, int groups, bool stride_one, bool idil_one) {
    const int n = static_cast<int>(x.ndim()) - 2;
    const int C = static_cast<int>(x.shape(n + 1));
    const int O = static_cast<int>(w.shape(0));
    AlignedConv out{std::move(x), std::move(w), O, false};
    if (groups != 1 || !idil_one || (n != 2 && n != 3) || C == 0 || O == 0)
        return out;
    const auto up16 = [](int v) { return (v + 15) / 16 * 16 - v; };
    const bool o_aligned = O <= 16 || O % 16 == 0;
    int extra_c = 0;
    int extra_o = 0;
    if (n == 2) {
        const auto k_area = out.w.shape(1) * out.w.shape(2);
        if (stride_one && k_area >= 9 && C > 4 && C % 16 != 0 && o_aligned)
            extra_c = up16(C);
    } else {
        if (C % 16 != 0)
            extra_c = up16(C);
        if (!o_aligned)
            extra_o = up16(O);
    }
    if (extra_c == 0 && extra_o == 0)
        return out;
    const auto zero_width = [](const ::mlx::core::array& a, int axis, int extra) {
        std::vector<std::pair<int, int>> width(a.ndim(), {0, 0});
        width[static_cast<std::size_t>(axis)] = {0, extra};
        return ::mlx::core::pad(a, width, ::mlx::core::array(static_cast<float>(0.0), a.dtype()));
    };
    if (extra_c > 0) {
        out.x = zero_width(out.x, n + 1, extra_c);
        out.w = zero_width(out.w, n + 1, extra_c);
    }
    if (extra_o > 0) {
        out.w = zero_width(out.w, 0, extra_o);
        out.sliced = true;
    }
    return out;
}

bool GpuBackend::gpu_conv_has_empty_operand(const ::mlx::core::array& x,
                                            const ::mlx::core::array& w,
                                            std::size_t out_numel) {
    return x.size() == 0 || w.size() == 0 || out_numel == 0;
}

::mlx::core::array GpuBackend::gpu_conv_bias_only(const ::mlx::core::array& b,
                                                  const Shape& out_shape) {
    ::mlx::core::Shape b_brd(out_shape.size(), 1);
    b_brd[1] = static_cast<::mlx::core::ShapeElem>(out_shape[1]);
    return ::mlx::core::contiguous(
        ::mlx::core::broadcast_to(::mlx::core::reshape(b, b_brd), gpu::to_mlx_shape(out_shape)));
}

std::vector<Storage> GpuBackend::gpu_conv_empty_grads(const ::mlx::core::array& x,
                                                      const ::mlx::core::array& w,
                                                      ::mlx::core::array db,
                                                      Dtype dt) {
    return {Storage{gpu::wrap_mlx_array(::mlx::core::zeros_like(x), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::zeros_like(w), dt)},
            Storage{gpu::wrap_mlx_array(std::move(db), dt)}};
}

::mlx::core::array GpuBackend::gpu_conv_general(const ::mlx::core::array& x,
                                                const ::mlx::core::array& w,
                                                const std::vector<int>& stride,
                                                const std::vector<int>& pad_lo,
                                                const std::vector<int>& pad_hi,
                                                const std::vector<int>& kernel_dilation,
                                                const std::vector<int>& input_dilation,
                                                int groups,
                                                bool flip) {
    const auto all_one = [](const std::vector<int>& v) {
        return std::all_of(v.begin(), v.end(), [](int e) { return e == 1; });
    };
    auto a = gpu_align_conv_channels(x, w, groups, all_one(stride), all_one(input_dilation));
    auto y = ::mlx::core::conv_general(a.x, a.w, stride, pad_lo, pad_hi, kernel_dilation,
                                       input_dilation, groups, flip);
    if (a.sliced)
        y = gpu_slice_axis(y, static_cast<int>(y.ndim()) - 1, 0, a.out_channels);
    return y;
}

::mlx::core::array GpuBackend::gpu_mlx_conv_grouped(const ::mlx::core::array& x_nhwc,
                                                    const ::mlx::core::array& W_nhwc,
                                                    const std::vector<int>& stride,
                                                    const std::vector<int>& pad,
                                                    const std::vector<int>& dil,
                                                    int groups,
                                                    int N) {
    const std::vector<int> ones_n(N, 1);
    if (groups > 1 && N == 3) {
        const int in_g = static_cast<int>(x_nhwc.shape(N + 1)) / groups;
        const int out_g = static_cast<int>(W_nhwc.shape(0)) / groups;
        std::vector<::mlx::core::array> parts;
        parts.reserve(static_cast<std::size_t>(groups));
        for (int j = 0; j < groups; ++j)
            parts.push_back(
                gpu_conv_general(gpu_slice_axis(x_nhwc, N + 1, j * in_g, (j + 1) * in_g),
                                 gpu_slice_axis(W_nhwc, 0, j * out_g, (j + 1) * out_g), stride, pad,
                                 pad, dil, ones_n, /*groups=*/1, /*flip=*/false));
        return ::mlx::core::concatenate(std::move(parts), /*axis=*/N + 1);
    }
    return gpu_conv_general(x_nhwc, W_nhwc, stride, pad, pad, dil, ones_n, groups,
                            /*flip=*/false);
}

::mlx::core::array GpuBackend::gpu_mlx_conv_transpose(const ::mlx::core::array& x_nhwc,
                                                      const ::mlx::core::array& W_nhwc,
                                                      const int* stride,
                                                      const int* pad,
                                                      const int* dil,
                                                      const int* opad,
                                                      int groups,
                                                      int N) {
    // MLX refuses grouping above two spatial dimensions ("Can only
    // handle groups != 1 in 1D or 2D convolutions"), so the rank-5
    // grouped case runs one ungrouped convolution per group over
    // channel slices.  Identical result, ``groups`` kernel launches.
    if (groups > 1 && N == 3) {
        const int in_g = static_cast<int>(x_nhwc.shape(N + 1)) / groups;
        const int out_g = static_cast<int>(W_nhwc.shape(0)) / groups;
        std::vector<::mlx::core::array> parts;
        parts.reserve(static_cast<std::size_t>(groups));
        for (int j = 0; j < groups; ++j)
            parts.push_back(gpu_mlx_conv_transpose(
                gpu_slice_axis(x_nhwc, N + 1, j * in_g, (j + 1) * in_g),
                gpu_slice_axis(W_nhwc, 0, j * out_g, (j + 1) * out_g), stride, pad, dil, opad,
                /*groups=*/1, N));
        return ::mlx::core::concatenate(std::move(parts), /*axis=*/N + 1);
    }
    if (N == 1)
        return ::mlx::core::conv_transpose1d(x_nhwc, W_nhwc, stride[0], pad[0], dil[0], opad[0],
                                             groups);
    // MLX runs a transposed convolution as a stride-1 one whose input is
    // dilated by ``stride``, so that is the convolution to align for.
    const bool idil_one = std::all_of(stride, stride + N, [](int e) { return e == 1; });
    auto a = gpu_align_conv_channels(x_nhwc, W_nhwc, groups, /*stride_one=*/true, idil_one);
    ::mlx::core::array y =
        N == 2 ? ::mlx::core::conv_transpose2d(a.x, a.w, std::pair<int, int>{stride[0], stride[1]},
                                               std::pair<int, int>{pad[0], pad[1]},
                                               std::pair<int, int>{dil[0], dil[1]},
                                               std::pair<int, int>{opad[0], opad[1]}, groups)
               : ::mlx::core::conv_transpose3d(
                     a.x, a.w, std::tuple<int, int, int>{stride[0], stride[1], stride[2]},
                     std::tuple<int, int, int>{pad[0], pad[1], pad[2]},
                     std::tuple<int, int, int>{dil[0], dil[1], dil[2]},
                     std::tuple<int, int, int>{opad[0], opad[1], opad[2]}, groups);
    if (a.sliced)
        y = gpu_slice_axis(y, N + 1, 0, a.out_channels);
    return y;
}

}  // namespace backend
}  // namespace lucid
