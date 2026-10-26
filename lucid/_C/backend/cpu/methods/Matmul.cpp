// lucid/_C/backend/cpu/methods/Matmul.cpp
// Out-of-line CpuBackend method definitions for the Matmul family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::matmul(const Storage& a, const Storage& b, const MatmulOpts& opts, Dtype dt) {
    // There is no integer BLAS either, and the product of two integer
    // matrices is an integer matrix — the reference framework and Metal
    // both answer it, and the CPU raised.  Taken in double and rounded
    // back: exact as long as each dot product stays inside 2^53, which
    // covers int32 and below at any realistic K.  int64 operands large
    // enough to break that would already have overflowed their own
    // type in a native integer GEMM.
    if (detail::is_int_like(dt)) {
        const std::size_t out_n = opts.batch * static_cast<std::size_t>(opts.M) * opts.N;
        const Storage wide = matmul(detail::as_f64(a), detail::as_f64(b), opts, Dtype::F64);
        return detail::f64_to_int(wide, out_n, dt);
    }
    // No half BLAS to call, so the product is taken in single and
    // rounded once.  Accumulating in the wider type is what a half
    // GEMM would do internally anyway.
    if (detail::is_half_like(dt)) {
        const std::size_t a_n = opts.batch * static_cast<std::size_t>(opts.M) * opts.K;
        const std::size_t b_n = opts.batch * static_cast<std::size_t>(opts.K) * opts.N;
        const std::size_t out_n = opts.batch * static_cast<std::size_t>(opts.M) * opts.N;
        const Storage wide =
            matmul(Storage{detail::widen_half(std::get<CpuStorage>(a), a_n)},
                   Storage{detail::widen_half(std::get<CpuStorage>(b), b_n)}, opts, Dtype::F32);
        return detail::narrow_half(wide, out_n, dt);
    }
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    const int M = opts.M, K = opts.K, N = opts.N;
    const std::size_t batch = opts.batch;
    std::size_t out_n = batch * static_cast<std::size_t>(M) * static_cast<std::size_t>(N);
    std::size_t nb = out_n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    std::memset(ptr.get(), 0, nb);

    // Degenerate dims (M / N / K == 0): the zeroed (or empty) output is
    // already the correct result — a K=0 contraction sums over nothing → 0 —
    // and cblas_*gemm asserts on these ("lda must be >= MAX(K,1)" for K=0),
    // so skip the BLAS call entirely.
    const bool run_gemm = M > 0 && N > 0 && K > 0;
    if (dt == Dtype::F32) {
        const float* ap = reinterpret_cast<const float*>(ca.ptr.get());
        const float* bp = reinterpret_cast<const float*>(cb.ptr.get());
        float* cp = reinterpret_cast<float*>(ptr.get());
        if (run_gemm) {
            for (std::size_t bi = 0; bi < batch; ++bi) {
                cpu::sgemm(opts.transA, opts.transB, M, N, K, 1.0f, ap + bi * M * K,
                           opts.transA ? M : K, bp + bi * K * N, opts.transB ? K : N, 0.0f,
                           cp + bi * M * N, N);
            }
        }
    } else if (dt == Dtype::F64) {
        const double* ap = reinterpret_cast<const double*>(ca.ptr.get());
        const double* bp = reinterpret_cast<const double*>(cb.ptr.get());
        double* cp = reinterpret_cast<double*>(ptr.get());
        if (run_gemm) {
            for (std::size_t bi = 0; bi < batch; ++bi) {
                cpu::dgemm(opts.transA, opts.transB, M, N, K, 1.0, ap + bi * M * K,
                           opts.transA ? M : K, bp + bi * K * N, opts.transB ? K : N, 0.0,
                           cp + bi * M * N, N);
            }
        }
    } else {
        ErrorBuilder("cpu_backend::matmul").not_implemented("dtype not supported");
    }
    Shape out_shape;
    if (batch > 1)
        out_shape.push_back(static_cast<std::int64_t>(batch));
    out_shape.push_back(M);
    out_shape.push_back(N);
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::linear(const Storage& x,
                           const Storage& weight,
                           const Storage& bias,
                           const Shape& x_shape,
                           const Shape& weight_shape,
                           const Shape& out_shape,
                           Dtype dt) {
    // The one that blocks a real half model: nn.Linear is the first
    // layer of almost everything, so `model.half()` ran on Metal and
    // raised on the CPU.  No half BLAS exists, so the product is taken
    // in single and rounded once — what a half GEMM does internally.
    if (detail::is_half_like(dt)) {
        const std::size_t x_n = shape_numel(x_shape);
        const std::size_t w_n = shape_numel(weight_shape);
        const auto& cb = std::get<CpuStorage>(bias);
        const bool has_bias = cb.ptr != nullptr && cb.nbytes > 0;
        const Storage wide = linear(
            Storage{detail::widen_half(std::get<CpuStorage>(x), x_n)},
            Storage{detail::widen_half(std::get<CpuStorage>(weight), w_n)},
            has_bias ? Storage{detail::widen_half(cb, cb.nbytes / sizeof(std::uint16_t))} : bias,
            x_shape, weight_shape, out_shape, Dtype::F32);
        return detail::narrow_half(wide, shape_numel(out_shape), dt);
    }
    const auto [M, K] = flatten_linear_x(x_shape);
    const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
    CpuStorage out{allocate_aligned_bytes(M * N * dtype_size(dt), Device::CPU),
                   M * N * dtype_size(dt), dt};
    if (M > 0 && N > 0 && K > 0) {
        MatmulOpts opts;
        opts.transA = false;
        opts.transB = true;
        opts.M = static_cast<int>(M);
        opts.K = static_cast<int>(K);
        opts.N = static_cast<int>(N);
        opts.batch = 1;
        out = std::get<CpuStorage>(matmul(x, weight, opts, dt));
    } else if (out.nbytes) {
        std::memset(out.ptr.get(), 0, out.nbytes);
    }
    if (M > 0 && N > 0)
        add_linear_bias(out, std::get<CpuStorage>(bias), M, N, dt);
    (void)out_shape;
    return Storage{std::move(out)};
}

std::vector<Storage> CpuBackend::linear_backward(const Storage& grad,
                                                 const Storage& x,
                                                 const Storage& weight,
                                                 const Shape& x_shape,
                                                 const Shape& weight_shape,
                                                 const Shape& bias_shape,
                                                 Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(linear_backward(detail::as_f32(grad), detail::as_f32(x),
                                                   detail::as_f32(weight), x_shape, weight_shape,
                                                   bias_shape, Dtype::F32),
                                   dt);
    const auto [M, K] = flatten_linear_x(x_shape);
    const std::size_t N = static_cast<std::size_t>(weight_shape[0]);
    const std::size_t elem = dtype_size(dt);
    CpuStorage dx{allocate_aligned_bytes(M * K * elem, Device::CPU), M * K * elem, dt};
    CpuStorage dW{allocate_aligned_bytes(N * K * elem, Device::CPU), N * K * elem, dt};
    CpuStorage db{allocate_aligned_bytes(shape_numel(bias_shape) * elem, Device::CPU),
                  shape_numel(bias_shape) * elem, dt};

    if (M > 0 && N > 0 && K > 0) {
        MatmulOpts dx_opts;
        dx_opts.M = static_cast<int>(M);
        dx_opts.K = static_cast<int>(N);
        dx_opts.N = static_cast<int>(K);
        dx_opts.batch = 1;
        dx = std::get<CpuStorage>(matmul(grad, weight, dx_opts, dt));

        MatmulOpts dW_opts;
        dW_opts.transA = true;
        dW_opts.M = static_cast<int>(N);
        dW_opts.K = static_cast<int>(M);
        dW_opts.N = static_cast<int>(K);
        dW_opts.batch = 1;
        dW = std::get<CpuStorage>(matmul(grad, x, dW_opts, dt));

        sum_linear_rows(std::get<CpuStorage>(grad), db, M, N, dt);
    } else {
        if (dx.nbytes)
            std::memset(dx.ptr.get(), 0, dx.nbytes);
        if (dW.nbytes)
            std::memset(dW.ptr.get(), 0, dW.nbytes);
        if (db.nbytes)
            std::memset(db.ptr.get(), 0, db.nbytes);
    }
    return {Storage{std::move(dx)}, Storage{std::move(dW)}, Storage{std::move(db)}};
}

Storage CpuBackend::bilinear_layer_forward(const Storage& x1,
                                           const Storage& x2,
                                           const Storage& weight,
                                           const Storage& bias,
                                           bool has_bias,
                                           const Shape& x1_shape,
                                           const Shape& x2_shape,
                                           const Shape& w_shape,
                                           Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(bilinear_layer_forward(detail::as_f32(x1), detail::as_f32(x2),
                                                          detail::as_f32(weight),
                                                          detail::as_f32(bias), has_bias, x1_shape,
                                                          x2_shape, w_shape, Dtype::F32),
                                   dt);
    std::size_t B = 1;
    for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
        B *= static_cast<std::size_t>(x1_shape[i]);
    const std::size_t D1 = static_cast<std::size_t>(x1_shape.back());
    const std::size_t D2 = static_cast<std::size_t>(x2_shape.back());
    const std::size_t Dout = static_cast<std::size_t>(w_shape[0]);
    const auto& xs1 = std::get<CpuStorage>(x1);
    const auto& xs2 = std::get<CpuStorage>(x2);
    const auto& ws = std::get<CpuStorage>(weight);
    const CpuStorage* bs = has_bias ? &std::get<CpuStorage>(bias) : nullptr;
    auto out_cpu = alloc_cpu(B * Dout, dt);
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* x1p = reinterpret_cast<const T*>(xs1.ptr.get());
        const T* x2p = reinterpret_cast<const T*>(xs2.ptr.get());
        const T* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const T* bp = bs ? reinterpret_cast<const T*>(bs->ptr.get()) : nullptr;
        T* yp = reinterpret_cast<T*>(out_cpu.ptr.get());
        for (std::size_t bi = 0; bi < B; ++bi) {
            const T* x1b = x1p + bi * D1;
            const T* x2b = x2p + bi * D2;
            T* yb = yp + bi * Dout;
            for (std::size_t k = 0; k < Dout; ++k) {
                const T* Wk = wp + k * D1 * D2;
                T acc = T{0};
                for (std::size_t i = 0; i < D1; ++i) {
                    T row = T{0};
                    for (std::size_t j = 0; j < D2; ++j)
                        row += Wk[i * D2 + j] * x2b[j];
                    acc += x1b[i] * row;
                }
                yb[k] = acc + (bp ? bp[k] : T{0});
            }
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::bilinear_layer_forward").not_implemented("dtype");
    return Storage{std::move(out_cpu)};
}

std::vector<Storage> CpuBackend::bilinear_layer_backward(const Storage& grad_out,
                                                         const Storage& x1,
                                                         const Storage& x2,
                                                         const Storage& weight,
                                                         const Shape& x1_shape,
                                                         const Shape& x2_shape,
                                                         const Shape& w_shape,
                                                         bool has_bias,
                                                         Dtype dt) {
    if (detail::is_half_like(dt))
        return detail::back_to_f16(bilinear_layer_backward(detail::as_f32(grad_out),
                                                           detail::as_f32(x1), detail::as_f32(x2),
                                                           detail::as_f32(weight), x1_shape,
                                                           x2_shape, w_shape, has_bias, Dtype::F32),
                                   dt);
    std::size_t B = 1;
    for (std::size_t i = 0; i + 1 < x1_shape.size(); ++i)
        B *= static_cast<std::size_t>(x1_shape[i]);
    const std::size_t D1 = static_cast<std::size_t>(x1_shape.back());
    const std::size_t D2 = static_cast<std::size_t>(x2_shape.back());
    const std::size_t Dout = static_cast<std::size_t>(w_shape[0]);
    const auto& xs1 = std::get<CpuStorage>(x1);
    const auto& xs2 = std::get<CpuStorage>(x2);
    const auto& ws = std::get<CpuStorage>(weight);
    const auto& gys = std::get<CpuStorage>(grad_out);
    auto dx1_cpu = alloc_cpu(B * D1, dt);
    auto dx2_cpu = alloc_cpu(B * D2, dt);
    auto dW_cpu = alloc_cpu(Dout * D1 * D2, dt);
    auto db_cpu = has_bias ? alloc_cpu(Dout, dt) : CpuStorage{};
    auto run = [&](auto tag) {
        using T = decltype(tag);
        const T* x1p = reinterpret_cast<const T*>(xs1.ptr.get());
        const T* x2p = reinterpret_cast<const T*>(xs2.ptr.get());
        const T* wp = reinterpret_cast<const T*>(ws.ptr.get());
        const T* gyp = reinterpret_cast<const T*>(gys.ptr.get());
        T* dx1p = reinterpret_cast<T*>(dx1_cpu.ptr.get());
        T* dx2p = reinterpret_cast<T*>(dx2_cpu.ptr.get());
        T* dwp = reinterpret_cast<T*>(dW_cpu.ptr.get());
        T* dbp = has_bias ? reinterpret_cast<T*>(db_cpu.ptr.get()) : nullptr;
        std::memset(dx1p, 0, sizeof(T) * B * D1);
        std::memset(dx2p, 0, sizeof(T) * B * D2);
        std::memset(dwp, 0, sizeof(T) * Dout * D1 * D2);
        if (dbp)
            std::memset(dbp, 0, sizeof(T) * Dout);
        for (std::size_t bi = 0; bi < B; ++bi) {
            const T* x1b = x1p + bi * D1;
            const T* x2b = x2p + bi * D2;
            const T* gb = gyp + bi * Dout;
            T* dx1b = dx1p + bi * D1;
            T* dx2b = dx2p + bi * D2;
            for (std::size_t k = 0; k < Dout; ++k) {
                const T g = gb[k];
                if (dbp)
                    dbp[k] += g;
                const T* Wk = wp + k * D1 * D2;
                T* dWk = dwp + k * D1 * D2;
                for (std::size_t i = 0; i < D1; ++i) {
                    T row = T{0};
                    for (std::size_t j = 0; j < D2; ++j) {
                        row += Wk[i * D2 + j] * x2b[j];
                        dx2b[j] += g * x1b[i] * Wk[i * D2 + j];
                        dWk[i * D2 + j] += g * x1b[i] * x2b[j];
                    }
                    dx1b[i] += g * row;
                }
            }
        }
    };
    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu::bilinear_layer_backward").not_implemented("dtype");
    std::vector<Storage> out;
    out.push_back(Storage{std::move(dx1_cpu)});
    out.push_back(Storage{std::move(dx2_cpu)});
    out.push_back(Storage{std::move(dW_cpu)});
    if (has_bias)
        out.push_back(Storage{std::move(db_cpu)});
    else {
        CpuStorage empty;
        empty.dtype = dt;
        empty.nbytes = 0;
        out.push_back(Storage{std::move(empty)});
    }
    return out;
}

std::pair<std::size_t, std::size_t> CpuBackend::flatten_linear_x(const Shape& x_shape) {
    std::size_t M = 1;
    for (std::size_t d = 0; d + 1 < x_shape.size(); ++d)
        M *= static_cast<std::size_t>(x_shape[d]);
    return {M, static_cast<std::size_t>(x_shape.back())};
}

template <typename T>
void CpuBackend::add_linear_bias_typed(T* y, const T* b, std::size_t M, std::size_t N) {
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n)
            y[m * N + n] += b[n];
    }
}

void CpuBackend::add_linear_bias(
    CpuStorage& y, const CpuStorage& bias, std::size_t M, std::size_t N, Dtype dt) {
    if (dt == Dtype::F32)
        add_linear_bias_typed(reinterpret_cast<float*>(y.ptr.get()),
                              reinterpret_cast<const float*>(bias.ptr.get()), M, N);
    else if (dt == Dtype::F64)
        add_linear_bias_typed(reinterpret_cast<double*>(y.ptr.get()),
                              reinterpret_cast<const double*>(bias.ptr.get()), M, N);
    else
        ErrorBuilder("cpu_backend::linear").not_implemented("dtype not supported");
}

template <typename T>
void CpuBackend::sum_linear_rows_typed(const T* grad, T* db, std::size_t M, std::size_t N) {
    for (std::size_t n = 0; n < N; ++n)
        db[n] = T{};
    for (std::size_t m = 0; m < M; ++m) {
        for (std::size_t n = 0; n < N; ++n)
            db[n] += grad[m * N + n];
    }
}

void CpuBackend::sum_linear_rows(
    const CpuStorage& grad, CpuStorage& db, std::size_t M, std::size_t N, Dtype dt) {
    if (dt == Dtype::F32)
        sum_linear_rows_typed(reinterpret_cast<const float*>(grad.ptr.get()),
                              reinterpret_cast<float*>(db.ptr.get()), M, N);
    else if (dt == Dtype::F64)
        sum_linear_rows_typed(reinterpret_cast<const double*>(grad.ptr.get()),
                              reinterpret_cast<double*>(db.ptr.get()), M, N);
    else
        ErrorBuilder("cpu_backend::linear_backward").not_implemented("dtype not supported");
}

Storage CpuBackend::inner(const Storage& a,
                          const Storage& b,
                          const Shape& a_shape,
                          const Shape& b_shape,
                          const Shape& out_shape,
                          Dtype dt) {
    // Accelerate has no half kernels, so the whole family widens at the door,
    // reuses the float path and rounds once on the way out.  as_f32 /
    // back_to_f16 key off each storage's own dtype, so index and mask
    // operands travel through untouched.  See detail::widen_half.
    if (detail::is_half_like(dt))
        return detail::back_to_f16(
            inner(detail::as_f32(a), detail::as_f32(b), a_shape, b_shape, out_shape, Dtype::F32),
            dt);
    const auto& ca = std::get<CpuStorage>(a);
    const auto& cb = std::get<CpuStorage>(b);
    const std::int64_t K = a_shape.back();
    const std::size_t pa = shape_numel(Shape(a_shape.begin(), a_shape.end() - 1));
    const std::size_t pb = shape_numel(Shape(b_shape.begin(), b_shape.end() - 1));
    std::size_t n = shape_numel(out_shape);
    std::size_t nb = n * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(nb, Device::CPU);
    auto run = [&](auto* op, const auto* ap, const auto* bp) {
        using T = std::remove_pointer_t<decltype(op)>;
        for (std::size_t i = 0; i < pa; ++i)
            for (std::size_t j = 0; j < pb; ++j) {
                T s{};
                for (std::int64_t k = 0; k < K; ++k)
                    s = s + ap[i * K + k] * bp[j * K + k];
                op[i * pb + j] = s;
            }
    };
    if (dt == Dtype::F32)
        run(reinterpret_cast<float*>(ptr.get()), reinterpret_cast<const float*>(ca.ptr.get()),
            reinterpret_cast<const float*>(cb.ptr.get()));
    else if (dt == Dtype::F64)
        run(reinterpret_cast<double*>(ptr.get()), reinterpret_cast<const double*>(ca.ptr.get()),
            reinterpret_cast<const double*>(cb.ptr.get()));
    else if (dt == Dtype::C64)
        run(reinterpret_cast<std::complex<float>*>(ptr.get()),
            reinterpret_cast<const std::complex<float>*>(ca.ptr.get()),
            reinterpret_cast<const std::complex<float>*>(cb.ptr.get()));
    else
        ErrorBuilder("cpu_backend::inner").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, nb, dt}};
}

Storage CpuBackend::tensordot(const Storage&,
                              const Storage&,
                              const Shape&,
                              const Shape&,
                              const Shape&,
                              const std::vector<int>&,
                              const std::vector<int>&,
                              Dtype) {
    ErrorBuilder("cpu_backend::tensordot").not_implemented("not routed through backend");
}

Storage CpuBackend::fused_linear_relu_forward(
    const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
    if (dt != Dtype::F32)
        return IBackend::fused_linear_relu_forward(x, w, b, out_shape, dt);

    const std::size_t elem = dtype_size(dt);
    std::size_t M = 1;
    for (std::size_t i = 0; i + 1 < out_shape.size(); ++i)
        M *= static_cast<std::size_t>(out_shape[i]);
    const std::size_t N_out = static_cast<std::size_t>(out_shape.back());
    const std::size_t K = std::get<CpuStorage>(x).nbytes / (elem * (M > 0 ? M : 1));
    const Shape x_shape = {static_cast<std::int64_t>(M), static_cast<std::int64_t>(K)};
    const Shape weight_shape = {static_cast<std::int64_t>(N_out), static_cast<std::int64_t>(K)};

    Storage lin_out = linear(x, w, b, x_shape, weight_shape, out_shape, dt);

#ifdef __APPLE__
    {
        auto& cpu = std::get<CpuStorage>(lin_out);
        const std::size_t n = cpu.nbytes / sizeof(float);
        float* p = reinterpret_cast<float*>(cpu.ptr.get());
        cpu::vrelu_f32(p, p, n);
    }
#else
    {
        auto& cpu = std::get<CpuStorage>(lin_out);
        const std::size_t n = cpu.nbytes / sizeof(float);
        float* p = reinterpret_cast<float*>(cpu.ptr.get());
        for (std::size_t i = 0; i < n; ++i)
            if (p[i] < 0.f)
                p[i] = 0.f;
    }
#endif
    return lin_out;
}

Storage CpuBackend::fused_linear_gelu_forward(
    const Storage& x, const Storage& w, const Storage& b, const Shape& out_shape, Dtype dt) {
    if (dt != Dtype::F32)
        return IBackend::fused_linear_gelu_forward(x, w, b, out_shape, dt);

    const std::size_t elem_g = dtype_size(dt);
    std::size_t M_g = 1;
    for (std::size_t i = 0; i + 1 < out_shape.size(); ++i)
        M_g *= static_cast<std::size_t>(out_shape[i]);
    const std::size_t N_g = static_cast<std::size_t>(out_shape.back());
    const std::size_t K_g = std::get<CpuStorage>(x).nbytes / (elem_g * (M_g > 0 ? M_g : 1));
    const Shape x_shape_g = {static_cast<std::int64_t>(M_g), static_cast<std::int64_t>(K_g)};
    const Shape weight_shape_g = {static_cast<std::int64_t>(N_g), static_cast<std::int64_t>(K_g)};

    Storage lin_out = linear(x, w, b, x_shape_g, weight_shape_g, out_shape, dt);

    auto& cpu = std::get<CpuStorage>(lin_out);
    const std::size_t n = cpu.nbytes / sizeof(float);
    float* p = reinterpret_cast<float*>(cpu.ptr.get());

    std::vector<float> scratch(n);
    constexpr float kSqrt2OverPi = 0.7978845608f;
    constexpr float kCoeff = 0.044715f;

    for (std::size_t i = 0; i < n; ++i) {
        const float xi = p[i];
        scratch[i] = kSqrt2OverPi * (xi + kCoeff * xi * xi * xi);
    }
    cpu::vtanh_unfixed_f32(scratch.data(), scratch.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        p[i] = 0.5f * p[i] * (1.f + scratch[i]);

    return lin_out;
}

}  // namespace backend
}  // namespace lucid
