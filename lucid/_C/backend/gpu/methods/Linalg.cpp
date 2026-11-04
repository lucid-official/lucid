// lucid/_C/backend/gpu/methods/Linalg.cpp
// Out-of-line GpuBackend method definitions for the Linalg family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::linalg_norm(const Storage& a,
                                const Shape&,
                                double ord,
                                const std::vector<int>& axes,
                                bool keepdims,
                                Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    // The sums of powers accumulate in float32 for a half input (see
    // ``widen_half``): the float16 2-norm of 65536 ones squared and
    // summed to inf, where the reference gives 256.
    const auto out_dt = ga.arr->dtype();
    const auto xw = widen_half(*ga.arr);
    if (axes.empty()) {
        // A full reduction is ENTRYWISE: the documented contract for
        // ``lucid.linalg.norm`` is the vector definition applied over every
        // element (Frobenius for a matrix, l2 for a vector) — the Python
        // wrapper maps ord="fro" to 2.0 and has no matrix-norm dispatch
        // (``matrix_norm`` implements spectral / nuclear itself).  MLX's
        // linalg::norm with no axis instead applies *matrix* semantics to a
        // 2-D input (ord=2 -> largest singular value, ord=1 -> max column
        // sum), which silently disagreed with the CPU (Accelerate) stream —
        // same call, different number, device-dependent.  Flatten first so
        // both streams evaluate the same vector norm.
        auto flat = ::mlx::core::reshape(xw, ::mlx::core::Shape{-1});
        auto raw =
            ::mlx::core::linalg::norm(flat, ord, std::nullopt, /*keepdims=*/false, k_linalg_stream);
        if (keepdims)
            raw =
                ::mlx::core::reshape(raw, ::mlx::core::Shape(static_cast<int>(ga.arr->ndim()), 1));
        return Storage{gpu::wrap_mlx_array(narrow_half(::mlx::core::contiguous(raw), out_dt), dt)};
    }
    if (axes.size() >= 2) {
        // MLX likewise switches to *matrix* norms as soon as two axes are
        // reduced together (``dim=[0, 1]``), so evaluate the entrywise
        // p-norm directly to stay consistent with the CPU stream.
        auto absx = ::mlx::core::abs(xw, k_linalg_stream);
        ::mlx::core::array red = absx;
        if (std::isinf(ord)) {
            red = ord > 0 ? ::mlx::core::max(absx, axes, keepdims, k_linalg_stream)
                          : ::mlx::core::min(absx, axes, keepdims, k_linalg_stream);
        } else if (ord == 1.0) {
            red = ::mlx::core::sum(absx, axes, keepdims, k_linalg_stream);
        } else if (ord == 2.0) {
            red = ::mlx::core::sqrt(::mlx::core::sum(::mlx::core::square(absx, k_linalg_stream),
                                                     axes, keepdims, k_linalg_stream),
                                    k_linalg_stream);
        } else {
            auto p = ::mlx::core::power(absx, ::mlx::core::array(static_cast<float>(ord)),
                                        k_linalg_stream);
            red = ::mlx::core::power(::mlx::core::sum(p, axes, keepdims, k_linalg_stream),
                                     ::mlx::core::array(static_cast<float>(1.0 / ord)),
                                     k_linalg_stream);
        }
        return Storage{gpu::wrap_mlx_array(narrow_half(::mlx::core::contiguous(red), out_dt), dt)};
    }
    std::optional<std::vector<int>> axis_opt = axes;
    auto raw = ::mlx::core::linalg::norm(xw, ord, axis_opt, keepdims, k_linalg_stream);
    return Storage{gpu::wrap_mlx_array(narrow_half(::mlx::core::contiguous(raw), out_dt), dt)};
}

Storage GpuBackend::linalg_cholesky(const Storage& a, const Shape&, bool upper, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto src = upper ? ::mlx::core::swapaxes(*ga.arr, -1, -2, k_linalg_stream) : *ga.arr;
    auto lower = ::mlx::core::linalg::cholesky(src, /*upper=*/false, k_linalg_stream);
    auto diag = ::mlx::core::diagonal(lower, 0, -2, -1, k_linalg_stream);
    auto positive =
        ::mlx::core::greater(diag, ::mlx::core::array(0.0f, diag.dtype()), k_linalg_stream);
    gpu::raise_failed_step(::mlx::core::logical_not(positive, k_linalg_stream), "cholesky",
                           k_linalg_stream);
    auto out = upper ? ::mlx::core::swapaxes(lower, -1, -2, k_linalg_stream) : lower;
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage GpuBackend::linalg_inv(const Storage& a, const Shape&, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto out = gpu_lu_inverse(*ga.arr, "inv");
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage GpuBackend::linalg_solve(
    const Storage& a, const Storage& b, const Shape& a_shape, const Shape& b_shape, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const auto& gb = std::get<GpuStorage>(b);
    auto a_inv = gpu_lu_inverse(*ga.arr, "solve");
    // A right-hand side one rank below A is a (batch of) vector(s).
    const bool b_is_vec = (b_shape.size() + 1 == a_shape.size());
    ::mlx::core::array out =
        b_is_vec
            ? ::mlx::core::squeeze(::mlx::core::matmul(a_inv, ::mlx::core::expand_dims(*gb.arr, -1),
                                                       k_linalg_stream),
                                   -1, k_linalg_stream)
            : ::mlx::core::matmul(a_inv, *gb.arr, k_linalg_stream);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(out), dt)};
}

Storage GpuBackend::linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    if (power == 0) {
        auto eye = ::mlx::core::eye(n, n, 0, gpu::to_mlx_dtype(dt));
        if (shape.size() > 2) {
            eye = ::mlx::core::broadcast_to(eye, gpu::to_mlx_shape(shape));
        }
        return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(eye), dt)};
    }
    const int reps = std::abs(power);
    auto base = (power < 0) ? gpu_lu_inverse(*ga.arr, "matrix_power") : *ga.arr;
    ::mlx::core::array result = base;
    for (int i = 1; i < reps; ++i)
        result = ::mlx::core::matmul(result, base);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dt)};
}

Storage GpuBackend::linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) {
    Storage a_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), shape)};
    Storage out_cpu = Dispatcher::for_device(Device::CPU).linalg_pinv(a_cpu, shape, dt);
    Shape out_shape(shape.begin(), shape.end() - 2);
    out_shape.push_back(shape[shape.size() - 1]);
    out_shape.push_back(shape[shape.size() - 2]);
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(out_cpu), out_shape)};
}

Storage GpuBackend::linalg_det(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    auto factors = ::mlx::core::linalg::lu(*ga.arr, k_linalg_stream);
    if (factors.size() < 3)
        ErrorBuilder("gpu_backend::linalg_det").fail("lu returned fewer than 3 factors");

    const auto& p = factors[0];
    const auto& u = factors[2];
    auto diag = ::mlx::core::diagonal(u, 0, -2, -1);
    // Reduce only along the last axis (the diagonal); a no-axis prod() would
    // collapse all batch dimensions into a single scalar.
    auto det_u = ::mlx::core::prod(diag, -1, false, k_linalg_stream);

    Shape out_shape(shape.begin(), shape.end() - 2);
    std::int64_t batch = 1;
    for (std::size_t i = 0; i + 2 < shape.size(); ++i)
        batch *= shape[i];

    auto p_eval = p;
    p_eval.eval();
    const std::size_t matrix_n = static_cast<std::size_t>(n);
    std::vector<float> signs(static_cast<std::size_t>(batch), 1.0f);
    MemoryTracker::track_host_sync(p_eval.nbytes());
    const auto* p_data = p_eval.data<std::uint32_t>();
    for (std::int64_t b = 0; b < batch; ++b)
        signs[static_cast<std::size_t>(b)] = perm_index_sign(p_data + b * matrix_n, matrix_n);

    ::mlx::core::array sign_arr(signs.data(), gpu::to_mlx_shape(out_shape), ::mlx::core::float32);
    if (dt != Dtype::F32)
        sign_arr = ::mlx::core::astype(sign_arr, gpu::to_mlx_dtype(dt));
    auto det_a = ::mlx::core::multiply(det_u, sign_arr);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(det_a), dt)};
}

StoragePair
GpuBackend::linalg_qr(const Storage& a, const Shape&, const Shape&, const Shape&, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto [q, r] = ::mlx::core::linalg::qr(*ga.arr, k_linalg_stream);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(q), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(r), dt)}};
}

StoragePair
GpuBackend::linalg_eig(const Storage& a, const Shape&, const Shape&, const Shape&, Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    // LAPACK's geev rejects a NaN as an illegal argument, and MLX throws
    // that from its worker thread, which aborts the process.  Checked
    // here instead, with the CPU stream's error.
    gpu_require_finite(*ga.arr, "eig");
    auto [w, v] = ::mlx::core::linalg::eig(*ga.arr, k_linalg_stream);
    // MLX returns complex64 here, and tagging it with the real input
    // dtype used to be enough to lose it: since ``wrap_mlx_array``
    // converts rather than relabels, the astype to float32 discarded
    // every imaginary part on the way out.  Same defect as the CPU
    // path had, by a different route.
    const Dtype out_dt = complex_for(dt);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(w), out_dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(v), out_dt)}};
}

StoragePair GpuBackend::linalg_eigh(
    const Storage& a, const Shape& shape, const Shape&, const Shape&, Dtype dt, bool upper) {
    const auto& ga = std::get<GpuStorage>(a);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    if (n >= 3) {
        auto tri = upper ? ::mlx::core::triu(*ga.arr, 0, k_linalg_stream)
                         : ::mlx::core::tril(*ga.arr, 0, k_linalg_stream);
        auto bad = gpu_nonfinite_matrices(tri);
        auto any_bad = ::mlx::core::any(bad, false, k_linalg_stream);
        any_bad.eval();
        MemoryTracker::track_host_sync(any_bad.nbytes());
        if (any_bad.item<bool>())
            check_lapack_info(2, "eigh");
    }
    auto [w, v] = ::mlx::core::linalg::eigh(*ga.arr, gpu::mlx_eigh_uplo(upper), k_linalg_stream);
    return {Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(w), dt)},
            Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(v), dt)}};
}

std::vector<Storage> GpuBackend::linalg_svd(const Storage& a,
                                            const Shape& shape,
                                            bool compute_uv,
                                            const Shape& u_shape,
                                            const Shape&,
                                            const Shape& vt_shape,
                                            Dtype dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto bad = gpu_nonfinite_matrices(*ga.arr);
    const int k = static_cast<int>(std::min(shape[shape.size() - 2], shape[shape.size() - 1]));
    if (compute_uv && k >= 3) {
        auto any_bad = ::mlx::core::any(bad, false, k_linalg_stream);
        any_bad.eval();
        MemoryTracker::track_host_sync(any_bad.nbytes());
        if (any_bad.item<bool>())
            check_lapack_info(2, "svd");
    }
    auto pieces =
        ::mlx::core::linalg::svd(gpu_zero_where(bad, *ga.arr), compute_uv, k_linalg_stream);
    std::vector<Storage> out;
    out.reserve(pieces.size());
    // MLX returns the FULL decomposition (U is m×m, Vh is n×n) whereas the
    // op wrapper's shapes describe the REDUCED one (U m×k, Vh k×n with
    // k = min(m, n)).  Without slicing, the TensorImpl's Shape disagrees
    // with its MLX array and later engine ops read the wrong extents.
    for (std::size_t i = 0; i < pieces.size(); ++i) {
        auto p = pieces[i];
        if (compute_uv && (i == 0 || i == 2)) {
            const Shape& want = (i == 0) ? u_shape : vt_shape;
            ::mlx::core::Shape lo(want.size(), 0);
            ::mlx::core::Shape hi = gpu::to_mlx_shape(want);
            if (hi != p.shape())
                p = ::mlx::core::slice(p, lo, hi);
        }
        // The singular values have one axis fewer than the mask.
        const bool is_values = !compute_uv || i == 1;
        p = gpu_nan_where(is_values ? ::mlx::core::squeeze(bad, -1) : bad, p);
        out.push_back(Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(p), dt)});
    }
    return out;
}

std::vector<Storage> GpuBackend::linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) {
    Storage a_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), shape)};
    auto out = Dispatcher::for_device(Device::CPU).linalg_lu_factor(a_cpu, shape, dt);
    // Must upload: the op wrapper tags every output with the *input*
    // device (Metal), so returning CPU storage desyncs the device tag from
    // the storage variant and every later engine op hits
    // bad_variant_access.  Shapes mirror lu_factor_op: LU keeps the input
    // shape, pivots are batch dims + min(m, n), info the batch dims.  The
    // CPU call has already refused a rank below 2.
    Shape pivot_shape(shape.begin(), shape.end() - 2);
    pivot_shape.push_back(std::min(shape[shape.size() - 2], shape[shape.size() - 1]));
    return gpu::upload_factor(out, shape, pivot_shape);
}

Storage GpuBackend::linalg_solve_triangular(const Storage& a,
                                            const Storage& b,
                                            const Shape& a_shape,
                                            const Shape& b_shape,
                                            bool upper,
                                            bool unitriangular,
                                            Dtype dt) {
    Storage a_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), a_shape)};
    Storage b_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(b), b_shape)};
    Storage x_cpu =
        Dispatcher::for_device(Device::CPU)
            .linalg_solve_triangular(a_cpu, b_cpu, a_shape, b_shape, upper, unitriangular, dt);
    // Must upload: the op wrapper tags the result with the *input* device
    // (Metal), so returning CpuStorage here desyncs the device tag from the
    // storage variant and every later engine op hits bad_variant_access.
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(x_cpu), b_shape)};
}

std::vector<Storage> GpuBackend::linalg_lstsq(
    const Storage& a, const Storage& b, const Shape& a_shape, const Shape& b_shape, Dtype dt) {
    Storage a_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), a_shape)};
    Storage b_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(b), b_shape)};
    auto res = Dispatcher::for_device(Device::CPU).linalg_lstsq(a_cpu, b_cpu, a_shape, b_shape, dt);
    Shape sol_shape(a_shape.begin(), a_shape.end() - 2);
    sol_shape.push_back(a_shape[a_shape.size() - 1]);
    if (b_shape.size() > 1 && b_shape.back() > 1)
        sol_shape.push_back(b_shape.back());
    std::vector<Storage> out;
    out.reserve(res.size());
    for (const auto& r : res)
        out.push_back(Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(r), sol_shape)});
    return out;
}

Storage GpuBackend::linalg_lu_solve(const Storage& LU,
                                    const Storage& pivots,
                                    const Storage& b,
                                    const Shape& lu_shape,
                                    const Shape& b_shape,
                                    Dtype dt) {
    Shape piv_shape(lu_shape.begin(), lu_shape.end() - 1);
    Storage lu_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(LU), lu_shape)};
    Storage piv_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(pivots), piv_shape)};
    Storage b_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(b), b_shape)};
    Storage result = Dispatcher::for_device(Device::CPU)
                         .linalg_lu_solve(lu_cpu, piv_cpu, b_cpu, lu_shape, b_shape, dt);
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(result), b_shape)};
}

Storage GpuBackend::linalg_householder_product(const Storage& H,
                                               const Storage& tau,
                                               const Shape& h_shape,
                                               Dtype dt) {
    const auto& gt = std::get<GpuStorage>(tau);
    Storage h_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(H), h_shape)};
    Storage t_cpu{gpu::download_gpu_to_cpu(gt, gpu::mlx_shape_to_lucid(gt.arr->shape()))};
    Storage result =
        Dispatcher::for_device(Device::CPU).linalg_householder_product(h_cpu, t_cpu, h_shape, dt);
    const std::int64_t m = h_shape[h_shape.size() - 2];
    const std::int64_t n = h_shape[h_shape.size() - 1];
    Shape q_shape(h_shape.begin(), h_shape.end() - 2);
    q_shape.push_back(m);
    q_shape.push_back(std::min(m, n));
    return Storage{gpu::upload_cpu_to_gpu(std::get<CpuStorage>(result), q_shape)};
}

std::vector<Storage> GpuBackend::linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) {
    Storage a_cpu{gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), shape)};
    auto out = Dispatcher::for_device(Device::CPU).linalg_ldl_factor(a_cpu, shape, dt);
    // Shapes mirror ldl_factor_op: LD keeps the input shape, pivots are
    // batch dims + n.  The MLX array must carry the batch dims too — it
    // holds only as many elements as its shape names, and every later
    // read of this tensor copies the TensorImpl's element count out of
    // it.
    return gpu::upload_factor(out, shape, Shape(shape.begin(), shape.end() - 1));
}

::mlx::core::array GpuBackend::gpu_nonfinite_matrices(const ::mlx::core::array& a) {
    auto finite =
        ::mlx::core::all(::mlx::core::isfinite(a, k_linalg_stream), std::vector<int>{-2, -1},
                         /*keepdims=*/true, k_linalg_stream);
    return ::mlx::core::logical_not(finite, k_linalg_stream);
}

::mlx::core::array GpuBackend::gpu_zero_where(const ::mlx::core::array& bad,
                                              const ::mlx::core::array& a) {
    return ::mlx::core::where(bad, ::mlx::core::zeros_like(a, k_linalg_stream), a, k_linalg_stream);
}

::mlx::core::array GpuBackend::gpu_nan_where(const ::mlx::core::array& bad,
                                             const ::mlx::core::array& a) {
    const ::mlx::core::array nan(std::numeric_limits<float>::quiet_NaN(), a.dtype());
    return ::mlx::core::where(bad, nan, a, k_linalg_stream);
}

void GpuBackend::gpu_require_finite(const ::mlx::core::array& a, const char* op) {
    auto finite =
        ::mlx::core::all(::mlx::core::isfinite(a, k_linalg_stream), false, k_linalg_stream);
    finite.eval();
    MemoryTracker::track_host_sync(finite.nbytes());
    if (!finite.item<bool>())
        ErrorBuilder(op).invalid_argument("input should not contain infs or NaNs");
}

::mlx::core::array GpuBackend::gpu_lu_inverse(const ::mlx::core::array& a, const char* op) {
    auto factors = ::mlx::core::linalg::lu(a, k_linalg_stream);
    if (factors.size() < 3)
        ErrorBuilder(op).fail("lu returned fewer than 3 factors");
    const auto& p = factors[0];
    const auto& l = factors[1];
    const auto& u = factors[2];

    auto pivots = ::mlx::core::diagonal(u, 0, -2, -1, k_linalg_stream);
    gpu::raise_failed_step(
        ::mlx::core::equal(pivots, ::mlx::core::array(0.0f, pivots.dtype()), k_linalg_stream), op,
        k_linalg_stream);

    auto u_inv = ::mlx::core::linalg::tri_inv(u, /*upper=*/true, k_linalg_stream);
    auto l_inv = ::mlx::core::linalg::tri_inv(l, /*upper=*/false, k_linalg_stream);
    auto m = ::mlx::core::matmul(u_inv, l_inv, k_linalg_stream);
    auto cols = ::mlx::core::broadcast_to(
        ::mlx::core::expand_dims(::mlx::core::astype(p, ::mlx::core::int32, k_linalg_stream), -2,
                                 k_linalg_stream),
        m.shape(), k_linalg_stream);
    return ::mlx::core::take_along_axis(m, cols, -1, k_linalg_stream);
}

float GpuBackend::perm_index_sign(const std::uint32_t* p, std::size_t n) {
    std::vector<bool> seen(n, false);
    std::size_t cycles = 0;
    for (std::size_t i = 0; i < n; ++i) {
        if (seen[i])
            continue;
        ++cycles;
        std::size_t j = i;
        while (!seen[j]) {
            seen[j] = true;
            j = p[j];
        }
    }
    return ((n - cycles) % 2 == 0) ? 1.0f : -1.0f;
}

}  // namespace backend
}  // namespace lucid
