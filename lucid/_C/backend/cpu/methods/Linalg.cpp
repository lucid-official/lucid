// lucid/_C/backend/cpu/methods/Linalg.cpp
// Out-of-line CpuBackend method definitions for the Linalg family.
#include "../CpuBackend.h"

namespace lucid {
namespace backend {

Storage CpuBackend::linalg_norm(const Storage& a,
                                const Shape& shape,
                                double ord,
                                const std::vector<int>& axes,
                                bool keepdims,
                                Dtype dt) {
    Shape out_shape = reduced_norm_shape(shape, axes, keepdims);
    const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);
    if (dt == Dtype::F32) {
        norm_typed(reinterpret_cast<const float*>(cs.ptr.get()),
                   reinterpret_cast<float*>(ptr.get()), shape, axes, ord, out_shape);
    } else if (dt == Dtype::F64) {
        norm_typed(reinterpret_cast<const double*>(cs.ptr.get()),
                   reinterpret_cast<double*>(ptr.get()), shape, axes, ord, out_shape);
    } else {
        ErrorBuilder("cpu_backend::linalg_norm").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::linalg_cholesky(const Storage& a, const Shape& shape, bool upper, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
    if (cs.nbytes > 0)
        std::memcpy(ptr.get(), cs.ptr.get(), cs.nbytes);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    const bool lower = !upper;
    int info = 0;
    if (dt == Dtype::F32) {
        auto* p = reinterpret_cast<float*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_cholesky_f32(p + b * per_mat, n, lower, &info);
            check_lapack_info(info, "cholesky", b, batch);
        }
    } else if (dt == Dtype::F64) {
        auto* p = reinterpret_cast<double*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_cholesky_f64(p + b * per_mat, n, lower, &info);
            check_lapack_info(info, "cholesky", b, batch);
        }
    } else {
        ErrorBuilder("cpu_backend::linalg_cholesky").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, cs.nbytes, dt}};
}

Storage CpuBackend::linalg_inv(const Storage& a, const Shape& shape, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
    if (cs.nbytes > 0)
        std::memcpy(ptr.get(), cs.ptr.get(), cs.nbytes);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    int info = 0;
    if (dt == Dtype::F32) {
        auto* p = reinterpret_cast<float*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_inv_f32(p + b * per_mat, n, &info);
            check_lapack_info(info, "inv", b, batch);
        }
    } else if (dt == Dtype::F64) {
        auto* p = reinterpret_cast<double*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_inv_f64(p + b * per_mat, n, &info);
            check_lapack_info(info, "inv", b, batch);
        }
    } else {
        ErrorBuilder("cpu_backend::linalg_inv").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, cs.nbytes, dt}};
}

Storage CpuBackend::linalg_solve(
    const Storage& a, const Storage& b, const Shape& a_shape, const Shape& b_shape, Dtype dt) {
    const auto& a_cpu = std::get<CpuStorage>(a);
    const auto& b_cpu = std::get<CpuStorage>(b);
    auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
    if (b_cpu.nbytes > 0)
        std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);

    const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
    const bool b_is_vec = (b_shape.size() == a_shape.size() - 1);
    const int nrhs = b_is_vec ? 1 : static_cast<int>(b_shape[b_shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(a_shape, 2);
    const std::size_t a_per = static_cast<std::size_t>(n) * n;
    const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;
    int info = 0;
    if (dt == Dtype::F32) {
        std::vector<float> A_local(a_per);
        const auto* a_p = reinterpret_cast<const float*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<float*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(float));
            cpu::lapack_solve_f32(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
            check_lapack_info(info, "solve", bi, batch);
        }
    } else if (dt == Dtype::F64) {
        std::vector<double> A_local(a_per);
        const auto* a_p = reinterpret_cast<const double*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<double*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            std::memcpy(A_local.data(), a_p + bi * a_per, a_per * sizeof(double));
            cpu::lapack_solve_f64(A_local.data(), x_p + bi * b_per, n, nrhs, &info);
            check_lapack_info(info, "solve", bi, batch);
        }
    } else {
        ErrorBuilder("cpu_backend::linalg_solve").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
}

Storage CpuBackend::linalg_matrix_power(const Storage& a, const Shape& shape, int power, Dtype dt) {
    const auto& cs = std::get<CpuStorage>(a);
    auto ptr = allocate_aligned_bytes(cs.nbytes, Device::CPU);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    const int reps = std::abs(power);

    auto run = [&](auto type_tag) {
        using T = decltype(type_tag);
        const T* in_p = reinterpret_cast<const T*>(cs.ptr.get());
        T* out_p = reinterpret_cast<T*>(ptr.get());
        std::vector<T> base(per_mat), tmp(per_mat);
        for (std::int64_t b = 0; b < batch; ++b) {
            T* result = out_p + b * per_mat;
            if (power == 0) {
                set_matrix_identity(result, n);
                continue;
            }
            std::memcpy(base.data(), in_p + b * per_mat, per_mat * sizeof(T));
            if (power < 0) {
                int info = 0;
                if constexpr (std::is_same_v<T, float>)
                    cpu::lapack_inv_f32(base.data(), n, &info);
                else
                    cpu::lapack_inv_f64(base.data(), n, &info);
                check_lapack_info(info, "matrix_power");
            }
            std::memcpy(result, base.data(), per_mat * sizeof(T));
            for (int i = 1; i < reps; ++i) {
                if constexpr (std::is_same_v<T, float>)
                    cpu::sgemm(false, false, n, n, n, 1.0f, result, n, base.data(), n, 0.0f,
                               tmp.data(), n);
                else
                    cpu::dgemm(false, false, n, n, n, 1.0, result, n, base.data(), n, 0.0,
                               tmp.data(), n);
                std::memcpy(result, tmp.data(), per_mat * sizeof(T));
            }
        }
    };

    if (dt == Dtype::F32)
        run(float{});
    else if (dt == Dtype::F64)
        run(double{});
    else
        ErrorBuilder("cpu_backend::linalg_matrix_power").not_implemented("dtype not supported");
    return Storage{CpuStorage{ptr, cs.nbytes, dt}};
}

Storage CpuBackend::linalg_pinv(const Storage& a, const Shape& shape, Dtype dt) {
    const int m = static_cast<int>(shape[shape.size() - 2]);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    Shape out_shape(shape.begin(), shape.end() - 2);
    out_shape.push_back(n);
    out_shape.push_back(m);

    const auto& cs = std::get<CpuStorage>(a);
    const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t out_per = static_cast<std::size_t>(n) * m;

    if (dt == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
        auto* out_p = reinterpret_cast<float*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b)
            pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
    } else if (dt == Dtype::F64) {
        const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
        auto* out_p = reinterpret_cast<double*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b)
            pinv_one(in_p + b * in_per, m, n, out_p + b * out_per);
    } else {
        ErrorBuilder("cpu_backend::linalg_pinv").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

Storage CpuBackend::linalg_det(const Storage& a, const Shape& shape, Dtype dt) {
    const int n = static_cast<int>(shape[shape.size() - 1]);
    Shape out_shape(shape.begin(), shape.end() - 2);
    const std::size_t out_nbytes = shape_numel(out_shape) * dtype_size(dt);
    auto ptr = allocate_aligned_bytes(out_nbytes, Device::CPU);

    const auto& cs = std::get<CpuStorage>(a);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    std::vector<int> ipiv(n);
    int info = 0;

    // A singular matrix (info > 0) still factors to the end, with an exact
    // zero on U's diagonal: the product through it is a signed zero, as on
    // Metal, where a literal 0 was not.  (The reference factors Aᵀ, so an
    // asymmetric singular matrix can take the other sign there.)
    if (dt == Dtype::F32) {
        std::vector<float> l(per_mat), u(per_mat);
        const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
        auto* out_p = reinterpret_cast<float*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_lu_f32(in_p + b * per_mat, n, ipiv.data(), l.data(), u.data(), &info);
            if (info < 0)
                check_lapack_info(info, "det");
            float det = ipiv_sign(ipiv.data(), n);
            for (int i = 0; i < n; ++i)
                det *= u[i * n + i];
            out_p[b] = det;
        }
    } else if (dt == Dtype::F64) {
        std::vector<double> l(per_mat), u(per_mat);
        const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
        auto* out_p = reinterpret_cast<double*>(ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_lu_f64(in_p + b * per_mat, n, ipiv.data(), l.data(), u.data(), &info);
            if (info < 0)
                check_lapack_info(info, "det");
            double det = ipiv_sign(ipiv.data(), n);
            for (int i = 0; i < n; ++i)
                det *= u[i * n + i];
            out_p[b] = det;
        }
    } else {
        ErrorBuilder("cpu_backend::linalg_det").not_implemented("dtype not supported");
    }
    return Storage{CpuStorage{ptr, out_nbytes, dt}};
}

StoragePair CpuBackend::linalg_qr(
    const Storage& a, const Shape& shape, const Shape& q_shape, const Shape& r_shape, Dtype dt) {
    const int m = static_cast<int>(shape[shape.size() - 2]);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const int k = std::min(m, n);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t q_per = static_cast<std::size_t>(m) * k;
    const std::size_t r_per = static_cast<std::size_t>(k) * n;

    const std::size_t q_nbytes = shape_numel(q_shape) * dtype_size(dt);
    const std::size_t r_nbytes = shape_numel(r_shape) * dtype_size(dt);
    auto q_ptr = allocate_aligned_bytes(q_nbytes, Device::CPU);
    auto r_ptr = allocate_aligned_bytes(r_nbytes, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);

    int info = 0;
    if (dt == Dtype::F32) {
        const auto* in_p = reinterpret_cast<const float*>(cs.ptr.get());
        auto* q_p = reinterpret_cast<float*>(q_ptr.get());
        auto* r_p = reinterpret_cast<float*>(r_ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_qr_f32(in_p + b * in_per, m, n, q_p + b * q_per, r_p + b * r_per, &info);
            check_lapack_info(info, "qr");
        }
    } else if (dt == Dtype::F64) {
        const auto* in_p = reinterpret_cast<const double*>(cs.ptr.get());
        auto* q_p = reinterpret_cast<double*>(q_ptr.get());
        auto* r_p = reinterpret_cast<double*>(r_ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            cpu::lapack_qr_f64(in_p + b * in_per, m, n, q_p + b * q_per, r_p + b * r_per, &info);
            check_lapack_info(info, "qr");
        }
    } else {
        ErrorBuilder("cpu_backend::linalg_qr").not_implemented("dtype not supported");
    }

    return {Storage{CpuStorage{q_ptr, q_nbytes, dt}}, Storage{CpuStorage{r_ptr, r_nbytes, dt}}};
}

template <typename T>
void CpuBackend::unpack_eig(
    const T* wr, const T* wi, const T* vr, int n, T* values_out, T* vectors_out) {
    for (int j = 0; j < n; ++j) {
        values_out[2 * j] = wr[j];
        values_out[2 * j + 1] = wi[j];
    }
    if (vectors_out == nullptr)
        return;

    const auto at = [&](int i, int col) { return vr[static_cast<std::size_t>(i) * n + col]; };
    const auto put = [&](int i, int col, T re, T im) {
        const std::size_t k = 2 * (static_cast<std::size_t>(i) * n + col);
        vectors_out[k] = re;
        vectors_out[k + 1] = im;
    };

    int j = 0;
    while (j < n) {
        if (wi[j] == T{0}) {
            for (int i = 0; i < n; ++i)
                put(i, j, at(i, j), T{0});
            ++j;
            continue;
        }
        // A conjugate pair: two real columns, four real numbers per
        // row, becoming two complex entries that are each other's
        // conjugate.  ``j + 1 < n`` is guaranteed by LAPACK, and
        // checked rather than assumed because reading past the last
        // column would be silent.
        if (j + 1 >= n) {
            for (int i = 0; i < n; ++i)
                put(i, j, at(i, j), T{0});
            ++j;
            continue;
        }
        for (int i = 0; i < n; ++i) {
            const T re = at(i, j);
            const T im = at(i, j + 1);
            put(i, j, re, im);
            put(i, j + 1, re, -im);
        }
        j += 2;
    }
}

StoragePair CpuBackend::linalg_eig(const Storage& a,
                                   const Shape& shape,
                                   const Shape& values_shape,
                                   const Shape& vectors_shape,
                                   Dtype dt) {
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n;
    const std::size_t per_w = static_cast<std::size_t>(n);

    // The eigenvalues of a real matrix are complex, so the output
    // dtype is not the input's.  ``complex_for`` keeps the lane width:
    // an f64 matrix whose spectrum came back as complex64 would shed
    // eight decimal digits, which is the same class of quiet loss as
    // dropping the imaginary part outright.
    const Dtype out_dt = complex_for(dt);
    const std::size_t values_nbytes = shape_numel(values_shape) * dtype_size(out_dt);
    const std::size_t vectors_nbytes = shape_numel(vectors_shape) * dtype_size(out_dt);
    auto values_ptr = allocate_aligned_bytes(values_nbytes, Device::CPU);
    auto vectors_ptr = allocate_aligned_bytes(vectors_nbytes, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);

    int info = 0;
    auto run_eig = [&](auto tag, auto fn) {
        using T = decltype(tag);
        const auto* in_p = reinterpret_cast<const T*>(cs.ptr.get());
        auto* w_p = reinterpret_cast<T*>(values_ptr.get());
        auto* v_p = reinterpret_cast<T*>(vectors_ptr.get());
        std::vector<T> wr(n), wi(n), vr(per_mat);
        for (std::int64_t b = 0; b < batch; ++b) {
            require_finite_eig(in_p + b * per_mat, per_mat);
            fn(in_p + b * per_mat, n, wr.data(), wi.data(), vr.data(), &info);
            check_lapack_info(info, "eig");
            unpack_eig<T>(wr.data(), wi.data(), vr.data(), n, w_p + 2 * b * per_w,
                          v_p + 2 * b * per_mat);
        }
    };
    if (dt == Dtype::F32)
        run_eig(float{}, cpu::lapack_eig_f32);
    else if (dt == Dtype::F64)
        run_eig(double{}, cpu::lapack_eig_f64);
    else
        ErrorBuilder("cpu_backend::linalg_eig").not_implemented("dtype not supported");

    return {Storage{CpuStorage{values_ptr, values_nbytes, out_dt}},
            Storage{CpuStorage{vectors_ptr, vectors_nbytes, out_dt}}};
}

StoragePair CpuBackend::linalg_eigh(const Storage& a,
                                    const Shape& shape,
                                    const Shape& values_shape,
                                    const Shape& vectors_shape,
                                    Dtype dt,
                                    bool upper) {
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t per_mat = static_cast<std::size_t>(n) * n,
                      per_w = static_cast<std::size_t>(n);
    auto w_cpu = alloc_cpu(shape_numel(values_shape), dt);
    auto v_cpu = alloc_cpu(shape_numel(vectors_shape), dt);
    const auto& cs = std::get<CpuStorage>(a);
    const char uplo = upper ? 'U' : 'L';
    int info = 0;
    auto run_eigh = [&](auto tag, auto fn) {
        using T = decltype(tag);
        const auto* in = reinterpret_cast<const T*>(cs.ptr.get());
        auto* wp = reinterpret_cast<T*>(w_cpu.ptr.get());
        auto* vp = reinterpret_cast<T*>(v_cpu.ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            const T* mat = in + b * per_mat;
            check_eigh_triangle_finite(mat, n, upper, b, batch);
            fn(mat, n, wp + b * per_w, vp + b * per_mat, &info, uplo);
            check_lapack_info(info, "eigh", b, batch);
        }
    };
    if (dt == Dtype::F32)
        run_eigh(float{}, cpu::lapack_eigh_f32);
    else if (dt == Dtype::F64)
        run_eigh(double{}, cpu::lapack_eigh_f64);
    else
        ErrorBuilder("cpu_backend::linalg_eigh").not_implemented("dtype not supported");
    return {Storage{std::move(w_cpu)}, Storage{std::move(v_cpu)}};
}

std::vector<Storage> CpuBackend::linalg_svd(const Storage& a,
                                            const Shape& shape,
                                            bool compute_uv,
                                            const Shape& u_shape,
                                            const Shape& s_shape,
                                            const Shape& vt_shape,
                                            Dtype dt) {
    const int m = static_cast<int>(shape[shape.size() - 2]);
    const int n = static_cast<int>(shape[shape.size() - 1]);
    const int k = std::min(m, n);
    const std::int64_t batch = leading_matrix_batch_count(shape, 2);
    const std::size_t in_per = static_cast<std::size_t>(m) * n;
    const std::size_t s_per = static_cast<std::size_t>(k);
    const std::size_t u_per = static_cast<std::size_t>(m) * k;
    const std::size_t vt_per = static_cast<std::size_t>(k) * n;
    const std::size_t s_nbytes = shape_numel(s_shape) * dtype_size(dt);
    auto s_ptr = allocate_aligned_bytes(s_nbytes, Device::CPU);
    const auto& cs = std::get<CpuStorage>(a);

    if (!compute_uv) {
        int info = 0;
        auto run_vals = [&](auto tag, auto fn) {
            using T = decltype(tag);
            std::vector<T> u(u_per), vt(vt_per);
            const auto* in_p = reinterpret_cast<const T*>(cs.ptr.get());
            auto* s_p = reinterpret_cast<T*>(s_ptr.get());
            for (std::int64_t b = 0; b < batch; ++b) {
                if (!all_finite(in_p + b * in_per, in_per)) {
                    fill_nan(s_p + b * s_per, s_per);
                    continue;
                }
                fn(in_p + b * in_per, m, n, false, u.data(), s_p + b * s_per, vt.data(), &info);
                check_lapack_info(info, "svd", b, batch);
            }
        };
        if (dt == Dtype::F32)
            run_vals(float{}, cpu::lapack_svd_f32);
        else if (dt == Dtype::F64)
            run_vals(double{}, cpu::lapack_svd_f64);
        else
            ErrorBuilder("cpu_backend::linalg_svd").not_implemented("dtype not supported");
        return {Storage{CpuStorage{s_ptr, s_nbytes, dt}}};
    }

    const std::size_t u_nbytes = shape_numel(u_shape) * dtype_size(dt);
    const std::size_t vt_nbytes = shape_numel(vt_shape) * dtype_size(dt);
    auto u_ptr = allocate_aligned_bytes(u_nbytes, Device::CPU);
    auto vt_ptr = allocate_aligned_bytes(vt_nbytes, Device::CPU);
    int info = 0;
    auto run_uv = [&](auto tag, auto fn) {
        using T = decltype(tag);
        const auto* in_p = reinterpret_cast<const T*>(cs.ptr.get());
        auto* u_p = reinterpret_cast<T*>(u_ptr.get());
        auto* s_p = reinterpret_cast<T*>(s_ptr.get());
        auto* vt_p = reinterpret_cast<T*>(vt_ptr.get());
        for (std::int64_t b = 0; b < batch; ++b) {
            if (!all_finite(in_p + b * in_per, in_per)) {
                if (k >= 3)
                    check_lapack_info(2, "svd", b, batch);
                fill_nan(u_p + b * u_per, u_per);
                fill_nan(s_p + b * s_per, s_per);
                fill_nan(vt_p + b * vt_per, vt_per);
                continue;
            }
            fn(in_p + b * in_per, m, n, false, u_p + b * u_per, s_p + b * s_per, vt_p + b * vt_per,
               &info);
            check_lapack_info(info, "svd", b, batch);
        }
    };
    if (dt == Dtype::F32)
        run_uv(float{}, cpu::lapack_svd_f32);
    else if (dt == Dtype::F64)
        run_uv(double{}, cpu::lapack_svd_f64);
    else
        ErrorBuilder("cpu_backend::linalg_svd").not_implemented("dtype not supported");
    return {Storage{CpuStorage{u_ptr, u_nbytes, dt}}, Storage{CpuStorage{s_ptr, s_nbytes, dt}},
            Storage{CpuStorage{vt_ptr, vt_nbytes, dt}}};
}

std::vector<Storage> CpuBackend::linalg_lu_factor(const Storage& a, const Shape& shape, Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_lu_factor").not_implemented("only F32/F64 supported");
    if (shape.size() < 2)
        ErrorBuilder("cpu_backend::linalg_lu_factor")
            .invalid_argument("input must be at least 2-D");
    return cpu::lu_factor(std::get<CpuStorage>(a), shape, dt);
}

Storage CpuBackend::linalg_solve_triangular(const Storage& a,
                                            const Storage& b,
                                            const Shape& a_shape,
                                            const Shape& b_shape,
                                            bool upper,
                                            bool unitriangular,
                                            Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_solve_triangular").not_implemented("only F32/F64");
    const auto& a_cpu = std::get<CpuStorage>(a);
    const auto& b_cpu = std::get<CpuStorage>(b);
    const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
    const bool b_is_vec = (b_shape.size() == a_shape.size() - 1);
    const int nrhs = b_is_vec ? 1 : static_cast<int>(b_shape[b_shape.size() - 1]);
    const std::int64_t batch = leading_matrix_batch_count(a_shape, 2);
    const std::size_t a_per = static_cast<std::size_t>(n) * n;
    const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;
    // trsm solves in place, so the solution starts as a copy of B.
    auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
    if (b_cpu.nbytes > 0)
        std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);
    if (dt == Dtype::F32) {
        const auto* a_p = reinterpret_cast<const float*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<float*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi)
            cpu::strsm(upper, unitriangular, n, nrhs, a_p + bi * a_per, n, x_p + bi * b_per, nrhs);
    } else {
        const auto* a_p = reinterpret_cast<const double*>(a_cpu.ptr.get());
        auto* x_p = reinterpret_cast<double*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi)
            cpu::dtrsm(upper, unitriangular, n, nrhs, a_p + bi * a_per, n, x_p + bi * b_per, nrhs);
    }
    return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
}

std::vector<Storage> CpuBackend::linalg_lstsq(
    const Storage& a, const Storage& b, const Shape& a_shape, const Shape& b_shape, Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_lstsq").not_implemented("only F32/F64");
    // Rank first: ``a_shape.size() - 2`` is unsigned, so a rank-1 A wraps to a
    // colossal index, reads past the shape, and hands the garbage on as ``m`` —
    // which then sizes every buffer here.  Seen as both a 1.8e19-byte allocation
    // request and a SIGSEGV, depending on what the garbage happened to be.
    if (a_shape.size() < 2)
        ErrorBuilder("cpu_backend::linalg_lstsq").invalid_argument("A must be at least 2-D");
    const auto& a_cpu = std::get<CpuStorage>(a);
    const auto& b_cpu = std::get<CpuStorage>(b);
    const int m = static_cast<int>(a_shape[a_shape.size() - 2]);
    const int n = static_cast<int>(a_shape[a_shape.size() - 1]);
    const int nrhs = (b_shape.size() > 1) ? static_cast<int>(b_shape[b_shape.size() - 1]) : 1;
    const int ldb = std::max(m, n);

    // A copy (sgels overwrites A)
    std::size_t a_nb = static_cast<std::size_t>(m) * n * dtype_size(dt);
    std::size_t b_nb = static_cast<std::size_t>(ldb) * nrhs * dtype_size(dt);
    auto a_ptr = allocate_aligned_bytes(a_nb, Device::CPU);
    // The copy below takes ``m`` rows out of B without asking how many B has.
    // When B was shorter the read ran off the end — returning a plausible answer
    // whenever it landed on mapped memory, and segfaulting inside ``memmove``
    // when it did not.
    if (b_cpu.nbytes < static_cast<std::size_t>(m) * nrhs * dtype_size(dt))
        ErrorBuilder("cpu_backend::linalg_lstsq").fail("B has fewer rows than A");

    auto b_ptr = allocate_aligned_bytes(b_nb, Device::CPU);
    std::memset(b_ptr.get(), 0, b_nb);  // zero-pad to max(m,n)

    int info = 0;
    if (dt == Dtype::F32) {
        std::memcpy(a_ptr.get(), a_cpu.ptr.get(), a_nb);
        // Copy B into first m rows
        auto* bp = reinterpret_cast<float*>(b_ptr.get());
        const auto* bsrc = reinterpret_cast<const float*>(b_cpu.ptr.get());
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < nrhs; ++c)
                bp[r * nrhs + c] = bsrc[r * nrhs + c];
        cpu::lapack_lstsq_f32(reinterpret_cast<float*>(a_ptr.get()), bp, m, n, nrhs, &info);
    } else {
        std::memcpy(a_ptr.get(), a_cpu.ptr.get(), a_nb);
        auto* bp = reinterpret_cast<double*>(b_ptr.get());
        const auto* bsrc = reinterpret_cast<const double*>(b_cpu.ptr.get());
        for (int r = 0; r < m; ++r)
            for (int c = 0; c < nrhs; ++c)
                bp[r * nrhs + c] = bsrc[r * nrhs + c];
        cpu::lapack_lstsq_f64(reinterpret_cast<double*>(a_ptr.get()), bp, m, n, nrhs, &info);
    }
    check_lapack_info(info, "lstsq");

    // Solution shape: (n, nrhs) — first n rows of B
    std::size_t sol_nb = static_cast<std::size_t>(n) * nrhs * dtype_size(dt);
    auto sol_ptr = allocate_aligned_bytes(sol_nb, Device::CPU);
    std::memcpy(sol_ptr.get(), b_ptr.get(), sol_nb);

    return {Storage{CpuStorage{sol_ptr, sol_nb, dt}}};
}

Storage CpuBackend::linalg_lu_solve(const Storage& LU,
                                    const Storage& pivots,
                                    const Storage& b,
                                    const Shape& lu_shape,
                                    const Shape& b_shape,
                                    Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_lu_solve").not_implemented("only F32/F64");
    const auto& lu_cpu = std::get<CpuStorage>(LU);
    const auto& piv_cpu = std::get<CpuStorage>(pivots);
    const auto& b_cpu = std::get<CpuStorage>(b);
    const int n = static_cast<int>(lu_shape[lu_shape.size() - 1]);
    const int nrhs = (b_shape.size() > 1) ? static_cast<int>(b_shape[b_shape.size() - 1]) : 1;
    const std::int64_t batch = leading_matrix_batch_count(lu_shape, 2);

    auto out_ptr = allocate_aligned_bytes(b_cpu.nbytes, Device::CPU);
    std::memcpy(out_ptr.get(), b_cpu.ptr.get(), b_cpu.nbytes);

    const std::size_t lu_per = static_cast<std::size_t>(n) * n;
    const std::size_t b_per = static_cast<std::size_t>(n) * nrhs;
    const auto* ipiv = reinterpret_cast<const int*>(piv_cpu.ptr.get());
    int info = 0;
    if (dt == Dtype::F32) {
        const auto* lup = reinterpret_cast<const float*>(lu_cpu.ptr.get());
        auto* xp = reinterpret_cast<float*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            cpu::lapack_lu_solve_f32(lup + bi * lu_per, ipiv + bi * n, xp + bi * b_per, n, nrhs,
                                     &info);
            check_lapack_info(info, "lu_solve", bi, batch);
        }
    } else {
        const auto* lup = reinterpret_cast<const double*>(lu_cpu.ptr.get());
        auto* xp = reinterpret_cast<double*>(out_ptr.get());
        for (std::int64_t bi = 0; bi < batch; ++bi) {
            cpu::lapack_lu_solve_f64(lup + bi * lu_per, ipiv + bi * n, xp + bi * b_per, n, nrhs,
                                     &info);
            check_lapack_info(info, "lu_solve", bi, batch);
        }
    }
    return Storage{CpuStorage{out_ptr, b_cpu.nbytes, dt}};
}

Storage CpuBackend::linalg_householder_product(const Storage& H,
                                               const Storage& tau,
                                               const Shape& h_shape,
                                               Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_householder_product").not_implemented("only F32/F64");
    // Same unsigned wrap as ``linalg_lstsq``: a rank-1 H indexed at
    // ``size() - 2`` reads past the shape, and what it finds becomes the row
    // count every buffer here is sized from.
    if (h_shape.size() < 2)
        ErrorBuilder("cpu_backend::linalg_householder_product")
            .invalid_argument("H must be at least 2-D");
    const auto& h_cpu = std::get<CpuStorage>(H);
    const auto& tau_cpu = std::get<CpuStorage>(tau);
    const int m = static_cast<int>(h_shape[h_shape.size() - 2]);
    const int n = static_cast<int>(h_shape[h_shape.size() - 1]);
    const int k = std::min(m, n);

    // Q shape: m × k
    const std::size_t q_nb = static_cast<std::size_t>(m) * k * dtype_size(dt);
    auto q_ptr = allocate_aligned_bytes(q_nb, Device::CPU);
    int info = 0;
    if (dt == Dtype::F32) {
        cpu::lapack_householder_product_f32(reinterpret_cast<const float*>(h_cpu.ptr.get()),
                                            reinterpret_cast<const float*>(tau_cpu.ptr.get()),
                                            reinterpret_cast<float*>(q_ptr.get()), m, n, k, &info);
    } else {
        cpu::lapack_householder_product_f64(reinterpret_cast<const double*>(h_cpu.ptr.get()),
                                            reinterpret_cast<const double*>(tau_cpu.ptr.get()),
                                            reinterpret_cast<double*>(q_ptr.get()), m, n, k, &info);
    }
    check_lapack_info(info, "householder_product");
    return Storage{CpuStorage{q_ptr, q_nb, dt}};
}

std::vector<Storage> CpuBackend::linalg_ldl_factor(const Storage& a, const Shape& shape, Dtype dt) {
    if (dt != Dtype::F32 && dt != Dtype::F64)
        ErrorBuilder("cpu_backend::linalg_ldl_factor").not_implemented("only F32/F64");
    return cpu::ldl_factor(std::get<CpuStorage>(a), shape, dt);
}

Shape CpuBackend::reduced_norm_shape(const Shape& shape,
                                     const std::vector<int>& axes,
                                     bool keepdims) {
    if (axes.empty()) {
        if (keepdims)
            return Shape(shape.size(), 1);
        return Shape{};
    }
    std::vector<bool> mask(shape.size(), false);
    for (int axis : axes) {
        const int wrapped = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
        mask[static_cast<std::size_t>(wrapped)] = true;
    }
    Shape out;
    for (std::size_t d = 0; d < shape.size(); ++d) {
        if (mask[d]) {
            if (keepdims)
                out.push_back(1);
        } else {
            out.push_back(shape[d]);
        }
    }
    return out;
}

template <typename T, typename Fn>
void CpuBackend::norm_elementwise_loop(const T* in,
                                       const Shape& shape,
                                       const std::vector<bool>& reduce_mask,
                                       const Shape& out_shape,
                                       Fn fn) {
    const std::size_t nd = shape.size();
    Stride in_stride(nd), out_stride(out_shape.size());
    if (nd > 0) {
        in_stride[nd - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(nd) - 2; i >= 0; --i)
            in_stride[static_cast<std::size_t>(i)] =
                in_stride[static_cast<std::size_t>(i) + 1] * shape[static_cast<std::size_t>(i) + 1];
    }
    if (!out_shape.empty()) {
        const std::size_t ond = out_shape.size();
        out_stride[ond - 1] = 1;
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(ond) - 2; i >= 0; --i)
            out_stride[static_cast<std::size_t>(i)] = out_stride[static_cast<std::size_t>(i) + 1] *
                                                      out_shape[static_cast<std::size_t>(i) + 1];
    }

    // ``out_shape`` keeps the reduced axes as size-1 under keepdims and drops
    // them otherwise.  A surviving input axis therefore maps to output axis
    // ``d`` in the keepdims layout, but to "number of surviving axes before
    // d" when they were dropped.  Using the packed counter for BOTH (the
    // previous behaviour) over-counted the stride under keepdims and wrote
    // past ``acc`` — e.g. (2,3,4) reduced over dim 1 with keepdims indexes up
    // to 16 into an 8-element buffer: a heap overflow that segfaulted.  A 2-D
    // input happened to survive because the mis-picked stride equals 1 there.
    const bool keepdims_layout = out_shape.size() == nd;
    std::vector<std::int64_t> coord(nd, 0);
    const std::size_t in_numel = shape_numel(shape);
    for (std::size_t f = 0; f < in_numel; ++f) {
        std::size_t in_flat = 0;
        std::size_t out_flat = 0;
        std::size_t out_axis = 0;
        for (std::size_t d = 0; d < nd; ++d) {
            in_flat += static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(in_stride[d]);
            if (!reduce_mask[d]) {
                const std::size_t oa = keepdims_layout ? d : out_axis;
                out_flat +=
                    static_cast<std::size_t>(coord[d]) * static_cast<std::size_t>(out_stride[oa]);
                ++out_axis;
            }
        }
        fn(in[in_flat], out_flat);
        for (std::ptrdiff_t d = static_cast<std::ptrdiff_t>(nd) - 1; d >= 0; --d) {
            if (++coord[static_cast<std::size_t>(d)] < shape[static_cast<std::size_t>(d)])
                break;
            coord[static_cast<std::size_t>(d)] = 0;
        }
    }
}

template <typename T>
void CpuBackend::norm_typed(const T* in,
                            T* out,
                            const Shape& shape,
                            const std::vector<int>& axes,
                            double ord,
                            const Shape& out_shape) {
    std::vector<bool> reduce_mask(shape.size(), false);
    if (axes.empty()) {
        for (std::size_t i = 0; i < shape.size(); ++i)
            reduce_mask[i] = true;
    } else {
        for (int axis : axes) {
            const int wrapped = axis < 0 ? axis + static_cast<int>(shape.size()) : axis;
            reduce_mask[static_cast<std::size_t>(wrapped)] = true;
        }
    }
    const std::size_t out_numel = std::max<std::size_t>(1, shape_numel(out_shape));
    if (std::isinf(ord)) {
        const bool positive = ord > 0;
        std::vector<T> acc(out_numel, positive ? T{0} : std::numeric_limits<T>::infinity());
        norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
            const T av = std::abs(v);
            acc[o] = positive ? std::max(acc[o], av) : std::min(acc[o], av);
        });
        std::memcpy(out, acc.data(), out_numel * sizeof(T));
        return;
    }
    if (ord == 0.0) {
        std::vector<T> acc(out_numel, T{0});
        norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
            if (v != T{0})
                acc[o] += T{1};
        });
        std::memcpy(out, acc.data(), out_numel * sizeof(T));
        return;
    }
    std::vector<T> acc(out_numel, T{0});
    norm_elementwise_loop<T>(in, shape, reduce_mask, out_shape, [&](T v, std::size_t o) {
        const T av = std::abs(v);
        if (ord == 2.0)
            acc[o] += v * v;
        else if (ord == 1.0)
            acc[o] += av;
        else
            acc[o] += std::pow(av, static_cast<T>(ord));
    });
    if (ord == 2.0) {
        for (std::size_t i = 0; i < out_numel; ++i)
            acc[i] = std::sqrt(acc[i]);
    } else if (ord != 1.0) {
        const T inv_ord = static_cast<T>(1.0 / ord);
        for (std::size_t i = 0; i < out_numel; ++i)
            acc[i] = std::pow(acc[i], inv_ord);
    }
    std::memcpy(out, acc.data(), out_numel * sizeof(T));
}

std::int64_t CpuBackend::leading_matrix_batch_count(const Shape& shape, std::size_t mat_dims) {
    std::int64_t batch = 1;
    for (std::size_t i = 0; i + mat_dims < shape.size(); ++i)
        batch *= shape[i];
    return batch;
}

template <typename T>
bool CpuBackend::all_finite(const T* p, std::size_t n) {
    return std::all_of(p, p + n, [](T v) { return std::isfinite(v); });
}

template <typename T>
void CpuBackend::fill_nan(T* p, std::size_t n) {
    std::fill(p, p + n, std::numeric_limits<T>::quiet_NaN());
}

template <typename T>
void CpuBackend::require_finite_eig(const T* p, std::size_t n) {
    if (!all_finite(p, n))
        ErrorBuilder("eig").invalid_argument("input should not contain infs or NaNs");
}

template <typename T>
void CpuBackend::set_matrix_identity(T* out, int n) {
    const std::size_t total = static_cast<std::size_t>(n) * n;
    std::memset(out, 0, total * sizeof(T));
    for (int i = 0; i < n; ++i)
        out[i * n + i] = T{1};
}

template <typename T>
void CpuBackend::pinv_one(const T* a, int m, int n, T* aplus) {
    const int k = std::min(m, n);
    // Past k = 2, gesdd never returns on an infinity (LCD-833), so a non-finite
    // matrix is refused before it gets there, as linalg_svd refuses it.
    if (k >= 3 && !all_finite(a, static_cast<std::size_t>(m) * n))
        check_lapack_info(2, "pinv");
    // A NaN makes gesdd report an illegal argument; infinities go through.
    if (std::any_of(a, a + static_cast<std::size_t>(m) * n, [](T v) { return std::isnan(v); })) {
        fill_nan(aplus, static_cast<std::size_t>(n) * m);
        return;
    }
    std::vector<T> u(static_cast<std::size_t>(m) * k);
    std::vector<T> s(k);
    std::vector<T> vt(static_cast<std::size_t>(k) * n);
    int info = 0;
    if constexpr (std::is_same_v<T, float>)
        cpu::lapack_svd_f32(a, m, n, false, u.data(), s.data(), vt.data(), &info);
    else
        cpu::lapack_svd_f64(a, m, n, false, u.data(), s.data(), vt.data(), &info);
    check_lapack_info(info, "pinv");

    const T smax = (k > 0) ? *std::max_element(s.begin(), s.end()) : T{0};
    const T rcond = std::numeric_limits<T>::epsilon() * static_cast<T>(std::max(m, n));
    const T cutoff = rcond * smax;

    std::vector<T> s_inv_ut(static_cast<std::size_t>(k) * m);
    for (int i = 0; i < k; ++i) {
        const T inv = (s[i] > cutoff) ? T{1} / s[i] : T{0};
        for (int j = 0; j < m; ++j)
            s_inv_ut[i * m + j] = inv * u[j * k + i];
    }

    if constexpr (std::is_same_v<T, float>) {
        cpu::sgemm(true, false, n, m, k, 1.0f, vt.data(), n, s_inv_ut.data(), m, 0.0f, aplus, m);
    } else {
        cpu::dgemm(true, false, n, m, k, 1.0, vt.data(), n, s_inv_ut.data(), m, 0.0, aplus, m);
    }
}

float CpuBackend::ipiv_sign(const int* ipiv, int n) {
    int swaps = 0;
    for (int i = 0; i < n; ++i)
        if (ipiv[i] != i + 1)
            ++swaps;
    return (swaps % 2 == 0) ? 1.0f : -1.0f;
}

}  // namespace backend
}  // namespace lucid
