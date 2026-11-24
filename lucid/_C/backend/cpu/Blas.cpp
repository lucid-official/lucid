// lucid/_C/backend/cpu/Blas.cpp
//
// Implements the BLAS wrapper functions declared in Blas.h by delegating
// directly to cblas_{s,d,c,z}gemm, cblas_{s,d}gemv, cblas_{s,d}axpy and
// cblas_{s,d,c,z}trsm from Apple Accelerate.  All calls use CblasRowMajor
// storage order because Lucid tensors are row-major by default.
//
// Empty extents
// -------------
// Every wrapper answers a zero extent itself and never hands one to
// Accelerate.  BLAS requires each leading dimension to be at least 1 even
// when the extent it describes is 0, and a row-major caller naturally
// passes the extent as the leading dimension — ``ldc = N`` for an
// ``M x 0`` product.  Accelerate's error handler does not return to the
// caller: it prints
//
//     BLAS error: Parameter number 14 passed to cblas_sgemm had an invalid value
//
// and exits the process with status 255.  That is how an attention with no
// keys killed the interpreter.  The arithmetic is defined for every empty
// case, so the wrappers apply it directly:
//
// - an output with no elements (``M == 0`` or ``N == 0``) needs no work;
// - an empty contraction (``K == 0``) makes ``alpha * A @ B`` the zero
//   matrix, leaving ``C = beta * C``.  With ``beta == 0`` BLAS does not
//   read ``C`` at all, so ``C`` is overwritten with zeros rather than
//   scaled — scaling would carry a NaN in uninitialised memory through
//   ``0 * NaN``;
// - a triangular solve with no rows or no right-hand sides (``M == 0`` or
//   ``N == 0``) has nothing to solve — a row-major ``M x 0`` right-hand
//   side would otherwise reach BLAS as ``ldb = 0``.

#include "Blas.h"

#include <cstdlib>

#include <Accelerate/Accelerate.h>

namespace lucid::backend::cpu {

namespace {
// Converts a bool transpose flag to the CBLAS enum expected by Accelerate.
inline CBLAS_TRANSPOSE T(bool t) {
    return t ? CblasTrans : CblasNoTrans;
}

// Converts the triangle and unit-diagonal flags of a triangular operand.
inline CBLAS_UPLO Uplo(bool upper) {
    return upper ? CblasUpper : CblasLower;
}

inline CBLAS_DIAG Diag(bool unit) {
    return unit ? CblasUnit : CblasNonUnit;
}

// ``C <- beta * C`` over a row-major ``M x N`` block with row stride ``ldc``,
// writing zeros when ``beta == 0`` (BLAS does not read ``C`` then).
template <typename F>
void scale_matrix(int M, int N, F beta, F* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        F* row = C + static_cast<std::ptrdiff_t>(i) * ldc;
        if (beta == F{0}) {
            for (int j = 0; j < N; ++j)
                row[j] = F{0};
        } else if (beta != F{1}) {
            for (int j = 0; j < N; ++j)
                row[j] *= beta;
        }
    }
}

// ``y <- beta * y`` over ``n`` elements spaced ``inc`` apart.  A negative
// stride walks the same elements in the opposite order, which scaling does
// not care about.
template <typename F>
void scale_vector(int n, F beta, F* y, int inc) {
    const std::ptrdiff_t step = std::abs(inc);
    for (int i = 0; i < n; ++i) {
        F& v = y[static_cast<std::ptrdiff_t>(i) * step];
        v = (beta == F{0}) ? F{0} : v * beta;
    }
}

// True when the GEMM was fully answered without calling BLAS.
template <typename F>
bool gemm_empty(int M, int N, int K, F beta, F* C, int ldc) {
    if (M <= 0 || N <= 0)
        return true;
    if (K <= 0) {
        scale_matrix(M, N, beta, C, ldc);
        return true;
    }
    return false;
}

// True when the GEMV was fully answered without calling BLAS.  ``A`` is
// ``M x N``; the output has ``M`` entries (``N`` when transposed) and the
// contraction runs over the other extent.
template <typename F>
bool gemv_empty(bool transA, int M, int N, F beta, F* y, int incy) {
    const int out_len = transA ? N : M;
    const int red_len = transA ? M : N;
    if (out_len <= 0)
        return true;
    if (red_len <= 0) {
        scale_vector(out_len, beta, y, incy);
        return true;
    }
    return false;
}
}  // namespace

void sgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           float alpha,
           const float* A,
           int lda,
           const float* B,
           int ldb,
           float beta,
           float* C,
           int ldc) {
    if (gemm_empty(M, N, K, beta, C, ldc))
        return;
    cblas_sgemm(CblasRowMajor, T(transA), T(transB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void dgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           double alpha,
           const double* A,
           int lda,
           const double* B,
           int ldb,
           double beta,
           double* C,
           int ldc) {
    if (gemm_empty(M, N, K, beta, C, ldc))
        return;
    cblas_dgemm(CblasRowMajor, T(transA), T(transB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void cgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           Complex64 alpha,
           const Complex64* A,
           int lda,
           const Complex64* B,
           int ldb,
           Complex64 beta,
           Complex64* C,
           int ldc) {
    if (gemm_empty(M, N, K, beta, C, ldc))
        return;
    // The complex routines take alpha and beta by address.
    cblas_cgemm(CblasRowMajor, T(transA), T(transB), M, N, K, &alpha, A, lda, B, ldb, &beta, C,
                ldc);
}

void zgemm(bool transA,
           bool transB,
           int M,
           int N,
           int K,
           Complex128 alpha,
           const Complex128* A,
           int lda,
           const Complex128* B,
           int ldb,
           Complex128 beta,
           Complex128* C,
           int ldc) {
    if (gemm_empty(M, N, K, beta, C, ldc))
        return;
    cblas_zgemm(CblasRowMajor, T(transA), T(transB), M, N, K, &alpha, A, lda, B, ldb, &beta, C,
                ldc);
}

void sgemv(bool transA,
           int M,
           int N,
           float alpha,
           const float* A,
           int lda,
           const float* x,
           int incx,
           float beta,
           float* y,
           int incy) {
    if (gemv_empty(transA, M, N, beta, y, incy))
        return;
    cblas_sgemv(CblasRowMajor, T(transA), M, N, alpha, A, lda, x, incx, beta, y, incy);
}

void dgemv(bool transA,
           int M,
           int N,
           double alpha,
           const double* A,
           int lda,
           const double* x,
           int incx,
           double beta,
           double* y,
           int incy) {
    if (gemv_empty(transA, M, N, beta, y, incy))
        return;
    cblas_dgemv(CblasRowMajor, T(transA), M, N, alpha, A, lda, x, incx, beta, y, incy);
}

// Unit strides throughout: the callers accumulate over whole contiguous
// buffers, so exposing incx/incy would be parameters nobody sets.
void saxpy(int n, float alpha, const float* x, float* y) {
    if (n <= 0)
        return;
    cblas_saxpy(n, alpha, x, 1, y, 1);
}

void daxpy(int n, double alpha, const double* x, double* y) {
    if (n <= 0)
        return;
    cblas_daxpy(n, alpha, x, 1, y, 1);
}

// Side Left, no transpose, alpha 1: the only form the callers solve — a
// right-side or transposed system is rewritten as this one above the
// backend.
void strsm(bool upper, bool unit_diag, int M, int N, const float* A, int lda, float* B, int ldb) {
    if (M <= 0 || N <= 0)
        return;
    cblas_strsm(CblasRowMajor, CblasLeft, Uplo(upper), CblasNoTrans, Diag(unit_diag), M, N, 1.0f, A,
                lda, B, ldb);
}

void dtrsm(bool upper, bool unit_diag, int M, int N, const double* A, int lda, double* B, int ldb) {
    if (M <= 0 || N <= 0)
        return;
    cblas_dtrsm(CblasRowMajor, CblasLeft, Uplo(upper), CblasNoTrans, Diag(unit_diag), M, N, 1.0, A,
                lda, B, ldb);
}

void ctrsm(
    bool upper, bool unit_diag, int M, int N, const Complex64* A, int lda, Complex64* B, int ldb) {
    if (M <= 0 || N <= 0)
        return;
    const Complex64 one{1.0f, 0.0f};
    cblas_ctrsm(CblasRowMajor, CblasLeft, Uplo(upper), CblasNoTrans, Diag(unit_diag), M, N, &one, A,
                lda, B, ldb);
}

void ztrsm(bool upper,
           bool unit_diag,
           int M,
           int N,
           const Complex128* A,
           int lda,
           Complex128* B,
           int ldb) {
    if (M <= 0 || N <= 0)
        return;
    const Complex128 one{1.0, 0.0};
    cblas_ztrsm(CblasRowMajor, CblasLeft, Uplo(upper), CblasNoTrans, Diag(unit_diag), M, N, &one, A,
                lda, B, ldb);
}

}  // namespace lucid::backend::cpu
