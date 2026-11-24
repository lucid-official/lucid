// lucid/_C/backend/cpu/Blas.h
//
// Thin wrappers around Apple Accelerate CBLAS routines used by the CPU backend
// for matrix multiplication, matrix-vector multiplication, scaled vector
// accumulation and triangular solves.  All functions assume row-major storage
// (CblasRowMajor) and map the bool flags to the CBLAS enums.  "s" prefix =
// float32; "d" prefix = float64; "c" / "z" = complex64 / complex128.

#pragma once

#include <complex>
#include <cstddef>

#include "../../api.h"

namespace lucid::backend::cpu {

// The complex dtypes' host types, as the complex BLAS routines take them.
using Complex64 = std::complex<float>;
using Complex128 = std::complex<double>;

// Single-precision general matrix multiply (GEMM).
//
// Computes $C \leftarrow \alpha (A B) + \beta C$ in row-major layout using
// Accelerate's ``cblas_sgemm``.  Each transpose flag is mapped to the
// corresponding ``CBLAS_TRANSPOSE`` enum (``CblasNoTrans`` / ``CblasTrans``)
// at zero copy cost — the BLAS kernel selects an alternate inner loop for
// transposed operands.
//
// Parameters
// ----------
// transA, transB : bool
//     Whether $A$ or $B$ should be transposed before the multiply.
// M, N, K : int
//     Output is $M \times N$; the contracted dimension is $K$, so
//     $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$
//     (before optional transposition).
// alpha, beta : float
//     Linear-combination coefficients.  Use $\alpha = 1, \beta = 0$ for a
//     pure multiply; nonzero $\beta$ enables fused accumulate-into-C.
// A, B : const float*
//     Row-major operand buffers.
// C : float*
//     Row-major output buffer; updated in place when $\beta \neq 0$.
// lda, ldb, ldc : int
//     Leading dimensions (row stride in elements) of $A$, $B$, $C$.
//
// Math
// ----
// $$ C_{ij} \leftarrow \alpha \sum_{k=0}^{K-1} A_{ik} B_{kj} + \beta C_{ij} $$
//
// Notes
// -----
// Single-threaded on Apple Silicon for small/medium sizes; Accelerate
// dispatches to AMX (matrix coprocessor) for large GEMMs automatically.
//
// References
// ----------
// BLAS Reference (Netlib), Accelerate.framework ``cblas_sgemm``.
LUCID_INTERNAL void sgemm(bool transA,
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
                          int ldc);

// Double-precision general matrix multiply (GEMM).
//
// Identical contract to ``sgemm`` but operates on ``double`` buffers and
// dispatches to ``cblas_dgemm``.
//
// Parameters
// ----------
// transA, transB : bool
//     Whether $A$ or $B$ should be transposed before the multiply.
// M, N, K : int
//     $C \in \mathbb{R}^{M \times N}$, $A \in \mathbb{R}^{M \times K}$,
//     $B \in \mathbb{R}^{K \times N}$ before optional transposition.
// alpha, beta : double
//     Linear-combination coefficients.
// A, B : const double*
//     Row-major operand buffers.
// C : double*
//     Row-major output buffer.
// lda, ldb, ldc : int
//     Leading dimensions.
//
// Math
// ----
// $$ C_{ij} \leftarrow \alpha \sum_{k=0}^{K-1} A_{ik} B_{kj} + \beta C_{ij} $$
//
// References
// ----------
// Accelerate.framework ``cblas_dgemm``.
LUCID_INTERNAL void dgemm(bool transA,
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
                          int ldc);

// Complex general matrix multiply (GEMM), single and double precision.
//
// ``cgemm`` is ``cblas_cgemm`` over ``complex64`` and ``zgemm`` is
// ``cblas_zgemm`` over ``complex128``; the contract is ``sgemm``'s.  A
// transpose flag is a plain transpose, not a conjugate transpose, as it is
// for the real routines: the engine's backward formulas conjugate around the
// product themselves (``autograd/ComplexGrad.h``).
LUCID_INTERNAL void cgemm(bool transA,
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
                          int ldc);

LUCID_INTERNAL void zgemm(bool transA,
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
                          int ldc);

// Single-precision general matrix-vector multiply (GEMV).
//
// Computes $y \leftarrow \alpha (A x) + \beta y$ in row-major layout via
// Accelerate's ``cblas_sgemv``.  Significantly cheaper than ``sgemm`` for
// vector right-hand sides because the inner loop stays in cache.
//
// Parameters
// ----------
// transA : bool
//     If true, compute $y \leftarrow \alpha A^T x + \beta y$ instead.
// M, N : int
//     $A \in \mathbb{R}^{M \times N}$.  When ``transA`` is false, $x$ has
//     $N$ elements and $y$ has $M$; reversed when ``transA`` is true.
// alpha, beta : float
//     Linear-combination coefficients.
// A : const float*
//     Row-major matrix buffer.
// x : const float*
//     Input vector buffer (stride ``incx``).
// y : float*
//     Output vector buffer (stride ``incy``); updated in place.
// lda : int
//     Leading dimension (row stride) of $A$.
// incx, incy : int
//     Element stride within the input/output vectors.
//
// Math
// ----
// $$ y_i \leftarrow \alpha \sum_{j=0}^{N-1} A_{ij} x_j + \beta y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_sgemv``.
LUCID_INTERNAL void sgemv(bool transA,
                          int M,
                          int N,
                          float alpha,
                          const float* A,
                          int lda,
                          const float* x,
                          int incx,
                          float beta,
                          float* y,
                          int incy);

// Double-precision general matrix-vector multiply (GEMV).
//
// Identical contract to ``sgemv`` but for ``double`` buffers; dispatches to
// ``cblas_dgemv``.
//
// Parameters
// ----------
// transA : bool
//     If true, multiply against $A^T$ instead of $A$.
// M, N : int
//     $A \in \mathbb{R}^{M \times N}$.
// alpha, beta : double
//     Linear-combination coefficients.
// A : const double*
//     Row-major matrix buffer.
// x : const double*
//     Input vector buffer.
// y : double*
//     Output vector buffer; updated in place.
// lda : int
//     Leading dimension of $A$.
// incx, incy : int
//     Element strides of $x$ and $y$.
//
// Math
// ----
// $$ y_i \leftarrow \alpha \sum_{j=0}^{N-1} A_{ij} x_j + \beta y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_dgemv``.
LUCID_INTERNAL void dgemv(bool transA,
                          int M,
                          int N,
                          double alpha,
                          const double* A,
                          int lda,
                          const double* x,
                          int incx,
                          double beta,
                          double* y,
                          int incy);

// Single-precision scaled vector accumulation, $y \leftarrow \alpha x + y$.
//
// The BLAS name for "add a scaled copy of one vector to another", and the
// building block of any linear combination of same-shaped buffers.  Preferred
// over a multiply into a temporary followed by an add: it needs no temporary,
// touches $y$ once per term instead of three times, and Accelerate's kernel
// contracts the multiply and the add into a single rounding.
//
// Parameters
// ----------
// n : int
//     Element count.
// alpha : float
//     Scale applied to $x$.
// x : const float*
//     Vector to accumulate; unchanged.
// y : float*
//     Accumulator, updated in place.
//
// Math
// ----
// $$ y_i \leftarrow \alpha x_i + y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_saxpy``.
LUCID_INTERNAL void saxpy(int n, float alpha, const float* x, float* y);

// Double-precision scaled vector accumulation, $y \leftarrow \alpha x + y$.
//
// Parameters
// ----------
// n : int
//     Element count.
// alpha : double
//     Scale applied to $x$.
// x : const double*
//     Vector to accumulate; unchanged.
// y : double*
//     Accumulator, updated in place.
//
// Math
// ----
// $$ y_i \leftarrow \alpha x_i + y_i $$
//
// References
// ----------
// Accelerate.framework ``cblas_daxpy``.
LUCID_INTERNAL void daxpy(int n, double alpha, const double* x, double* y);

// Single-precision triangular solve with a matrix right-hand side (TRSM).
//
// Overwrites $B$ with $X = A^{-1} B$, where $A$ is an $M \times M$
// triangular matrix on the left and $B$ is $M \times N$, both row-major.
// Only the triangle named by ``upper`` is read; ``unit_diag`` treats the
// diagonal as ones without reading it.  Dispatches to Accelerate's
// ``cblas_strsm`` with side Left, no transpose and $\alpha = 1$.
//
// Parameters
// ----------
// upper : bool
//     Whether $A$ is upper (back substitution) or lower (forward
//     substitution) triangular.
// unit_diag : bool
//     Treat the diagonal of $A$ as all ones.
// M, N : int
//     $A \in \mathbb{R}^{M \times M}$, $B \in \mathbb{R}^{M \times N}$.
// A : const float*
//     Row-major triangular matrix buffer.
// lda : int
//     Leading dimension (row stride) of $A$.
// B : float*
//     Row-major right-hand sides, replaced by the solution.
// ldb : int
//     Leading dimension (row stride) of $B$.
//
// Math
// ----
// For upper $A$, back substitution:
// $$ x_i = \Big(b_i - \sum_{j > i} A_{ij} x_j\Big) / A_{ii} $$
//
// Notes
// -----
// No singularity check.  An exactly-zero diagonal entry divides by zero,
// so the solution carries the IEEE result ($\pm\infty$, or NaN for
// $0 / 0$) — the reference framework's answer for a singular triangle.
// LAPACK's ``?trtrs`` driver checks the diagonal first and reports
// ``info > 0`` without solving; that is the difference that matters here.
//
// References
// ----------
// Accelerate.framework ``cblas_strsm``.
LUCID_INTERNAL void
strsm(bool upper, bool unit_diag, int M, int N, const float* A, int lda, float* B, int ldb);

// Double-precision triangular solve with a matrix right-hand side (TRSM).
//
// Identical contract to ``strsm`` but for ``double`` buffers; dispatches to
// ``cblas_dtrsm``.
//
// Parameters
// ----------
// upper : bool
//     Whether $A$ is upper or lower triangular.
// unit_diag : bool
//     Treat the diagonal of $A$ as all ones.
// M, N : int
//     $A \in \mathbb{R}^{M \times M}$, $B \in \mathbb{R}^{M \times N}$.
// A : const double*
//     Row-major triangular matrix buffer.
// lda : int
//     Leading dimension of $A$.
// B : double*
//     Row-major right-hand sides, replaced by the solution.
// ldb : int
//     Leading dimension of $B$.
//
// Math
// ----
// $$ X = A^{-1} B $$
//
// References
// ----------
// Accelerate.framework ``cblas_dtrsm``.
LUCID_INTERNAL void
dtrsm(bool upper, bool unit_diag, int M, int N, const double* A, int lda, double* B, int ldb);

// Complex triangular solve with a matrix right-hand side (TRSM), single and
// double precision.
//
// ``ctrsm`` is ``cblas_ctrsm`` over ``complex64`` and ``ztrsm`` is
// ``cblas_ztrsm`` over ``complex128``; the contract is ``strsm``'s (side Left,
// no transpose, $\alpha = 1$).  The substitution divides by the complex
// diagonal, so an exactly-zero diagonal entry gives the IEEE result there too.
LUCID_INTERNAL void
ctrsm(bool upper, bool unit_diag, int M, int N, const Complex64* A, int lda, Complex64* B, int ldb);

LUCID_INTERNAL void ztrsm(
    bool upper, bool unit_diag, int M, int N, const Complex128* A, int lda, Complex128* B, int ldb);

}  // namespace lucid::backend::cpu
