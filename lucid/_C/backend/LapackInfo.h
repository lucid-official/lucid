// lucid/_C/backend/LapackInfo.h
//
// The one reading of a factorisation's status, for both streams.
//
// LAPACK reports through ``info``: ``0`` succeeded, ``-k`` means argument
// ``k`` was illegal (always Lucid's bug — it marshalled the call wrongly),
// and ``k > 0`` means the matrix itself could not be factored at step ``k``
// (a zero pivot, a leading minor that is not positive definite, an
// iteration that did not converge).  The last is ``LinAlgError``; the first
// stays a plain ``LucidError``.
//
// The CPU stream gets ``info`` from LAPACK.  MLX reports nothing, so the
// Metal stream reads the failed step off the factor (a zero pivot of
// ``U``, a non-positive diagonal entry of ``L``) and comes here with it —
// the two streams raise the same class with the same text, naming the same
// matrix of a batch.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

#include "../core/ErrorBuilder.h"

namespace lucid::backend {

// What ``info = k > 0`` means for ``op``, worded as ``lucid.linalg`` words
// the same failure from a ``*_ex`` variant — one message per failure,
// whichever layer reports it.
inline std::string lapack_failure_reason(std::string_view op, int k) {
    const std::string at = std::to_string(k);
    if (op == "cholesky")
        return "The factorization could not be completed because the input is not "
               "positive-definite (the leading minor of order " +
               at + " is not positive-definite).";
    if (op == "solve")
        return "The diagonal element " + at +
               " is zero, the solve could not be completed because the input matrix is singular.";
    if (op == "inv" || op == "matrix_power")
        return "The diagonal element " + at +
               " is zero, the inversion could not be completed because the input matrix is "
               "singular.";
    static constexpr std::pair<std::string_view, std::string_view> kSingularOps[] = {
        {"solve_triangular", ""}, {"lu_solve", ""}, {"lu_factor", ""},
        {"ldl_factor", ""},       {"det", ""},
    };
    for (const auto& [name, _] : kSingularOps) {
        if (op == name)
            return "The diagonal element " + at +
                   " is zero, the factorization could not be completed because the input matrix is "
                   "singular.";
    }
    static constexpr std::pair<std::string_view, std::string_view> kConvergeOps[] = {
        {"eigh", "eigenvalues"},     {"eigvalsh", "eigenvalues"},  {"svd", "singular values"},
        {"pinv", "singular values"}, {"lstsq", "singular values"},
    };
    for (const auto& [name, what] : kConvergeOps) {
        if (op == name)
            return "The algorithm failed to converge because the input matrix is ill-conditioned "
                   "or has too many repeated " +
                   std::string(what) + " (error code: " + at + ").";
    }
    if (op == "eig" || op == "eigvals")
        return "The QR algorithm failed to compute all eigenvalues (error code: " + at + ").";
    return "The algorithm failed to converge (error code: " + at + ").";
}

// Raise for a non-zero LAPACK ``info`` of matrix ``index`` among ``batch``.
// The matrix is named, ``(Batch element b): ``, only when there is more than
// one.  Call it inside the batch loop: ``info`` is overwritten per matrix, so
// checked once after the loop it reports only the last matrix, and an
// earlier failure leaves its output block unwritten and returned as if it
// were an answer.
inline void
check_lapack_info(int info, const char* op, std::int64_t index = 0, std::int64_t batch = 1) {
    if (info == 0)
        return;
    const std::string where =
        batch > 1 ? "(Batch element " + std::to_string(index) + "): " : std::string{};
    if (info < 0)
        ErrorBuilder(op).fail(where + "LAPACK invalid argument index" + std::to_string(-info));
    ErrorBuilder(op).linalg_error(where + lapack_failure_reason(op, info));
}

// Raise for the first set flag of ``flags``, a row-major ``(batch, n)``
// grid with one flag per factorisation step, as LAPACK would have: the
// first failed matrix, and ``info`` the 1-based step it failed at.  Returns
// when no flag is set.
inline void check_step_flags(const bool* flags, std::size_t count, std::size_t n, const char* op) {
    if (n == 0)
        return;
    std::size_t first = 0;
    while (first < count && !flags[first])
        ++first;
    if (first == count)
        return;
    check_lapack_info(static_cast<int>(first % n) + 1, op, static_cast<std::int64_t>(first / n),
                      static_cast<std::int64_t>(count / n));
}

// Raise for the first non-finite entry in the active triangle of ``mat``
// when ``n >= 3`` (matching the reference's ``eigh`` / ``eigvalsh`` check).
template <typename T>
inline void
check_eigh_triangle_finite(const T* mat, int n, bool upper, std::int64_t b, std::int64_t batch) {
    if (n < 3)
        return;
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            if ((upper ? (i <= j) : (i >= j)) && !(mat[i * n + j] - mat[i * n + j] == T{0}))
                check_lapack_info(2, "eigh", b, batch);
}

}  // namespace lucid::backend
