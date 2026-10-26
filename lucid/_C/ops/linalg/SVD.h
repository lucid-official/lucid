// lucid/_C/ops/linalg/SVD.h
//
// Singular Value Decomposition (SVD) forward op for arbitrary-shape matrices.
//
// Factorises any (possibly rectangular) input $A \in \mathbb{R}^{m \times n}$
// into $A = U \Sigma V^\top$ where $U$ has orthonormal columns, $V$ has
// orthonormal columns, and $\Sigma$ is a diagonal matrix of non-negative
// singular values in descending order.  The op returns the "reduced" (a.k.a.
// economy / thin) SVD: $U$ is $(m, k)$, $V^\top$ is $(k, n)$, and $\Sigma$
// collapses to a length-$k$ vector with $k = \min(m, n)$.  The full-square
// variant (where $U$ is $(m, m)$ and $V^\top$ is $(n, n)$) is assembled by
// the Python wrapper from this reduced form, not by this op.
//
// Math
// ----
// $$
//   A = U\,\Sigma\,V^\top,
//   \qquad U^\top U = I_k,\quad V^\top V = I_k,\quad
//   \Sigma = \mathrm{diag}(\sigma_1, \ldots, \sigma_k),\quad
//   \sigma_1 \ge \cdots \ge \sigma_k \ge 0.
// $$
//
// Notes
// -----
// CPU backend dispatches to LAPACK's divide-and-conquer ``*gesdd`` driver,
// which is asymptotically faster than the QR-iteration based ``*gesvd`` for
// large matrices at the cost of a larger temporary workspace.  GPU backend
// calls ``mlx::core::linalg::svd`` (which itself routes through the CPU
// stream — MLX has no native Metal SVD kernel yet, see H3 carve-out).
//
// Autograd is **not** wired at the C++ level.  Differentiation is handled by
// the Python-side ``_SVD`` multi-output ``Function`` wrapper that implements
// the Giles 2008 closed form in terms of the Loewner matrix
// $F_{ij} = \sigma_i / (\sigma_i^2 - \sigma_j^2)$.  That
// closed form is well-conditioned for distinct singular values but diverges
// near repeated $\sigma$'s; caller responsibility to regularise in that
// regime (e.g. by adding a small diagonal jitter to $A$).
//
// References
// ----------
// - Papadopoulo & Lourakis, "Estimating the Jacobian of the SVD" (2000).
// - Giles, "Collected Matrix Derivative Results for Forward and Reverse
//   Mode Algorithmic Differentiation" (2008).
// - Townsend, "Differentiating the SVD" (2016).
//
// See Also
// --------
// - ``Eigh.h`` — symmetric eigendecomposition uses an analogous Loewner
//   structure for its backward.
// - ``Pinv.h`` — Moore-Penrose pseudoinverse built on top of the SVD.

#pragma once

#include <vector>

#include "../../api.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"

namespace lucid {

// Compute the reduced singular value decomposition of a rectangular matrix.
//
// Parameters
// ----------
// a : TensorImplPtr
//     Input matrix of shape ``(..., m, n)`` with leading batch dims.  Must
//     be at least 2-D and have a floating-point dtype.
// compute_uv : bool, optional
//     When ``true`` (default) returns the full reduced factorisation
//     $\{U, \Sigma, V^\top\}$.  When ``false`` returns $\{\Sigma\}$ only,
//     which lets the backend skip assembly of the singular vectors —
//     roughly 2x faster and used by ``svdvals`` / ``matrix_rank`` / ``cond``.
//
// Returns
// -------
// std::vector<TensorImplPtr>
//     If ``compute_uv=true``: ``{U, S, Vh}`` with
//     $U$ shape ``(..., m, k)``, $S$ shape ``(..., k)``,
//     $V^\top$ shape ``(..., k, n)``, $k = \min(m, n)$.
//     If ``compute_uv=false``: ``{S}`` only.
//
// Shape
// -----
// - Input ``a``: ``(..., m, n)``.
// - ``U``:  ``(..., m, k)`` — left singular vectors as columns.
// - ``S``:  ``(..., k)``    — singular values in descending order.
// - ``Vh``: ``(..., k, n)`` — right singular vectors as rows (note: ``Vh``
//   is $V^\top$, not $V$, matching NumPy/reference-framework convention).
//
// Raises
// ------
// std::runtime_error
//     If ``a`` is null, has dtype that is not floating point, or has fewer
//     than two dimensions.
//
// Notes
// -----
// Singular values are guaranteed non-negative and sorted in descending
// order on output.  Outputs are leaf nodes in the autograd graph at the
// C++ level — the Python wrapper attaches the gradient functions.
//
// Examples
// --------
// >>> // CPU equivalent of: U, S, Vh = svd(A, full_matrices=False)
// >>> auto out = svd_op(a_impl, /*compute_uv=*/true);
// >>> auto& U  = out[0];  auto& S = out[1];  auto& Vh = out[2];
LUCID_API std::vector<TensorImplPtr> svd_op(const TensorImplPtr& a, bool compute_uv = true);

}  // namespace lucid
