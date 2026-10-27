// lucid/_C/test/helpers/numeric_assert.h
//
// GTest matchers, byte-level inspectors, and assertion macros for
// :class:`TensorImpl` objects.
//
// Companion to :file:`tensor_factory.h` — that header *creates* fixture
// tensors, this one *inspects* them.  Together they form the canonical
// "build a tensor, run an op, assert on the result" pipeline used
// throughout the C++ unit-test suite under :file:`lucid/_C/test/`.
//
// The matchers always pull the tensor's bytes into a flat
// ``std::vector<float>`` (see :func:`to_float_vec`) and compare in F32
// regardless of the source dtype.  This trades a tiny precision loss
// for a uniform assertion surface that works for both CPU and GPU
// tensors and both ``F32`` and ``F64`` storage.
//
// Notes
// -----
// All comparisons run on the CPU after a one-shot
// :func:`IBackend::to_cpu` round-trip if the tensor lives on the GPU —
// test-only behaviour, never used in production hot paths.
//
// Examples
// --------
// ```
// auto t = some_op(x);
// EXPECT_TENSOR_SHAPE(t, (Shape{3, 4}));
// EXPECT_TENSOR_DTYPE(t, Dtype::F32);
// EXPECT_TENSOR_NEAR(t, 1.0f, 1e-5f);   // all elements near scalar
// ASSERT_TENSOR_FINITE(t);              // no NaN / Inf
// EXPECT_TENSORS_NEAR(a, b, 1e-5f);     // element-wise diff
// ```
//
// See Also
// --------
// :file:`tensor_factory.h` — companion header that produces the
//     :type:`TensorImplPtr` values consumed by these matchers.

#pragma once

#include <cmath>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

#include <gtest/gtest.h>

#include "../../backend/Dispatcher.h"
#include "../../core/Dtype.h"
#include "../../core/Half.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "../../core/fwd.h"

namespace lucid::test {

// Extract a :class:`TensorImpl` 's raw bytes as a flat
// ``std::vector<float>`` for numeric inspection in unit tests.
//
// For CPU tensors, reinterprets the :class:`CpuStorage` buffer directly
// — zero copy.  For GPU tensors, performs a one-shot
// :func:`IBackend::to_cpu` transfer then reads.  Always returns the
// "as-if F32" view, casting from the tensor's native dtype as needed
// so the caller can compare without dtype branching.
//
// Parameters
// ----------
// t : TensorImplPtr
//     The tensor to flatten.  Any shape, any device, dtype ``F32`` or
//     ``F64``.
//
// Returns
// -------
// vector<float>
//     Length equals ``t->numel()``.  Elements are in row-major order
//     matching the tensor's logical layout.  ``F64`` storage is cast
//     down element-wise to ``float``.
//
// Notes
// -----
// Test-only — production code never round-trips GPU→CPU just to read
// values.  Use :class:`Storage` accessors or the public
// :func:`Tensor::numpy` bridge instead.
//
// The F64 → F32 narrowing is lossy by design: the matcher tolerance
// (``tol`` parameters on the assertion macros) is expected to absorb
// the rounding noise.
//
// Examples
// --------
// ```
// auto t = cpu_full({3}, 2.5);
// auto v = to_float_vec(t);             // {2.5f, 2.5f, 2.5f}
// ```
//
// See Also
// --------
// :func:`grad_to_float_vec`, :func:`EXPECT_TENSOR_NEAR`.
inline std::vector<float> to_float_vec(const TensorImplPtr& t) {
    std::size_t n = t->numel();

    CpuStorage cpu_st;
    if (t->device() == Device::CPU) {
        cpu_st = std::get<CpuStorage>(t->storage());
    } else {
        auto& b = backend::Dispatcher::for_device(Device::CPU);
        cpu_st = b.to_cpu(t->storage(), t->shape());
    }

    if (t->dtype() == Dtype::F64) {
        const auto* raw = reinterpret_cast<const double*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i)
            result[i] = static_cast<float>(raw[i]);
        return result;
    }
    if (is_half_float(t->dtype())) {
        const auto* raw = reinterpret_cast<const std::uint16_t*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        backend::detail::half_or_bfloat16_to_float_bulk(raw, result.data(), n,
                                                        t->dtype() == Dtype::BF16);
        return result;
    }
    const auto* raw = reinterpret_cast<const float*>(cpu_st.ptr.get());
    return std::vector<float>(raw, raw + n);
}

// Whether the tensor currently holds a populated gradient buffer.
//
// Thin predicate over :func:`TensorImpl::grad_storage` — returns
// ``true`` once :func:`Engine::backward` has accumulated into ``t``,
// ``false`` for any tensor that has never received a backward signal
// (including freshly constructed ``requires_grad = true`` tensors
// before the first backward pass).
//
// Parameters
// ----------
// t : TensorImplPtr
//     Any tensor.  Need not have ``requires_grad = true``.
//
// Returns
// -------
// bool
//     ``true`` if ``t->grad_storage().has_value()``, ``false``
//     otherwise.
//
// See Also
// --------
// :func:`grad_to_float_vec`, :func:`grad_numel`.
inline bool has_grad(const TensorImplPtr& t) {
    return t->grad_storage().has_value();
}

// Extract a tensor's accumulated gradient as a flat ``std::vector<float>``.
//
// Mirror of :func:`to_float_vec` but reads from
// :func:`TensorImpl::grad_storage` instead of :func:`TensorImpl::storage`.
// Returns an empty vector when no gradient has been accumulated — the
// caller is responsible for distinguishing that from a genuinely
// zero-length gradient via :func:`has_grad` if needed.
//
// Parameters
// ----------
// t : TensorImplPtr
//     The tensor whose gradient is read.  Must have already been
//     touched by :func:`Engine::backward` (verify with :func:`has_grad`).
//
// Returns
// -------
// vector<float>
//     Flat ``float`` view of the gradient buffer in row-major order.
//     ``F64`` gradient storage is cast down element-wise.  Empty
//     vector if the tensor has no gradient yet.
//
// Notes
// -----
// Assumes the gradient lives on the CPU (the normal :func:`Engine::backward`
// path).  GPU-resident gradients would need an explicit
// :func:`IBackend::to_cpu` transfer that this helper does *not* perform
// — extend the helper if your test exercises that path.
//
// Examples
// --------
// ```
// Engine::backward(loss);
// auto g = grad_to_float_vec(param);
// EXPECT_NEAR(g[0], 0.5f, 1e-5f);
// ```
//
// See Also
// --------
// :func:`has_grad`, :func:`grad_numel`, :func:`to_float_vec`.
inline std::vector<float> grad_to_float_vec(const TensorImplPtr& t) {
    const auto& g = t->grad_storage();
    if (!g.has_value())
        return {};
    const auto& cpu_st = std::get<CpuStorage>(*g);
    std::size_t n = cpu_st.nbytes / dtype_size(t->dtype());
    if (t->dtype() == Dtype::F64) {
        const auto* raw = reinterpret_cast<const double*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        for (std::size_t i = 0; i < n; ++i)
            result[i] = static_cast<float>(raw[i]);
        return result;
    }
    if (is_half_float(t->dtype())) {
        const auto* raw = reinterpret_cast<const std::uint16_t*>(cpu_st.ptr.get());
        std::vector<float> result(n);
        backend::detail::half_or_bfloat16_to_float_bulk(raw, result.data(), n,
                                                        t->dtype() == Dtype::BF16);
        return result;
    }
    const auto* raw = reinterpret_cast<const float*>(cpu_st.ptr.get());
    return std::vector<float>(raw, raw + n);
}

// Number of elements stored in the tensor's gradient buffer.
//
// Derived from :func:`CpuStorage::nbytes` divided by
// :func:`dtype_size(t->dtype())`.  Returns ``0`` when no gradient has
// been accumulated — same sentinel as a genuinely empty gradient, so
// pair with :func:`has_grad` when the distinction matters.
//
// Parameters
// ----------
// t : TensorImplPtr
//     The tensor whose gradient size is queried.
//
// Returns
// -------
// size_t
//     Element count of the gradient buffer (``nbytes / dtype_size``),
//     or ``0`` if the tensor has no gradient.
//
// See Also
// --------
// :func:`has_grad`, :func:`grad_to_float_vec`.
inline std::size_t grad_numel(const TensorImplPtr& t) {
    const auto& g = t->grad_storage();
    if (!g.has_value())
        return 0;
    const auto& cpu_st = std::get<CpuStorage>(*g);
    return cpu_st.nbytes / dtype_size(t->dtype());
}

// Assert that a tensor's shape matches the expected :class:`Shape`
// exactly.
//
// Fails the enclosing GTest case with a formatted diff when the shapes
// differ.  Wrap parentheses around list-initialised shape literals so
// the preprocessor doesn't mis-split on the inner commas:
// ``EXPECT_TENSOR_SHAPE(t, (Shape{3, 4}))``.
//
// Parameters
// ----------
// tensor_impl_ptr : TensorImplPtr
//     Tensor under test.  Stored in a local so its lifetime covers the
//     :func:`TensorImpl::shape` reference — avoids UB with temporaries
//     such as ``exp_op(x)``.
// expected_shape : Shape
//     Expected shape.  Must be parenthesised when constructed inline
//     from a brace-initialised :class:`Shape`.
//
// Notes
// -----
// Comparison is exact — no broadcasting tolerance.  Use this for the
// final post-op shape check; for rank-only checks use a separate
// GTest matcher.
//
// Examples
// --------
// ```
// EXPECT_TENSOR_SHAPE(out, (Shape{2, 3, 4}));
// EXPECT_TENSOR_SHAPE(reduce_sum(x, 0), (Shape{3, 4}));
// ```
#define EXPECT_TENSOR_SHAPE(tensor_impl_ptr, expected_shape)                                       \
    do {                                                                                           \
        const auto& _t = (tensor_impl_ptr);                                                        \
        const auto& _s = _t->shape();                                                              \
        const Shape _e = (expected_shape);                                                         \
        EXPECT_EQ(_s, _e) << "Tensor shape mismatch.";                                             \
    } while (0)

// Assert that a tensor's dtype matches the expected :enum:`Dtype`.
//
// Fails the enclosing GTest case with a formatted diff when the dtypes
// differ.  Useful after promotion rules (e.g. ``F32 + F64 → F64``) or
// after explicit :func:`astype_op` calls to lock in the expected
// element type.
//
// Parameters
// ----------
// tensor_impl_ptr : TensorImplPtr
//     Tensor under test.
// expected_dtype : Dtype
//     Expected element type.
//
// Examples
// --------
// ```
// EXPECT_TENSOR_DTYPE(y, Dtype::F64);
// EXPECT_TENSOR_DTYPE(astype_op(x, Dtype::F32), Dtype::F32);
// ```
#define EXPECT_TENSOR_DTYPE(tensor_impl_ptr, expected_dtype)                                       \
    EXPECT_EQ((tensor_impl_ptr)->dtype(), (expected_dtype)) << "Tensor dtype mismatch."

// Assert that every element of a tensor is within ``tol`` of a scalar.
//
// Flattens the tensor via :func:`to_float_vec` (CPU or GPU) and runs
// ``EXPECT_NEAR`` over each element against the broadcast scalar.
// Reports the failing index in the GTest output for easy triage.
//
// Parameters
// ----------
// tensor_impl_ptr : TensorImplPtr
//     Tensor under test.
// scalar_val : float
//     Expected value broadcast to every element.  Cast to ``float``
//     internally.
// tol : float
//     Absolute tolerance.  Each ``|t[i] - scalar_val|`` must be
//     ``<= tol`` for the assertion to pass.
//
// Math
// ----
// $$ \forall i, \; |t_i - s| \le \tau $$
//
// Examples
// --------
// ```
// auto x = cpu_ones({4});
// EXPECT_TENSOR_NEAR(x, 1.0f, 1e-6f);   // all ones, tight tolerance
// ```
//
// See Also
// --------
// :func:`EXPECT_TENSORS_NEAR` for element-wise comparison of two tensors.
#define EXPECT_TENSOR_NEAR(tensor_impl_ptr, scalar_val, tol)                                       \
    do {                                                                                           \
        auto _vals = lucid::test::to_float_vec(tensor_impl_ptr);                                   \
        float _expected = static_cast<float>(scalar_val);                                          \
        float _tol = static_cast<float>(tol);                                                      \
        for (std::size_t _i = 0; _i < _vals.size(); ++_i) {                                        \
            EXPECT_NEAR(_vals[_i], _expected, _tol) << "Element [" << _i << "] out of tolerance."; \
        }                                                                                          \
    } while (0)

// Assert that every element of a tensor is finite (no NaN, no Inf).
//
// Flattens via :func:`to_float_vec` and runs ``std::isfinite`` on each
// element.  Aborts the enclosing test case on the first non-finite
// value (``ASSERT_*`` semantics) — used to guard against numerical
// blow-ups before subsequent assertions read garbage.
//
// Parameters
// ----------
// tensor_impl_ptr : TensorImplPtr
//     Tensor under test.
//
// Raises
// ------
// GTest fatal failure
//     If any element is ``NaN`` or ``Inf``.  Reports the offending
//     index.
//
// Examples
// --------
// ```
// auto y = some_op(x);
// ASSERT_TENSOR_FINITE(y);           // gate before further checks
// EXPECT_TENSOR_NEAR(y, expected, 1e-5f);
// ```
#define ASSERT_TENSOR_FINITE(tensor_impl_ptr)                                                      \
    do {                                                                                           \
        auto _vals = lucid::test::to_float_vec(tensor_impl_ptr);                                   \
        for (std::size_t _i = 0; _i < _vals.size(); ++_i) {                                        \
            ASSERT_TRUE(std::isfinite(_vals[_i])) << "Element [" << _i << "] is NaN or Inf.";      \
        }                                                                                          \
    } while (0)

// Assert that two tensors are element-wise close within ``tol``.
//
// Flattens both tensors via :func:`to_float_vec`, asserts they have
// the same total element count (``ASSERT_EQ`` — fatal), then runs
// ``EXPECT_NEAR`` over each index pair.  The canonical matcher for
// "op output matches a hand-computed reference tensor".
//
// Parameters
// ----------
// lhs : TensorImplPtr
//     Tensor under test.
// rhs : TensorImplPtr
//     Expected reference tensor.  Shape need not equal :paramref:`lhs`
//     's, only total ``numel()`` must match — the matcher works on
//     flattened buffers.
// tol : float
//     Absolute tolerance per element.  Each
//     ``|lhs.flat[i] - rhs.flat[i]|`` must be ``<= tol``.
//
// Math
// ----
// $$ \forall i, \; |a_i - b_i| \le \tau $$
//
// Notes
// -----
// Shape mismatch is *not* checked — only element-count.  Pair with
// :func:`EXPECT_TENSOR_SHAPE` when shape-equality matters (it usually
// does).
//
// Examples
// --------
// ```
// auto got = matmul_op(a, b);
// auto exp = reference_matmul(a, b);
// EXPECT_TENSOR_SHAPE(got, exp->shape());
// EXPECT_TENSORS_NEAR(got, exp, 1e-5f);
// ```
//
// See Also
// --------
// :func:`EXPECT_TENSOR_NEAR` for comparing a tensor against a scalar.
#define EXPECT_TENSORS_NEAR(lhs, rhs, tol)                                                         \
    do {                                                                                           \
        auto _a = lucid::test::to_float_vec(lhs);                                                  \
        auto _b = lucid::test::to_float_vec(rhs);                                                  \
        ASSERT_EQ(_a.size(), _b.size()) << "Tensor size mismatch.";                                \
        float _tol = static_cast<float>(tol);                                                      \
        for (std::size_t _i = 0; _i < _a.size(); ++_i) {                                           \
            EXPECT_NEAR(_a[_i], _b[_i], _tol) << "Element [" << _i << "] mismatch.";               \
        }                                                                                          \
    } while (0)

}  // namespace lucid::test
