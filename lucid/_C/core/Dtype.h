// lucid/_C/core/Dtype.h
//
// Engine-side element type tag plus constexpr helpers covering size, naming,
// and category predicates.
//
// Keeping the helpers ``constexpr`` lets the compiler fold dtype-driven
// arithmetic (stride computation, allocation sizes, kernel dispatch) entirely
// at compile time whenever the dtype is known statically — common in
// template-instantiated op kernels.
//
// The Python-side :class:`lucid.dtype` is a thin object wrapper around this
// enum; :attr:`Tensor.dtype` ultimately reads :class:`TensorMeta::dtype`.
//
// See Also
// --------
// :class:`Device` — device tag tracked alongside dtype on every tensor.
// :file:`Storage.h` — every storage variant carries a :class:`Dtype` field.

#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

// Element type of a tensor's storage buffer.
//
// The underlying ``uint8_t`` representation is part of the persistent ABI:
// values are written into Lucid checkpoint files and crossed across the
// Python/C++ pybind11 boundary.  **Append new entries at the end** — never
// reorder, never insert in the middle, never repurpose a value.
//
// Attributes
// ----------
// Bool : Dtype
//     1-byte boolean (0 / 1).  Truthy semantics; excluded from
//     :func:`is_integral` because it has no arithmetic meaning.
// I8 : Dtype
//     Signed 8-bit integer.
// I16 : Dtype
//     Signed 16-bit integer.
// I32 : Dtype
//     Signed 32-bit integer.  Common index dtype on the GPU stream.
// I64 : Dtype
//     Signed 64-bit integer.  Default integer dtype on the CPU stream
//     and standard index dtype on the CPU side.
// F16 : Dtype
//     IEEE-754 binary16 half-precision float.  Used for inference / AMP.
// F32 : Dtype
//     IEEE-754 binary32 single-precision float.  Default floating dtype.
// F64 : Dtype
//     IEEE-754 binary64 double-precision float.
// C64 : Dtype
//     Complex pair of two ``float32`` lanes (real, imag), totalling 8
//     bytes per element.  Same :func:`dtype_size` as :attr:`F64` but
//     fundamentally different numeric semantics — :func:`is_floating_point`
//     returns ``false`` for it.
//
// Notes
// -----
// BF16 carries float32's exponent range in 16 bits, trading mantissa for
// dynamic range — which is the whole reason to reach for it over F16, and
// the reason aliasing the two is not an option: code that picks bfloat16
// to avoid overflow would silently get the format that overflows.
enum class Dtype : std::uint8_t {
    Bool,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    C64,
    // Appended rather than slotted next to F16: the enum's numeric values
    // are written into checkpoints and traces, and inserting in the middle
    // would renumber everything after it.
    BF16,
    // Complex pair of two ``float64`` lanes, 16 bytes per element.
    //
    // Added for ``eig``, which is the one op whose output dtype is not a
    // choice: a real matrix has complex eigenvalues, and the only real
    // matrices that do not are the ones whose characteristic polynomial
    // happens to split over R.  Without a double-width complex type the
    // eigenvalues of an ``f64`` matrix had nowhere to go, and the answer
    // was the real parts alone — silently, with the imaginary halves
    // dropped and no indication that a conjugate pair had been there.
    //
    // CPU only.  MLX has no complex128, so the GPU path refuses it the
    // same way it already refuses ``f64``.
    C128,
};

// Returns the size in bytes of a single element of the given dtype.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to query.
//
// Returns
// -------
// std::size_t
//     Element size in bytes.  ``1`` for :attr:`Bool` / :attr:`I8`, ``2``
//     for :attr:`I16` / :attr:`F16`, ``4`` for :attr:`I32` / :attr:`F32`,
//     ``8`` for :attr:`I64` / :attr:`F64` / :attr:`C64`.
//
// Raises
// ------
// std::logic_error
//     The enum was extended without updating this switch.
//
// Notes
// -----
// ``constexpr`` so that ``nbytes = numel * dtype_size(dt)`` folds away at
// compile time in kernel templates instantiated against a known dtype.
constexpr std::size_t dtype_size(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return 1;
    case Dtype::I8:
        return 1;
    case Dtype::I16:
        return 2;
    case Dtype::I32:
        return 4;
    case Dtype::I64:
        return 8;
    case Dtype::F16:
        return 2;
    case Dtype::F32:
        return 4;
    case Dtype::F64:
        return 8;
    case Dtype::C64:
        return 8;
    case Dtype::C128:
        return 16;
    case Dtype::BF16:
        return 2;
    }
    throw std::logic_error("dtype_size: unknown Dtype");
}

// Returns the canonical string name of a dtype.
//
// This is the form used in error messages, ``repr(tensor)`` rendering, and
// the Python :class:`lucid.dtype` wrapper's ``__repr__``.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to convert.
//
// Returns
// -------
// std::string_view
//     One of ``"bool"``, ``"int8"``, ``"int16"``, ``"int32"``, ``"int64"``,
//     ``"float16"``, ``"float32"``, ``"float64"``, ``"complex64"``.  The
//     view points into a static literal — safe to store across calls.
//
// Raises
// ------
// std::logic_error
//     The enum was extended without updating this switch.
constexpr std::string_view dtype_name(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return "bool";
    case Dtype::I8:
        return "int8";
    case Dtype::I16:
        return "int16";
    case Dtype::I32:
        return "int32";
    case Dtype::I64:
        return "int64";
    case Dtype::F16:
        return "float16";
    case Dtype::F32:
        return "float32";
    case Dtype::F64:
        return "float64";
    case Dtype::C64:
        return "complex64";
    case Dtype::C128:
        return "complex128";
    case Dtype::BF16:
        return "bfloat16";
    }
    throw std::logic_error("dtype_name: unknown Dtype");
}

// Predicate: is the dtype a real floating-point type?
//
// :attr:`Dtype::C64` is **not** floating point under this predicate even
// though its lanes are float32 — complex tensors take a separate dispatch
// path in most kernels.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to test.
//
// Returns
// -------
// bool
//     ``true`` for :attr:`F16`, :attr:`F32`, :attr:`F64`; ``false``
//     otherwise.
//
// Notes
// -----
// Must be kept in sync with the :class:`Dtype` enum: if BF16 is added,
// it should be included here.
constexpr bool is_floating_point(Dtype dt) {
    return dt == Dtype::F16 || dt == Dtype::BF16 || dt == Dtype::F32 || dt == Dtype::F64;
}

// Predicate: is the dtype one of the two 16-bit float formats?
//
// A property of the dtype, so it lives beside :func:`is_floating_point`
// rather than in any one backend.  Accelerate has kernels for neither
// format, so both take the same widen-reuse-narrow route through
// float32 across the CPU backend — and the random generators, which
// have no input to widen, use it to decide to draw wide and round.
//
// Parameters
// ----------
// dt : Dtype
//     The dtype to test.
//
// Returns
// -------
// bool
//     ``true`` for :attr:`F16` and :attr:`BF16`, ``false`` otherwise.
constexpr bool is_half_float(Dtype dt) {
    return dt == Dtype::F16 || dt == Dtype::BF16;
}

// Predicate: is the dtype a signed integer type?
//
// :attr:`Dtype::Bool` is intentionally **excluded** — booleans are stored
// as 1-byte integers but have no meaningful arithmetic interpretation in
// the integer sense, so promotion / index validation paths should not
// treat them as integers.
//
// Parameters
// ----------
// dt : Dtype
//     Dtype to test.
//
// Returns
// -------
// bool
//     ``true`` for :attr:`I8`, :attr:`I16`, :attr:`I32`, :attr:`I64`;
//     ``false`` for everything else (including :attr:`Bool`).
constexpr bool is_integral(Dtype dt) {
    return dt == Dtype::I8 || dt == Dtype::I16 || dt == Dtype::I32 || dt == Dtype::I64;
}

// Predicate: does the dtype hold a complex pair?
//
// Both complex types answer ``true`` and neither is floating point under
// :func:`is_floating_point`, which is deliberate: their lanes are floats
// but they take a separate dispatch, and a real-only kernel handed a
// complex buffer would read two elements as one.
//
// Parameters
// ----------
// dt : Dtype
//     The dtype to test.
//
// Returns
// -------
// bool
//     ``true`` for :attr:`C64` and :attr:`C128`.
constexpr bool is_complex(Dtype dt) {
    return dt == Dtype::C64 || dt == Dtype::C128;
}

// The complex type whose lanes match a given real float type.
//
// ``f32 -> C64``, ``f64 -> C128``, and the half formats widen to
// :attr:`C64` because there is no half-precision complex.  Used by
// ``eig`` to pick an output dtype that loses nothing: an ``f64`` matrix
// whose eigenvalues came back as :attr:`C64` would silently shed eight
// decimal digits, which is the failure this type was added to avoid.
constexpr Dtype complex_for(Dtype real) {
    return real == Dtype::F64 ? Dtype::C128 : Dtype::C64;
}

// The real type of one lane of a complex dtype.
//
// The inverse of :func:`complex_for` on the range where it is
// invertible; the identity on anything already real.
constexpr Dtype real_lane_of(Dtype dt) {
    if (dt == Dtype::C128)
        return Dtype::F64;
    if (dt == Dtype::C64)
        return Dtype::F32;
    return dt;
}

// Predicate: can the GPU stream hold this dtype?
//
// MLX has no double precision, so the two dtypes with 64-bit lanes stay on the
// CPU.  The one list of them: ``to_mlx_dtype`` refuses what this says no to, and
// so does a phantom, which stands in for a tensor the device could hold
// (``core/PhantomMode.h``).
constexpr bool metal_holds(Dtype dt) {
    return dt != Dtype::F64 && dt != Dtype::C128;
}

}  // namespace lucid
