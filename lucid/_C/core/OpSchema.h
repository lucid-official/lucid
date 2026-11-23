// lucid/_C/core/OpSchema.h
//
// Static metadata descriptor for a registered op.
//
// Every op in Lucid is described by a constexpr :class:`OpSchema` instance —
// conventionally a public static member named ``schema_v1`` on the op's
// backward class.  The schema records the op's name, version, declared AMP
// policy, determinism contract, arity, and a list of "stable" inputs that
// autograd must guard against in-place mutation.
//
// The schema is registered with the global :class:`OpRegistry` at static
// initialisation time (see :file:`OpRegistry.h`'s ``LUCID_REGISTER_OP``
// macro) and consulted by :class:`SchemaGuard` at every op dispatch to:
//
//   0. Refuse an input dtype the reference refuses (``refused``).
//   1. Reject calls that violate the current determinism setting.
//   2. Compute the effective compute dtype — under AMP from the op's row in
//      the per-device policy table (``AmpPolicyTable.def``, looked up by
//      ``name``), and for an integer input from ``real_valued`` /
//      ``declared_amp_policy``.
//
// Versioning contract
// -------------------
// ``schema_v1`` is the version-1 schema; bumping to ``schema_v2`` is a wire-
// format break — saved checkpoints and traced graphs that reference the
// older version must be migrated.  Bump the ``version`` field whenever the
// op's signature, semantics, or AMP/determinism contract changes in a way
// that an old caller could not handle.
//
// See Also
// --------
// :class:`OpRegistry`     — the global name → schema map.
// :class:`SchemaGuard`    — runtime gate that consumes the schema.
// :func:`amp::policy_for` — the per-device autocast policy table.

#pragma once

#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "../api.h"
#include "AmpPolicy.h"
#include "Dtype.h"

namespace lucid {

// How an op turns a dtype away, and with which of the reference's two error
// classes.
//
// The reference's CPU backend refuses a bool or integer operand on many ops,
// two ways: a kernel with no instantiation for the dtype raises
// ``NotImplementedError``, and a guard on the input dtype raises a plain
// ``RuntimeError``.  A program that catches one does not catch the other, so
// the class is part of the answer.  It is one class per op: the reference's
// class does not change with the integer width, only whether the width is
// refused does (``avg_pool2d`` answers int64 and refuses the narrower ones).
//
// ``Kernel`` -> :class:`NotImplementedError` (``engine.NotImplementedError``,
//               a ``RuntimeError`` and the builtin ``NotImplementedError``).
// ``Guard``  -> :class:`DtypeMismatch` (``engine.DtypeMismatch``, a
//               ``RuntimeError`` and a ``TypeError``).
enum class DtypeRefusal : std::uint8_t { None = 0, Kernel, Guard };

// What the refusal tells the caller to cast to.  Mirrors the reference's dtype
// list for the op: an op that answers integers as well as floats says "an
// integer or floating dtype", one that answers only floats says "a floating
// dtype".
enum class CastHint : std::uint8_t { Number = 0, Floating, Integer };

// The cast a refusal's message offers.  The wording is the Python table's
// (``lucid/_ops/_bool_refusal.py``), so the engine and the composites that
// still refuse in Python say the same thing; the ``OpSchema`` bindings read
// it from here too.
constexpr std::string_view cast_hint_text(CastHint hint) noexcept {
    switch (hint) {
    case CastHint::Floating:
        return "a floating dtype, e.g. x.float()";
    case CastHint::Integer:
        return "an integer dtype, e.g. x.long()";
    case CastHint::Number:
        break;
    }
    return "an integer or floating dtype, e.g. x.long() or x.float()";
}

// A set of dtypes, one bit per :class:`Dtype` code (``1 << code``).
//
// The codes are the checkpoint-stable enum values, 0..10 today, so a 16-bit
// word holds every dtype with room for five more.
using DtypeMask = std::uint16_t;

static_assert(static_cast<unsigned>(Dtype::C128) < 16 && static_cast<unsigned>(Dtype::BF16) < 16,
              "DtypeMask is 16 bits wide: widen it before adding a dtype past code 15");

constexpr DtypeMask dtype_bit(Dtype dt) noexcept {
    return static_cast<DtypeMask>(DtypeMask{1} << static_cast<unsigned>(dt));
}

// The signed integer widths Lucid has (there is no unsigned type).
constexpr DtypeMask kSignedIntegers =
    dtype_bit(Dtype::I8) | dtype_bit(Dtype::I16) | dtype_bit(Dtype::I32) | dtype_bit(Dtype::I64);

// The widths narrower than 64 bits: the reference's ``avg_pool`` and
// ``conv_transpose`` have a kernel for int64 and none for these.
constexpr DtypeMask kNarrowSigned =
    dtype_bit(Dtype::I8) | dtype_bit(Dtype::I16) | dtype_bit(Dtype::I32);

// A bool counts as an integer for every op the reference turns away both of.
constexpr DtypeMask kBoolAndSignedIntegers = dtype_bit(Dtype::Bool) | kSignedIntegers;

// Per-op metadata record stored by pointer in the :class:`OpRegistry`.
//
// Schemas are intended to be ``constexpr`` statics defined alongside each
// op implementation; the registry stores raw pointers and assumes the
// pointee outlives the registry (true for process-lifetime statics).
// Constructing schemas with dynamic storage duration is supported but
// discouraged.
//
// Attributes
// ----------
// name : std::string_view
//     Canonical op name; used as the :class:`OpRegistry` key and in
//     diagnostic / error messages.  Must point into storage that lives at
//     least as long as the registry (constexpr literals are ideal).
// version : int
//     Monotonic schema version starting at 1.  Bump on backwards-
//     incompatible changes — bumping invalidates checkpoints / traces that
//     reference the previous version.
// declared_amp_policy : AmpPolicy
//     The policy the op author declared.  Autocast does not read it: an
//     op's autocast behaviour is its row in the per-device table
//     (:func:`amp::policy_for`), which can differ between Metal and CPU
//     and which one field could not express.  What the field still
//     decides is integer-input promotion outside autocast — ``ForceFP32``
//     here, like ``real_valued``, promotes an integer input to float32.
// deterministic : bool
//     ``true`` if the op produces bit-identical results given identical
//     inputs and configuration.  ``false`` for ops that rely on atomics,
//     non-associative parallel reductions, or other sources of run-to-run
//     variation.  Consulted by :func:`check_schema_determinism` /
//     :class:`SchemaGuard` when ``Determinism::is_enabled()``.
// determinism_note : std::string_view
//     Human-readable explanation of why the op is non-deterministic
//     (e.g. ``"uses atomic scatter-add"``).  Included verbatim in the
//     error message raised under ``set_deterministic(True)``.  Empty for
//     deterministic ops.
// input_arity : int
//     Expected number of input tensors.  ``-1`` means variadic (the op
//     accepts any number of inputs; the op is responsible for its own
//     arity validation).
// output_arity : int
//     Number of output tensors produced.  Defaults to ``1``; most ops are
//     single-output.
// stable_input_indices : std::vector<int>
//     Indices into the input list whose values are saved for backward.
//     Autograd's version-counter logic uses these to detect illegal
//     in-place mutations between forward and backward that would corrupt
//     the saved tensors.
// internal : bool
//     ``true`` for ops that are not part of the public Python API
//     (implementation-detail kernels, fused / internal helpers).  The
//     Python binding layer hides these from ``dir(lucid)`` etc.
// refusal : DtypeRefusal
//     ``None`` unless the reference CPU refuses some dtype for this op; then
//     the error class it raises.  One class per op.
// refused : DtypeMask
//     The dtypes the op refuses, one bit per :class:`Dtype` code.
//     :class:`SchemaGuard` reads it first, so every entry that builds a guard
//     (CPU, Metal, the in-place twin, a direct engine call, a trace) refuses
//     the same way at the cost of one shift and one test.  An op whose other
//     operands are not seen by the guard calls :func:`check_schema_dtype` for
//     them.
// cast : CastHint
//     What the refusal's message tells the caller to cast to.  Read only when
//     ``refused`` is non-empty.
struct LUCID_API OpSchema {
    std::string_view name;
    int version;

    AmpPolicy declared_amp_policy;
    bool deterministic;
    std::string_view determinism_note;

    // -1 indicates the op accepts a variable number of inputs.
    int input_arity = -1;

    int output_arity = 1;

    // Indices into the input list whose values are saved for backward;
    // SchemaGuard / autograd version-check logic uses these to identify
    // which inputs must remain unmodified between forward and backward.
    std::vector<int> stable_input_indices;

    // When true, the op should not be exposed in the public Python API.
    bool internal = false;

    // When true, this op's result is a real number whatever its input was,
    // so an integer input is promoted to float rather than truncated or
    // refused.  ``exp(1)`` is 2.718, not 2.
    //
    // Deliberately not inferred from the AMP policy.  The two say different
    // things: the policy answers "*which* float under autocast", and
    // ``AmpPolicy::Promote`` is worn by ``matmul``, whose integer answer is
    // an integer.  Forcing these ops to ``ForceFP32`` instead would have
    // been the smaller edit and the wrong one — it would drag ``tanh`` and
    // ``sigmoid`` out of F16 inside an autocast scope, costing half-precision
    // training memory and speed to fix an integer-input bug.
    bool real_valued = false;

    // When true, an integer or bool input keeps its own dtype although
    // ``declared_amp_policy`` is ``ForceFP32``: the op has an exact answer in that
    // type, and the policy speaks for floating inputs only.  ``pow`` is the
    // case.  ``int64 ** 2`` is an integer, exact past 2**53 where float32 is
    // not, while ``f16 ** f16`` under autocast runs in float32 as the
    // reference runs it.  Without the flag one policy had to serve both, and
    // switching ``pow`` to ``Promote`` for the first broke the second.
    bool integral_closed = false;

    // The dtypes this op turns away, as the reference CPU does.  Declared with
    // :func:`refusing` so the op's schema line names the dtypes, the class and
    // the cast hint in one place; read by :class:`SchemaGuard`.
    DtypeRefusal refusal = DtypeRefusal::None;
    CastHint cast = CastHint::Number;
    DtypeMask refused = 0;

    // Minimal constructor for deterministic ops with default arity.
    //
    // Parameters
    // ----------
    // n : std::string_view
    //     Canonical op name.
    // v : int
    //     Schema version (start at 1).
    // ap : AmpPolicy
    //     Sets ``declared_amp_policy``.
    // det : bool, default=true
    //     Determinism flag.
    // note : std::string_view, default=""
    //     Non-determinism explanation; ignored when ``det`` is true.
    // real : bool, default=false
    //     Sets ``real_valued``.
    // closed : bool, default=false
    //     Sets ``integral_closed``.
    //
    // Notes
    // -----
    // Leaves ``input_arity = -1`` (variadic), ``output_arity = 1``,
    // ``stable_input_indices`` empty, and ``internal = false``.  Use the
    // full constructor below when these defaults are not appropriate.
    constexpr OpSchema(std::string_view n,
                       int v,
                       AmpPolicy ap,
                       bool det = true,
                       std::string_view note = "",
                       bool real = false,
                       bool closed = false)
        : name(n),
          version(v),
          declared_amp_policy(ap),
          deterministic(det),
          determinism_note(note),
          real_valued(real),
          integral_closed(closed) {}

    // Full constructor for ops that need explicit arity, saved-input
    // tracking, or the ``internal`` flag set.
    //
    // Parameters
    // ----------
    // n : std::string_view
    //     Canonical op name.
    // v : int
    //     Schema version.
    // ap : AmpPolicy
    //     Sets ``declared_amp_policy``.
    // det : bool
    //     Determinism flag.
    // note : std::string_view
    //     Non-determinism explanation.
    // in_arity : int
    //     Expected input count, or ``-1`` for variadic.
    // out_arity : int
    //     Number of outputs produced.
    // stable_ins : std::vector<int>
    //     Indices of inputs whose values are saved for backward.
    // is_internal : bool, default=false
    //     Hide the op from the public Python surface when true.
    OpSchema(std::string_view n,
             int v,
             AmpPolicy ap,
             bool det,
             std::string_view note,
             int in_arity,
             int out_arity,
             std::vector<int> stable_ins,
             bool is_internal = false,
             bool real = false)
        : name(n),
          version(v),
          declared_amp_policy(ap),
          deterministic(det),
          determinism_note(note),
          input_arity(in_arity),
          output_arity(out_arity),
          stable_input_indices(std::move(stable_ins)),
          internal(is_internal),
          real_valued(real) {}

    // Returns a copy of this schema that also refuses ``mask``.
    //
    // Chains onto a schema's constructor so the declaration stays one line:
    //
    //     const OpSchema SoftmaxBackward::schema_v1 =
    //         OpSchema{"softmax", 1, AmpPolicy::ForceFP32, true}.refusing(
    //             kBoolAndSignedIntegers, DtypeRefusal::Kernel);
    //
    // An op has one refusal class and one cast hint, so a second call may add
    // dtypes but may not change either.  The functions are ``constexpr``: a
    // schema that does is rejected by the compiler where the declaration is a
    // constant expression, and by the first load of the library where it is
    // not (an exception out of a static initialiser ends the process, so the
    // mistake cannot ship unseen).
    //
    // Parameters
    // ----------
    // mask : DtypeMask
    //     The dtypes to refuse, e.g. ``kSignedIntegers``.  Non-empty.
    // kind : DtypeRefusal
    //     The reference's error class for those dtypes.  Not ``None``.
    // hint : CastHint, default=Floating
    //     What the message tells the caller to cast to.
    constexpr OpSchema
    refusing(DtypeMask mask, DtypeRefusal kind, CastHint hint = CastHint::Floating) const {
        if (mask == 0 || kind == DtypeRefusal::None)
            throw std::logic_error("OpSchema::refusing needs a dtype and a refusal class");
        if (refusal != DtypeRefusal::None && (refusal != kind || cast != hint))
            throw std::logic_error("OpSchema::refusing: an op has one refusal class and one hint");
        OpSchema out = *this;
        out.refusal = kind;
        out.cast = hint;
        out.refused = static_cast<DtypeMask>(refused | mask);
        return out;
    }

    // Shorthand for ``refusing(dtype_bit(Dtype::Bool), kind, hint)``: the ops
    // that turn away a bool and nothing else (``round``, ``relu``).
    constexpr OpSchema refusing_bool(DtypeRefusal kind, CastHint hint) const {
        return refusing(dtype_bit(Dtype::Bool), kind, hint);
    }

    // Whether this op turns ``dt`` away.
    constexpr bool refuses(Dtype dt) const noexcept {
        return ((refused >> static_cast<unsigned>(dt)) & 1u) != 0;
    }
};

// Computes a 64-bit FNV-1a fingerprint of an op's behavioural schema.
//
// The hash mixes the schema's ``name``, ``version``, the op's autocast
// policy on each device (its row in the policy table, Metal then CPU), and
// the ``deterministic`` flag — what together defines how the op behaves.
// ``stable_input_indices``, ``internal``, and arity counts are excluded
// because they describe shape rather than semantics.
//
// Used for change-detection: cached graph compilations, profiler entries,
// and checkpoint metadata embed this hash so a schema bump (or any
// behaviour-relevant change) invalidates the cache automatically.
//
// Parameters
// ----------
// s : const OpSchema&
//     The schema to hash.
//
// Returns
// -------
// std::uint64_t
//     A 64-bit FNV-1a digest.  Stable across processes for the same
//     schema fields; not cryptographic.
//
// Math
// ----
// $$ h_0 = \text{0xcbf29ce484222325}, \qquad
//    h_{i+1} = (h_i \oplus b_i) \cdot \text{0x100000001b3} \pmod{2^{64}}, $$
//
// where $b_i$ ranges over the bytes of ``name``, a null separator,
// the little-endian bytes of ``version``, the Metal and CPU policy bytes,
// and finally the ``deterministic`` flag (0 or 1).
//
// Notes
// -----
// The null byte between ``name`` and ``version`` prevents length-
// extension collisions of the form ``"ab"`` vs ``"a" + 'b'``.
LUCID_API std::uint64_t schema_hash(const OpSchema& s);

}  // namespace lucid
