// lucid/_C/core/SchemaGuard.h
//
// Op-dispatch entry gate that combines dtype refusal, determinism enforcement
// and AMP dtype resolution.
//
// Every Lucid op (forward and backward) begins its dispatch sequence by
// constructing a :class:`SchemaGuard` from its :class:`OpSchema` and the
// dtype/device of its primary input.  The guard's constructor does three
// things in one pass:
//
//   0. Dtype refusal — first statement.  When ``input_dtype`` is in the
//      schema's ``refused`` mask, the constructor throws before anything else
//      is read: ``DtypeRefusal::Kernel`` raises :class:`NotImplementedError`
//      (the reference's "no kernel for this dtype"), ``DtypeRefusal::Guard``
//      raises :class:`DtypeMismatch` (its dtype guard).  Both messages read
//      ``<op> does not accept <dtype> — cast to <hint>`` with the hint from
//      ``cast``.  An op that refuses a dtype declares it on its schema, so the
//      CPU and Metal entries, the in-place twin, a direct engine call and a
//      trace share one check at the cost of one shift and one test.  The guard
//      sees one operand: an op with another operand the refusal covers calls
//      :func:`check_schema_dtype` for it, and an op built from other ops (a
//      composite with no schema of its own) calls :func:`throw_dtype_refusal`
//      itself.
//
//   1. Determinism gate — if the global :func:`Determinism::is_enabled`
//      flag is set and the schema's ``deterministic`` field is false, a
//      :class:`LucidError` is raised before any computation begins.  The
//      error message includes the op's ``determinism_note`` so users
//      know which aspect of the op is non-reproducible.
//
//   2. Effective dtype resolution — if AMP is active for ``device`` on the
//      current thread and ``input_dtype`` is autocast-eligible (``F16``,
//      ``BF16``, or ``F32``), the input dtype is transformed according to
//      the op's row in the per-device policy table
//      (:func:`amp::policy_for` on ``schema.name``):
//        Promote       — switch to the autocast dtype, except that
//                        ``(CPU, F16)`` is clamped to F32 because Apple
//                        Accelerate has no native F16 BLAS path.
//        KeepInput     — return ``input_dtype`` unchanged (also every op
//                        without a row).
//        ForceFP32     — always return :class:`Dtype::F32`.
//        PromoteWidest — ``input_dtype``: one input is its own widest.
//      Non-eligible dtypes (``F64``, ``C64``, ``C128``, integers, ``Bool``)
//      and calls outside AMP return ``input_dtype`` unchanged (or ``F32``
//      when an integral/bool input enters a real-valued / declared
//      ``ForceFP32`` op whose schema is not ``integral_closed``).
//
// While AMP is active the guard also turns it off on ``device`` for its own
// lifetime, as the reference framework's autocast wrappers do: the op has
// chosen its dtype at the entry, and the ops its body is built from must
// not apply their own rows again.  That covers the body only when the guard
// lives as long as the body does — true of the kernel templates' forwards,
// not of ``promote_for_schema``, whose guard is gone before the body runs.
//
// :func:`check_schema_determinism` exposes the determinism gate as a
// standalone function for callers (e.g. custom-function wrappers,
// graph-capture-style trace builders) that need only the determinism
// check without dtype resolution.
//
// See Also
// --------
// :class:`OpSchema`        — static metadata that drives the guard.
// :func:`amp::policy_for`  — the per-device autocast policy table.

#pragma once

#include <optional>
#include <string_view>

#include "../api.h"
#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "OpSchema.h"
#include "fwd.h"

namespace lucid {

// Determinism-only gate for a schema.
//
// Equivalent to the first half of :class:`SchemaGuard`'s constructor.
// Exists so callers that do not need AMP dtype resolution can perform
// just the determinism check without paying for the (small) extra work.
//
// Parameters
// ----------
// schema : const OpSchema&
//     The op's schema record.  Only the ``deterministic``,
//     ``determinism_note``, and ``name`` fields are read.
//
// Raises
// ------
// LucidError
//     Thrown when :func:`Determinism::is_enabled` is true and
//     ``schema.deterministic`` is false.  The message starts with
//     ``"non-deterministic op called under set_deterministic(True)"``
//     and appends ``schema.determinism_note`` in parentheses when
//     non-empty.
//
// Notes
// -----
// When determinism is not globally enabled, this function is a no-op
// and adds only a single relaxed atomic load to the call site.
LUCID_API void check_schema_determinism(const OpSchema& schema);

// Throws the refusal of an op that does not accept a dtype.
//
// The one place the message and the class are built: :class:`SchemaGuard`'s
// constructor and :func:`check_schema_dtype` call it for a schema whose
// ``refused`` mask holds the dtype, and a composite that has no schema
// (``frac``, the linalg solvers) calls it by name.
//
// Parameters
// ----------
// op : std::string_view
//     The op's name, as the caller used it.
// dt : Dtype
//     The refused dtype; its :func:`dtype_name` is in the message.
// kind : DtypeRefusal
//     ``Kernel`` throws :class:`NotImplementedError`, ``Guard`` throws
//     :class:`DtypeMismatch` (expected ``a non-bool dtype`` for a bool and
//     ``a floating dtype`` otherwise, got the dtype's name).  ``None`` is
//     treated as ``Kernel``: the function never returns.
// hint : CastHint
//     Picks the cast hint the message ends with.
//
// Raises
// ------
// NotImplementedError / DtypeMismatch
//     Always.  The message is ``<op> does not accept <dtype> — cast to <hint>``.
[[noreturn]] LUCID_API void
throw_dtype_refusal(std::string_view op, Dtype dt, DtypeRefusal kind, CastHint hint);

// Refuses ``dtype`` if the schema's ``refused`` mask holds it.
//
// What :class:`SchemaGuard`'s constructor does with its one input, for the
// operands it does not see and for a path it should not run on: the second
// operand of ``solve`` / ``nextafter``, the weight of ``embedding_bag``, a
// dropout that only draws a mask when ``training && p > 0``.  A pass-through
// call must keep a dtype the reference lets through, which a guard built at
// the entry would refuse.
//
// Parameters
// ----------
// schema : const OpSchema&
//     The op's schema.
// dtype : Dtype
//     The operand's dtype.
//
// Raises
// ------
// NotImplementedError / DtypeMismatch
//     Per ``schema.refusal``, when ``dtype`` is refused.
LUCID_API void check_schema_dtype(const OpSchema& schema, Dtype dtype);

// Combined determinism + AMP dtype-resolution guard.
//
// Constructed once at the top of an op's dispatch routine.  The
// constructor performs both checks; subsequent code reads the resolved
// dtype via :func:`effective_dtype`.
//
// Inside an autocast scope the guard is an RAII scope: autocast stays off
// on the op's device until the guard is destroyed, so it must live as long
// as the op's body.  Outside one it owns nothing.
//
// Examples
// --------
// .. code-block:: cpp
//
//     // Inside a forward op
//     SchemaGuard guard(MyOp::schema_v1, input->dtype(), input->device());
//     const Dtype dt = guard.effective_dtype();
//     auto out = backend->allocate(out_shape, dt);
//     dispatch_kernel(input, out, dt);
//
// See Also
// --------
// :func:`check_schema_determinism` — determinism-only variant.
class LUCID_API SchemaGuard {
public:
    // Constructs the guard and runs both gates.
    //
    // Parameters
    // ----------
    // schema : const OpSchema&
    //     Static metadata for the op being dispatched.
    // input_dtype : Dtype
    //     The dtype of the op's primary input — the starting point for
    //     AMP dtype resolution.
    // device : Device, default=Device::CPU
    //     The device on which the op will execute.  Selects the table's
    //     column, and applies the CPU-F16 → F32 demotion rule under
    //     ``AmpPolicy::Promote``, since Accelerate has no native F16 path.
    //
    // Raises
    // ------
    // LucidError
    //     Thrown when the op is non-deterministic and the global
    //     deterministic mode is active.  See
    //     :func:`check_schema_determinism` for the exact message
    //     format.
    SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device = Device::CPU);

    SchemaGuard(const SchemaGuard&) = delete;
    SchemaGuard& operator=(const SchemaGuard&) = delete;

    // Returns the dtype that the op should use for its computation.
    //
    // May differ from the ``input_dtype`` passed at construction when
    // AMP is active and the op's policy on the device is ``Promote``
    // (returns the autocast dtype, possibly demoted on CPU) or
    // ``ForceFP32`` (returns :class:`Dtype::F32`).  Equals
    // ``input_dtype`` for ``KeepInput`` / ``PromoteWidest`` and when AMP
    // is not active.
    //
    // Returns
    // -------
    // Dtype
    //     The effective compute dtype for the op.
    //
    // Notes
    // -----
    // Marked ``noexcept`` — purely a field read, no allocations or
    // exceptions.
    Dtype effective_dtype() const noexcept { return effective_dtype_; }

private:
    Dtype effective_dtype_;
    // Engaged only while AMP was active at construction.
    std::optional<amp::AutocastGuard> autocast_off_;
};

}  // namespace lucid
