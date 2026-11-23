// lucid/_C/core/SchemaGuard.cpp
//
// Implementation of the determinism gate and AMP dtype-resolution logic.
// check_schema_determinism() is also a standalone entry point for callers
// that need only the determinism gate without spinning up a full
// SchemaGuard (e.g. custom-function wrappers).

#include "SchemaGuard.h"

#include <string>

#include "AmpPolicy.h"
#include "Determinism.h"
#include "Device.h"
#include "Dtype.h"
#include "Error.h"
#include "ErrorBuilder.h"

namespace lucid {

void check_schema_determinism(const OpSchema& schema) {
    if (Determinism::is_enabled() && !schema.deterministic) {
        std::string msg = "non-deterministic op called under set_deterministic(True)";
        // Append the op-specific note (e.g. "uses atomic scatter-add") so the
        // user understands which aspect of the op causes non-determinism.
        if (!schema.determinism_note.empty()) {
            msg += " (";
            msg += schema.determinism_note;
            msg += ")";
        }
        ErrorBuilder(schema.name).fail(msg);
    }
}

namespace {

// Whether an integer or bool input becomes float32: the op answers in real
// numbers (``real_valued``, or the declared ``ForceFP32`` family) and has no
// exact answer of its own in the integer type (``integral_closed``).
constexpr bool promotes_integral_input(const OpSchema& schema) noexcept {
    return (schema.real_valued || schema.declared_amp_policy == AmpPolicy::ForceFP32) &&
           !schema.integral_closed;
}

// The dtype an input the autocast policy does not touch computes in.
//
// ``ForceFP32`` says this op needs real numbers: exp, log, sqrt, tanh,
// sigmoid, erfinv.  It was only consulted inside an autocast scope, so
// outside one an integer input took two different wrong roads — half the
// family computed in the integer type and truncated (``exp(1)`` answered
// ``2``), and the other half raised ``NotImplementedError`` because no
// integer kernel existed.
//
// The reference framework promotes to float32 for all of them, and that is
// the only answer that is not simply wrong: the value of ``exp(1)`` is not
// an integer.  Promoting here covers the whole family at once rather than
// op by op, and leaves the *result* dtype float, which is what the caller
// then observes.  ``is_integral`` deliberately excludes Bool, so it is named
// here: the reference promotes a bool input for these ops too, and
// ``sqrt(True)`` is 1.0 rather than a type error.
constexpr Dtype untouched_dtype(const OpSchema& schema, Dtype input_dtype) noexcept {
    if (promotes_integral_input(schema) && (is_integral(input_dtype) || input_dtype == Dtype::Bool))
        return Dtype::F32;
    return input_dtype;
}

}  // namespace

void throw_dtype_refusal(std::string_view op, Dtype dt, DtypeRefusal kind, CastHint hint) {
    const std::string got{dtype_name(dt)};
    std::string msg{op};
    msg += " does not accept ";
    msg += got;
    msg += " — cast to ";
    msg += cast_hint_text(hint);
    if (kind == DtypeRefusal::Guard) {
        std::string context{op};
        context += ": ";
        context += msg;
        throw DtypeMismatch(dt == Dtype::Bool ? "a non-bool dtype" : "a floating dtype", got,
                            std::move(context));
    }
    ErrorBuilder(op).not_implemented(msg);
}

void check_schema_dtype(const OpSchema& schema, Dtype dtype) {
    if (schema.refuses(dtype)) [[unlikely]]
        throw_dtype_refusal(schema.name, dtype, schema.refusal, schema.cast);
}

SchemaGuard::SchemaGuard(const OpSchema& schema, Dtype input_dtype, Device device) {
    // First, and one shift and one test: a schema that refuses a dtype says so by
    // data, so the CPU and Metal entries, the in-place twin, a direct engine call
    // and a trace all stop here with the same class and the same message.
    if (schema.refuses(input_dtype)) [[unlikely]]
        throw_dtype_refusal(schema.name, input_dtype, schema.refusal, schema.cast);

    check_schema_determinism(schema);

    if (!amp::is_active(device)) {
        effective_dtype_ = untouched_dtype(schema, input_dtype);
        return;
    }

    const Dtype autocast_dt = *amp::active_dtype(device);
    // The op's own body runs with autocast off, as the reference's autocast
    // wrappers run theirs: an op that chose float32 at its entry and is
    // built from ``mul`` and ``add`` must not have those cast again, and an
    // op with no row must not have its parts apply theirs.
    autocast_off_.emplace(autocast_dt, device, /*enabled=*/false);

    if (!amp::is_autocast_eligible(input_dtype)) {
        effective_dtype_ = untouched_dtype(schema, input_dtype);
        return;
    }
    effective_dtype_ =
        amp::policy_dtype(amp::policy_for(schema.name, device), input_dtype, autocast_dt, device);
}

}  // namespace lucid
