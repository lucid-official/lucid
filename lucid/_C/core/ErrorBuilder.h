// lucid/_C/core/ErrorBuilder.h
//
// Fluent error-construction helpers and a thread-local operation call-stack
// for enriching error messages with context about which ops were active at
// the point of failure.
//
// ErrorContext maintains a thread-local stack of operation names.  Each op
// that wants to annotate errors with its name pushes onto the stack via
// ErrorContextGuard (RAII) or the manual push/pop API.  When an error is
// thrown, ErrorBuilder::format_with_context() appends the current trace
// string (e.g. "linear > matmul > gemm") to the message.
//
// Usage:
//   ErrorContextGuard guard("my_op");
//   ErrorBuilder("my_op").shape_mismatch(expected, got);
//
// ErrorBuilder does NOT hold a reference to any tensor; its only job is to
// call the right throw expression with a well-formatted message.

#pragma once

#include <string>
#include <string_view>
#include <vector>

#include "../api.h"
#include "Device.h"
#include "Dtype.h"
#include "Error.h"
#include "Shape.h"

namespace lucid {

// Thread-local call-stack of operation names used to annotate error
// messages.
//
// Each entry is the human-readable name of an op (e.g. ``"conv2d"``); the
// stack records the current chain of in-flight ops so that any exception
// can be formatted with a ``[trace=...]`` suffix identifying exactly where
// the failure occurred.  The stack is intentionally a flat
// ``std::vector<std::string>`` — the depth is bounded by the model's
// effective call chain and the cost of an extra ``emplace_back`` /
// ``pop_back`` is negligible compared to the op itself.
//
// All members are ``static`` and act on a thread-local container, so no
// instance is required and no synchronisation is performed.  Each thread
// owns its own private stack; the stack is **not** propagated across
// thread boundaries (e.g. when work is dispatched to a background executor),
// so callers in worker threads should push their own context entries.
//
// See Also
// --------
// :class:`ErrorContextGuard` : RAII wrapper that is preferred to manual
//     push/pop in any scope that may throw.
// :class:`ErrorBuilder` : Consumes :cpp:func:`trace` when formatting
//     thrown exceptions.
class LUCID_API ErrorContext {
public:
    // Pushes an op name onto the calling thread's context stack.
    //
    // Parameters
    // ----------
    // op_name : std::string
    //     Identifier for the op entering scope.  Moved into the stack;
    //     subsequent :cpp:func:`trace` calls will include it until a
    //     matching :cpp:func:`pop`.
    static void push(std::string op_name);

    // Pops the most recently pushed name from the calling thread's stack.
    //
    // Notes
    // -----
    // No-op if the stack is empty — callers do not need to guard against
    // over-pop in error-recovery paths.
    static void pop();

    // Returns the current call stack joined by ``" > "``.
    //
    // Returns
    // -------
    // std::string
    //     ``"outer > inner > leaf"`` style summary, or an empty string when
    //     the stack is empty.  Callers can therefore test for emptiness to
    //     decide whether to emit the ``[trace=...]`` suffix at all.
    static std::string trace();

    // Clears the calling thread's stack.
    //
    // Notes
    // -----
    // Intended for test teardown or error-recovery paths that need to drop
    // a partially-built trace (e.g. after catching across a long-running
    // dispatch loop).  Production code should normally rely on
    // :class:`ErrorContextGuard` for balanced push/pop.
    static void reset();
};

// RAII wrapper that pushes an op name on construction and pops it on
// destruction.
//
// Preferred over the manual :cpp:func:`ErrorContext::push` /
// :cpp:func:`ErrorContext::pop` pair in any scope that may throw — the
// destructor guarantees the stack is balanced even when an exception
// propagates through.  Move and copy are deleted to forbid transferring
// ownership of a stack frame between scopes, which would break the LIFO
// invariant.
//
// Parameters
// ----------
// op_name : std::string
//     Identifier pushed onto the thread's :class:`ErrorContext` stack on
//     construction.  The corresponding pop is performed by the destructor.
//
// Examples
// --------
// ```
// {
//     ErrorContextGuard guard("conv2d");
//     // ... any throw inside this scope is tagged "[trace=conv2d]" ...
// }
// ```
class LUCID_API ErrorContextGuard {
public:
    // Pushes op_name onto the thread's :class:`ErrorContext` stack.
    //
    // Parameters
    // ----------
    // op_name : std::string
    //     Identifier for the in-flight op; moved into the stack.
    explicit ErrorContextGuard(std::string op_name);

    // Pops the most recently pushed entry.
    ~ErrorContextGuard();

    ErrorContextGuard(const ErrorContextGuard&) = delete;
    ErrorContextGuard& operator=(const ErrorContextGuard&) = delete;
};

// Constructs and throws typed exceptions, annotating each message with the
// owning op name and the current :class:`ErrorContext` trace.
//
// Every throw helper formats its message as
// ``"<op>: <msg> [trace=<chain>]"`` (the ``[trace=...]`` suffix is omitted
// when the context stack is empty) and then dispatches to the appropriate
// :class:`LucidError` subclass.  Helpers are marked ``[[noreturn]]`` so
// callers can use them as drop-in replacements for ``throw`` without
// confusing the compiler's control-flow analysis.
//
// :class:`ErrorBuilder` is stateless beyond the owning op name — it never
// holds a reference to a tensor and is safe to construct on the stack at
// the throw site.
//
// Parameters
// ----------
// op_name : std::string, std::string_view, or const char*
//     Identifier of the op responsible for the throw.  Three overloads are
//     provided so call sites can pass whichever literal type is most
//     convenient.
//
// Examples
// --------
// ```
// ErrorBuilder("my_op").shape_mismatch(expected, got, "weight tensor");
// ErrorBuilder("conv2d").not_implemented("dilation>1 on integer dtypes");
// ```
//
// See Also
// --------
// :class:`ErrorContext` : Owns the thread-local trace that ``ErrorBuilder``
//     consumes.
// :class:`Validator` : Layered on top of ``ErrorBuilder`` for tensor
//     precondition checks.
class LUCID_API ErrorBuilder {
public:
    // Constructs a builder owning the given op name.
    //
    // Parameters
    // ----------
    // op_name : std::string
    //     Identifier moved into ``op_``.  Used as the prefix of every
    //     formatted message produced by this builder.
    explicit ErrorBuilder(std::string op_name) : op_(std::move(op_name)) {}

    // Constructs a builder from a string view (copied into ``op_``).
    //
    // Parameters
    // ----------
    // op_name : std::string_view
    //     Identifier copied into ``op_``.
    explicit ErrorBuilder(std::string_view op_name) : op_(op_name) {}

    // Constructs a builder from a C string (copied into ``op_``).
    //
    // Parameters
    // ----------
    // op_name : const char*
    //     Identifier copied into ``op_``.  Must be null-terminated.
    explicit ErrorBuilder(const char* op_name) : op_(op_name) {}

    // Throws a generic :class:`LucidError` with the message prepended by the
    // op name and the current trace.
    //
    // Parameters
    // ----------
    // msg : const std::string&
    //     Human-readable explanation of the failure.
    //
    // Raises
    // ------
    // LucidError
    //     Always.  This function does not return.
    [[noreturn]] void fail(const std::string& msg) const;

    // Throws a :class:`NotImplementedError` with the op name and trace.
    //
    // Use for combinations the engine recognises but does not yet support
    // (e.g. an unsupported dtype on a specific backend).
    //
    // Parameters
    // ----------
    // msg : const std::string&
    //     Description of the missing implementation.
    //
    // Raises
    // ------
    // NotImplementedError
    //     Always.  This function does not return.
    [[noreturn]] void not_implemented(const std::string& msg) const;

    // Throws an :class:`IndexError` with the op name and trace.
    //
    // Parameters
    // ----------
    // msg : const std::string&
    //     Description of the out-of-bounds access (typically the offending
    //     index and the valid range).
    //
    // Raises
    // ------
    // IndexError
    //     Always.  This function does not return.
    [[noreturn]] void index_error(const std::string& msg) const;

    // Throws an :class:`InvalidArgument` with the op name and trace.
    //
    // Parameters
    // ----------
    // msg : const std::string&
    //     Which argument and the value it may not have.
    //
    // Raises
    // ------
    // InvalidArgument
    //     Always.  This function does not return.
    [[noreturn]] void invalid_argument(const std::string& msg) const;

    // Throws a :class:`LinAlgError` with the op name and trace.
    //
    // For a matrix the routine cannot factor (LAPACK's ``info > 0``), not
    // for a malformed argument.
    //
    // Parameters
    // ----------
    // msg : const std::string&
    //     The failed step, e.g. the ``info`` LAPACK reported.
    //
    // Raises
    // ------
    // LinAlgError
    //     Always.  This function does not return.
    [[noreturn]] void linalg_error(const std::string& msg) const;

    // Throws a :class:`ShapeMismatch` with the op name, optional detail,
    // and current trace combined as the exception's ``context`` field.
    //
    // Parameters
    // ----------
    // expected : const Shape&
    //     Shape the op required.
    // got : const Shape&
    //     Shape that was actually supplied.
    // detail : const std::string&, optional
    //     Extra annotation (e.g. ``"rhs"``) appended after the op name when
    //     non-empty.  Defaults to ``""``.
    //
    // Raises
    // ------
    // ShapeMismatch
    //     Always.  This function does not return.
    [[noreturn]] void
    shape_mismatch(const Shape& expected, const Shape& got, const std::string& detail = "") const;

    // Throws a :class:`DtypeMismatch`, converting :cpp:enum:`Dtype`
    // enumerators to their canonical string names before construction.
    //
    // Parameters
    // ----------
    // expected : Dtype
    //     Required dtype.
    // got : Dtype
    //     Dtype actually supplied.
    // detail : const std::string&, optional
    //     Extra annotation appended after the op name when non-empty.
    //     Defaults to ``""``.
    //
    // Raises
    // ------
    // DtypeMismatch
    //     Always.  This function does not return.
    [[noreturn]] void
    dtype_mismatch(Dtype expected, Dtype got, const std::string& detail = "") const;

    // Throws a :class:`DeviceMismatch`, converting :cpp:enum:`Device`
    // enumerators to their canonical string names before construction.
    //
    // Parameters
    // ----------
    // expected : Device
    //     Required device.
    // got : Device
    //     Device actually supplied.
    // detail : const std::string&, optional
    //     Extra annotation appended after the op name when non-empty.
    //     Defaults to ``""``.
    //
    // Raises
    // ------
    // DeviceMismatch
    //     Always.  This function does not return.
    [[noreturn]] void
    device_mismatch(Device expected, Device got, const std::string& detail = "") const;

private:
    std::string op_;
    // Returns ``"<op_>: <msg> [trace=<chain>]"`` (the ``[trace=...]`` suffix
    // is omitted when the current :class:`ErrorContext` trace is empty).
    // Used by the throw helpers that take a free-form message.
    std::string format_with_context(const std::string& msg) const;
};

// The refusal of a dtype :func:`metal_holds` says no to, whichever way it tried
// to reach Metal: a real tensor through ``to_mlx_dtype``, a phantom through
// :func:`refuse_dtype_device_cannot_hold`.  Internal to the engine, so it is
// not exported.
//
// Raises
// ------
// NotImplementedError
//     Always.  This function does not return.
[[noreturn]] void refuse_metal_dtype(Dtype dt);

}  // namespace lucid
