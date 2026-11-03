// lucid/_C/core/GradMode.h
//
// Thread-local gradient-computation mode for the autograd dispatcher.
//
// The op dispatch layer queries :func:`GradMode::is_enabled` before any
// autograd bookkeeping (attaching ``grad_fn``, saving inputs for the
// backward pass, propagating ``requires_grad`` to outputs).  When the
// mode is ``false`` ops execute in pure-inference style: no autograd
// node is created, no saved-for-backward tensors are retained, and the
// produced tensors carry ``requires_grad = false`` regardless of their
// inputs.  The mode is the C++ source of truth that the Python
// :class:`lucid.no_grad` / :func:`lucid.set_grad_enabled` helpers flip.
//
// The flag is stored in ``thread_local`` storage so each thread (e.g. a
// DataLoader worker vs the main training thread) carries an independent
// setting.  This avoids the race conditions of a global atomic flag and
// matches the reference-framework's per-thread semantics.
//
// Notes
// -----
// Toggle the mode through the RAII guard :class:`NoGradGuard` whenever
// possible — explicit ``set_enabled(false)`` followed by manual
// ``set_enabled(true)`` does not survive exceptions.
//
// See Also
// --------
// :class:`lucid.no_grad` — Python context manager / decorator.
// :func:`lucid.set_grad_enabled` — Python imperative setter.

#pragma once

namespace lucid {

// Thread-local switch controlling whether autograd records ops on the
// current thread.
//
// All methods are ``static``: there is no instance state — the class is
// a namespace-style holder over a single ``thread_local bool`` defined
// in :file:`GradMode.cpp`.  Querying or setting the flag is a single
// TLS load/store with no synchronisation cost.
//
// Notes
// -----
// The default value at thread startup is ``true`` (gradients tracked).
// Threads created via :func:`std::thread` or worker pools each begin
// with their own fresh ``true``-valued flag.
//
// See Also
// --------
// :class:`NoGradGuard` — RAII helper for scoped disabling.
class GradMode {
public:
    // Returns whether autograd is currently active on the calling thread.
    //
    // Returns
    // -------
    // bool
    //     ``true`` when ops should build autograd nodes, ``false`` when
    //     they should run in pure-inference mode.
    //
    // Notes
    // -----
    // The value defaults to ``true`` for every newly-spawned thread.
    static bool is_enabled();

    // Sets the autograd activity flag for the calling thread.
    //
    // Parameters
    // ----------
    // value : bool
    //     ``true`` to enable autograd recording, ``false`` to suppress
    //     it.  The change is only visible to the calling thread.
    //
    // Notes
    // -----
    // Prefer :class:`NoGradGuard` over manual paired ``set_enabled``
    // calls — the RAII form correctly restores the previous flag when
    // an exception unwinds the stack.
    //
    // See Also
    // --------
    // :class:`NoGradGuard` — exception-safe scoped toggle.
    static void set_enabled(bool value);

    // Returns the calling thread's raw grad-mode flag, ignoring inference mode.
    static bool raw_enabled();

    // Returns whether inference mode is currently active on the calling thread.
    static bool inference_enabled();

    // Sets the inference-mode flag for the calling thread.
    static void set_inference_enabled(bool value);
};

// RAII guard that disables gradient tracking for the lifetime of the
// object on the calling thread.
//
// Constructing the guard captures the current value of
// :func:`GradMode::is_enabled` and forces the flag to ``false``; the
// destructor unconditionally restores the saved value.  This nests
// correctly with any outer guards because each instance only restores
// what it itself saw — multiple nested guards rebuild the original
// state level by level on unwind.
//
// Attributes
// ----------
// prev_ : bool
//     Snapshot of :func:`GradMode::is_enabled` taken at construction
//     time and restored verbatim by the destructor.
//
// Notes
// -----
// The C++ counterpart of Python's :class:`lucid.no_grad` context
// manager.  Bindings that need a ``with`` block on the Python side
// instantiate one of these and tie its lifetime to ``__enter__`` /
// ``__exit__``.  The guard is intentionally non-copyable so that the
// "exactly one restore per construction" invariant cannot be broken.
//
// Examples
// --------
// Disable autograd for a small inference subroutine without touching
// the surrounding training loop's mode::
//
//     {
//         NoGradGuard nograd;
//         auto y = model->forward(x);  // no graph built
//     }
//     // grad mode automatically restored here
//
// See Also
// --------
// :class:`GradMode` — underlying flag accessors.
// :class:`lucid.no_grad` — Python equivalent.
class NoGradGuard {
public:
    // Captures the current grad-mode flag and sets it to ``false``.
    NoGradGuard();

    // Restores the grad-mode flag to the value captured at construction.
    ~NoGradGuard();

    NoGradGuard(const NoGradGuard&) = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

// RAII guard that sets gradient tracking to ``enabled`` for its lifetime on
// the calling thread, and restores the flag it found.  A backward pass runs
// under one: on only when it records the graph of its own gradients.
class GradModeGuard {
public:
    explicit GradModeGuard(bool enabled);
    ~GradModeGuard();

    GradModeGuard(const GradModeGuard&) = delete;
    GradModeGuard& operator=(const GradModeGuard&) = delete;

private:
    bool prev_;
};

}  // namespace lucid
