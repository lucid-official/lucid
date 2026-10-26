// lucid/_C/core/Generator.h
//
// Counter-based pseudo-random number generator (Philox-4x32-10) and
// the process-wide default-generator accessor.
//
// Philox is a block cipher used as a stateless PRNG: given a
// ``(counter, seed)`` pair it deterministically yields four
// independent 32-bit outputs per round in a fixed 10-round mixing
// schedule.  Because every call's output depends only on the explicit
// counter — never on hidden state from previous calls — two threads
// holding the same seed can advance disjoint counter ranges without
// any lock contention.  This makes Philox the natural choice for the
// parallel sampling kernels in :file:`ops/gfunc/`.
//
// Every distribution op (``randn``, ``uniform``, ``bernoulli``,
// ``randint`` …) threads a :class:`Generator` through its kwargs.
// When the user does not pass one, the op falls back to the global
// :func:`default_generator` singleton — the same convention as the
// reference framework, so checkpoint state captured by other libraries
// can be replayed bit-for-bit on Lucid.
//
// Notes
// -----
// Sampling calls (:func:`Generator::next_uint32x4`,
// :func:`Generator::next_uniform_float`) advance the counter and are
// *not* internally synchronised; concurrent callers must acquire
// :func:`Generator::mutex` or use a ``std::lock_guard``.  The mutex is
// exposed so the Python binding layer can hold it across a batch of
// draws and amortise the lock-acquire overhead.
//
// See Also
// --------
// :func:`lucid.manual_seed` — Python convenience that seeds the
//     default generator.
// :func:`default_generator` — global singleton accessor.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "../api.h"

namespace lucid {

// Philox-4x32-10 counter-based pseudo-random number generator.
//
// Owns a ``(seed, counter)`` pair plus a mutex guarding both fields.
// Each call to :func:`next_uint32x4` produces four independent 32-bit
// uniforms and increments the counter by one; the (seed, counter) tuple
// uniquely identifies every emitted block, which is what makes the
// stream reproducible across process invocations.
//
// Attributes
// ----------
// seed_ : uint64_t
//     The seed value last set via :func:`set_seed` (or the constructor).
//     Read-only after construction except through :func:`set_seed`.
// counter_ : uint64_t
//     Monotonically increasing call counter; reset to ``0`` by
//     :func:`set_seed`.  May be overridden directly via
//     :func:`set_counter` for checkpoint restore.
// mu_ : std::mutex
//     Guards both ``seed_`` and ``counter_`` so concurrent samplers
//     observe a consistent state.
//
// Notes
// -----
// Invariants enforced by the implementation:
//
// 1. ``counter_`` is monotonically increasing within a seed epoch.
// 2. ``(seed_, counter_)`` uniquely identifies each produced block,
//    making the sequence fully reproducible.
// 3. ``mu_`` must be held by all mutating accessors when the generator
//    is shared across threads.
//
// Sampling calls are not thread-safe on their own — wrap them in a
// ``std::lock_guard<std::mutex>`` over :func:`mutex` when sharing.
//
// Examples
// --------
// Reproducible draws across two runs::
//
//     Generator g(42);
//     uint32_t out[4];
//     g.next_uint32x4(out);
//     // second run, same seed → identical out[0..3]
//
// See Also
// --------
// :func:`default_generator` — global Generator used when ops are
//     called without an explicit ``generator=`` kwarg.
class LUCID_API Generator {
public:
    // Constructs a generator with the given seed and counter starting
    // at zero.
    //
    // Parameters
    // ----------
    // seed : uint64_t, optional
    //     Initial seed value.  Defaults to ``0``, matching the global
    //     :func:`default_generator`.
    explicit Generator(std::uint64_t seed = 0);

    // Sets the seed and resets the counter to zero, restarting the
    // sequence from a fresh deterministic origin.
    //
    // Parameters
    // ----------
    // seed : uint64_t
    //     New seed.  Replaces the existing seed atomically with the
    //     counter reset (both protected by ``mu_``).
    //
    // Notes
    // -----
    // The standard re-seed-then-resample idiom for checkpointing.
    // After this call the next four uniforms depend only on ``seed``,
    // independent of any previous use of the generator.
    void set_seed(std::uint64_t seed);

    // Returns the seed value last set via the constructor or
    // :func:`set_seed`.
    //
    // Returns
    // -------
    // uint64_t
    //     Current seed.  Not synchronised with concurrent
    //     :func:`set_seed` calls — wrap in a lock if a consistent
    //     snapshot is required.
    std::uint64_t seed() const { return seed_; }

    // Generates four independent 32-bit uniforms using Philox-4x32-10
    // on the current ``(counter_, seed_)`` state, then increments
    // ``counter_`` by one.
    //
    // Parameters
    // ----------
    // out : uint32_t[4]
    //     Output buffer; must point to storage for at least four
    //     ``uint32_t`` values.  The caller owns the memory.
    //
    // Notes
    // -----
    // Not thread-safe on its own.  When shared across threads, acquire
    // :func:`mutex` (or use a ``std::lock_guard``) before calling.
    //
    // See Also
    // --------
    // :func:`next_uniform_float` — single-float variant.
    void next_uint32x4(std::uint32_t out[4]);

    // Generates a single uniform float in ``[0, 1)`` from the upper 24
    // bits of one Philox output word.
    //
    // Returns
    // -------
    // float
    //     Uniformly distributed value in :math:`[0, 1)`.
    //
    // Notes
    // -----
    // Internally derives one ``uint32_t`` from
    // :func:`next_uint32x4`, drops the lower 8 bits for uniformity in
    // the IEEE-754 ``float`` mantissa, and divides by :math:`2^{24}`.
    // The three unused 32-bit words from the Philox block are
    // discarded; callers that need higher throughput should batch
    // through :func:`next_uint32x4`.
    //
    // Not thread-safe — see :func:`next_uint32x4`.
    float next_uniform_float();

    // Returns the internal mutex so callers can hold it across batched
    // sampling calls.
    //
    // Returns
    // -------
    // std::mutex&
    //     Reference to the mutex guarding ``seed_`` and ``counter_``.
    //
    // Notes
    // -----
    // Exposed so the Python binding can wrap a whole vector draw under
    // one lock acquisition rather than re-locking on every uniform.
    std::mutex& mutex() { return mu_; }

    // Returns the current counter value.
    //
    // Returns
    // -------
    // uint64_t
    //     The number of Philox blocks consumed since the last
    //     :func:`set_seed` (modulo any direct :func:`set_counter`
    //     override).
    //
    // Notes
    // -----
    // Useful when serialising RNG state for a checkpoint: capture
    // ``(seed(), counter())`` and restore via :func:`set_seed` +
    // :func:`set_counter` on reload.
    std::uint64_t counter() const { return counter_; }

    // Directly overrides the counter — used to restore a saved RNG
    // state from a checkpoint.
    //
    // Parameters
    // ----------
    // c : uint64_t
    //     New counter value.  Replaces ``counter_`` atomically with
    //     respect to other field accesses guarded by ``mu_``.
    //
    // Notes
    // -----
    // Normal use should call :func:`set_seed` first to fix the seed,
    // then :func:`set_counter` to restore the position within that
    // seed epoch.  This pair reproduces any previously observed state
    // exactly.
    void set_counter(std::uint64_t c) { counter_ = c; }

private:
    std::uint64_t seed_;
    std::uint64_t counter_;
    std::mutex mu_;
};

// Returns the process-wide default :class:`Generator` instance.
//
// Returns
// -------
// Generator&
//     Reference to the global singleton, lazily constructed on first
//     access with ``seed = 0``.  Lifetime persists for the duration of
//     the process.
//
// Notes
// -----
// All random ops that are not given an explicit ``generator=`` kwarg
// route through this singleton.  Python's
// :func:`lucid.manual_seed(seed)` is implemented by calling
// :func:`Generator::set_seed` on the object returned here.
//
// Examples
// --------
// Re-seed every distribution-op call across the whole process::
//
//     default_generator().set_seed(42);
//     // every subsequent randn / uniform / bernoulli sees seed=42
//
// See Also
// --------
// :func:`lucid.manual_seed` — Python convenience wrapper.
LUCID_API Generator& default_generator();

}  // namespace lucid
