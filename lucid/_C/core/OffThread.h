// lucid/_C/core/OffThread.h
//
// Tensors another thread makes while a recording is open.
//
// The tracer is per thread, so what a worker thread computes during a traced
// call is no op of the recording, and its result reaches an op of the tracing
// thread as a tensor nobody told the trace of.  Taken for an outside input it is
// pinned at the first call's bytes, and every compiled call after the first
// answers with them.  A grad node would give it away, but under ``no_grad`` there
// is none, and nothing else in a tensor says when or where it was made.
//
// While at least one recording asks for it (``compile::Tracer::refuse_off_thread_values``)
// the TensorImpl constructor reports each tensor a thread with no recording of
// its own makes, here, with the epoch of the latest recording to have opened.  A
// recording refuses a feed registered at or after its own epoch: made during it,
// on another thread.  A tensor made before it — a parameter, a closure's, one a
// pool computed ahead of the call — carries an older epoch or none, and stays a
// normal input.
//
// The registry is keyed by address and has no destructor hook, so an entry can
// outlive its tensor.  A new tensor at that address goes through the same
// constructor hook, which overwrites or erases the entry, so a stale one never
// answers for a live tensor.  The registry is dropped when the last recording
// closes.  With no recording open the cost is one relaxed load per construction.

#pragma once

#include <atomic>
#include <cstdint>

#include "../api.h"
#include "fwd.h"

namespace lucid::off_thread {

// How many recordings are open; read on every TensorImpl construction.
inline std::atomic<int> g_open{0};

// A recording opens on the calling thread and answers its epoch (> 0).  The
// thread leads it until :func:`close`: what it makes is the trace's own.
LUCID_API std::uint64_t open();

// Closes a recording opened on the thread ``opener`` (:func:`thread_token`).
// Called from another thread, the opener keeps leading until it ends, which only
// means it is not watched.
LUCID_API void close(std::uintptr_t opener);

// An identity of the calling thread for :func:`close`.
LUCID_API std::uintptr_t thread_token();

// A tensor was constructed at ``impl`` (out of line past the one branch below).
LUCID_API void note_made(const TensorImpl* impl);

inline void on_construct(const TensorImpl* impl) {
    if (g_open.load(std::memory_order_relaxed) != 0)
        note_made(impl);
}

// Whether ``impl`` was made on a thread with no recording of its own, during a
// recording whose epoch is ``epoch`` or later.
LUCID_API bool made_since(const TensorImpl* impl, std::uint64_t epoch);

}  // namespace lucid::off_thread
