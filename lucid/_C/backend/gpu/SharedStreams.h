// lucid/_C/backend/gpu/SharedStreams.h
//
// One MLX GPU stream and one MLX CPU stream for the whole process.
//
// MLX gives every thread its own default stream and keeps each stream's
// command encoder in thread-local storage.  A graph node records the stream
// it was built on, so a tensor whose work is still pending could be run only
// by the thread that built it.  Any other thread failed with "There is no
// Stream(gpu, N) in current thread", and a lazy tensor returned by a worker
// that then exited could never run at all.  Lucid promises one Metal stream
// per process (``lucid.metal.MetalStream``), so every thread that builds
// Lucid work makes these two streams its defaults.  The CPU one serves the
// GPU graphs' CPU-stream nodes: MLX runs linalg and FFT only there.
//
// The streams come from ``new_thread_unsafe_stream``: any thread may run
// them, but two threads must never encode onto one at the same time — four
// threads evaluating on one such stream without a lock crashed in each of
// four runs.  The GIL is that lock: the engine encodes MLX work only while
// holding it, and releases it only to wait for a run already encoded (see
// ``run_then_wait`` in bind_tensor.cpp).
//
// Header-only so the dispatcher can call it without a new translation unit:
// an inline function's static and thread_local variables are one object
// across the program.

#pragma once

#include <mlx/device.h>
#include <mlx/stream.h>

namespace lucid::gpu {

// Make the shared streams the calling thread's MLX defaults.  Returns true
// on the call that did so, false once the thread is already bound.
//
// The first call in the process creates the streams; the engine makes it at
// import, from the importing thread, so later calls cannot fail on that.
inline bool bind_thread_streams() {
    thread_local bool bound = false;
    if (bound) [[likely]]
        return false;
    static const ::mlx::core::Stream gpu =
        ::mlx::core::new_thread_unsafe_stream(::mlx::core::Device::gpu);
    static const ::mlx::core::Stream cpu =
        ::mlx::core::new_thread_unsafe_stream(::mlx::core::Device::cpu);
    ::mlx::core::set_default_stream(gpu);
    ::mlx::core::set_default_stream(cpu);
    bound = true;
    return true;
}

}  // namespace lucid::gpu
