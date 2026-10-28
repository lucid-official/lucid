// lucid/_C/backend/gpu/mps/MpsBridge.mm
//
// Obj-C++ implementation of the MLX ↔ MPSGraph bridge.  See MpsBridge.h.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <atomic>
#include <mutex>

#include <mlx/array.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>

#include "MpsBridge.h"

namespace lucid::gpu::mps {

namespace {

std::once_flag g_init_once;
id<MTLDevice> g_device = nil;
id<MTLCommandQueue> g_queue = nil;
std::atomic<bool> g_bridge_ok{false};

void init_bridge_once() {
    std::call_once(g_init_once, []() {
        // Apple Silicon has one MTLDevice per system; MTLCreateSystemDefaultDevice
        // returns the same object MLX uses internally.  Going through Apple's
        // API directly avoids depending on MLX's metal-cpp include path.
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device)
            return;
        g_queue = [g_device newCommandQueue];
        if (!g_queue)
            return;
        [g_queue setLabel:@"lucid-mps-dispatch"];
        g_bridge_ok.store(true, std::memory_order_release);
    });
}

}  // namespace

void* shared_mtl_device() {
    init_bridge_once();
    return (__bridge void*)g_device;
}

void* shared_mtl_queue() {
    init_bridge_once();
    return (__bridge void*)g_queue;
}

std::size_t metal_device_allocated_bytes() {
    // Total GPU memory the Metal device currently holds — captures MLX, the
    // compiled MPSGraph executable's internal activations, and run_executable's
    // output buffers alike (unlike MLX's allocator-only peak, which is blind to
    // MPSGraph).  Used to measure the compiled training step's true footprint.
    init_bridge_once();
    return (std::size_t)[g_device currentAllocatedSize];
}

BufferView array_to_buffer(const ::mlx::core::array& arr) {
    // Force materialisation + wait for completion when the array is not yet
    // evaluated.  We need the kernel that produced `arr` to finish before
    // MPSGraph reads the buffer — MLX's command queue and ours are distinct,
    // so Metal will not insert an implicit fence.  `wait()` is the explicit
    // cross-queue barrier.  Skipping `eval()` + `wait()` when `is_available()`
    // is already true avoids 100+ redundant MLX scheduler lock round-trips per
    // step for static parameter and constant buffers.
    auto& mutable_arr = const_cast<::mlx::core::array&>(arr);
    if (!mutable_arr.is_available()) {
        mutable_arr.eval();
        mutable_arr.wait();
    }

    // Critical: MPSGraph reads the MTLBuffer linearly assuming row-major
    // storage matching the declared `nsShape`, but MLX upstream ops
    // (notably Conv2d) return *lazy transpose views* — the underlying
    // buffer is in a permuted layout (e.g. NHWC physical for an NCHW
    // logical shape).  Feeding such a buffer to MPSGraph as-is shuffles
    // elements silently and gives wrong results.  Force a contiguous
    // materialisation here so the MPSGraph executable gets a
    // buffer whose physical layout matches its logical shape.
    //
    // This is a no-op (single MLX flag check) when arr is already
    // row-contiguous — `mlx::core::contiguous` returns the input view
    // unchanged in that case.  Only the truly non-contig case pays a
    // memcpy.
    if (!mutable_arr.flags().row_contiguous) {
        mutable_arr = ::mlx::core::contiguous(mutable_arr);
        mutable_arr.eval();
        mutable_arr.wait();
    }

    // A row-contiguous array can still begin past the start of its buffer:
    // a leading-axis slice ``x[3:]`` is one.  Every caller hands the buffer
    // to MPSGraph, whose tensor data takes a buffer and no offset, so such
    // an array was read from the buffer's start — a compiled module fed
    // ``x[3:]`` computed on ``x[:n]``, with no error.  Give it a buffer of
    // its own.  Not ``copy`` or ``contiguous``: both return a contiguous
    // input as it is, sharing the buffer and its offset.  Not ``add`` of
    // zero, which turns -0.0 into +0.0; multiplying by one changes no value.
    if (mutable_arr.offset() != 0) {
        mutable_arr =
            ::mlx::core::multiply(mutable_arr, ::mlx::core::array(1, mutable_arr.dtype()));
        mutable_arr.eval();
        mutable_arr.wait();
    }

    const auto& buf = arr.buffer();
    // Buffer::ptr() on a const Buffer returns const void*; cast away const
    // because the underlying MTLBuffer is mutable and we hand it to
    // MPSGraph which expects id<MTLBuffer>.
    void* mtl_buffer_raw = const_cast<void*>(buf.ptr());
    // MLX keeps the offset in bytes (``data()`` adds it to a ``char*``).
    // It was multiplied by the item size here as well, so the DLPack export
    // of a float32 slice three elements in claimed byte_offset 48.  Zero
    // after the block above, but stated in the right unit.
    const std::size_t offset_bytes = static_cast<std::size_t>(arr.offset());
    const std::size_t nbytes = arr.nbytes();
    return BufferView{mtl_buffer_raw, offset_bytes, nbytes};
}

::mlx::core::array
buffer_to_array(void* mtl_buffer, std::vector<int> shape, Dtype dt, std::size_t offset_bytes) {
    // Wrap as an allocator::Buffer with a deleter that transfers ownership
    // back to ARC.  We don't go through MetalAllocator::make_buffer because
    // that path is for buffers MLX itself allocates — ours came from
    // [MTLBuffer newBufferWithLength:] in the kernel layer.
    ::mlx::core::allocator::Buffer wrapped{mtl_buffer};

    auto deleter = [](::mlx::core::allocator::Buffer b) {
        // `(__bridge_transfer id<MTLBuffer>)` moves the +1 retain into ARC's
        // scope; assigning to nil immediately releases.
        @autoreleasepool {
            id<MTLBuffer> dropped = (__bridge_transfer id<MTLBuffer>)b.ptr();
            (void)dropped;
        }
    };

    ::mlx::core::Shape mlx_shape;
    mlx_shape.reserve(shape.size());
    for (int d : shape)
        mlx_shape.push_back(d);

    // Map Lucid Dtype → mlx::core::Dtype.  Reuse the existing helper from
    // MlxBridge but without pulling its full header (which would create a
    // dependency cycle).  The mapping is small and stable.
    ::mlx::core::Dtype mlx_dt = ::mlx::core::float32;
    switch (dt) {
    case Dtype::F32:
        mlx_dt = ::mlx::core::float32;
        break;
    case Dtype::F16:
        mlx_dt = ::mlx::core::float16;
        break;
    case Dtype::F64:
        mlx_dt = ::mlx::core::float64;
        break;
    case Dtype::I32:
        mlx_dt = ::mlx::core::int32;
        break;
    case Dtype::I64:
        mlx_dt = ::mlx::core::int64;
        break;
    // MLX's ``complex64`` is a pair of float32 lanes, the same bytes
    // Lucid's C64 and MPSGraph's ``MPSDataTypeComplexFloat32`` hold, so
    // the buffer crosses unchanged.
    case Dtype::C64:
        mlx_dt = ::mlx::core::complex64;
        break;
    // The rest of the table.  These were absent rather than excluded:
    // ``to_mps_dtype`` already builds graphs that produce them, so a
    // compiled model returning a comparison — ``x > 0`` — reached this
    // switch and raised at call time instead of answering.
    case Dtype::Bool:
        mlx_dt = ::mlx::core::bool_;
        break;
    case Dtype::I8:
        mlx_dt = ::mlx::core::int8;
        break;
    case Dtype::I16:
        mlx_dt = ::mlx::core::int16;
        break;
    case Dtype::BF16:
        mlx_dt = ::mlx::core::bfloat16;
        break;
    // C128 stops here on purpose: neither MPSGraph nor MLX has a
    // double-precision complex, so there is nothing to map it onto.
    default:
        throw std::runtime_error(
            "lucid::gpu::mps::buffer_to_array: unsupported Dtype for MPSGraph");
    }

    ::mlx::core::array out(wrapped, std::move(mlx_shape), mlx_dt, deleter);
    // Offset support: if the caller wants a view starting at offset_bytes,
    // we'd need to slice — for the pilot, require offset_bytes == 0.
    if (offset_bytes != 0) {
        throw std::runtime_error(
            "lucid::gpu::mps::buffer_to_array: nonzero offset_bytes not yet supported");
    }
    return out;
}

void wait_all() {
    init_bridge_once();
    if (!g_bridge_ok.load(std::memory_order_acquire))
        return;
    // Issue an empty command buffer + waitUntilCompleted.  This drains
    // the queue without inspecting individual ops.
    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    [cb commit];
    [cb waitUntilCompleted];
}

}  // namespace lucid::gpu::mps
