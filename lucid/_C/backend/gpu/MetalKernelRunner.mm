// lucid/_C/backend/gpu/MetalKernelRunner.mm
//
// Implements compile_metal_kernel and run_metal_kernel declared in
// MetalKernelRunner.h.
//
// Pipeline cache:
//   g_pipeline_cache maps a (source, function_name) hash to a PipelineEntry.
//   g_cache_mutex serialises cache reads and writes.  Cache entries are never
//   evicted; the cache is process-lifetime.  cache_key uses djb2-mix to
//   combine the two string hashes.
//
// resolve_storage_to_mtl:
//   Converts any Storage variant to an id<MTLBuffer> for use as a kernel
//   argument.  SharedStorage provides a pre-existing MTLBuffer at zero cost.
//   GpuStorage must be eval()'d first (to materialise the lazy graph), then
//   a *copy* into a shared buffer is made via newBufferWithBytes — not
//   newBufferWithBytesNoCopy — because MLX-managed Metal allocations are not
//   guaranteed to be page-aligned, which is a requirement of the no-copy API.
//   CpuStorage takes the same copy path.  is_temp==true buffers are created
//   inline and must not be released by the caller.
//
// run_metal_kernel buffer indexing convention (matches MSL [[buffer(i)]]):
//   [0 .. n-1]     input buffers (one per element of inputs[])
//   [n .. n+k-1]   scalar constants (passed via setBytes:length:atIndex:)
//   [n+k]          output buffer

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/MemoryStats.h"
#include "../../core/MemoryStats.h"
#include "../../core/PhantomMode.h"
#include "MlxBridge.h"
#include "MetalAllocator.h"
#include "MetalKernelRunner.h"

namespace lucid::gpu {

namespace {

// Returns the process-wide default Metal device, created once on first call.
id<MTLDevice> runner_device() {
    static id<MTLDevice> dev = nil;
    static dispatch_once_t once;
    dispatch_once(&once, ^{
        dev = MTLCreateSystemDefaultDevice();
    });
    return dev;
}

// Cached pipeline state and command queue for a compiled kernel.
struct PipelineEntry {
    id<MTLComputePipelineState> pso  = nil;
    id<MTLCommandQueue>         cq   = nil;
};

// Global pipeline cache and its mutex.  The cache is never evicted; entries
// accumulate for the lifetime of the process.
std::mutex g_cache_mutex;
std::unordered_map<std::size_t, PipelineEntry> g_pipeline_cache;

// Hashes (source, fn_name) into a single std::size_t using the boost-style
// hash combine formula to reduce collision probability.
std::size_t cache_key(const std::string& source, const std::string& fn_name) {
    std::size_t h = std::hash<std::string>{}(source);
    // XOR with a golden-ratio-based mix to combine the two hashes.
    h ^= std::hash<std::string>{}(fn_name) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

}

void MetalKernel::release_() noexcept {
    if (pipeline_state) {
        CFRelease(pipeline_state);
        pipeline_state = nullptr;
    }
    if (command_queue) {
        CFRelease(command_queue);
        command_queue = nullptr;
    }
}

MetalKernel compile_metal_kernel(const std::string& source,
                                 const std::string& function_name) {
    const std::size_t key = cache_key(source, function_name);

    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        auto it = g_pipeline_cache.find(key);
        if (it != g_pipeline_cache.end()) {
            MetalKernel k;
            k.name = function_name;

            k.pipeline_state = (__bridge_retained void*)it->second.pso;
            k.command_queue  = (__bridge_retained void*)it->second.cq;
            return k;
        }
    }

    id<MTLDevice> dev = runner_device();
    if (!dev)
        return {};

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:source.c_str()];
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:nil error:&err];
    if (!lib) {
        ErrorBuilder("compile_metal_kernel")
            .fail(std::string("MTL library compilation failed: ") +
                  (err ? [err.localizedDescription UTF8String] : "unknown"));
    }

    NSString* fn_ns = [NSString stringWithUTF8String:function_name.c_str()];
    id<MTLFunction> fn = [lib newFunctionWithName:fn_ns];
    if (!fn) {
        ErrorBuilder("compile_metal_kernel")
            .fail("function '" + function_name + "' not found in shader source");
    }

    id<MTLComputePipelineState> pso =
        [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!pso) {
        ErrorBuilder("compile_metal_kernel")
            .fail(std::string("pipeline state creation failed: ") +
                  (err ? [err.localizedDescription UTF8String] : "unknown"));
    }

    id<MTLCommandQueue> cq = [dev newCommandQueue];
    if (!cq) {
        ErrorBuilder("compile_metal_kernel").fail("failed to create command queue");
    }

    {
        std::lock_guard<std::mutex> lk(g_cache_mutex);
        PipelineEntry entry;
        entry.pso = pso;
        entry.cq  = cq;
        g_pipeline_cache.emplace(key, entry);
    }

    MetalKernel k;
    k.name = function_name;
    k.pipeline_state = (__bridge_retained void*)pso;
    k.command_queue  = (__bridge_retained void*)cq;
    return k;
}

namespace {

// A resolved Metal buffer with a flag indicating whether it was allocated
// temporarily (for GpuStorage and CpuStorage copies) and must be released
// after the command buffer completes.  SharedStorage buffers are not temporary.
struct BoundBuffer {
    id<MTLBuffer> buf     = nil;
    bool          is_temp = false;
};

// Resolves any Storage variant to a Metal buffer suitable for passing to a
// compute kernel.  For SharedStorage the existing MTLBuffer is reused;
// for GpuStorage and CpuStorage a temporary shared buffer is created by copy.
BoundBuffer resolve_storage_to_mtl(const Storage& s, id<MTLDevice> dev) {
    // A phantom input has no bytes to bind.  Under phantom mode the kernel
    // never gets here (the phantom backend answers); a phantom used after its
    // block does.
    if (storage_is_phantom(s))
        refuse_phantom_read("run_metal_kernel");
    if (storage_is_metal_shared(s)) {
        const auto& sh = storage_metal_shared(s);

        return {(__bridge id<MTLBuffer>)sh.mtl_handle, false};
    }
    if (storage_is_gpu(s)) {
        const auto& gs = storage_gpu(s);
        if (!gs.arr) {
            ErrorBuilder("run_metal_kernel").fail("null GpuStorage array");
        }

        gs.arr->eval();
        MemoryTracker::track_host_sync(gs.nbytes);
        const void* ptr = gs.arr->data<std::uint8_t>();
        if (!ptr) {
            ErrorBuilder("run_metal_kernel").fail("MLX array data pointer is null after eval");
        }
        id<MTLBuffer> tmp = [dev newBufferWithBytes:ptr
                                            length:gs.nbytes
                                           options:MTLResourceStorageModeShared];
        if (!tmp) {
            ErrorBuilder("run_metal_kernel").fail("failed to allocate Metal buffer for GpuStorage");
        }
        return {tmp, true};
    }

    const auto& cs = storage_cpu(s);
    id<MTLBuffer> tmp = [dev newBufferWithBytes:cs.ptr.get()
                                        length:cs.nbytes
                                       options:MTLResourceStorageModeShared];
    return {tmp, true};
}

}

Storage run_metal_kernel(const MetalKernel&              kernel,
                         const std::vector<Storage>&     inputs,
                         const Shape&                    output_shape,
                         Dtype                           output_dtype,
                         const KernelLaunchConfig&       config,
                         const std::vector<KernelConstant>& constants) {
    if (!kernel.is_valid())
        ErrorBuilder("run_metal_kernel").fail("kernel is invalid (compilation failed?)");

    id<MTLDevice> dev = runner_device();
    auto pso = (__bridge id<MTLComputePipelineState>)kernel.pipeline_state;
    auto cq  = (__bridge id<MTLCommandQueue>)kernel.command_queue;

    const std::size_t out_numel = shape_numel(output_shape);
    const std::size_t out_bytes = out_numel * dtype_size(output_dtype);
    OwnedMetalBuffer out_owned  = make_metal_shared(out_bytes);
    if (!out_owned.buf.cpu_ptr) {
        ErrorBuilder("run_metal_kernel").fail("failed to allocate output Metal buffer");
    }
    id<MTLBuffer> out_buf =
        (__bridge id<MTLBuffer>)out_owned.buf.mtl_handle;

    std::vector<BoundBuffer> bound;
    bound.reserve(inputs.size());
    for (const auto& s : inputs)
        bound.push_back(resolve_storage_to_mtl(s, dev));

    id<MTLCommandBuffer>        cmd  = [cq commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

    [enc setComputePipelineState:pso];

    for (std::size_t i = 0; i < bound.size(); ++i)
        [enc setBuffer:bound[i].buf offset:0 atIndex:static_cast<NSUInteger>(i)];

    const std::size_t const_base = bound.size();
    for (std::size_t ci = 0; ci < constants.size(); ++ci) {
        std::visit(
            [&](auto v) {
                auto typed = v;
                [enc setBytes:&typed
                       length:sizeof(typed)
                      atIndex:static_cast<NSUInteger>(const_base + ci)];
            },
            constants[ci]);
    }

    const NSUInteger out_idx =
        static_cast<NSUInteger>(bound.size() + constants.size());
    [enc setBuffer:out_buf offset:0 atIndex:out_idx];

    MTLSize g  = MTLSizeMake(config.grid[0], config.grid[1], config.grid[2]);
    MTLSize tg = MTLSizeMake(config.threads[0], config.threads[1], config.threads[2]);
    [enc dispatchThreadgroups:g threadsPerThreadgroup:tg];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    SharedStorage result;
    result.cpu_ptr    = out_owned.buf.cpu_ptr;
    result.mtl_handle = out_owned.buf.mtl_handle;
    result.nbytes     = out_bytes;
    result.dtype      = output_dtype;
    result.owner      = std::move(out_owned.owner);

    return Storage{std::move(result)};
}

}
