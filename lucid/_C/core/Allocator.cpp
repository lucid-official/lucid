// lucid/_C/core/Allocator.cpp
//
// Thread-local small-block pool allocator for CPU tensors.
//
// Design rationale:
//   ML workloads create many short-lived intermediate tensors (activations,
//   gradients, temporaries) in a narrow set of sizes.  Going to the OS for
//   every allocation is expensive.  This file implements a per-thread free
//   list (ThreadPool) with 23 size classes, each a power of two from
//   kMinClass (64 B) up to kMaxClass (4 MB).  Each class holds at most
//   kMaxDepth (32) free blocks.
//
//   When a block is freed, the custom deleter tries to push it back onto the
//   appropriate free list.  If the list is full, the block is returned to the
//   OS via std::free.  Because the pool is per-thread, there is no locking.
//
//   Allocations larger than kMaxClass (> 4 MB) bypass the pool and are
//   served directly by posix_memalign; they are freed with std::free.
//
//   GPU-device allocations (device == Device::GPU) also bypass the pool — the
//   pool only handles ordinary virtual-memory pages; GPU-private MLX buffers
//   would be mishandled if placed here.

#include "Allocator.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

#include <sys/sysctl.h>

#include "Error.h"
#include "MemoryStats.h"

namespace lucid {

namespace {

// Size-class boundaries — the smallest pooled allocation is kMinClass bytes
// (equal to kCpuAlignment so every block satisfies the alignment requirement).
constexpr std::size_t kMinClass = kCpuAlignment;    // 64 B — one cache line
constexpr std::size_t kMaxClass = 4 * 1024 * 1024;  // 4 MB — above this, bypass pool
// Maximum number of free blocks retained per size class per thread.  Higher
// values reduce posix_memalign calls at the cost of more virtual-memory
// retention; 32 was chosen to keep total retained memory below ~100 MB
// across all size classes while still absorbing burst allocation patterns.
constexpr std::size_t kMaxDepth = 32;
// Number of power-of-two size classes from kMinClass to kMaxClass inclusive.
// log2(4 MB / 64 B) + 1 = 16 + 1 = 17 classes… but the implementation
// allocates 23 slots to leave headroom for future kMinClass/kMaxClass changes.
constexpr std::size_t kNumClasses = 23;

// Rounds n up to the nearest power-of-two size class >= kMinClass.
// A block of this size can hold n bytes and still satisfies kCpuAlignment
// because kMinClass == kCpuAlignment and all powers of two above it are also
// multiples of kCpuAlignment.
inline std::size_t round_up_pool(std::size_t n) noexcept {
    std::size_t s = kMinClass;
    while (s < n)
        s <<= 1;
    return s;
}

// Maps a rounded size (already a power of two >= kMinClass) to its zero-based
// class index.  Index 0 corresponds to kMinClass, index 22 to kMaxClass.
inline int class_index(std::size_t rounded) noexcept {
    int idx = 0;
    std::size_t s = kMinClass;
    while (s < rounded && idx < static_cast<int>(kNumClasses) - 1) {
        s <<= 1;
        ++idx;
    }
    return idx;
}

// Per-thread free-list pool.  Each of the kNumClasses entries is a vector of
// raw pointers to posix_memalign blocks.  The destructor drains all lists on
// thread exit to avoid leaks.
struct ThreadPool {
    std::array<std::vector<void*>, kNumClasses> lists;

    // Pops the most recently returned block from class cls, or nullptr if the
    // list is empty.
    void* pop(int cls) noexcept {
        auto& v = lists[static_cast<std::size_t>(cls)];
        if (v.empty())
            return nullptr;
        void* p = v.back();
        v.pop_back();
        return p;
    }

    // Pushes block p onto class cls.  Returns false (and does not take
    // ownership) if the list is already at kMaxDepth capacity, in which case
    // the caller must std::free the block.
    bool push(int cls, void* p) noexcept {
        auto& v = lists[static_cast<std::size_t>(cls)];
        if (v.size() >= kMaxDepth)
            return false;
        v.push_back(p);
        return true;
    }

    // Frees all cached blocks when this thread's pool is destroyed (i.e.
    // thread exits).
    ~ThreadPool() {
        for (auto& v : lists)
            for (void* p : v)
                std::free(p);
    }
};

// One ThreadPool instance per thread.  Constructed lazily on first use;
// destroyed (and all cached blocks freed) when the owning thread exits.
thread_local ThreadPool t_pool;

// The machine's physical memory, read once.  No single buffer can be held
// resident past it, and macOS does not say so at allocation: it reserves
// address space for up to 16 TiB, and the process is killed when the pages
// are touched — ``zeros(2**40)`` (4 TiB) died zero-filling, SIGKILL instead
// of an error.  Unknown (the sysctl failed) means no limit.
std::size_t physical_memory_bytes() noexcept {
    static const std::size_t bytes = [] {
        std::uint64_t mem = 0;
        std::size_t len = sizeof(mem);
        if (::sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) != 0 || mem == 0)
            return std::numeric_limits<std::size_t>::max();
        return static_cast<std::size_t>(mem);
    }();
    return bytes;
}

// ``why``, when given, goes in front of the counters the exception reports.
[[noreturn]] void
throw_out_of_memory(std::size_t nbytes, Device device, const std::string& why = {}) {
    const auto s = MemoryTracker::get_stats(device);
    OutOfMemory err(nbytes, s.current_bytes, s.peak_bytes, std::string(device_name(device)));
    if (!why.empty())
        err.prefix_message(why);
    throw err;
}

}  // namespace

std::shared_ptr<std::byte[]> allocate_aligned_bytes(std::size_t nbytes, Device device) {
    if (nbytes == 0)
        return {};
    if (nbytes > physical_memory_bytes()) [[unlikely]]
        throw_out_of_memory(nbytes, device,
                            "one buffer cannot be larger than the machine's " +
                                std::to_string(physical_memory_bytes()) +
                                " bytes of physical memory: ");

    // Only CPU allocations up to kMaxClass go through the thread-local pool.
    // GPU-device allocations bypass the pool because this allocator owns only
    // ordinary virtual-memory pages; MLX GPU buffers are managed by the MLX
    // runtime and must not be recycled through this pool.
    const bool poolable = (device == Device::CPU) && (nbytes <= kMaxClass);
    void* raw = nullptr;

    if (poolable) {
        const std::size_t rounded = round_up_pool(nbytes);
        const int cls = class_index(rounded);
        // Try to reuse a previously freed block from the pool.
        raw = t_pool.pop(cls);
        if (!raw) {
            // Pool miss: fall back to a fresh posix_memalign call.
            if (::posix_memalign(&raw, kCpuAlignment, rounded) != 0 || !raw)
                throw_out_of_memory(nbytes, device);
        }
        MemoryTracker::track_alloc(nbytes, device);

        // Capture rounded_cap by value so the deleter can compute the correct
        // class index without holding a reference to a stack variable that no
        // longer exists when the deleter fires.
        const std::size_t rounded_cap = rounded;
        auto deleter = [nbytes, device, rounded_cap](std::byte* p) {
            const int c = class_index(rounded_cap);
            // Try to return to pool; if the list is at capacity, free directly.
            if (!t_pool.push(c, p))
                std::free(p);
            MemoryTracker::track_free(nbytes, device);
        };
        return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
    }

    // Large allocation (> 4 MB) or non-CPU device: go straight to posix_memalign.
    // The deleter always calls std::free because there is no pool for these.
    if (::posix_memalign(&raw, kCpuAlignment, nbytes) != 0 || !raw)
        throw_out_of_memory(nbytes, device);
    MemoryTracker::track_alloc(nbytes, device);
    auto deleter = [nbytes, device](std::byte* p) {
        std::free(p);
        MemoryTracker::track_free(nbytes, device);
    };
    return std::shared_ptr<std::byte[]>(static_cast<std::byte*>(raw), deleter);
}

}  // namespace lucid
