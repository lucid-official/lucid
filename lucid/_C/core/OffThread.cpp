// lucid/_C/core/OffThread.cpp
//
// See OffThread.h.

#include "OffThread.h"

#include <mutex>
#include <unordered_map>

namespace lucid::off_thread {

namespace {

// Recordings the calling thread leads.
thread_local int t_leads = 0;

struct Registry {
    std::mutex mutex;
    std::unordered_map<const TensorImpl*, std::uint64_t> made;
    std::uint64_t latest = 0;  // the epoch of the last recording to open
    // ``made.size()``, readable without the lock: a leading thread's constructor
    // has nothing to erase when it is zero.
    std::atomic<std::size_t> size{0};
};

Registry& registry() {
    static Registry r;
    return r;
}

}  // namespace

std::uint64_t open() {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    g_open.fetch_add(1, std::memory_order_relaxed);
    ++t_leads;
    return ++r.latest;
}

void close(std::uintptr_t opener) {
    Registry& r = registry();
    std::lock_guard<std::mutex> lock(r.mutex);
    if (opener == thread_token() && t_leads > 0)
        --t_leads;
    if (g_open.fetch_sub(1, std::memory_order_relaxed) == 1) {
        r.made.clear();
        r.size.store(0, std::memory_order_relaxed);
    }
}

std::uintptr_t thread_token() {
    return reinterpret_cast<std::uintptr_t>(&t_leads);
}

void note_made(const TensorImpl* impl) {
    Registry& r = registry();
    if (t_leads > 0) {
        if (r.size.load(std::memory_order_relaxed) == 0)
            return;
        std::lock_guard<std::mutex> lock(r.mutex);
        r.made.erase(impl);
        r.size.store(r.made.size(), std::memory_order_relaxed);
        return;
    }
    std::lock_guard<std::mutex> lock(r.mutex);
    r.made[impl] = r.latest;
    r.size.store(r.made.size(), std::memory_order_relaxed);
}

bool made_since(const TensorImpl* impl, std::uint64_t epoch) {
    Registry& r = registry();
    if (r.size.load(std::memory_order_relaxed) == 0)
        return false;
    std::lock_guard<std::mutex> lock(r.mutex);
    const auto it = r.made.find(impl);
    return it != r.made.end() && it->second >= epoch;
}

}  // namespace lucid::off_thread
