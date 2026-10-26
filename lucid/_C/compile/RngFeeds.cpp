// lucid/_C/compile/RngFeeds.cpp
//
// See RngFeeds.h.  The registry is keyed by the tensor's address and
// holds a weak reference: a draw whose tensor has died cannot match a
// new tensor that happens to reuse the address, because the weak
// reference no longer locks to it.

#include "RngFeeds.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <utility>

#include "../autograd/Helpers.h"
#include "../core/Generator.h"
#include "../core/TensorImpl.h"
#include "../random/Random.h"
#include "Tracer.h"

namespace lucid::compile {

namespace {

struct Entry {
    std::weak_ptr<TensorImpl> ref;
    RngRecipe recipe;
    std::uint64_t seq = 0;
};

struct Registry {
    std::mutex mu;
    std::unordered_map<const TensorImpl*, Entry> entries;
    std::uint64_t next_seq = 0;
};

Registry& registry() {
    static Registry r;
    return r;
}

TensorImplPtr draw(const RngRecipe& r) {
    Generator* gen = &default_generator();
    switch (r.kind) {
    case RngRecipe::Kind::DropoutMask: {
        Storage mask = bernoulli_mask_storage_shape(r.a, r.shape, r.dtype, r.device, *gen);
        Storage scaled = mul_scalar_storage(mask, r.b, shape_numel(r.shape), r.dtype, r.device);
        return std::make_shared<TensorImpl>(std::move(scaled), r.shape, r.dtype, r.device, false);
    }
    case RngRecipe::Kind::Uniform:
        return uniform_op(r.shape, r.a, r.b, r.dtype, r.device, gen);
    case RngRecipe::Kind::Normal:
        return normal_op(r.shape, r.a, r.b, r.dtype, r.device, gen);
    case RngRecipe::Kind::Randint:
        return randint_op(r.shape, r.lo, r.hi, r.dtype, r.device, gen);
    case RngRecipe::Kind::Bernoulli:
        return bernoulli_op(r.shape, r.a, r.dtype, r.device, gen);
    }
    return nullptr;
}

// Detaches the calling thread's tracer for the duration of the draws, so
// a compiled call made while another trace records does not add them to
// it.
class TracerDetach {
public:
    TracerDetach() : saved_(current_tracer()) { set_current_tracer(nullptr); }
    ~TracerDetach() { set_current_tracer(saved_); }
    TracerDetach(const TracerDetach&) = delete;
    TracerDetach& operator=(const TracerDetach&) = delete;

private:
    Tracer* saved_;
};

}  // namespace

void note_rng_feed(const TensorImplPtr& out, RngRecipe recipe) {
    if (!out)
        return;
    Registry& reg = registry();
    std::lock_guard<std::mutex> lock(reg.mu);
    // Drop the draws whose tensors are gone before the map grows further.
    if (reg.entries.size() >= 256) {
        for (auto it = reg.entries.begin(); it != reg.entries.end();) {
            if (it->second.ref.expired())
                it = reg.entries.erase(it);
            else
                ++it;
        }
    }
    reg.entries.insert_or_assign(out.get(), Entry{out, std::move(recipe), reg.next_seq++});
}

std::vector<TensorImplPtr> redraw_rng_feeds(const std::vector<TensorImplPtr>& feeds) {
    std::vector<std::pair<std::uint64_t, std::size_t>> order;
    std::vector<RngRecipe> recipes(feeds.size());
    {
        Registry& reg = registry();
        std::lock_guard<std::mutex> lock(reg.mu);
        if (reg.entries.empty())
            return feeds;
        for (std::size_t i = 0; i < feeds.size(); ++i) {
            const auto it = reg.entries.find(feeds[i].get());
            if (it == reg.entries.end())
                continue;
            if (it->second.ref.lock() != feeds[i]) {
                reg.entries.erase(it);
                continue;
            }
            order.emplace_back(it->second.seq, i);
            recipes[i] = it->second.recipe;
        }
    }
    if (order.empty())
        return feeds;
    std::sort(order.begin(), order.end());
    std::vector<TensorImplPtr> out = feeds;
    TracerDetach detach;
    for (const auto& [seq, i] : order)
        out[i] = draw(recipes[i]);
    return out;
}

}  // namespace lucid::compile
