// lucid/_C/compile/ExecutableCache.cpp
//
// Pure-C++ implementation of the executable cache.  Holds raw
// :class:`CompiledExecutable` pointers (the Objective-C++ object is
// hidden behind ``void*`` in :file:`CompiledExecutable.mm`) and routes
// eviction through :func:`destroy_executable`, which is the free
// function defined in the .mm to release the ARC-strong reference.

#include "ExecutableCache.h"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdlib>
#include <utility>

namespace lucid::compile {

namespace {
// Boost-style hash combiner — small enough to inline here and used
// uniformly across every CacheKey field.
inline void hash_combine(std::size_t& seed, std::size_t v) noexcept {
    seed ^= v + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

// Hash a single :type:`AttributeValue`.  The variant is small (i64 /
// vector<i64> / double / bool / string) so a switch over alternatives
// is cheaper than a polymorphic visitor here.
inline std::size_t hash_attribute_value(const AttributeValue& v) noexcept {
    if (const auto* p = std::get_if<std::int64_t>(&v))
        return std::hash<std::int64_t>{}(*p);
    if (const auto* p = std::get_if<std::vector<std::int64_t>>(&v)) {
        std::size_t h = 0xA110C8ULL;
        for (auto d : *p)
            hash_combine(h, std::hash<std::int64_t>{}(d));
        return h;
    }
    if (const auto* p = std::get_if<double>(&v))
        return std::hash<std::uint64_t>{}(std::bit_cast<std::uint64_t>(*p));
    if (const auto* p = std::get_if<bool>(&v))
        return std::hash<bool>{}(*p);
    if (const auto* p = std::get_if<std::string>(&v))
        return std::hash<std::string>{}(*p);
    return 0;
}

// A double attribute is a constant baked into the executable, so two keys
// match only when its bits do.  ``==`` calls 0.0 and -0.0 equal — handing
// ``t + (-0.0)`` the executable compiled for ``t + 0.0`` — and NaN unequal to
// itself, so a NaN attribute never hit (LCD-446).
inline bool same_attribute_value(const AttributeValue& a, const AttributeValue& b) noexcept {
    const auto* da = std::get_if<double>(&a);
    const auto* db = std::get_if<double>(&b);
    if (da != nullptr && db != nullptr)
        return std::bit_cast<std::uint64_t>(*da) == std::bit_cast<std::uint64_t>(*db);
    return a == b;
}

inline bool same_op_attrs(const std::vector<std::pair<std::string, AttributeValue>>& a,
                          const std::vector<std::pair<std::string, AttributeValue>>& b) noexcept {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const auto& x, const auto& y) {
        return x.first == y.first && same_attribute_value(x.second, y.second);
    });
}

inline std::int64_t intern_canonical_id(TraceIdMap& map, TensorId id) {
    if (id.v < 0)
        return id.v;
    auto [it, inserted] =
        map.to_canonical.emplace(id.v, static_cast<std::int64_t>(map.to_raw.size()));
    if (inserted)
        map.to_raw.push_back(id);
    return it->second;
}

inline std::vector<std::pair<std::string, AttributeValue>> sorted_node_attrs(const OpNode& node) {
    std::vector<std::pair<std::string, AttributeValue>> sorted(node.attrs.begin(),
                                                               node.attrs.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    return sorted;
}

inline void append_node_to_key(CacheKey& key,
                               TraceIdMap& local_map,
                               const OpNode& node,
                               Device* out_device = nullptr) {
    key.op_names.push_back(node.name);
    std::vector<std::int64_t> inputs;
    inputs.reserve(node.inputs.size());
    for (const auto& id : node.inputs)
        inputs.push_back(intern_canonical_id(local_map, id));
    key.op_inputs.push_back(std::move(inputs));
    for (const auto& out : node.outputs)
        intern_canonical_id(local_map, out.id);
    if (!node.outputs.empty()) {
        const auto& meta = node.outputs[0];
        key.output_shapes.push_back(meta.shape);
        key.output_dtypes.push_back(meta.dtype);
        if (out_device != nullptr)
            *out_device = meta.device;
    } else {
        key.output_shapes.emplace_back();
        key.output_dtypes.push_back(Dtype::F32);
    }
    key.op_attrs.push_back(sorted_node_attrs(node));
}

inline void append_vjp_region_to_key(CacheKey& key, TraceIdMap& local_map, const VjpRegion& r) {
    key.op_names.emplace_back("__vjp_region__");
    std::vector<std::int64_t> header_ids;
    header_ids.reserve(r.input_ids.size() + r.output_metas.size() + r.cotangent_ids.size() +
                       r.input_grad_ids.size());
    for (TensorId id : r.input_ids)
        header_ids.push_back(intern_canonical_id(local_map, id));
    for (const auto& om : r.output_metas)
        header_ids.push_back(intern_canonical_id(local_map, om.id));
    for (TensorId id : r.cotangent_ids)
        header_ids.push_back(intern_canonical_id(local_map, id));
    for (TensorId id : r.input_grad_ids)
        header_ids.push_back(intern_canonical_id(local_map, id));
    key.op_inputs.push_back(std::move(header_ids));
    key.output_shapes.push_back(Shape{static_cast<std::int64_t>(r.fwd_begin),
                                      static_cast<std::int64_t>(r.fwd_end),
                                      static_cast<std::int64_t>(r.input_ids.size()),
                                      static_cast<std::int64_t>(r.output_metas.size())});
    key.output_dtypes.push_back(Dtype::I64);
    key.op_attrs.emplace_back();
    for (const auto& node : r.bwd_ops)
        append_node_to_key(key, local_map, node);
}

inline std::vector<FeedSig>
build_feed_sigs(const std::unordered_map<TensorId, std::pair<Shape, Dtype>>& feed_meta,
                TraceIdMap& map) {
    // Any external feed not referenced by ``graph.ops`` or ``explicit_outputs``
    // (for example, an unused RNG draw that still occupies a placeholder slot
    // in ``MpsBuilder::compile_trace``) receives a deterministic canonical id
    // in ascending raw-id order so the key distinguishes traces with extra
    // feeds and ``map`` covers every slot of ``exe->input_ids``.
    std::vector<TensorId> unreferenced;
    for (const auto& [tid, _] : feed_meta) {
        if (map.to_canonical.find(tid.v) == map.to_canonical.end())
            unreferenced.push_back(tid);
    }
    std::sort(unreferenced.begin(), unreferenced.end());
    for (TensorId tid : unreferenced)
        intern_canonical_id(map, tid);

    std::vector<FeedSig> feeds;
    feeds.reserve(feed_meta.size());
    for (const auto& [tid, meta] : feed_meta) {
        const auto it = map.to_canonical.find(tid.v);
        if (it != map.to_canonical.end())
            feeds.push_back(FeedSig{it->second, meta.first, meta.second});
    }
    std::sort(feeds.begin(), feeds.end(),
              [](const FeedSig& a, const FeedSig& b) { return a.canonical_id < b.canonical_id; });
    return feeds;
}

inline bool remap_to_canonical(std::vector<TensorId>& ids, const TraceIdMap& id_map) {
    for (auto& tid : ids) {
        const auto it = id_map.to_canonical.find(tid.v);
        if (it == id_map.to_canonical.end())
            return false;
        tid = TensorId{it->second};
    }
    return true;
}

inline bool remap_to_raw(std::vector<TensorId>& ids, const TraceIdMap& id_map) {
    for (auto& tid : ids) {
        if (tid.v < 0 || static_cast<std::size_t>(tid.v) >= id_map.to_raw.size())
            return false;
        tid = id_map.to_raw[static_cast<std::size_t>(tid.v)];
    }
    return true;
}
}  // namespace

bool CacheKey::operator==(const CacheKey& other) const noexcept {
    return device == other.device && op_names == other.op_names && op_inputs == other.op_inputs &&
           output_shapes == other.output_shapes && output_dtypes == other.output_dtypes &&
           std::equal(op_attrs.begin(), op_attrs.end(), other.op_attrs.begin(),
                      other.op_attrs.end(), same_op_attrs) &&
           feeds == other.feeds && outputs == other.outputs;
}

std::size_t CacheKeyHash::operator()(const CacheKey& key) const noexcept {
    std::size_t h = std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(key.device));
    for (const auto& name : key.op_names)
        hash_combine(h, std::hash<std::string>{}(name));
    // Wiring — a sentinel after each op keeps {0,1},{2} apart from {0},{1,2}.
    for (const auto& inputs : key.op_inputs) {
        for (auto i : inputs)
            hash_combine(h, std::hash<std::int64_t>{}(i));
        hash_combine(h, 0x5EEDF00DULL);
    }
    for (const auto& shape : key.output_shapes) {
        for (auto d : shape)
            hash_combine(h, std::hash<std::int64_t>{}(d));
        // Trailing sentinel so [a, b] and [a*b] hash differently.
        hash_combine(h, 0xCAFEBABEULL);
    }
    for (auto dt : key.output_dtypes)
        hash_combine(h, std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(dt)));
    // Attributes — sorted-key serialisation keeps the hash
    // deterministic across runs.  Sentinel between ops prevents
    // [{a:1}, {b:2}] colliding with [{a:1,b:2}, {}].
    for (const auto& per_op : key.op_attrs) {
        for (const auto& [k, v] : per_op) {
            hash_combine(h, std::hash<std::string>{}(k));
            hash_combine(h, hash_attribute_value(v));
        }
        hash_combine(h, 0xDEADBEEFULL);
    }
    for (const auto& f : key.feeds) {
        hash_combine(h, std::hash<std::int64_t>{}(f.canonical_id));
        for (auto d : f.shape)
            hash_combine(h, std::hash<std::int64_t>{}(d));
        hash_combine(h, std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(f.dtype)));
        hash_combine(h, 0xFEEDFACEULL);
    }
    for (auto o : key.outputs)
        hash_combine(h, std::hash<std::int64_t>{}(o));
    hash_combine(h, 0x0B7B075ULL);
    return h;
}

CacheKey make_cache_key(const TraceGraph& graph,
                        const std::unordered_map<TensorId, std::pair<Shape, Dtype>>& feed_meta,
                        const std::vector<TensorId>& explicit_outputs,
                        TraceIdMap* id_map) {
    CacheKey key;
    key.op_names.reserve(graph.ops.size());
    key.op_inputs.reserve(graph.ops.size());
    key.output_shapes.reserve(graph.ops.size());
    key.output_dtypes.reserve(graph.ops.size());
    key.op_attrs.reserve(graph.ops.size());

    TraceIdMap local_map;
    Device device = Device::CPU;
    for (const auto& node : graph.ops)
        append_node_to_key(key, local_map, node, &device);
    for (const auto& r : graph.vjp_regions)
        append_vjp_region_to_key(key, local_map, r);
    key.device = device;

    // Requested outputs in the same numbering; an id the walk never saw
    // (a feed returned as-is) still gets a stable number before ``feed_meta``
    // is canonicalized below.
    key.outputs.reserve(explicit_outputs.size());
    for (const auto& id : explicit_outputs)
        key.outputs.push_back(intern_canonical_id(local_map, id));

    key.feeds = build_feed_sigs(feed_meta, local_map);
    if (id_map != nullptr)
        *id_map = std::move(local_map);
    return key;
}

SharedExecutable share_executable(CompiledExecutable* exec) {
    if (exec == nullptr)
        return {};
    return SharedExecutable(exec, [](CompiledExecutable* e) { destroy_executable(e); });
}

SharedExecutable canonicalize_executable(const CompiledExecutable* exe, const TraceIdMap& id_map) {
    if (exe == nullptr)
        return {};
    std::vector<TensorId> canon_in = executable_input_ids(exe);
    std::vector<TensorId> canon_out = executable_output_ids(exe);
    if (!remap_to_canonical(canon_in, id_map) || !remap_to_canonical(canon_out, id_map))
        return {};
    return share_executable(relabel_executable(exe, std::move(canon_in), std::move(canon_out)));
}

SharedExecutable localize_executable(const CompiledExecutable* exe, const TraceIdMap& id_map) {
    if (exe == nullptr)
        return {};
    std::vector<TensorId> raw_in = executable_input_ids(exe);
    std::vector<TensorId> raw_out = executable_output_ids(exe);
    if (!remap_to_raw(raw_in, id_map) || !remap_to_raw(raw_out, id_map))
        return {};
    return share_executable(relabel_executable(exe, std::move(raw_in), std::move(raw_out)));
}

ExecutableCache::~ExecutableCache() {
    clear();
}

SharedExecutable ExecutableCache::find(const CacheKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = map_.find(key);
    if (it == map_.end())
        return {};

    // Bump the matching entry to the back of the LRU vector (most
    // recent).  Linear scan is acceptable because ``max_entries_`` is
    // O(10–100).
    auto vit = std::find(lru_order_.begin(), lru_order_.end(), key);
    if (vit != lru_order_.end()) {
        CacheKey saved = std::move(*vit);
        lru_order_.erase(vit);
        lru_order_.push_back(std::move(saved));
    }
    return it->second;
}

SharedExecutable ExecutableCache::insert(CacheKey key, SharedExecutable exec) {
    std::lock_guard<std::mutex> lock(mu_);

    // Replace any existing entry under the same key; a caller still
    // holding the previous executable keeps it alive.
    map_[key] = exec;

    // Bring the key to the front of "most-recent".
    auto vit = std::find(lru_order_.begin(), lru_order_.end(), key);
    if (vit != lru_order_.end())
        lru_order_.erase(vit);
    lru_order_.push_back(std::move(key));

    // Evict from the front until back at capacity.
    while (lru_order_.size() > max_entries_) {
        const CacheKey victim = std::move(lru_order_.front());
        lru_order_.erase(lru_order_.begin());
        map_.erase(victim);
    }
    return exec;
}

void ExecutableCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    map_.clear();
    lru_order_.clear();
}

std::size_t ExecutableCache::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return map_.size();
}

void ExecutableCache::set_max_entries(std::size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    if (n == 0)
        n = 1;
    max_entries_ = n;
}

ExecutableCache& ExecutableCache::session() {
    // Heap-allocated singleton — process-lifetime cache, no shutdown
    // ordering issues with the rest of the engine.
    static ExecutableCache* instance = [] {
        auto* c = new ExecutableCache();
        if (const char* s = std::getenv("LUCID_COMPILE_MAX_CACHE")) {
            char* end = nullptr;
            unsigned long v = std::strtoul(s, &end, 10);
            if (end != s && v > 0)
                c->set_max_entries(static_cast<std::size_t>(v));
        }
        return c;
    }();
    return *instance;
}

}  // namespace lucid::compile
