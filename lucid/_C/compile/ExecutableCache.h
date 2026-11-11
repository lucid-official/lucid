// lucid/_C/compile/ExecutableCache.h
//
// Process-global cache of compiled MPSGraph executables, keyed by the
// structural signature of the trace that produced them.  Two traces
// that emit the same op sequence with the same per-op output shapes
// and dtypes on the same device share one executable.
//
// The cache is the owner of each :class:`CompiledExecutable` inserted
// into it: eviction (LRU when ``max_entries`` is exceeded) and
// :func:`clear` both release the entries via
// :func:`destroy_executable`.
//
// Notes
// -----
// Phase 1.2 ships an LRU implementation with default ``max_entries =
// 32``, override-able by setting ``LUCID_COMPILE_MAX_CACHE=N`` in the
// environment before the first :func:`session` call.
//
// The cache is thread-safe (one mutex protects the underlying map +
// LRU vector); calls from the autograd Engine or DataLoader workers
// share one cache.

#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "CompiledExecutable.h"  // forward + destroy_executable
#include "TraceIR.h"

namespace lucid::compile {

// Cache key — the structural signature of a :class:`TraceGraph`.
//
// Two traces share an executable iff every field below matches.  The
// input *values* are not part of the key; only the per-op output meta
// influences the compiled graph (because MPSGraph placeholders are
// shape-static in Phase 1.2 and Phase 1.6 will introduce a separate
// ``dynamic`` key variant).
//
// Attributes
// ----------
// op_names : std::vector<std::string>
//     Op-name sequence in dispatch order.  Compared element-wise.
// op_inputs : std::vector<std::vector<std::int64_t>>
//     Per-op input wiring: every tensor id renumbered by first appearance
//     across the walk, so the numbers depend only on how the graph is
//     connected and not on the ids a particular trace drew.  ``a * b``
//     fed one tensor twice reads ``{0, 0}``, fed two tensors ``{0, 1}``.
//     Without this both hashed alike and one caller ran the other's
//     executable — one feed bound where two were passed.
// output_shapes : std::vector<Shape>
//     Per-op single-output shape.  Phase 1.2 supports one output per
//     node; Phase 1.3+ may append entries for multi-output ops.
// output_dtypes : std::vector<Dtype>
//     Per-op single-output dtype.
// op_attrs : std::vector<std::vector<std::pair<std::string, AttributeValue>>>
//     Per-op attribute map, serialised in sorted-key order so the
//     hash is deterministic.  Required for emitter-divergent
//     attributes that don't show up in shapes (e.g. ``dropout``'s
//     ``training`` / ``p``, ``softmax``'s ``axis``).  Without this
//     two traces with identical structure but different attributes
//     would collide on the same cache entry and one caller would
//     silently receive the other's executable.
// feeds : std::vector<FeedSig>
//     Shape + dtype of every external feed (user input or parameter),
//     ordered by the same first-appearance numbering as ``op_inputs``.
//     The executable's placeholders are typed and shaped by its feeds and
//     ``run_executable`` refuses a mismatch, so two traces whose ops agree
//     but whose inputs do not must not share an entry.  Without this,
//     ``isnan(x) + 1`` traced on float32 and then on int64 — every op
//     output the same shape and dtype — hit the float executable and
//     raised "feed slot 0 expects dtype float32, got int64".
// outputs : std::vector<std::int64_t>
//     The tensors the caller asked for, in order and in the same numbering.
//     The graph alone does not say which of its values are returned: two
//     callables tracing ``(t + 1, t * 2)`` and ``(t * 2, t + 1)`` build the
//     same ops, and without this the second ran the first's executable and
//     got its outputs in the first's order — as ``topk(t)[0]`` got the
//     indices of an earlier ``topk(t)[1]``.
// device : Device
//     Device on which the executable was compiled (all ops in a trace
//     must agree on one device — Phase 1.2 builder enforces this).
// policy : std::vector<std::int64_t>
//     The process settings that change what a compile emits for the same
//     trace (see :func:`compile_policy`).  They are read when the key is
//     made, so an executable built under one setting never answers a
//     lookup made under another: toggling ``LUCID_COMPILE_RANK_CANON_MIN_LEAD``
//     mid-process used to hand back the executable built under the old
//     value, and the pass looked as if it had not run (LCD-922).  The disk
//     cache names its files by this key's hash, so it follows.
struct LUCID_API FeedSig {
    std::int64_t canonical_id = 0;
    Shape shape;
    Dtype dtype = Dtype::F32;
    bool operator==(const FeedSig& other) const noexcept {
        return canonical_id == other.canonical_id && shape == other.shape && dtype == other.dtype;
    }
};

struct LUCID_API CacheKey {
    std::vector<std::string> op_names;
    std::vector<std::vector<std::int64_t>> op_inputs;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;
    std::vector<std::vector<std::pair<std::string, AttributeValue>>> op_attrs;
    std::vector<FeedSig> feeds;
    std::vector<std::int64_t> outputs;
    Device device = Device::CPU;
    std::vector<std::int64_t> policy;

    // Structural equality — every field above must match element-wise.
    // Used as the unordered_map bucket-equality predicate.
    bool operator==(const CacheKey& other) const noexcept;
};

// std-style hasher for :class:`CacheKey`; used as the bucket function
// for :class:`ExecutableCache`'s internal map.
struct LUCID_API CacheKeyHash {
    // Hash combine over op_names + output_shapes + output_dtypes +
    // device.  Two structurally-equal keys produce the same hash.
    std::size_t operator()(const CacheKey& key) const noexcept;
};

// The environment settings, read now, that change what a compile emits: every
// one that is resolved at each compile rather than once per process.
//   [0] ``rank_canon_min_lead()``    (``LUCID_COMPILE_RANK_CANON_MIN_LEAD``)
//   [1] ``use_manual_vjp()``         (``LUCID_MANUAL_VJP``)
//   [2] ``use_manual_vjp_require()`` (``LUCID_MANUAL_VJP_REQUIRE``)
// Logging switches (``LUCID_COMPILE_VERBOSE``, ``LUCID_MANUAL_VJP_DEBUG``) are not
// here, nor are the settings latched at first read (``LUCID_COMPILE_ASYNC``,
// ``LUCID_COMPILE_ALIAS_INPLACE``, ``LUCID_COMPILE_FUSED_SDPA``): those cannot
// change inside a process, or change how an executable runs, not what is in it.
// Python's per-module cache keys on this too (``signature_of``).  The attention
// workaround state is not part of it: the probe flips it once, after the first
// Python key of a process has been made.  ``make_cache_key`` appends it.
LUCID_API std::vector<std::int64_t> compile_policy();

// Bidirectional mapping between a trace's raw :type:`TensorId` values and
// the canonical first-appearance numbering minted by :func:`make_cache_key`.
//
// The session :class:`ExecutableCache` stores each cached executable with
// its ``input_ids`` and ``output_ids`` translated into canonical indices
// ``0 .. to_raw.size() - 1`` via :func:`canonicalize_executable`, and each
// caller of ``compile_or_cached`` receives a copy relabelled into that
// caller's own raw :type:`TensorId` space via :func:`localize_executable`.
// Without this translation, a session-cache hit returned an executable
// whose ``input_ids`` / ``output_ids`` carried the *first* trace's raw ids
// — which either failed feed lookup in the second trace (silently dropping
// the second callable to eager) or bound feeds in the wrong order when an
// earlier RNG draw or folded constant shifted raw ids (LCD-454).
struct LUCID_API TraceIdMap {
    std::unordered_map<std::int64_t, std::int64_t> to_canonical;
    std::vector<TensorId> to_raw;
};

// Build a :class:`CacheKey` from a :class:`TraceGraph`.
//
// Walks ``graph.ops`` once, copying the name + first-output meta from
// each node.  Device is taken from the most recent non-empty output;
// it is the caller's responsibility to ensure the trace is
// single-device (the MpsBuilder rejects mixed-device traces before
// hashing).
//
// ``feed_meta`` maps each external-feed id to its shape and dtype; it
// fills :attr:`CacheKey::feeds`.
// ``explicit_outputs`` are the trace ids the caller will read back, in
// order; they fill :attr:`CacheKey::outputs`.
// When ``id_map`` is non-null, it is populated with the raw-to-canonical
// and canonical-to-raw :type:`TensorId` bijection for this trace.
LUCID_API CacheKey
make_cache_key(const TraceGraph& graph,
               const std::unordered_map<TensorId, std::pair<Shape, Dtype>>& feed_meta = {},
               const std::vector<TensorId>& explicit_outputs = {},
               TraceIdMap* id_map = nullptr);

// Shared ownership of a compiled executable.  The last holder — the cache,
// or a Python wrapper that took it out of the cache — releases it through
// :func:`destroy_executable`.
using SharedExecutable = std::shared_ptr<CompiledExecutable>;

// Take ownership of ``exec`` (``nullptr`` gives an empty pointer).
LUCID_API SharedExecutable share_executable(CompiledExecutable* exec);

// Return a copy of ``exe`` whose ``input_ids`` and ``output_ids`` have been
// translated from the trace's raw :type:`TensorId` space into canonical
// indices via ``id_map.to_canonical``.  Returns an empty pointer if ``exe``
// is null or any id is absent from ``id_map``.
LUCID_API SharedExecutable canonicalize_executable(const CompiledExecutable* exe,
                                                   const TraceIdMap& id_map);

// Return a copy of a canonicalized ``exe`` whose ``input_ids`` and
// ``output_ids`` have been translated from canonical indices back into the
// caller's raw :type:`TensorId` space via ``id_map.to_raw``.  Returns an
// empty pointer if ``exe`` is null or any canonical index is out of range.
LUCID_API SharedExecutable localize_executable(const CompiledExecutable* exe,
                                               const TraceIdMap& id_map);

// Bounded-LRU cache of compiled MPSGraph executables.
//
// Thread-safe.  Entries are shared: eviction drops the cache's reference
// only.  It used to hand out borrowed raw pointers and delete the
// executable on eviction, so a module holding one — every CompiledModule
// keeps its executables — called freed memory once more than
// ``max_entries`` signatures had been compiled in the process.
//
// An executable whose first run was given up on (:file:`FirstRunWatchdog.h`) is never
// the one evicted, so the refusal is remembered for every later hit; the watchdog's
// circuit breaker keeps their number small (its limit plus the first runs in flight
// when it opened).
class LUCID_API ExecutableCache {
public:
    // Construct an empty cache with the default LRU cap (32 entries).
    // Use :func:`set_max_entries` to override before inserting.
    ExecutableCache() = default;

    // Destructor — releases every cached :class:`CompiledExecutable`
    // via :func:`destroy_executable` before the map is torn down.
    ~ExecutableCache();

    ExecutableCache(const ExecutableCache&) = delete;
    ExecutableCache& operator=(const ExecutableCache&) = delete;
    ExecutableCache(ExecutableCache&&) = delete;
    ExecutableCache& operator=(ExecutableCache&&) = delete;

    // Look up an executable.
    //
    // Returns a shared reference on hit (and bumps the entry to
    // most-recent in the LRU order) or an empty pointer on miss.
    SharedExecutable find(const CacheKey& key);

    // Insert ``exec`` under ``key`` and return it.  Replaces any existing
    // entry (dropping the cache's reference to the previous executable).
    // If the resulting size exceeds ``max_entries``, drops the
    // least-recently used entries until back at capacity.
    SharedExecutable insert(CacheKey key, SharedExecutable exec);

    // Drop the cache's reference to every stored executable.
    void clear();

    // Number of currently-cached entries.
    std::size_t size() const;

    // Override the LRU cap.  Honoured by subsequent inserts; existing
    // entries are not evicted retroactively.  Default is 32.
    void set_max_entries(std::size_t n);

    // Process-global session cache.  First call honours the
    // ``LUCID_COMPILE_MAX_CACHE`` environment variable (positive
    // integer).  Subsequent calls return the same instance.
    static ExecutableCache& session();

private:
    mutable std::mutex mu_;
    std::unordered_map<CacheKey, SharedExecutable, CacheKeyHash> map_;
    // Most-recent at the back; front = next eviction candidate.
    std::vector<CacheKey> lru_order_;
    std::size_t max_entries_ = 32;
};

}  // namespace lucid::compile
