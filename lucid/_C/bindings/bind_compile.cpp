// lucid/_C/bindings/bind_compile.cpp
//
// Phase 1.1 Python bindings for :mod:`lucid.compile`.
//
// Exposes:
//   - TensorMeta / OpNode / TraceGraph (read-only views of the recorded IR).
//   - Tracer (the thread-local op recorder).
//   - set_current_tracer() / current_tracer() — install or detach the
//     process-wide Tracer pointer that OpScopeFull's hook reads on each
//     op entry.
//
// The shapes here are intentionally minimal — Phase 1.2 layers MpsBuilder /
// ExecutableCache on top using the same Tracer.graph() output, and the JSON
// trace dump invoked by LUCID_COMPILE_DEBUG=1 consumes the read-only fields
// surfaced below.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/stat.h>

#include "../backend/gpu/mps/MpsBridge.h"
#include "../compile/CompiledExecutable.h"
#include "../compile/ExecutableCache.h"
#include "../compile/MpsBuilder.h"
#include "../compile/OpEmitters/OpEmitter.h"
#include "../compile/RngFeeds.h"
#include "../compile/Tracer.h"
#include "../compile/VjpEmitters/VjpEmitter.h"
#include "../core/TensorImpl.h"
#include "../version.h"

namespace py = pybind11;

// pybind11 type caster for :class:`lucid::compile::TraceId`.  Maps
// Python ``int`` ⇆ ``TraceId{int64_t}`` at the boundary.  Without
// this, ``std::vector<TraceId>`` / ``std::unordered_map<TraceId, …>``
// cross the C++↔Python boundary as unregistered opaque types.
namespace pybind11 {
namespace detail {
template <>
struct type_caster<::lucid::compile::TraceId> {
public:
    PYBIND11_TYPE_CASTER(::lucid::compile::TraceId, _("int"));

    bool load(handle src, bool /*convert*/) {
        PyObject* obj = src.ptr();
        if (obj == nullptr || !PyLong_Check(obj))
            return false;
        long long v = PyLong_AsLongLong(obj);
        if (v == -1 && PyErr_Occurred()) {
            PyErr_Clear();
            return false;
        }
        value = ::lucid::compile::TraceId{static_cast<std::int64_t>(v)};
        return true;
    }

    static handle
    cast(::lucid::compile::TraceId src, return_value_policy /*policy*/, handle /*parent*/) {
        return PyLong_FromLongLong(static_cast<long long>(src.v));
    }
};
}  // namespace detail
}  // namespace pybind11

namespace lucid::bindings {

namespace {

// MPSGraph reads a tensor's Metal buffer in place, so ``array_to_buffer``
// first runs the tensor's pending MLX work and gives it a contiguous buffer
// of its own.  That is MLX encoding onto the streams every thread shares,
// which two threads must not do at once (SharedStreams.h).  Doing it here,
// with the GIL held, leaves the run that follows without the GIL nothing to
// encode: every buffer it reads is ready.  Four threads running compiled
// models crashed in each of four runs when the run did it unlocked.  A view
// is skipped: a run never reads a view's buffer in place.
void ready_buffers(const std::vector<lucid::TensorImplPtr>& tensors) {
    for (const auto& t : tensors) {
        if (!t || t->device() != lucid::Device::GPU || t->family_key() != nullptr)
            continue;
        if (const auto* gs = std::get_if<lucid::GpuStorage>(&t->storage()); gs && gs->arr)
            lucid::gpu::mps::array_to_buffer(*gs->arr);
    }
}

// The feeds a compiled run reads once the GIL is released.  A Metal view
// reads its value out of its family's root and caches it in the tensor, and
// another Python thread may write the family meanwhile — replacing the root
// and dropping that cache — so each view is resolved here, with the GIL
// held, into a tensor that owns its value.  Traced RNG draws are redrawn
// first so a draw whose trace also took a view of it (joining a family)
// still matches its original TensorImpl in the RNG feed registry.  The
// feeds, and an in-place run's ``targets``, leave with their buffers ready.
std::vector<lucid::TensorImplPtr>
resolve_feeds(const std::vector<lucid::TensorImplPtr>& inputs,
              const std::vector<lucid::TensorImplPtr>& targets = {}) {
    const auto redrawn = lucid::compile::redraw_rng_feeds(inputs);
    std::vector<lucid::TensorImplPtr> feeds;
    feeds.reserve(redrawn.size());
    for (const auto& t : redrawn) {
        if (t && t->family_key() != nullptr)
            feeds.push_back(std::make_shared<lucid::TensorImpl>(t->storage(), t->shape(),
                                                                t->dtype(), t->device(), false));
        else
            feeds.push_back(t);
    }
    ready_buffers(feeds);
    ready_buffers(targets);
    return feeds;
}

// Python-side wrapper holding a shared reference to a compiled executable;
// the last holder releases it via :func:`destroy_executable`.
class PyCompiledExecutable {
public:
    explicit PyCompiledExecutable(lucid::compile::SharedExecutable exe) : exe_(std::move(exe)) {}

    PyCompiledExecutable(const PyCompiledExecutable&) = delete;
    PyCompiledExecutable& operator=(const PyCompiledExecutable&) = delete;

    lucid::compile::CompiledExecutable* raw() const { return exe_.get(); }

    std::size_t num_inputs() const { return lucid::compile::executable_num_inputs(exe_.get()); }
    std::size_t num_outputs() const { return lucid::compile::executable_num_outputs(exe_.get()); }
    std::vector<lucid::compile::TensorId> input_ids() const {
        return lucid::compile::executable_input_ids(exe_.get());
    }
    std::vector<lucid::compile::TensorId> output_ids() const {
        return lucid::compile::executable_output_ids(exe_.get());
    }
    std::vector<lucid::compile::TensorId> grad_output_ids() const {
        return lucid::compile::executable_grad_output_ids(exe_.get());
    }
    std::vector<std::vector<std::int64_t>> input_shapes() const {
        std::vector<std::vector<std::int64_t>> out;
        for (const auto& s : lucid::compile::executable_input_shapes(exe_.get()))
            out.emplace_back(s.begin(), s.end());
        return out;
    }
    std::vector<std::string> input_dtypes() const {
        std::vector<std::string> out;
        for (lucid::Dtype dt : lucid::compile::executable_input_dtypes(exe_.get()))
            out.emplace_back(lucid::dtype_name(dt));
        return out;
    }
    std::vector<std::size_t> feed_order() const {
        return lucid::compile::executable_feed_order(exe_.get());
    }
    std::vector<std::string> feed_names() const {
        return lucid::compile::executable_feed_names(exe_.get());
    }

private:
    // Shared with the session cache: eviction there no longer frees an
    // executable a module still runs.
    lucid::compile::SharedExecutable exe_;
};

void bind_trace_nodes(py::module_& m) {
    using lucid::compile::OpNode;
    using lucid::compile::TensorMeta;
    using lucid::compile::TraceGraph;
    using lucid::compile::VjpRegion;

    py::class_<TensorMeta>(m, "TensorMeta")
        .def_readonly("id", &TensorMeta::id)
        .def_readonly("shape", &TensorMeta::shape)
        .def_readonly("dtype", &TensorMeta::dtype)
        .def_readonly("device", &TensorMeta::device)
        .def("__repr__",
             [](const TensorMeta& t) { return "TensorMeta(id=" + std::to_string(t.id) + ")"; });

    py::class_<OpNode>(m, "OpNode")
        .def_readonly("name", &OpNode::name)
        .def_readonly("inputs", &OpNode::inputs)
        .def_readonly("outputs", &OpNode::outputs)
        .def_readonly("attrs", &OpNode::attrs)
        .def("__repr__", [](const OpNode& n) {
            return "OpNode(name='" + n.name + "', outputs=" + std::to_string(n.outputs.size()) +
                   ", attrs=" + std::to_string(n.attrs.size()) + ")";
        });

    py::class_<VjpRegion>(m, "VjpRegion")
        .def_readonly("fwd_begin", &VjpRegion::fwd_begin)
        .def_readonly("fwd_end", &VjpRegion::fwd_end)
        .def_readonly("input_ids", &VjpRegion::input_ids)
        .def_readonly("output_metas", &VjpRegion::output_metas)
        .def_readonly("cotangent_ids", &VjpRegion::cotangent_ids)
        .def_readonly("input_grad_ids", &VjpRegion::input_grad_ids)
        .def_readonly("bwd_ops", &VjpRegion::bwd_ops);

    py::class_<TraceGraph>(m, "TraceGraph")
        .def_readonly("ops", &TraceGraph::ops)
        .def_readonly("next_id", &TraceGraph::next_id)
        .def_readonly("unsupported", &TraceGraph::unsupported)
        .def_readonly("vjp_regions", &TraceGraph::vjp_regions)
        .def("__len__", [](const TraceGraph& g) { return g.ops.size(); });
}

void commit_tracer_vjp_region(lucid::compile::Tracer& t,
                              std::size_t fwd_begin,
                              std::size_t fwd_end,
                              std::vector<lucid::compile::TensorId> input_ids,
                              const std::vector<lucid::compile::TensorId>& output_ids,
                              const std::vector<lucid::TensorImplPtr>& output_impls,
                              std::vector<lucid::compile::TensorId> cotangent_ids,
                              std::vector<lucid::compile::TensorId> input_grad_ids,
                              std::size_t bwd_begin) {
    std::vector<lucid::compile::TensorMeta> metas;
    metas.reserve(output_ids.size());
    for (std::size_t i = 0; i < output_ids.size() && i < output_impls.size(); ++i) {
        const auto& impl = output_impls[i];
        metas.push_back(lucid::compile::TensorMeta{output_ids[i], impl->shape(), impl->dtype(),
                                                   impl->device()});
    }
    t.commit_vjp_region(fwd_begin, fwd_end, std::move(input_ids), std::move(metas),
                        std::move(cotangent_ids), std::move(input_grad_ids), bwd_begin);
}

void bind_tracer_class(py::module_& m) {
    using lucid::compile::Tracer;

    py::class_<Tracer>(m, "Tracer")
        .def(py::init<>())
        .def_property_readonly("graph", &Tracer::graph, py::return_value_policy::reference_internal)
        .def_property("redraw_rng", &Tracer::redraw_rng, &Tracer::set_redraw_rng,
                      "Whether default-generator draws become feeds drawn again on "
                      "every run (compile) or stay ops in the recording (Core ML).")
        .def_property("fold_constants", &Tracer::fold_constants, &Tracer::set_fold_constants,
                      "Whether an op whose inputs were all made inside the trace is left "
                      "out, its output a constant feed (compile) or recorded (every other "
                      "tracer).")
        .def(
            "made_here",
            [](const Tracer& t, const lucid::TensorImplPtr& impl) {
                return t.made_here(impl.get());
            },
            py::arg("impl"),
            "Whether ``impl`` is a constant the trace made itself, shared with "
            "nothing outside it.")
        .def(
            "outside_writes",
            [](const Tracer& t) {
                py::list out;
                for (const auto& [tid, impl] : t.outside_writes())
                    out.append(py::make_tuple(tid, impl));
                return out;
            },
            "(current id, TensorImpl) for every tensor from outside the trace "
            "that the trace wrote in place, in first-write order.")
        .def("restore_outside_writes", &Tracer::restore_outside_writes,
             "Put every tensor from outside the trace that it wrote in place "
             "back to its value from before the trace.")
        .def("watch", &Tracer::watch, py::arg("impl"),
             "Declare an argument of the traced call: a host read of it marks the "
             "recording unsupported even before an op has consumed it.")
        .def("mark_unsupported", &Tracer::mark_unsupported, py::arg("why"),
             "Mark the recording as one no executable can stand for; every "
             "compile entry point then refuses it.")
        .def_property_readonly(
            "external_feeds",
            [](const Tracer& t) {
                py::dict d;
                for (const auto& [tid, impl] : t.external_feeds())
                    d[py::cast(tid)] = impl;
                return d;
            },
            "id → TensorImpl map for inputs not produced by any traced "
            "op (model parameters + user inputs).")
        .def_property_readonly(
            "retained_values",
            [](const Tracer& t) {
                py::dict d;
                for (const auto& [tid, impl] : t.retained_values())
                    d[py::cast(tid)] = impl;
                return d;
            },
            "id → TensorImpl map for every traced value the tracer still "
            "holds, for a consumer that needs contents and not only shape.")
        .def(
            "lookup_id",
            [](const Tracer& t, const lucid::TensorImplPtr& impl) -> py::object {
                const lucid::compile::TensorId id = t.lookup_id(impl.get());
                if (id < 0)
                    return py::none();
                return py::cast(id);
            },
            py::arg("impl"),
            "Reverse-lookup the trace TensorId for an observed TensorImpl. "
            "Returns None if ``impl`` is not in the trace.")
        .def("op_count", &Tracer::op_count)
        .def("outside_write_count", &Tracer::outside_write_count)
        .def("ensure_id", &Tracer::ensure_id, py::arg("impl"))
        .def("register_cotangent", &Tracer::register_cotangent, py::arg("impl"))
        .def("discard_ops_from", &Tracer::discard_ops_from, py::arg("idx"))
        .def("commit_vjp_region", &commit_tracer_vjp_region, py::arg("fwd_begin"),
             py::arg("fwd_end"), py::arg("input_ids"), py::arg("output_ids"),
             py::arg("output_impls"), py::arg("cotangent_ids"), py::arg("input_grad_ids"),
             py::arg("bwd_begin"));

    m.def("set_current_tracer", &lucid::compile::set_current_tracer, py::arg("tracer").none(true));
    m.def("current_tracer", &lucid::compile::current_tracer, py::return_value_policy::reference);
}

}  // namespace

// Registers the compile-time tracing surface as a sub-module of `_C.engine`.
//
// Mirrors :func:`register_profiler` — the Python wrapper holds the only
// owning reference to each :class:`Tracer`; the engine retains a raw
// pointer installed via :func:`set_current_tracer`.
void register_compile(py::module_& m) {
    using lucid::compile::TraceGraph;

    bind_trace_nodes(m);
    bind_tracer_class(m);

    // ── Phase 1.2 step 1: compile / run / cache surface ──────────────────

    // Opaque executable handle.  Python owns one PyCompiledExecutable per
    // compile_trace() return; destruction releases the MPSGraphExecutable
    // via the wrapper's destructor.
    py::class_<PyCompiledExecutable, std::shared_ptr<PyCompiledExecutable>>(m, "CompiledExecutable")
        .def_property_readonly("num_inputs", &PyCompiledExecutable::num_inputs)
        .def_property_readonly("num_outputs", &PyCompiledExecutable::num_outputs)
        .def_property_readonly("input_ids", &PyCompiledExecutable::input_ids,
                               "Feed-order TensorId list — run_executable() "
                               "expects inputs in this order.")
        .def_property_readonly("output_ids", &PyCompiledExecutable::output_ids,
                               "Forward-output TensorId list.  run_executable() "
                               "returns these first, followed by grad outputs.")
        .def_property_readonly("grad_output_ids", &PyCompiledExecutable::grad_output_ids,
                               "Per-parameter gradient TensorId list (Phase 1.3 "
                               "backward path).  Empty for forward-only executables. "
                               "run_executable() returns one gradient tensor per "
                               "id here, immediately after the forward outputs.")
        .def_property_readonly("input_shapes", &PyCompiledExecutable::input_shapes,
                               "Feed-order input shapes frozen at trace time.  Slot i "
                               "of run_executable() must receive this shape (on a "
                               "dynamic-batch executable, only the trailing axes of a "
                               "non-parameter slot are binding).")
        .def_property_readonly("input_dtypes", &PyCompiledExecutable::input_dtypes,
                               "Feed-order input dtype names ('float32', 'int64', ...) "
                               "— companion to input_shapes.")
        .def_property_readonly("feed_order", &PyCompiledExecutable::feed_order,
                               "Bind order against MPSGraph's feed array: inputsArray "
                               "position k carries feed slot feed_order[k].  Empty when "
                               "the two orders coincide (freshly compiled executables); "
                               "populated after load_executable().")
        .def_property_readonly("feed_names", &PyCompiledExecutable::feed_names,
                               "Placeholder names MPSGraph reports for the feed tensors, "
                               "in its own order.  Diagnostic.");

    // Compile a TraceGraph into an MPSGraph executable.
    // Returns None on any abort condition (unsupported op, mixed-device,
    // dtype error, …) — the caller falls back to eager dispatch.
    m.def(
        "compile_trace",
        [](const lucid::compile::TraceGraph& graph,
           const py::dict& external_feeds_py) -> py::object {
            if (!graph.unsupported.empty())
                return py::none();  // the documented abort signal
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_trace(graph, feeds, &err);
            if (exe == nullptr)
                return py::none();
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("graph"), py::arg("external_feeds"),
        "Lower a TraceGraph to a compiled MPSGraph executable; "
        "returns None when any op in the trace is unsupported.");

    // Run a compiled executable with feed-order input tensors. Returns a
    // list of fresh GPU-backed TensorImpls in the executable's target
    // order.  Forward declaration of run_executable lives in
    // CompiledExecutable.mm.
    m.def(
        "run_executable",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper,
           const std::vector<lucid::TensorImplPtr>& inputs) -> std::vector<lucid::TensorImplPtr> {
            if (!wrapper)
                throw std::invalid_argument("run_executable: null wrapper");
            const auto feeds = resolve_feeds(inputs);
            // The run waits for the GPU; other Python threads need not.
            py::gil_scoped_release release;
            return lucid::compile::run_executable(wrapper->raw(), feeds);
        },
        py::arg("executable"), py::arg("inputs"),
        "Execute a CompiledExecutable; inputs in input_ids order, "
        "outputs in output_ids order.");

    m.def(
        "run_executable_inplace",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper,
           const std::vector<lucid::TensorImplPtr>& inputs,
           const std::vector<lucid::TensorImplPtr>& output_targets) -> void {
            if (!wrapper)
                throw std::invalid_argument("run_executable_inplace: null wrapper");
            const auto feeds = resolve_feeds(inputs, output_targets);
            std::vector<lucid::GpuStorage> fresh_outputs;
            {
                py::gil_scoped_release release;
                fresh_outputs = lucid::compile::run_executable_inplace_outputs(
                    wrapper->raw(), feeds, output_targets);
            }
            // The swap replaces a Metal view family's root and drops its
            // members' cached views: run it with the GIL held again so no
            // Python thread reading a sibling races it.
            lucid::compile::swap_executable_inplace_targets(output_targets, fresh_outputs);
        },
        py::arg("executable"), py::arg("inputs"), py::arg("output_targets"),
        "Execute a CompiledExecutable, writing each output directly into "
        "the corresponding pre-allocated tensor in ``output_targets`` "
        "instead of returning fresh tensors.  Used by the compiled-"
        "optimizer step path — avoids the per-output allocation + copy_ "
        "round-trip when the targets are parameters / state buffers "
        "that already live on the GPU.");

    // Session cache surface — `find` / `insert` are internal (Phase 1.4 will
    // wrap them inside CompiledModule); for Phase 1.2 we expose only size +
    // clear so tests can observe / reset cache state.
    // Diagnostics: report whether a given op name has an emitter
    // registered, and whether that emitter is real-emit or a stub.
    // Used by ``tools/`` and the docstring coverage tracker — NOT
    // a public end-user API.
    m.def(
        "emitter_registered",
        [](const std::string& name) -> bool {
            return lucid::compile::find_emitter(name) != nullptr;
        },
        py::arg("name"),
        "Return True iff an emitter is registered for ``name``.  A "
        "registered emitter may still return nullptr at emit time "
        "(stub semantics) — this helper only distinguishes "
        "'registered but possibly a stub' from 'no emitter at all'.");

    m.def("session_cache_size", [] { return lucid::compile::ExecutableCache::session().size(); });
    m.def("session_cache_clear", [] { lucid::compile::ExecutableCache::session().clear(); });

    // Attention fused-pass workaround capability flag (see OpEmitter.h +
    // lucid.compile._core.attention_probe).  -1 unprobed / 0 unaffected /
    // 1 affected.  The Python probe sets it before the first real attention
    // compile so the expensive transpose only fires on affected GPUs.
    m.def("attention_workaround_state", &lucid::compile::attention_workaround_state);
    m.def("set_attention_workaround_state", &lucid::compile::set_attention_workaround_state,
          py::arg("state"));

    // Phase 1.2 secondary acceptance gate: cache-aware compile.  On the
    // first call with a given structural signature the trace is compiled
    // and inserted into the session-global cache; subsequent calls with
    // a matching CacheKey return the same executable, shared with the
    // cache so an eviction cannot free it under a module that holds it.
    m.def(
        "compile_or_cached",
        [](const lucid::compile::TraceGraph& graph, const py::dict& external_feeds_py,
           bool dynamic_batch, const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& explicit_outputs) -> py::object {
            if (!graph.unsupported.empty())
                throw std::runtime_error("compile_or_cached: " + graph.unsupported);
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }

            if (dynamic_batch) {
                // Phase 1.6 dynamic path: bypass the session cache.
                // ``make_cache_key`` derives from concrete trace shapes
                // which would mint a fresh entry per batch size — the
                // very thing dynamic mode is meant to avoid.  The
                // caller (CompiledModule) holds its own
                // dynamic-batch-aware Python cache.
                std::string err;
                lucid::compile::CompiledExecutable* exe = lucid::compile::compile_trace(
                    graph, feeds, &err, /*dynamic_batch=*/true, param_ids, explicit_outputs);
                if (exe == nullptr) {
                    if (!err.empty())
                        throw std::runtime_error(err);
                    return py::none();
                }
                return py::cast(
                    std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
            }

            auto& cache = lucid::compile::ExecutableCache::session();
            std::unordered_map<lucid::compile::TensorId, std::pair<lucid::Shape, lucid::Dtype>>
                feed_meta;
            feed_meta.reserve(feeds.size());
            for (const auto& [tid, impl] : feeds)
                if (impl)
                    feed_meta.emplace(tid, std::make_pair(impl->shape(), impl->dtype()));
            lucid::compile::TraceIdMap id_map;
            lucid::compile::CacheKey key =
                lucid::compile::make_cache_key(graph, feed_meta, explicit_outputs, &id_map);

            if (auto hit = cache.find(key)) {
                if (auto local = lucid::compile::localize_executable(hit.get(), id_map)) {
                    return py::cast(std::make_shared<PyCompiledExecutable>(std::move(local)));
                }
            }

            // ``LUCID_COMPILE_DISK_CACHE=1`` opt-in: before falling
            // through to a fresh compile, try to load a previously-
            // serialised executable from
            // ``$LUCID_COMPILE_DISK_CACHE_DIR`` (default
            // ``$TMPDIR/lucid_mpsgraph_cache``).  Filename is keyed by
            // the cache key's hash + the engine ABI so an SDK or
            // engine version bump invalidates stale caches without
            // explicit user action.  Misses (file absent, format
            // version mismatch, parse error) fall through to compile
            // + save.
            static const bool disk_cache_enabled = []() {
                const char* s = std::getenv("LUCID_COMPILE_DISK_CACHE");
                return s && std::string(s) == "1";
            }();
            std::string disk_path;
            if (disk_cache_enabled) {
                const char* dir_env = std::getenv("LUCID_COMPILE_DISK_CACHE_DIR");
                std::string dir;
                if (dir_env && *dir_env) {
                    dir = dir_env;
                } else {
                    const char* tmp = std::getenv("TMPDIR");
                    dir = std::string(tmp ? tmp : "/tmp") + "lucid_mpsgraph_cache";
                }
                // Ensure trailing slash + create directory if missing.
                if (!dir.empty() && dir.back() != '/')
                    dir.push_back('/');
                // ``mkdir -p`` via POSIX ``mkdir`` (one level is enough
                // because $TMPDIR / user-provided dir already exists).
                (void)::mkdir(dir.c_str(), 0755);
                lucid::compile::CacheKeyHash h;
                std::size_t hashv = h(key);
                char buf[128];
                std::snprintf(buf, sizeof(buf), "%slucid_abi%d_%016zx", dir.c_str(),
                              static_cast<int>(LUCID_ABI_VERSION), hashv);
                disk_path = buf;

                std::string load_err;
                lucid::compile::CompiledExecutable* disk_hit =
                    lucid::compile::load_executable(disk_path, &load_err);
                if (disk_hit != nullptr) {
                    auto shared_disk = lucid::compile::share_executable(disk_hit);
                    if (auto local =
                            lucid::compile::localize_executable(shared_disk.get(), id_map)) {
                        cache.insert(key, std::move(shared_disk));
                        return py::cast(std::make_shared<PyCompiledExecutable>(std::move(local)));
                    }
                }
            }

            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_trace(graph, feeds, &err,
                                              /*dynamic_batch=*/false, param_ids, explicit_outputs);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            auto raw_shared = lucid::compile::share_executable(exe);
            if (auto canon_shared =
                    lucid::compile::canonicalize_executable(raw_shared.get(), id_map)) {
                auto hit = cache.insert(key, std::move(canon_shared));
                if (disk_cache_enabled && !disk_path.empty())
                    (void)lucid::compile::save_executable(hit.get(), disk_path);
            }
            return py::cast(std::make_shared<PyCompiledExecutable>(std::move(raw_shared)));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("dynamic_batch") = false,
        py::arg("param_ids") = std::vector<lucid::compile::TensorId>{},
        py::arg("explicit_outputs") = std::vector<lucid::compile::TensorId>{},
        "Cache-aware compile: on a cache miss the trace is compiled and "
        "inserted into the session-global ExecutableCache; on a hit the "
        "cached executable is returned (Python wrapper is borrowed — the "
        "cache retains ownership).  With ``dynamic_batch=True`` (Phase 1.6) "
        "the session cache is bypassed (caller manages a dynamic-aware "
        "Python cache) and ``param_ids`` declares which feed ids are "
        "parameters whose first dim must stay static.  ``explicit_outputs`` "
        "(when non-empty) overrides the auto-detect graph-output heuristic "
        "with exactly the listed ids — used by CompiledModule to mask "
        "Python-discarded intermediates from the executable's output list.");

    // Phase 1.10 AOT export: expose the previously-internal
    // ``save_executable`` / ``load_executable`` C++ APIs as the
    // Python-facing user surface.  Stable user-facing pickle / AOT
    // serialisation now flows through these.
    m.def(
        "save_executable",
        [](const std::shared_ptr<PyCompiledExecutable>& wrapper, const std::string& path) -> bool {
            if (!wrapper)
                throw std::invalid_argument("save_executable: null wrapper");
            return lucid::compile::save_executable(wrapper->raw(), path);
        },
        py::arg("executable"), py::arg("path"),
        "Serialise the executable to ``<path>.mpsgraphpackage`` (Apple-native "
        "MPSGraphExecutable archive, macOS 14+) and ``<path>.meta`` (Lucid I/O "
        "plan).  Returns ``True`` on success.  Both files together constitute "
        "one saved compile artifact — ``load_executable`` expects them paired.  "
        "Same on-disk format as the internal ``LUCID_COMPILE_DISK_CACHE`` "
        "path.  Throws on null wrapper; returns ``False`` on I/O / "
        "serialisation failure.");

    m.def(
        "load_executable",
        [](const std::string& path) -> py::object {
            std::string err;
            lucid::compile::CompiledExecutable* exe = lucid::compile::load_executable(path, &err);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("path"),
        "Reload a CompiledExecutable previously saved via "
        "``save_executable(executable, path)``.  Expects both "
        "``<path>.mpsgraphpackage`` and ``<path>.meta`` to exist.  Returns "
        "the executable on success, ``None`` on missing files / format "
        "mismatch; throws on a corrupted .meta sidecar.  ABI-version mismatch "
        "(SDK or engine bump) is rejected at load time — callers must "
        "recompile from the source trace.");

    // Phase 1.3: forward + backward in one executable.
    m.def(
        "compile_trace_with_backward",
        [](const lucid::compile::TraceGraph& graph, const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id, const std::vector<lucid::compile::TensorId>& param_ids,
           bool dynamic_batch,
           const std::vector<lucid::compile::TensorId>& extra_output_ids) -> py::object {
            if (!graph.unsupported.empty())
                throw std::runtime_error("compile_trace_with_backward: " + graph.unsupported);
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe = lucid::compile::compile_trace_with_backward(
                graph, feeds, loss_id, param_ids, &err, dynamic_batch, extra_output_ids);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"), py::arg("param_ids"),
        py::arg("dynamic_batch") = false,
        py::arg("extra_output_ids") = std::vector<lucid::compile::TensorId>{},
        "Lower a TraceGraph + (loss_id, param_ids) into a single MPSGraph "
        "executable that computes the forward (loss) and the auto-derived "
        "gradients of loss w.r.t. each parameter.  Returns None on any "
        "unsupported op (eager fallback).  Output of run_executable: "
        "[loss_tensor, grad_for_param_ids[0], grad_for_param_ids[1], ..., "
        "extra_output_ids[0], extra_output_ids[1], ...] — ``extra_output_ids`` "
        "are explicit non-gradient trace outputs (e.g. BatchNorm running-stat "
        "EMA new_rm/new_rv) the caller writes back into stateful buffers each "
        "run.  With ``dynamic_batch=True`` (Phase 1.6) the leading dim of every "
        "non-parameter feed becomes a symbolic placeholder (-1), so one "
        "executable handles variable batch size.");

    // Phase 1.7: forward + backward + optimizer update in one executable.
    // Used by lucid.compile.fused_step(model, loss_fn, optimizer).
    py::enum_<lucid::compile::OptimizerSpec::Kind>(m, "OptimizerKind")
        .value("SGD", lucid::compile::OptimizerSpec::Kind::SGD)
        .value("ADAM", lucid::compile::OptimizerSpec::Kind::ADAM)
        .value("ADAMW", lucid::compile::OptimizerSpec::Kind::ADAMW);

    py::class_<lucid::compile::OptimizerSpec>(m, "OptimizerSpec")
        .def(py::init<>())
        .def_readwrite("kind", &lucid::compile::OptimizerSpec::kind)
        .def_readwrite("lr", &lucid::compile::OptimizerSpec::lr)
        .def_readwrite("momentum", &lucid::compile::OptimizerSpec::momentum)
        .def_readwrite("dampening", &lucid::compile::OptimizerSpec::dampening)
        .def_readwrite("weight_decay", &lucid::compile::OptimizerSpec::weight_decay)
        .def_readwrite("nesterov", &lucid::compile::OptimizerSpec::nesterov)
        .def_readwrite("beta1", &lucid::compile::OptimizerSpec::beta1)
        .def_readwrite("beta2", &lucid::compile::OptimizerSpec::beta2)
        .def_readwrite("eps", &lucid::compile::OptimizerSpec::eps);

    m.def(
        "compile_fused_training_step",
        [](const lucid::compile::TraceGraph& graph, const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id, const std::vector<lucid::compile::TensorId>& param_ids,
           const lucid::compile::OptimizerSpec& opt_spec,
           const std::vector<std::vector<lucid::compile::TensorId>>& state_buf_ids_per_param,
           const std::vector<lucid::compile::TensorId>& scalar_input_ids) -> py::object {
            if (!graph.unsupported.empty())
                throw std::runtime_error("compile_fused_training_step: " + graph.unsupported);
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe = lucid::compile::compile_fused_training_step(
                graph, feeds, loss_id, param_ids, opt_spec, state_buf_ids_per_param,
                scalar_input_ids, &err);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"), py::arg("param_ids"),
        py::arg("opt_spec"), py::arg("state_buf_ids_per_param"), py::arg("scalar_input_ids"),
        "Lower (forward + backward + optimizer.step) into a single MPSGraph "
        "executable.  Outputs (in order): [loss, new_param_0, new_param_1, "
        "..., new_state_0_0, new_state_0_1, ..., new_state_N_0, ...].  Used "
        "by lucid.compile.fused_step.");

    // Phase 1.8: generic fused step — the optimizer update is captured
    // in the same TraceGraph (via ghost-grad placeholders), so any
    // optimizer expressible as Lucid tensor ops works without C++ math
    // duplication.
    m.def(
        "compile_generic_fused_step",
        [](const lucid::compile::TraceGraph& graph, const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id, const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& ghost_grad_ids,
           const std::vector<lucid::compile::TensorId>& output_target_ids) -> py::object {
            if (!graph.unsupported.empty())
                throw std::runtime_error("compile_generic_fused_step: " + graph.unsupported);
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            // Note (2026-05-25): a disk-cache variant of this path was
            // prototyped — see git log for the experiment — but
            // measured slower than recompile on the fused_step case
            // (warm cache hit 129 ms vs fresh compile 132 ms; first
            // cold hit 828 ms while the OS file cache warmed).  The
            // forward-only ``compile_or_cached`` disk cache still
            // wins (6× cold-start) because forward MPSGraph packages
            // are smaller and serialise faster; fused_step packages
            // include the autograd-derived backward chain which
            // appears expensive to deserialise.  Decision: in-memory
            // session cache only for fused_step until the package
            // format improves.
            std::string err;
            lucid::compile::CompiledExecutable* exe = lucid::compile::compile_generic_fused_step(
                graph, feeds, loss_id, param_ids, ghost_grad_ids, output_target_ids, &err);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"), py::arg("param_ids"),
        py::arg("ghost_grad_ids"), py::arg("output_target_ids"),
        "Generic fused-step compile.  The TraceGraph contains both the "
        "forward+loss path and the optimizer update.  ``ghost_grad_ids`` "
        "are placeholder tensor ids that the Python wrapper reserved for "
        "the optimizer's grad inputs; this function derives those grads "
        "via MPSGraph autograd after emitting the forward, then binds "
        "each ghost grad to its derived gradient before emitting the "
        "remaining (optimizer) ops.  ``output_target_ids`` is the flat "
        "ordered list of new_param / new_state ids that become the "
        "executable's targets (loss is always target 0).");

    // Phase 1.9: stateful-variables variant.  Designated feeds become
    // ``variableWithData:`` (initialized from the Lucid Tensor's
    // current MTLBuffer contents at compile time); the matching
    // output_target_ids become ``assignVariable:`` operations + a
    // ``readVariable:`` tensor that flushes the post-assign value
    // into the Lucid Tensor's existing MTLBuffer on each call.
    m.def(
        "compile_generic_fused_step_with_vars",
        [](const lucid::compile::TraceGraph& graph, const py::dict& external_feeds_py,
           lucid::compile::TensorId loss_id, const std::vector<lucid::compile::TensorId>& param_ids,
           const std::vector<lucid::compile::TensorId>& ghost_grad_ids,
           const std::vector<lucid::compile::TensorId>& output_target_ids,
           const std::vector<std::pair<lucid::compile::TensorId, lucid::compile::TensorId>>&
               variable_pairs) -> py::object {
            if (!graph.unsupported.empty())
                throw std::runtime_error("compile_generic_fused_step_with_vars: " +
                                         graph.unsupported);
            std::unordered_map<lucid::compile::TensorId, lucid::TensorImplPtr> feeds;
            feeds.reserve(external_feeds_py.size());
            for (auto item : external_feeds_py) {
                auto tid = py::cast<lucid::compile::TensorId>(item.first);
                auto impl = py::cast<lucid::TensorImplPtr>(item.second);
                feeds.emplace(tid, std::move(impl));
            }
            std::string err;
            lucid::compile::CompiledExecutable* exe =
                lucid::compile::compile_generic_fused_step_with_vars(
                    graph, feeds, loss_id, param_ids, ghost_grad_ids, output_target_ids,
                    variable_pairs, &err);
            if (exe == nullptr) {
                if (!err.empty())
                    throw std::runtime_error(err);
                return py::none();
            }
            return py::cast(
                std::make_shared<PyCompiledExecutable>(lucid::compile::share_executable(exe)));
        },
        py::arg("graph"), py::arg("external_feeds"), py::arg("loss_id"), py::arg("param_ids"),
        py::arg("ghost_grad_ids"), py::arg("output_target_ids"), py::arg("variable_pairs"),
        "Stateful-variables variant of compile_generic_fused_step.  "
        "``variable_pairs`` is a list of ``(feed_id, write_id)`` tuples: "
        "the feed becomes a persistent MPSGraph variable initialised "
        "from the Lucid Tensor's current buffer contents, and the "
        "matching write_id is bound to an ``assignVariable:`` + "
        "``readVariable:`` pair so subsequent calls update the variable "
        "internally and flush the new value back into the Lucid Tensor's "
        "existing MTLBuffer (no per-call newBufferWithLength).  Empty "
        "variable_pairs is equivalent to compile_generic_fused_step.");

    // ── Manual-VJP coverage diagnostics ─────────────────────────────
    //
    // The 3-state classification + per-op registration query that
    // backs :func:`lucid.compile.diagnose`.  Pure introspection — no
    // graph state is touched.
    py::enum_<lucid::compile::VjpRegistration>(
        m, "VjpRegistration",
        "Three-state classification of an op's manual-VJP coverage.  "
        "``Registered`` = real backward (a :class:`VjpEmitter` exists); "
        "``GradSink`` = the walker stops gradient flow here by design "
        "(factories, integer casts, comparisons, arg-reduce); "
        "``Missing`` = no emitter, not a sink — soft-fallback to "
        "``gradientForPrimaryTensor:`` (or hard-fail under "
        "``LUCID_MANUAL_VJP_REQUIRE=1``).")
        .value("Registered", lucid::compile::VjpRegistration::Registered)
        .value("GradSink", lucid::compile::VjpRegistration::GradSink)
        .value("Missing", lucid::compile::VjpRegistration::Missing);

    m.def(
        "vjp_registration_status",
        [](const std::string& op_name) { return lucid::compile::vjp_registration_status(op_name); },
        py::arg("op_name"),
        "Look up an op's manual-VJP coverage status.  Returns a "
        ":class:`VjpRegistration` value.  Used by "
        ":func:`lucid.compile.diagnose` to build per-trace coverage "
        "reports without re-running the model.");

    m.def("use_manual_vjp", &lucid::compile::use_manual_vjp,
          "Return ``True`` if manual VJP is enabled "
          "(``LUCID_MANUAL_VJP`` env var is unset or set to a truthy "
          "value).  Production default: ON.");

    m.def("use_manual_vjp_require", &lucid::compile::use_manual_vjp_require,
          "Return ``True`` if hard-fail-on-gap is enabled "
          "(``LUCID_MANUAL_VJP_REQUIRE=1``).  When ON, a coverage gap "
          "raises ``RuntimeError`` instead of falling back to "
          "``gradientForPrimaryTensor:``.");

    m.def("use_manual_vjp_debug", &lucid::compile::use_manual_vjp_debug,
          "Return ``True`` if structured stderr logging on coverage "
          "gap is enabled (``LUCID_MANUAL_VJP_DEBUG=1``).  Off by "
          "default; turn on to diagnose why a model falls back from "
          "manual VJP to MPSGraph autograd.");
}

namespace {
// No-op symbols pulled from this TU keep each emitter family's
// file-scope registrar alive against aggressive dead-code stripping
// when the bindings are linked into the engine MODULE.  Each emitter
// is otherwise self-contained (only side effect is the static
// registrar in its .mm), and `--gc-sections` + ``-fvisibility=hidden``
// would otherwise drop the entire .mm.
void touch_emitters() {
    (void)lucid::compile::find_emitter("linear");
    (void)lucid::compile::find_emitter("relu");
    (void)lucid::compile::find_emitter("sigmoid");
    (void)lucid::compile::find_emitter("tanh");
    (void)lucid::compile::find_emitter("add");
    (void)lucid::compile::find_emitter("sub");
    (void)lucid::compile::find_emitter("mul");
    (void)lucid::compile::find_emitter("div");
    (void)lucid::compile::find_emitter("view");
    (void)lucid::compile::find_emitter("reshape");
    (void)lucid::compile::find_emitter("squeeze");
    (void)lucid::compile::find_emitter("unsqueeze");
    (void)lucid::compile::find_emitter("flatten");
    (void)lucid::compile::find_emitter("contiguous");
    (void)lucid::compile::find_emitter("permute");
    (void)lucid::compile::find_emitter("sum");
    (void)lucid::compile::find_emitter("mean");
    (void)lucid::compile::find_emitter("neg");
    (void)lucid::compile::find_emitter("abs");
    (void)lucid::compile::find_emitter("exp");
    (void)lucid::compile::find_emitter("log");
    (void)lucid::compile::find_emitter("sqrt");
    (void)lucid::compile::find_emitter("square");
    (void)lucid::compile::find_emitter("reciprocal");
    (void)lucid::compile::find_emitter("rsqrt");
    (void)lucid::compile::find_emitter("softmax");
    (void)lucid::compile::find_emitter("log_softmax");
    (void)lucid::compile::find_emitter("sin");
    (void)lucid::compile::find_emitter("cos");
    (void)lucid::compile::find_emitter("tan");
    (void)lucid::compile::find_emitter("sinh");
    (void)lucid::compile::find_emitter("cosh");
    (void)lucid::compile::find_emitter("silu");
    (void)lucid::compile::find_emitter("gelu");
    (void)lucid::compile::find_emitter("gelu_exact");
    (void)lucid::compile::find_emitter("pow");
    (void)lucid::compile::find_emitter("matmul");
    (void)lucid::compile::find_emitter("max_pool2d");
    (void)lucid::compile::find_emitter("avg_pool2d");
    (void)lucid::compile::find_emitter("layer_norm");
    (void)lucid::compile::find_emitter("rms_norm");
    (void)lucid::compile::find_emitter("conv2d");
    (void)lucid::compile::find_emitter("batch_norm_eval");
    (void)lucid::compile::find_emitter("max");
    (void)lucid::compile::find_emitter("min");
    (void)lucid::compile::find_emitter("prod");
    (void)lucid::compile::find_emitter("concatenate");
    (void)lucid::compile::find_emitter("batch_norm");
    (void)lucid::compile::find_emitter("batch_norm1d");
    (void)lucid::compile::find_emitter("batch_norm3d");
    (void)lucid::compile::find_emitter("group_norm");
    (void)lucid::compile::find_emitter("full");
    (void)lucid::compile::find_emitter("eye");
    (void)lucid::compile::find_emitter("astype");
    (void)lucid::compile::find_emitter("gather");
    (void)lucid::compile::find_emitter("equal");
    (void)lucid::compile::find_emitter("not_equal");
    (void)lucid::compile::find_emitter("greater");
    (void)lucid::compile::find_emitter("greater_equal");
    (void)lucid::compile::find_emitter("less");
    (void)lucid::compile::find_emitter("less_equal");
    // R1 chunk 1
    (void)lucid::compile::find_emitter("arccos");
    (void)lucid::compile::find_emitter("arcsin");
    (void)lucid::compile::find_emitter("arctan");
    (void)lucid::compile::find_emitter("ceil");
    (void)lucid::compile::find_emitter("floor");
    (void)lucid::compile::find_emitter("round");
    (void)lucid::compile::find_emitter("cube");
    (void)lucid::compile::find_emitter("cube_root");
    (void)lucid::compile::find_emitter("log2");
    (void)lucid::compile::find_emitter("sign");
    (void)lucid::compile::find_emitter("erf");
    (void)lucid::compile::find_emitter("clip");
    (void)lucid::compile::find_emitter("elu");
    (void)lucid::compile::find_emitter("leaky_relu");
    (void)lucid::compile::find_emitter("selu");
    (void)lucid::compile::find_emitter("mish");
    (void)lucid::compile::find_emitter("softplus");
    (void)lucid::compile::find_emitter("hard_sigmoid");
    (void)lucid::compile::find_emitter("hard_swish");
    (void)lucid::compile::find_emitter("relu6");
    (void)lucid::compile::find_emitter("maximum");
    (void)lucid::compile::find_emitter("minimum");
    (void)lucid::compile::find_emitter("floordiv");
    (void)lucid::compile::find_emitter("nextafter");
    (void)lucid::compile::find_emitter("pow_scalar");
    (void)lucid::compile::find_emitter("rpow_scalar");
    (void)lucid::compile::find_emitter("invert");
    (void)lucid::compile::find_emitter("var");
    // R1 chunk 2
    (void)lucid::compile::find_emitter("where");
    (void)lucid::compile::find_emitter("masked_fill");
    (void)lucid::compile::find_emitter("flip");
    (void)lucid::compile::find_emitter("roll");
    (void)lucid::compile::find_emitter("broadcast_to");
    (void)lucid::compile::find_emitter("pad");
    (void)lucid::compile::find_emitter("tile");
    (void)lucid::compile::find_emitter("repeat");
    (void)lucid::compile::find_emitter("stack");
    // R2 chunk 1 — loss family
    (void)lucid::compile::find_emitter("mse_loss");
    (void)lucid::compile::find_emitter("huber_loss");
    (void)lucid::compile::find_emitter("bce_loss");
    (void)lucid::compile::find_emitter("bce_with_logits");
    (void)lucid::compile::find_emitter("embedding");
    (void)lucid::compile::find_emitter("nll_loss");
    (void)lucid::compile::find_emitter("cross_entropy_loss");
    // R3 cumulative
    (void)lucid::compile::find_emitter("cumsum");
    (void)lucid::compile::find_emitter("cumprod");
    (void)lucid::compile::find_emitter("cummax");
    (void)lucid::compile::find_emitter("cummin");
    // MoreOps — sort family + linalg long tail + stubs
    (void)lucid::compile::find_emitter("sort");
    (void)lucid::compile::find_emitter("argsort");
    (void)lucid::compile::find_emitter("argmax");
    (void)lucid::compile::find_emitter("argmin");
    (void)lucid::compile::find_emitter("all");
    (void)lucid::compile::find_emitter("any");
    (void)lucid::compile::find_emitter("conj");
    (void)lucid::compile::find_emitter("real");
    (void)lucid::compile::find_emitter("imag");
    (void)lucid::compile::find_emitter("trace");
    (void)lucid::compile::find_emitter("inner");
    (void)lucid::compile::find_emitter("outer");
    (void)lucid::compile::find_emitter("dot");
    (void)lucid::compile::find_emitter("tril");
    (void)lucid::compile::find_emitter("triu");
    (void)lucid::compile::find_emitter("diagonal");
    (void)lucid::compile::find_emitter("nan_to_num");
    (void)lucid::compile::find_emitter("one_hot");
    (void)lucid::compile::find_emitter("scaled_dot_product_attention");
    (void)lucid::compile::find_emitter("lp_normalize");
    (void)lucid::compile::find_emitter("global_response_norm");
    (void)lucid::compile::find_emitter("scatter_add");
    (void)lucid::compile::find_emitter("scatter_amax");
    (void)lucid::compile::find_emitter("scatter_amin");
    (void)lucid::compile::find_emitter("scatter_prod");
    // Tier 1
    (void)lucid::compile::find_emitter("norm");
    (void)lucid::compile::find_emitter("matrix_power");
    (void)lucid::compile::find_emitter("det");
    (void)lucid::compile::find_emitter("inv");
    (void)lucid::compile::find_emitter("solve_triangular");
    (void)lucid::compile::find_emitter("cholesky");
    (void)lucid::compile::find_emitter("qr");
    (void)lucid::compile::find_emitter("eigh");
    (void)lucid::compile::find_emitter("svd");
    (void)lucid::compile::find_emitter("pinv");
    (void)lucid::compile::find_emitter("bilinear_layer");
    (void)lucid::compile::find_emitter("rotary_pos_embedding");
    (void)lucid::compile::find_emitter("tensordot");
    // Tier 2
    (void)lucid::compile::find_emitter("conv1d");
    (void)lucid::compile::find_emitter("conv3d");
    (void)lucid::compile::find_emitter("conv_transpose1d");
    (void)lucid::compile::find_emitter("conv_transpose2d");
    (void)lucid::compile::find_emitter("max_pool1d");
    (void)lucid::compile::find_emitter("avg_pool1d");
    // Tier 3
    (void)lucid::compile::find_emitter("interpolate_bilinear");
    (void)lucid::compile::find_emitter("interpolate_nearest_2d");
    // Tier 4 partial
    (void)lucid::compile::find_emitter("unfold");
    (void)lucid::compile::find_emitter("fold");
    (void)lucid::compile::find_emitter("affine_grid");
    (void)lucid::compile::find_emitter("unfold_dim");
}
[[maybe_unused]] auto kEmitterAnchor = (touch_emitters(), 0);
}  // namespace

}  // namespace lucid::bindings
