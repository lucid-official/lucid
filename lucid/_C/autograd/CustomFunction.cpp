// lucid/_C/autograd/CustomFunction.cpp
//
// Implements PythonBackwardNode::apply() and the pybind11 registration
// function register_custom_function().  Together they allow Python code to
// inject a custom backward computation into the C++ autograd graph.

#include "CustomFunction.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../core/Error.h"
#include "../core/ErrorBuilder.h"
#include "../core/GradMode.h"
#include "GraphOps.h"
#include "Helpers.h"
#include "TensorHooks.h"

namespace py = pybind11;

namespace lucid {

namespace {

// How a gradient ``backward`` returned is named in the TypeError for one that
// is neither a tensor nor ``None``.  Its shape, dtype and device were held to
// the input it is for by ``_python_node._validate`` before it got here.
constexpr const char* kBackwardResult = "a custom Function's backward result";

}  // namespace

FunctionCtx::~FunctionCtx() {
    clear();
}

namespace {

// Build a non-family detached alias of ``t`` for a saved output slot so the
// node's context does not keep ``t`` (and through ``t->grad_fn()``, the node
// itself) alive.  On the CPU, rebind ``t``'s and the alias's ``CpuStorage::ptr``
// onto distinct control blocks that each own the underlying buffer: ``t``'s
// own ``ptr.use_count()`` stays equal to its view-family count so an in-place
// op on ``t`` reaches the version check instead of refusing ``t`` as storage
// shared outside its views.
std::shared_ptr<TensorImpl> make_saved_output_alias(const std::shared_ptr<TensorImpl>& t) {
    Storage split = t->saved_output_storage();
    auto alias = TensorImpl::make_view(t, t->shape(), t->stride(), 0, /*join_family=*/false);
    alias->set_detached_alias(true);
    if (auto* a_cpu = std::get_if<CpuStorage>(&const_cast<Storage&>(alias->raw_storage()));
        a_cpu && a_cpu->ptr && std::holds_alternative<CpuStorage>(split)) {
        a_cpu->ptr =
            std::shared_ptr<std::byte[]>(std::get<CpuStorage>(split).ptr, a_cpu->ptr.get());
    }
    return alias;
}

}  // namespace

void FunctionCtx::save_for_backward(std::vector<std::shared_ptr<TensorImpl>> tensors) {
    const auto owner = owner_node_.lock();
    saved_.clear();
    saved_.reserve(tensors.size());
    for (auto& t : tensors) {
        if (t && owner && t->grad_fn() == owner) {
            saved_.push_back(SavedVariable{
                make_saved_output_alias(t),
                t,
                owner,
                t->version_sink(),
                t->grad_output_nr(),
                t->version(),
                true,
            });
        } else {
            saved_.push_back(SavedVariable{std::move(t), {}, {}, {}, 0, 0, false});
        }
    }
}

void FunctionCtx::set_owner_node(const std::shared_ptr<Node>& owner) {
    owner_node_ = owner;
    if (!owner)
        return;
    for (auto& entry : saved_) {
        if (!entry.is_output && entry.data && entry.data->grad_fn() == owner) {
            auto t = std::move(entry.data);
            entry.data = make_saved_output_alias(t);
            entry.live_output = t;
            entry.grad_fn = owner;
            entry.version_sink = t->version_sink();
            entry.output_nr = t->grad_output_nr();
            entry.saved_version = t->version();
            entry.is_output = true;
        }
    }
}

std::vector<std::shared_ptr<TensorImpl>> FunctionCtx::saved_tensors() const {
    std::vector<std::shared_ptr<TensorImpl>> out;
    out.reserve(saved_.size());
    for (const auto& entry : saved_) {
        if (!entry.is_output || !entry.data) {
            out.push_back(entry.data);
            continue;
        }
        const auto fn = entry.grad_fn.lock();
        const std::int64_t current_version =
            entry.version_sink ? entry.version_sink->load() : entry.saved_version;
        if (auto live = entry.live_output.lock()) {
            if (live->version() != entry.saved_version || current_version != entry.saved_version ||
                (live->grad_fn() == fn && live->grad_output_nr() == entry.output_nr)) {
                while (live->version() < current_version)
                    live->bump_version();
                out.push_back(std::move(live));
                continue;
            }
        }
        auto rewrapped = TensorImpl::make_view(entry.data, entry.data->shape(),
                                               entry.data->stride(), 0, /*join_family=*/false);
        if (fn) {
            rewrapped->set_grad_fn(fn);
            rewrapped->set_grad_output_nr(entry.output_nr);
            rewrapped->set_leaf(false);
            rewrapped->set_requires_grad(true);
        }
        while (rewrapped->version() < current_version)
            rewrapped->bump_version();
        out.push_back(std::move(rewrapped));
    }
    return out;
}

void FunctionCtx::for_each_owned_tensor(const std::function<void(const TensorImplPtr&)>& fn) const {
    for (const auto& entry : saved_) {
        if (entry.data)
            fn(entry.data);
    }
}

void FunctionCtx::clear() noexcept {
    auto saved = std::move(saved_);
    saved_.clear();
    owner_node_.reset();
    auto ext = std::move(extras_);
    extras_.clear();
    saved.clear();
    for (auto& kv : ext)
        release_py_object(kv.second);
}

PythonBackwardNode::~PythonBackwardNode() {
    slots_ = {};
    release_py_object(py_ctx);
    release_py_object(py_backward_fn);
}

void PythonBackwardNode::release_saved() {
    Node::release_saved();
    slots_ = {};
    // What an outer pass set aside here goes too: that pass finds the node
    // released and never asks for it back.
    passes_ = {};
    release_py_object(py_ctx);
    release_py_object(py_backward_fn);
}

void PythonBackwardNode::for_each_owned_tensor(
    const std::function<void(const TensorImplPtr&)>& fn) const {
    for (const auto& g : slots_.graph) {
        if (g)
            fn(g);
    }
}

void PythonBackwardNode::enter_pass() {
    if (passes_.is_running())
        return;
    Slots empty;
    empty.eager.resize(out_shapes.size());
    empty.graph.resize(out_shapes.size());
    passes_.enter(slots_, std::move(empty));
}

// Invoke the Python backward function and collect the resulting gradients.
//
// Steps:
//   1. Acquire the GIL; the Python interpreter may not be active on the
//      backward thread.
//   2. Wrap grad_out in a temporary TensorImpl so Python can treat it as a
//      normal tensor.  out_shape is used as the shape; if it is empty the
//      shape is inferred from nbytes / element size.
//   3. Call py_backward_fn(py_ctx, grad_tensor).  A Python exception is left
//      to propagate as the py::error_already_set pybind11 throws (see
//      invoke_backward).
//   4. Unpack the result: a tuple or list yields one Storage per item; a
//      single tensor yields one Storage.  Python None entries become empty
//      CpuStorage{} values representing "no gradient for this input"; any
//      other value that is not a tensor raises TypeError (tensor_from_python).
std::vector<Storage> PythonBackwardNode::apply(Storage grad_out) {
    py::gil_scoped_acquire gil;

    if (!py_backward_fn || py_backward_fn.is_none()) {
        ErrorBuilder("PythonBackwardNode::apply").fail("backward function is not set");
    }

    Dtype grad_dt = storage_dtype(grad_out);
    Device grad_dev = storage_device(grad_out);

    // out_shape is always recorded by _register_python_backward_node,
    // including {} for a 0-D scalar output.
    auto grad_impl =
        std::make_shared<TensorImpl>(std::move(grad_out), out_shape, grad_dt, grad_dev, false);

    py::tuple grads(1);
    grads[0] = py::cast(grad_impl);
    return invoke_backward(grads);
}

std::vector<TensorImplPtr> PythonBackwardNode::graph_results(const py::object& result) const {
    std::vector<TensorImplPtr> grads;
    if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result)) {
        for (auto item : result)
            grads.push_back(tensor_from_python(item, kBackwardResult));
    } else {
        grads.push_back(tensor_from_python(result, kBackwardResult));
    }
    return grads;
}

std::vector<TensorImplPtr> PythonBackwardNode::apply_for_graph(const TensorImplPtr& grad_out) {
    py::gil_scoped_acquire gil;
    if (!py_backward_fn || py_backward_fn.is_none())
        ErrorBuilder("PythonBackwardNode::apply_for_graph").fail("backward function is not set");
    if (once_differentiable)
        ErrorBuilder("custom Function")
            .not_implemented("create_graph=True through a Function whose backward is "
                             "once_differentiable — it does not record its own graph");

    // An exception raised by ``backward`` propagates untouched, as in
    // invoke_backward.
    return graph_results(py_backward_fn(py_ctx, py::cast(grad_out)));
}

void PythonBackwardNode::accumulate_barrier_grad(std::uint32_t input_nr, Storage grad) {
    if (saved_released())
        return;
    enter_pass();
    if (input_nr >= slots_.eager.size())
        return;

    auto& slot = slots_.eager[input_nr];
    if (!slot.has_value()) {
        slot = std::move(grad);
        return;
    }
    // The first buffer may be one the engine also routed elsewhere; a CPU
    // add would write into it in place, so sum into a copy this slot owns.
    Storage sum = own_grad_copy(*slot);
    accumulate_into(sum, grad);
    slot = std::move(sum);
}

std::vector<Storage> PythonBackwardNode::apply_barrier() {
    py::gil_scoped_acquire gil;

    if (!py_backward_fn || py_backward_fn.is_none()) {
        ErrorBuilder("PythonBackwardNode::apply_barrier").fail("backward function is not set");
    }
    enter_pass();

    const bool mat = !py_ctx || !py::isinstance<FunctionCtx>(py_ctx) ||
                     py::cast<const FunctionCtx&>(py_ctx).materialize_grads();
    const std::size_t n = out_shapes.size();
    py::tuple grads(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (!slots_.eager[i].has_value() && !mat) {
            grads[i] = py::none();
            continue;
        }
        // An output the loss never used contributes nothing, but ``backward``
        // is still written against a fixed argument list, so give it zeros
        // rather than ``None`` and let the arithmetic take care of itself.
        Storage g = slots_.eager[i].has_value()
                        ? std::move(*slots_.eager[i])
                        : make_zero_storage(out_shapes[i], out_dtypes[i], out_devices[i]);
        grads[i] = py::cast(std::make_shared<TensorImpl>(std::move(g), out_shapes[i], out_dtypes[i],
                                                         out_devices[i], false));
    }
    std::fill(slots_.eager.begin(), slots_.eager.end(), std::nullopt);

    return invoke_backward(grads);
}

void PythonBackwardNode::accumulate_barrier_grad_for_graph(std::uint32_t input_nr,
                                                           TensorImplPtr grad) {
    if (released() || !grad)
        return;
    enter_pass();
    if (input_nr >= slots_.graph.size())
        return;
    auto& slot = slots_.graph[input_nr];
    slot = slot ? add_op(slot, grad) : std::move(grad);
}

std::vector<TensorImplPtr> PythonBackwardNode::apply_barrier_for_graph() {
    py::gil_scoped_acquire gil;
    if (!py_backward_fn || py_backward_fn.is_none())
        ErrorBuilder("PythonBackwardNode::apply_barrier_for_graph")
            .fail("backward function is not set");
    if (once_differentiable)
        ErrorBuilder("custom Function")
            .not_implemented("create_graph=True through a Function whose backward is "
                             "once_differentiable — it does not record its own graph");
    enter_pass();

    const bool mat = !py_ctx || !py::isinstance<FunctionCtx>(py_ctx) ||
                     py::cast<const FunctionCtx&>(py_ctx).materialize_grads();
    const std::size_t n = out_shapes.size();
    py::tuple grads(n);
    for (std::size_t i = 0; i < n; ++i) {
        // An output no gradient reached is handed zeros, as in apply_barrier;
        // they are a constant, so nothing differentiates through them.
        TensorImplPtr g = std::move(slots_.graph[i]);
        if (!g && !mat) {
            grads[i] = py::none();
            continue;
        }
        if (!g)
            g = std::make_shared<TensorImpl>(
                make_zero_storage(out_shapes[i], out_dtypes[i], out_devices[i]), out_shapes[i],
                out_dtypes[i], out_devices[i], false);
        grads[i] = py::cast(std::move(g));
    }
    return graph_results(py_backward_fn(py_ctx, *grads));
}

std::vector<Storage> PythonBackwardNode::invoke_backward(const py::tuple& grads) {
    // An exception raised by ``backward`` is not caught here.  pybind11
    // carries it out as py::error_already_set, which holds the original
    // Python exception — type, message, traceback, ``__cause__``.  Nothing
    // between here and the Python caller catches it (the engine has no
    // handler, and lucid's translator in bind_errors.cpp matches only
    // LucidError types), so the binding's dispatcher restores it and the
    // caller of backward() sees the ValueError or VersionMismatch the user
    // code raised.  Rewrapping it as std::runtime_error, as this did,
    // delivered every such error as a RuntimeError with the type folded
    // into the message.
    //
    // The GIL is held for the whole traversal (engine_backward and
    // engine_grad do not release it), and error_already_set re-acquires it
    // anyway when it is read or destroyed.
    py::object result = py_backward_fn(py_ctx, *grads);

    std::vector<Storage> storages;
    const ErrorBuilder err("custom Function backward");

    // One Storage per returned value: ``None`` — no gradient for that input —
    // is an empty storage, a tensor its buffer.
    auto collect_one = [&](py::handle item) {
        const TensorImplPtr g = tensor_from_python(item, kBackwardResult);
        storages.push_back(g ? present_gradient(grad_slot_buffer(err, *g)) : Storage{CpuStorage{}});
    };

    // Unpack tuple/list returns (multiple inputs) or a single tensor return.
    if (py::isinstance<py::tuple>(result) || py::isinstance<py::list>(result)) {
        for (auto item : result)
            collect_one(item);
    } else {
        collect_one(result);
    }

    return storages;
}

// -------------------------------------------------------------------------
// pybind11 bindings
// -------------------------------------------------------------------------

namespace {

void record_python_output_metas(const std::vector<std::shared_ptr<TensorImpl>>& outputs,
                                PythonBackwardNode& node) {
    // Single-output metadata is kept as-is so that path stays on the engine's
    // direct (non-barrier) route.  The vectors are what make the node a
    // barrier, and they are filled for multi-output calls.
    node.out_shape = outputs.front()->shape();
    node.out_dtype = outputs.front()->dtype();
    node.out_device = outputs.front()->device();

    node.out_shapes.clear();
    node.out_dtypes.clear();
    node.out_devices.clear();
    if (outputs.size() > 1) {
        node.out_shapes.reserve(outputs.size());
        node.out_dtypes.reserve(outputs.size());
        node.out_devices.reserve(outputs.size());
        for (const auto& out : outputs) {
            node.out_shapes.push_back(out->shape());
            node.out_dtypes.push_back(out->dtype());
            node.out_devices.push_back(out->device());
        }
    }
}

void record_python_input_edges(const std::vector<std::shared_ptr<TensorImpl>>& inputs,
                               PythonBackwardNode& node) {
    std::vector<Edge> edges;
    edges.reserve(inputs.size());
    std::vector<std::int64_t> versions;
    versions.reserve(inputs.size());

    for (const auto& inp : inputs) {
        if (!inp || !inp->requires_grad()) {
            edges.emplace_back(nullptr, 0);
            versions.push_back(0);
            continue;
        }
        if (inp->is_leaf() && !inp->grad_fn())
            inp->set_grad_fn(std::make_shared<AccumulateGrad>(inp));
        edges.emplace_back(inp->grad_fn(), inp->grad_output_nr());
        versions.push_back(inp->version());
    }

    node.set_next_edges(std::move(edges));
    node.set_saved_versions(std::move(versions));
}

}  // namespace

void register_custom_function(py::module_& m) {
    // FunctionCtx — Python-visible as lucid._C.engine.FunctionCtx.
    // save_for_backward accepts *args of tensors (TensorImpl or Python
    // wrappers with an .impl attribute).  saved_tensors returns a tuple.
    // Arbitrary attributes are stored/retrieved via __setattr__/__getattr__.
    py::class_<FunctionCtx, std::shared_ptr<FunctionCtx>>(
        m, "FunctionCtx", py::custom_type_setup(enable_function_ctx_gc))
        .def(py::init<>())
        .def(
            "save_for_backward",
            [](FunctionCtx& ctx, py::args tensors) {
                std::vector<std::shared_ptr<TensorImpl>> v;
                v.reserve(tensors.size());
                for (auto t : tensors)
                    v.push_back(tensor_from_python(t, "a value passed to save_for_backward"));
                ctx.save_for_backward(std::move(v));
            },
            "Save tensors for backward.  Mirrors ctx.save_for_backward().")
        .def_property_readonly(
            "saved_tensors",
            [](const FunctionCtx& ctx) -> py::tuple {
                const auto saved = ctx.saved_tensors();
                py::tuple t(saved.size());
                for (std::size_t i = 0; i < saved.size(); ++i)
                    t[i] = py::cast(saved[i]);
                return t;
            },
            "Tuple of tensors saved during forward.")
        .def("__setattr__", [](FunctionCtx& ctx, const std::string& key,
                               py::object val) { ctx.store(key, std::move(val)); })
        .def("__getattr__", [](const FunctionCtx& ctx, const std::string& key) -> py::object {
            auto v = ctx.load(key);
            if (v.is_none())
                throw py::attribute_error(key);
            return v;
        });

    // _PythonBackwardNode — internal type; Python sets ctx and backward_fn
    // fields, then passes the node to _register_python_backward_node.
    py::class_<PythonBackwardNode, Node, std::shared_ptr<PythonBackwardNode>>(
        m, "_PythonBackwardNode", py::custom_type_setup(enable_node_autograd_gc))
        .def(py::init<>())
        .def_readwrite("ctx", &PythonBackwardNode::py_ctx)
        .def_readwrite("backward_fn", &PythonBackwardNode::py_backward_fn)
        .def_readwrite("once_differentiable", &PythonBackwardNode::once_differentiable);

    // _register_python_backward_node(output, node, inputs)
    //
    // Wire node into the autograd graph as the grad_fn of output.
    // For each input in inputs:
    //   - If the input is a leaf requiring grad, create an AccumulateGrad
    //     node for it (if not already present) and record an edge to it.
    //   - If the input does not require grad, record a null edge.
    // Saves the version counters for in-place mutation detection.
    // Marks output as a non-leaf tensor requiring gradients.
    m.def(
        "_register_python_backward_node",
        [](const std::vector<std::shared_ptr<TensorImpl>>& outputs,
           std::shared_ptr<PythonBackwardNode> node,
           const std::vector<std::shared_ptr<TensorImpl>>& inputs) {
            if (!node)
                ErrorBuilder("_register_python_backward_node").fail("null argument");
            if (outputs.empty())
                ErrorBuilder("_register_python_backward_node").fail("no outputs");
            for (const auto& out : outputs) {
                if (!out)
                    ErrorBuilder("_register_python_backward_node").fail("null argument");
            }

            record_python_output_metas(outputs, *node);
            record_python_input_edges(inputs, *node);

            // Every output shares the one node; ``grad_output_nr`` is what
            // tells the engine which slot an arriving gradient belongs to,
            // and it is the index the barrier reads back.
            for (std::size_t i = 0; i < outputs.size(); ++i) {
                outputs[i]->set_grad_fn(node);
                outputs[i]->set_grad_output_nr(static_cast<std::uint32_t>(i));
                outputs[i]->set_leaf(false);
                outputs[i]->set_requires_grad(true);
            }
            if (node->py_ctx && py::isinstance<FunctionCtx>(node->py_ctx)) {
                py::cast<FunctionCtx&>(node->py_ctx).set_owner_node(node);
            }
        },
        py::arg("outputs"), py::arg("node"), py::arg("inputs"),
        "Wire a PythonBackwardNode into the autograd graph for `outputs`.");
}

}  // namespace lucid
