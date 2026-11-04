// lucid/_C/bindings/bind_autograd.cpp
//
// Exposes the autograd graph and backward engine to Python.  Registers:
//   - Node / AccumulateGrad — the two graph node types visible from Python
//   - engine_backward() / engine_grad() — one reverse-mode pass from roots
//   - register_custom_function() — wires up lucid.autograd.Function so Python
//     subclasses can define custom forward/backward passes (delegates to
//     CustomFunction.h for the heavy lifting)
//   - _tensor_hook_runner() / _has_tensor_hooks() — the engine side of
//     Tensor.register_hook (autograd/TensorHooks.h)

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <vector>

#include "../autograd/AccumulateGrad.h"
#include "../autograd/CustomFunction.h"
#include "../autograd/Engine.h"
#include "../autograd/Helpers.h"
#include "../autograd/ModuleHookNode.h"
#include "../autograd/Node.h"
#include "../autograd/TensorHooks.h"
#include "../core/ErrorBuilder.h"
#include "../core/TensorImpl.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// The roots of a pass with their seeds, one seed per root.
//
// Raises
// ------
// LucidError
//     The counts differ, or a root is None.
std::vector<BackwardRoot>
backward_roots(const char* op,
               const std::vector<std::shared_ptr<TensorImpl>>& roots,
               const std::vector<std::optional<std::shared_ptr<TensorImpl>>>& seeds) {
    if (roots.size() != seeds.size())
        ErrorBuilder(op).fail("got " + std::to_string(roots.size()) + " roots and " +
                              std::to_string(seeds.size()) + " seeds");
    std::vector<BackwardRoot> out;
    out.reserve(roots.size());
    for (std::size_t i = 0; i < roots.size(); ++i) {
        if (!roots[i])
            ErrorBuilder(op).fail("roots[" + std::to_string(i) + "] is None");
        out.push_back({roots[i], seeds[i].value_or(nullptr)});
    }
    return out;
}

// The graph node types Python sees.
void register_nodes(py::module_& m) {
    // Node is the abstract base for all backward-graph vertices.  It is held
    // as shared_ptr<Node> because backward-graph edges are also shared_ptrs;
    // Python may hold a reference longer than the graph itself.
    py::class_<Node, std::shared_ptr<Node>>(m, "Node",
                                            py::custom_type_setup(enable_node_autograd_gc))
        .def_property_readonly("sequence_nr", &Node::sequence_nr)
        .def_property_readonly(
            "next_functions",
            [](const Node& n) {
                const auto& edges = n.next_edges();
                py::tuple out(edges.size());
                for (std::size_t i = 0; i < edges.size(); ++i) {
                    const auto& e = edges[i];
                    out[i] = py::make_tuple(e.node ? py::cast(e.node) : py::none(),
                                            py::int_(e.input_nr));
                }
                return out;
            },
            "Tuple of ``(grad_fn, output_nr)`` pairs for the upstream autograd nodes.")
        .def(
            "register_prehook",
            [](const std::shared_ptr<Node>& n, py::object hook) {
                return py::module_::import("lucid.autograd._hooks")
                    .attr("_register_node_prehook")(n, std::move(hook));
            },
            py::arg("hook"),
            "Register a backward prehook on this node. Called with ``(grad_outputs,)``.")
        .def(
            "register_hook",
            [](const std::shared_ptr<Node>& n, py::object hook) {
                return py::module_::import("lucid.autograd._hooks")
                    .attr("_register_node_hook")(n, std::move(hook));
            },
            py::arg("hook"),
            "Register a backward hook on this node. Called with ``(grad_inputs, grad_outputs)``.")
        .def("name", &Node::node_name,
             "The operation this node differentiates, e.g. ``\"MulBackward\"``.")
        .def(
            "_owned_tensors",
            [](const Node& n) {
                std::vector<TensorImplPtr> owned;
                n.for_each_owned_tensor([&](const TensorImplPtr& t) { owned.push_back(t); });
                return owned;
            },
            "The tensors this node holds a strong reference to, for the cyclic collector.")
        .def("__repr__", [](const Node& n) {
            return "<lucid.Node " + n.node_name() + " seq=" + std::to_string(n.sequence_nr()) + ">";
        });

    // AccumulateGrad is the leaf-variable sink node; Python code can detect it
    // via isinstance(tensor.grad_fn, engine.AccumulateGrad).
    py::class_<AccumulateGrad, Node, std::shared_ptr<AccumulateGrad>>(
        m, "AccumulateGrad", py::custom_type_setup(enable_node_autograd_gc))
        .def_property_readonly(
            "variable",
            [](const AccumulateGrad& acc) -> py::object {
                const auto leaf = acc.leaf().lock();
                if (!leaf)
                    return py::none();
                return py::module_::import("lucid._dispatch").attr("_wrap")(py::cast(leaf));
            },
            "The leaf Tensor whose gradient this node accumulates, or None if collected.");
}

// The engine's two passes, each from a list of roots and their seeds.
void register_passes(py::module_& m) {
    // engine_backward runs one pass from every root, each seeded with its
    // gradient — ones of its shape where that is None — as the reference
    // runs ``backward([a, b], [ga, gb])``: the graph the roots share is
    // walked once, and the hooks of a tensor in it see its whole gradient.
    m.def(
        "engine_backward",
        [](const std::vector<std::shared_ptr<TensorImpl>>& roots,
           const std::vector<std::optional<std::shared_ptr<TensorImpl>>>& grad_tensors,
           bool retain_graph, bool create_graph) {
            Engine::backward(backward_roots("engine_backward", roots, grad_tensors), retain_graph,
                             create_graph);
        },
        py::arg("roots"), py::arg("grad_tensors"), py::arg("retain_graph") = false,
        py::arg("create_graph") = false,
        "Run one backward pass from `roots`, each seeded with its entry of "
        "`grad_tensors` (None: ones of the root's shape). If create_graph is "
        "True, the backward pass itself is tracked in the autograd graph, "
        "enabling second-order derivatives (MAML, Hessians, etc.).");

    // engine_grad is the functional counterpart: it returns the gradients
    // instead of accumulating them, and touches no tensor's ``.grad`` at
    // all — not the requested inputs', and not any other leaf's.  Emulating
    // it in Python (full backward, then restore ``.grad``) can only restore
    // the tensors the caller named, so every other leaf in the graph keeps
    // whatever the traversal deposited.
    m.def(
        "engine_grad",
        [](const std::vector<std::shared_ptr<TensorImpl>>& roots,
           const std::vector<std::shared_ptr<TensorImpl>>& inputs,
           const std::vector<std::optional<std::shared_ptr<TensorImpl>>>& grad_outputs,
           bool retain_graph, bool create_graph) {
            return Engine::grad(backward_roots("engine_grad", roots, grad_outputs), inputs,
                                retain_graph, create_graph);
        },
        py::arg("roots"), py::arg("inputs"), py::arg("grad_outputs"),
        py::arg("retain_graph") = false, py::arg("create_graph") = false,
        "Gradients of `roots` w.r.t. `inputs` from one pass, each root seeded "
        "with its entry of `grad_outputs` (None: ones), returned rather than "
        "accumulated. Never reads or writes a leaf's .grad (a non-leaf that "
        "asked for retain_grad keeps what reaches it). An entry is None when "
        "that input lies outside the graph.");
}

void register_hook_bindings(py::module_& m) {
    // The engine half of ``Tensor.register_hook``.  A tensor's hooks run where
    // its gradient is complete — its producer's output slot, or a leaf's
    // accumulation — through one Python runner per slot, which Python
    // installs here and keeps the hook list in.
    m.def(
        "_tensor_hook_runner",
        [](const std::shared_ptr<TensorImpl>& t, const py::object& make) {
            return lucid::tensor_hook_runner(t, make);
        },
        py::arg("tensor"), py::arg("make"),
        "The runner of `tensor`'s gradient hooks, installed from `make()` the "
        "first time. A leaf's runner is its own; a non-leaf's sits on its "
        "producer's output slot, so every consumer's gradient goes through it. "
        "The engine calls `runner(grad)` once the slot's gradient is complete, "
        "with grad mode on only under create_graph; a returned tensor replaces "
        "the gradient, None keeps it. Raises if `tensor` does not require grad.");

    m.def(
        "_has_tensor_hooks",
        [](const std::shared_ptr<TensorImpl>& t) { return t && lucid::has_tensor_hooks(*t); },
        py::arg("tensor"), "Whether a hook runner is installed on `tensor`'s gradient slot.");

    m.def(
        "_tensor_post_accumulate_grad_hook_runner",
        [](const std::shared_ptr<TensorImpl>& t, const py::object& make) {
            return lucid::tensor_post_accumulate_grad_hook_runner(t, make);
        },
        py::arg("tensor"), py::arg("make"),
        "The runner of `tensor`'s post-accumulate-grad hooks, installed from `make()` the "
        "first time.");

    m.def(
        "_has_post_accumulate_grad_hooks",
        [](const std::shared_ptr<TensorImpl>& t) {
            return t && lucid::has_post_accumulate_grad_hooks(*t);
        },
        py::arg("tensor"), "Whether a post-accumulate-grad hook runner is installed on `tensor`.");

    m.def(
        "_run_post_accumulate_grad_hooks",
        [](const std::shared_ptr<TensorImpl>& t, bool graph) {
            if (t)
                lucid::run_leaf_post_hooks(*t, graph);
        },
        py::arg("tensor"), py::arg("graph") = false,
        "Run `tensor`'s registered post-accumulate-grad hooks.");

    m.def(
        "_node_prehook_runner",
        [](const std::shared_ptr<Node>& n, const py::object& make) {
            if (!n)
                ErrorBuilder("Node.register_prehook").invalid_argument("expected a node");
            return lucid::node_prehook_runner(*n, make);
        },
        py::arg("node"), py::arg("make"),
        "The runner of `node`'s backward prehooks, installed from `make()` the first time.");

    m.def(
        "_node_hook_runner",
        [](const std::shared_ptr<Node>& n, const py::object& make) {
            if (!n)
                ErrorBuilder("Node.register_hook").invalid_argument("expected a node");
            return lucid::node_hook_runner(*n, make);
        },
        py::arg("node"), py::arg("make"),
        "The runner of `node`'s backward hooks, installed from `make()` the first time.");

    m.def("_push_saved_tensors_hooks", &lucid::push_saved_tensors_hooks, py::arg("pack_hook"),
          py::arg("unpack_hook"), "Push a (pack_hook, unpack_hook) pair for built-in ops.");
    m.def("_pop_saved_tensors_hooks", &lucid::pop_saved_tensors_hooks,
          "Pop the active (pack_hook, unpack_hook) pair for built-in ops.");
}

}  // namespace

// Registers the backward engine, autograd node types, and the custom-function
// registration helper.
void register_autograd(py::module_& m) {
    register_nodes(m);
    register_passes(m);

    m.def(
        "_accumulate_grad",
        [](const std::shared_ptr<TensorImpl>& leaf, const std::shared_ptr<TensorImpl>& grad) {
            if (!leaf)
                ErrorBuilder("_accumulate_grad").fail("leaf is null");
            if (!grad)
                ErrorBuilder("_accumulate_grad").fail("grad is null");
            leaf->accumulate_grad_impl(grad);
        },
        py::arg("leaf"), py::arg("grad"),
        "Accumulate a gradient TensorImpl into `leaf`'s graph-mode .grad slot.");

    register_hook_bindings(m);

    // register_custom_function installs the Python-side CustomFunction class
    // and the _register_python_backward_node() hook used by lucid.autograd.Function.
    lucid::register_custom_function(m);
    lucid::register_module_hook_nodes(m);

    // Process-wide opt-in flag for ``check_version_match`` — mirrored as
    // ``lucid.autograd.graph.allow_mutation_on_saved_tensors`` on the Python
    // side.  The user takes responsibility for not corrupting saved
    // activations when this is enabled.
    m.def("is_mutation_on_saved_allowed", &lucid::is_mutation_on_saved_allowed,
          "Return whether the version-mismatch check is currently disabled.");
    m.def("set_mutation_on_saved_allowed", &lucid::set_mutation_on_saved_allowed, py::arg("allow"),
          "Enable or disable the autograd version-mismatch check.  When "
          "True, in-place mutations of saved tensors no longer raise "
          "VersionMismatch — gradients become the user's responsibility.");

    // Process-wide flag for ``lucid.autograd.detect_anomaly`` /
    // ``lucid.autograd.set_detect_anomaly``.
    m.def("is_anomaly_enabled", &lucid::is_anomaly_enabled,
          "Return whether autograd anomaly detection (NaN gradient checking) is enabled.");
    m.def("set_anomaly_enabled", &lucid::set_anomaly_enabled, py::arg("enabled"),
          "Enable or disable autograd anomaly detection (NaN gradient checking) in the engine.");
}

}  // namespace lucid::bindings
