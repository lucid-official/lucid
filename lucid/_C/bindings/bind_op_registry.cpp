// lucid/_C/bindings/bind_op_registry.cpp
//
// Exposes the global OpRegistry and OpSchema to Python for introspection and
// tooling.  Python code can iterate all registered ops, check their AMP policy,
// verify input/output arity, and use `internal` to determine which ops should
// appear in generated stubs.  None of these bindings are required at op
// dispatch time; they are used by BindingGen and test infrastructure.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "../core/AmpPolicy.h"
#include "../core/Dtype.h"
#include "../core/OpRegistry.h"
#include "../core/OpSchema.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

using RefusalView = std::optional<std::pair<std::string, std::string>>;

RefusalView refusal_view(const OpSchema& s, Dtype dt) {
    if (!s.refuses(dt))
        return std::nullopt;
    return std::pair{std::string(s.refusal == DtypeRefusal::Kernel ? "kernel" : "guard"),
                     std::string(cast_hint_text(s.cast))};
}

}  // namespace

// Registers the OpSchema read-only descriptor and the two registry query
// functions: op_lookup and op_registry_all.
void register_op_registry(py::module_& m) {
    // OpSchema contains only compile-time-constant string_views and plain POD
    // fields.  The `name` and `determinism_note` fields are string_view, so
    // they are copied into std::string to avoid dangling references in Python.
    py::class_<OpSchema>(m, "OpSchema")
        .def_property_readonly("name", [](const OpSchema& s) { return std::string(s.name); })
        .def_readonly("version", &OpSchema::version)
        // ``amp_policy`` is what autocast does with the op on Metal, read
        // from the policy table; ``declared_amp_policy`` is the schema's
        // own field, which now only drives integer-input promotion.
        .def_property_readonly(
            "amp_policy", [](const OpSchema& s) { return amp::policy_for(s.name, Device::GPU); })
        .def_property_readonly(
            "amp_policy_cpu",
            [](const OpSchema& s) { return amp::policy_for(s.name, Device::CPU); })
        .def_readonly("declared_amp_policy", &OpSchema::declared_amp_policy)
        .def_readonly("deterministic", &OpSchema::deterministic)
        .def_property_readonly("determinism_note",
                               [](const OpSchema& s) { return std::string(s.determinism_note); })
        .def_readonly("input_arity", &OpSchema::input_arity)
        .def_readonly("output_arity", &OpSchema::output_arity)
        .def_readonly("stable_input_indices", &OpSchema::stable_input_indices)
        // `internal` is true for ops that are implementation details and should
        // not appear in generated .pyi stubs or Python-facing documentation.
        .def_readonly("internal", &OpSchema::internal)
        // ``None`` for an op that accepts a bool; otherwise ``(kind, hint)``:
        // ``kind`` is ``"kernel"`` (the engine raises ``NotImplementedError``) or
        // ``"guard"`` (``DtypeMismatch``), ``hint`` the cast the message offers.
        // The bool column of ``refusal_for``, kept for the callers that only
        // ask about a bool.
        .def_property_readonly("bool_refusal",
                               [](const OpSchema& s) { return refusal_view(s, Dtype::Bool); })
        // Every dtype the op refuses, in dtype-code order.
        .def_property_readonly("refused_dtypes",
                               [](const OpSchema& s) {
                                   std::vector<Dtype> out;
                                   for (unsigned code = 0; code < 16; ++code)
                                       if ((s.refused >> code) & 1u)
                                           out.push_back(static_cast<Dtype>(code));
                                   return out;
                               })
        // ``None`` when the op answers ``dtype``; otherwise the same
        // ``(kind, hint)`` as ``bool_refusal``.
        .def(
            "refusal_for", [](const OpSchema& s, Dtype dtype) { return refusal_view(s, dtype); },
            py::arg("dtype"))
        .def("__repr__", [](const OpSchema& s) {
            return "OpSchema(name='" + std::string(s.name) +
                   "', version=" + std::to_string(s.version) +
                   ", input_arity=" + std::to_string(s.input_arity) + ")";
        });

    // op_lookup returns None if the name is not registered rather than raising
    // so callers can use a simple `if schema := engine.op_lookup(name)` idiom.
    // The returned pointer is borrowed from the static registry; reference
    // policy prevents pybind11 from trying to delete it.
    m.def(
        "op_lookup",
        [](std::string_view name) -> py::object {
            const auto* s = OpRegistry::lookup(name);
            if (!s)
                return py::none();
            return py::cast(s, py::return_value_policy::reference);
        },
        py::arg("name"));

    // op_registry_all returns all registered schemas as a list of borrowed
    // pointers.  The vector is constructed on demand and its elements are
    // pointers into the static registry — do not store them past the extension
    // module's lifetime.
    m.def(
        "op_registry_all",
        []() {
            std::vector<const OpSchema*> all = OpRegistry::all();
            return all;
        },
        py::return_value_policy::reference);
}

}  // namespace lucid::bindings
