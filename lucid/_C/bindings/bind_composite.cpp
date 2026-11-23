// lucid/_C/bindings/bind_composite.cpp
//
// Registers ops from ``ops/composite/`` plus a handful of pure aliases for
// existing primitives that share an engine implementation under a different
// public name (e.g. ``eq → equal``).  Aliases live here
// rather than in their owning bind_*.cpp because they are part of the same
// "convenience surface" the composite ops expose.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/TensorImpl.h"
#include "../ops/bfunc/Compare.h"
#include "../ops/bfunc/Floordiv.h"
#include "../ops/bfunc/Matmul.h"
#include "../ops/composite/Indexing.h"
#include "../ops/composite/Layout.h"
#include "../ops/composite/Logical.h"
#include "../ops/composite/Math.h"
#include "../ops/composite/Matrix.h"
#include "../ops/composite/Reductions.h"
#include "../ops/composite/Search.h"
#include "../ops/composite/Stats.h"
#include "../ops/ufunc/Discrete.h"
#include "../ops/ufunc/Trig.h"
#include "../ops/utils/Concat.h"
#include "../ops/utils/Nextafter.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/View.h"

namespace py = pybind11;

namespace lucid::bindings {

// Register every composite op plus the aliases listed above.
void register_composite(py::module_& m) {
    // ── elementwise math compositions ───────────────────────────────────────
    m.def("log10", &log10_op, py::arg("a"));
    m.def("log1p", &log1p_op, py::arg("a"));
    m.def("exp2", &exp2_op, py::arg("a"));
    m.def("trunc", &trunc_op, py::arg("a"));
    m.def("frac", &frac_op, py::arg("a"));

    m.def("atan2", &atan2_op, py::arg("y"), py::arg("x"));
    m.def("fmod", &fmod_op, py::arg("a"), py::arg("b"));
    m.def("remainder", &remainder_op, py::arg("a"), py::arg("b"));
    m.def("hypot", &hypot_op, py::arg("a"), py::arg("b"));
    m.def("logaddexp", &logaddexp_op, py::arg("a"), py::arg("b"));
    m.def("isclose", &isclose_op, py::arg("a"), py::arg("b"), py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8);

    // ── reductions ──────────────────────────────────────────────────────────
    m.def("logsumexp", &logsumexp_op, py::arg("a"), py::arg("dim"), py::arg("keepdims"));

    // ── linear algebra ──────────────────────────────────────────────────────
    m.def("mm", &mm_op, py::arg("a"), py::arg("b"));
    m.def("bmm", &bmm_op, py::arg("a"), py::arg("b"));
    m.def("kron", &kron_op, py::arg("a"), py::arg("b"));

    // ── logical ─────────────────────────────────────────────────────────────
    m.def("logical_and", &logical_and_op, py::arg("a"), py::arg("b"));
    m.def("logical_or", &logical_or_op, py::arg("a"), py::arg("b"));
    m.def("logical_xor", &logical_xor_op, py::arg("a"), py::arg("b"));
    m.def("logical_not", &logical_not_op, py::arg("a"));

    // ── indexing ────────────────────────────────────────────────────────────
    m.def("take", &take_op, py::arg("a"), py::arg("indices"));
    m.def("index_select", &index_select_op, py::arg("a"), py::arg("dim"), py::arg("indices"));
    m.def("narrow", &narrow_op, py::arg("a"), py::arg("dim"), py::arg("start"), py::arg("length"));
    m.def("scatter", &scatter_op, py::arg("base"), py::arg("dim"), py::arg("indices"),
          py::arg("src"));
    m.def("kthvalue", &kthvalue_op, py::arg("a"), py::arg("k"), py::arg("dim"), py::arg("keepdim"));

    // ── layout ──────────────────────────────────────────────────────────────
    m.def("movedim", &movedim_op, py::arg("a"), py::arg("source"), py::arg("destination"));
    m.def("unflatten", &unflatten_op, py::arg("a"), py::arg("dim"), py::arg("sizes"));

    // ── stats / combinatorial ───────────────────────────────────────────────
    m.def("histc", &histc_op, py::arg("a"), py::arg("bins"), py::arg("lo"), py::arg("hi"));
    m.def("cartesian_prod", &cartesian_prod_op, py::arg("tensors"));

    // ── sorted-array search ─────────────────────────────────────────────────
    m.def("searchsorted", &searchsorted_op, py::arg("sorted_1d"), py::arg("values"),
          py::arg("right") = false);
    m.def("bucketize", &bucketize_op, py::arg("values"), py::arg("boundaries"),
          py::arg("right") = false);

    // ── pure aliases of existing primitives ─────────────────────────────────
    // Comparison: short reference-framework-style names alongside the long
    // forms already bound by ``register_bfunc``.
    m.def("eq", &equal_op, py::arg("a"), py::arg("b"));
    m.def("ne", &not_equal_op, py::arg("a"), py::arg("b"));
    m.def("lt", &less_op, py::arg("a"), py::arg("b"));
    m.def("le", &less_equal_op, py::arg("a"), py::arg("b"));
    m.def("gt", &greater_op, py::arg("a"), py::arg("b"));
    m.def("ge", &greater_equal_op, py::arg("a"), py::arg("b"));

    // Trigonometric short names — share kernels with arcsin/arccos/arctan.
    m.def("asin", &arcsin_op, py::arg("a"));
    m.def("acos", &arccos_op, py::arg("a"));
    m.def("atan", &arctan_op, py::arg("a"));

    // Bitwise NOT alias — engine kernel is named ``invert``.
    m.def("bitwise_not", &invert_op, py::arg("a"));

    // Shape aliases.
    // ``view`` is reshape that refuses to copy (view_op).
    m.def(
        "view", [](const TensorImplPtr& a, const Shape& shape) { return view_op(a, shape); },
        py::arg("a"), py::arg("shape"));
    m.def("_bitcast", &bitcast_op, py::arg("a"), py::arg("dtype"));
    m.def("fma", &fma_op, py::arg("a"), py::arg("b"), py::arg("c"));

    // Top-level ``cross`` and ``norm`` are aliases for the linalg sub-module
    // entries.  They live in Python (lucid.linalg.{cross, norm}); the Python
    // ops registry installs the top-level forwarders so we don't need a
    // duplicate C++ binding here.
}

}  // namespace lucid::bindings
