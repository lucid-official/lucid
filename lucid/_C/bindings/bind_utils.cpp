// lucid/_C/bindings/bind_utils.cpp
//
// Registers shape manipulation, indexing, sorting, and histogram utilities on
// the top-level engine module.  These ops do not fit the uniform unary/binary
// patterns in bind_ufunc.cpp and bind_bfunc.cpp because they have heterogeneous
// signatures.  Groups:
//   1. View / shape: reshape, squeeze, squeeze_all, unsqueeze, contiguous,
//      flatten, broadcast_to, expand, expand_dims, ravel.
//   2. Concatenation / splitting: concatenate, stack, hstack, vstack, split,
//      split_at, chunk, unbind.
//   3. Tiling / padding: repeat, tile, pad.
//   4. Triangular masking: tril, triu.
//   5. Conditional selection: where, masked_fill, roll, gather, diagonal.
//   6. Sorting and indexing: sort, argsort, argmax, argmin, nonzero, unique,
//      topk (returns (values, indices) tuple).
//   7. Meshgrid.
//   8. Histogram ops: histogram, histogram2d, histogramdd.  Each returns a
//      (counts, edges) tuple because pybind11 cannot marshal std::array<> of
//      TensorImplPtr as a Python tuple automatically.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>

#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "../ops/utils/Concat.h"
#include "../ops/utils/Contiguous.h"
#include "../ops/utils/Histogram.h"
#include "../ops/utils/Layout.h"
#include "../ops/utils/Meshgrid.h"
#include "../ops/utils/Nextafter.h"
#include "../ops/utils/Pad.h"
#include "../ops/utils/Repeat.h"
#include "../ops/utils/Select.h"
#include "../ops/utils/Sort.h"
#include "../ops/utils/Tri.h"
#include "../ops/utils/View.h"
#include "BindingGen.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// Converts a Python list of int64 dimensions to the internal Shape typedef.
Shape vec_to_shape(const std::vector<std::int64_t>& v) {
    return Shape(v.begin(), v.end());
}

}  // namespace

// Registers all shape-manipulation, indexing, sorting, and histogram ops.
void register_utils(py::module_& m) {
    // View / shape ops.  contiguous is registered via bind_unary<> because its
    // backward node follows the standard unary schema.
    m.def("reshape", &reshape_op, py::arg("a"), py::arg("new_shape"),
          "Reshape — total numel must match. -1 wildcard supported.");
    m.def("squeeze", &squeeze_op, py::arg("a"), py::arg("dim"),
          "Remove a single size-1 dim at `dim`.");
    m.def("squeeze_all", &squeeze_all_op, py::arg("a"), "Remove all size-1 dims.");
    m.def("unsqueeze", &unsqueeze_op, py::arg("a"), py::arg("dim"),
          "Insert a size-1 dim at position `dim`.");
    bind_unary<ContiguousBackward>(m, &contiguous_op, "Force a contiguous copy.");

    // Concatenation and splitting.
    m.def("concatenate", &concatenate_op, py::arg("arrays"), py::arg("dim") = 0);
    m.def("stack", &stack_op, py::arg("arrays"), py::arg("dim") = 0);
    m.def("hstack", &hstack_op, py::arg("arrays"));
    m.def("vstack", &vstack_op, py::arg("arrays"));
    m.def("split", &split_op, py::arg("a"), py::arg("num_splits"), py::arg("dim") = 0);
    // split_at takes a Python list of indices; std::move transfers ownership
    // into the C++ op to avoid an extra copy.
    m.def(
        "split_at",
        [](const TensorImplPtr& a, std::vector<std::int64_t> indices, int dim) {
            return split_at_op(a, std::move(indices), dim);
        },
        py::arg("a"), py::arg("indices"), py::arg("dim") = 0);
    m.def("chunk", &chunk_op, py::arg("a"), py::arg("chunks"), py::arg("dim") = 0);
    m.def("unbind", &unbind_op, py::arg("a"), py::arg("dim") = 0);

    // Tiling and padding.
    m.def("repeat", &repeat_op, py::arg("a"), py::arg("repeats"), py::arg("dim") = 0);
    // tile accepts a Python list of per-axis repetition counts.
    m.def(
        "tile",
        [](const TensorImplPtr& a, std::vector<std::int64_t> reps) {
            return tile_op(a, std::move(reps));
        },
        py::arg("a"), py::arg("reps"));

    // pad accepts a list of (before, after) pairs — one per axis.
    m.def(
        "pad",
        [](const TensorImplPtr& a, std::vector<std::pair<std::int64_t, std::int64_t>> pad_width,
           double constant) { return pad_op(a, std::move(pad_width), constant); },
        py::arg("a"), py::arg("pad_width"), py::arg("constant") = 0.0);

    m.def("flatten", &flatten_op, py::arg("a"), py::arg("start_axis") = 0,
          py::arg("end_axis") = -1);
    // broadcast_to and expand take a Python int-list; vec_to_shape converts to
    // the internal Shape before calling the C++ op.
    m.def(
        "broadcast_to",
        [](const TensorImplPtr& a, std::vector<std::int64_t> shape) {
            return broadcast_to_op(a, vec_to_shape(shape));
        },
        py::arg("a"), py::arg("shape"));
    m.def(
        "expand",
        [](const TensorImplPtr& a, std::vector<std::int64_t> shape) {
            return expand_op(a, vec_to_shape(shape));
        },
        py::arg("a"), py::arg("shape"));
    m.def(
        "as_strided",
        [](const TensorImplPtr& a, std::vector<std::int64_t> size, std::vector<std::int64_t> stride,
           std::optional<std::int64_t> storage_offset) {
            return as_strided_op(a, vec_to_shape(size), stride, storage_offset.value_or(-1));
        },
        py::arg("a"), py::arg("size"), py::arg("stride"), py::arg("storage_offset") = py::none());

    // Triangular masking.  k is the diagonal offset (0 = main diagonal).
    m.def("tril", &tril_op, py::arg("a"), py::arg("k") = 0);
    m.def("triu", &triu_op, py::arg("a"), py::arg("k") = 0);

    // Conditional selection, indexing, and scatter.
    m.def("where", &where_op, py::arg("cond"), py::arg("x"), py::arg("y"));
    m.def("masked_fill", &masked_fill_op, py::arg("a"), py::arg("mask"), py::arg("value"));
    // roll and split_at take Python lists; std::move avoids copying the vectors.
    m.def(
        "roll",
        [](const TensorImplPtr& a, std::vector<std::int64_t> shifts, std::vector<int> dims) {
            return roll_op(a, std::move(shifts), std::move(dims));
        },
        py::arg("a"), py::arg("shifts"), py::arg("dims"));
    m.def("gather", &gather_op, py::arg("a"), py::arg("indices"), py::arg("dim") = -1);
    m.def("diagonal", &diagonal_op, py::arg("a"), py::arg("offset") = 0, py::arg("dim1") = -2,
          py::arg("dim2") = -1);
    m.def("flip", &flip_op, py::arg("a"), py::arg("dims"),
          "Reverse tensor along the given dims. CPU: loop copy. GPU: take with reversed indices.");
    m.def("masked_select", &masked_select_op, py::arg("a"), py::arg("mask"),
          "Boolean masked selection: returns 1-D tensor of elements where mask==True.");

    // Sorting and indexing.
    m.def("sort", &sort_op, py::arg("a"), py::arg("dim") = -1);
    m.def("argsort", &argsort_op, py::arg("a"), py::arg("dim") = -1);
    m.def("argmax", &argmax_op, py::arg("a"), py::arg("dim") = -1, py::arg("keepdims") = false);
    m.def("argmin", &argmin_op, py::arg("a"), py::arg("dim") = -1, py::arg("keepdims") = false);
    m.def("nonzero", &nonzero_op, py::arg("a"));
    m.def("unique", &unique_op, py::arg("a"));

    // topk returns a (values, indices) tuple; the C++ op returns a 2-element
    // vector which is unpacked here so Python sees a native tuple.
    m.def(
        "topk",
        [](const TensorImplPtr& a, std::int64_t k, int dim) {
            auto res = topk_op(a, k, dim);
            return py::make_tuple(res[0], res[1]);
        },
        py::arg("a"), py::arg("k"), py::arg("dim") = -1);

    m.def("meshgrid", &meshgrid_op, py::arg("arrays"), py::arg("indexing_xy") = false);

    // Convenience aliases.
    m.def("expand_dims", &unsqueeze_op, py::arg("a"), py::arg("dim"),
          "Insert a size-1 dim at position `dim` (alias for unsqueeze).");
    // ravel is implemented as a reshape to [-1] rather than a separate C++ op.
    m.def(
        "ravel",
        [](const TensorImplPtr& a) {
            return reshape_op(a, {a ? static_cast<std::int64_t>(a->numel()) : -1});
        },
        py::arg("a"), "Flatten to 1-D (alias for reshape(a, [-1])).");

    // Histogram2d returns (counts_tensor, edges_a_tensor, edges_b_tensor).
    m.def(
        "histogram2d",
        [](const TensorImplPtr& a, const TensorImplPtr& b, std::int64_t bins_a, std::int64_t bins_b,
           double lo_a, double hi_a, double lo_b, double hi_b, bool density) {
            auto out = histogram2d_op(a, b, bins_a, bins_b, lo_a, hi_a, lo_b, hi_b, density);
            return py::make_tuple(out.at(0), out.at(1));
        },
        py::arg("a"), py::arg("b"), py::arg("bins_a"), py::arg("bins_b"), py::arg("lo_a"),
        py::arg("hi_a"), py::arg("lo_b"), py::arg("hi_b"), py::arg("density") = false);

    m.def(
        "histogramdd",
        [](const TensorImplPtr& a, std::vector<std::int64_t> bins,
           std::vector<std::pair<double, double>> ranges, bool density) {
            auto out = histogramdd_op(a, std::move(bins), std::move(ranges), density);
            return py::make_tuple(out.at(0), out.at(1));
        },
        py::arg("a"), py::arg("bins"), py::arg("ranges"), py::arg("density") = false);

    // IEEE-754 next representable float — CPU-only, GPU inputs round-trip.
    m.def("nextafter", &nextafter_op, py::arg("a"), py::arg("b"));
}

}  // namespace lucid::bindings
