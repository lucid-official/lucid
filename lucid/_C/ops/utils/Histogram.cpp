// lucid/_C/ops/utils/Histogram.cpp
//
// Implements histogram_op, histogram2d_op, and histogramdd_op.  All three
// functions count data points into fixed-width bins and optionally normalise
// the counts to form a probability density.  Computation is always performed
// on the CPU; GPU inputs are transferred to host memory before counting
// because the random-access accumulation pattern does not vectorise well on
// the MLX compute graph.  F64 outputs remain on CPU (MLX has no float64
// support); other dtypes are transferred back to the original device after
// counting completes.

#include "Histogram.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <variant>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "_Detail.h"

namespace lucid {

namespace {

using utils_detail::allocate_cpu;
using utils_detail::fresh;

// Transfer `a` to host memory, regardless of its current device.
CpuStorage to_cpu(const TensorImplPtr& a) {
    return backend::Dispatcher::for_device(a->device()).to_cpu(a->storage(), a->shape());
}

// Move a CPU buffer to `target_device`, unless the dtype is F64 (which MLX
// does not support) in which case the storage remains on CPU.
Storage to_device_storage(CpuStorage&& cpu, Device target_device, const Shape& shape) {
    if (target_device == Device::GPU && cpu.dtype != Dtype::F64) {
        return backend::Dispatcher::for_device(Device::GPU).from_cpu(cpu, shape);
    }
    return Storage{std::move(cpu)};
}

// Determine the actual output device.  F64 outputs must stay on CPU because
// MLX (the GPU backend) does not support 64-bit floating-point buffers.
Device pick_out_device(Device requested, Dtype dt) {
    if (requested == Device::GPU && dt == Dtype::F64)
        return Device::CPU;
    return requested;
}

// Read the i-th scalar element from a CpuStorage as a double, casting from
// whatever element dtype the storage holds.  Used by histogram2d and
// histogramdd where the loop-level type must be uniform (double) regardless
// of the input tensor's dtype.
double read_double(const CpuStorage& s, std::size_t i, Dtype dt) {
    switch (dt) {
    case Dtype::F16:
        return backend::detail::half_bits_to_float(
            reinterpret_cast<const std::uint16_t*>(s.ptr.get())[i]);
    case Dtype::BF16:
        return backend::detail::bfloat16_bits_to_float(
            reinterpret_cast<const std::uint16_t*>(s.ptr.get())[i]);
    case Dtype::F32:
        return reinterpret_cast<const float*>(s.ptr.get())[i];
    case Dtype::F64:
        return reinterpret_cast<const double*>(s.ptr.get())[i];
    case Dtype::I8:
        return reinterpret_cast<const std::int8_t*>(s.ptr.get())[i];
    case Dtype::I16:
        return reinterpret_cast<const std::int16_t*>(s.ptr.get())[i];
    case Dtype::I32:
        return reinterpret_cast<const std::int32_t*>(s.ptr.get())[i];
    case Dtype::I64:
        return reinterpret_cast<const std::int64_t*>(s.ptr.get())[i];
    default:
        ErrorBuilder("histogram").not_implemented("dtype not supported");
    }
}

// Allocate a 1-D F64 tensor of shape (bins+1,) containing uniformly spaced
// bin boundary values from `lo` to `hi` inclusive.  The last entry is forced
// to exactly `hi` to avoid floating-point accumulation drift at the right edge.
TensorImplPtr build_edges(double lo, double hi, std::int64_t bins) {
    Shape sh{bins + 1};
    auto cpu = allocate_cpu(sh, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(cpu.ptr.get());
    const double step = (hi - lo) / static_cast<double>(bins);
    for (std::int64_t i = 0; i <= bins; ++i)
        dst[i] = lo + static_cast<double>(i) * step;
    // Assign the last edge exactly to avoid rounding errors that could leave
    // the rightmost boundary slightly below `hi`.
    dst[bins] = hi;
    return fresh(Storage{std::move(cpu)}, std::move(sh), Dtype::F64, Device::CPU);
}

// The bin width along each axis of ``histogramdd``, once the counts and the
// ranges have been checked.  A zero count left the counts tensor without a
// buffer, and the first sample in range was added through a null pointer.
std::vector<double> histogramdd_steps(const std::vector<std::int64_t>& bins,
                                      const std::vector<std::pair<double, double>>& ranges) {
    std::vector<double> step(bins.size());
    for (std::size_t d = 0; d < bins.size(); ++d) {
        if (bins[d] <= 0)
            ErrorBuilder("histogramdd").invalid_argument("bins must be > 0");
        if (ranges[d].second <= ranges[d].first)
            ErrorBuilder("histogramdd").invalid_argument("each range hi must be > lo");
        step[d] = (ranges[d].second - ranges[d].first) / static_cast<double>(bins[d]);
    }
    return step;
}

}  // namespace

// Count elements of `a` into `bins` uniform bins over [lo, hi).  Delegates
// the per-element bucketing to the CPU backend's histogram_forward kernel.
// After counting, the edge array is constructed via build_edges and both
// outputs are optionally transferred back to the input device (except when
// dtype is F64, which must remain on CPU).
std::vector<TensorImplPtr>
histogram_op(const TensorImplPtr& a, std::int64_t bins, double lo, double hi, bool density) {
    Validator::input(a, "histogram.a").non_null();
    if (bins <= 0)
        ErrorBuilder("histogram").invalid_argument("bins must be > 0");
    if (hi <= lo)
        ErrorBuilder("histogram").invalid_argument("hi must be > lo");
    OpScopeFull scope{"histogram", a->device(), a->dtype(), a->shape()};

    const auto cpu = to_cpu(a);
    auto& cpu_be = backend::Dispatcher::for_device(Device::CPU);
    Storage counts_storage =
        cpu_be.histogram_forward(Storage{cpu}, a->shape(), a->dtype(), lo, hi, bins, density);
    Shape counts_shape{bins};

    Device out_dev = pick_out_device(a->device(), Dtype::F64);

    auto final_counts_storage =
        to_device_storage(std::move(storage_cpu(counts_storage)), out_dev, counts_shape);
    auto counts_t =
        fresh(std::move(final_counts_storage), std::move(counts_shape), Dtype::F64, out_dev);
    // Recorded here, immediately after ``counts_t`` exists and *before*
    // the edges are built.  ``on_op_io`` writes to the most recently
    // entered node, and ``build_edges`` traces ops of its own — calling
    // this after it would hand the histogram's I/O to the edge builder
    // and leave this node empty, which is what froze the result into
    // the compiled graph as a trace-time feed.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, counts_t);
    auto edges = build_edges(lo, hi, bins);

    // If the output device is GPU (and dtype is not F64) transfer the edge
    // tensor to the GPU so both outputs live on the same device.
    if (out_dev == Device::GPU) {
        edges = fresh(backend::Dispatcher::for_device(Device::GPU)
                          .from_cpu(storage_cpu(edges->storage()), edges->shape()),
                      edges->shape(), Dtype::F64, Device::GPU);
    }
    return {counts_t, edges};
}

// Joint 2-D histogram of paired observations (a[i], b[i]).  Both inputs must
// have identical shapes (they are treated as flat parallel arrays of N samples).
// The count tensor has shape (bins_a, bins_b); element [ia, ib] accumulates
// points whose x-value falls in x-bin ia and whose y-value falls in y-bin ib.
// The edge tensor concatenates the x-edges (length bins_a+1) followed by the
// y-edges (length bins_b+1) into a single 1-D output of length
// (bins_a + bins_b + 2) for compact return.
// Samples outside either range are silently skipped.  Elements that land
// exactly on the right boundary are clamped to the last bin.
std::vector<TensorImplPtr> histogram2d_op(const TensorImplPtr& a,
                                          const TensorImplPtr& b,
                                          std::int64_t bins_a,
                                          std::int64_t bins_b,
                                          double lo_a,
                                          double hi_a,
                                          double lo_b,
                                          double hi_b,
                                          bool density) {
    if (!a || !b)
        ErrorBuilder("histogram2d").fail("null input");
    if (a->shape() != b->shape())
        throw ShapeMismatch(a->shape(), b->shape(), "histogram2d");
    if (bins_a <= 0 || bins_b <= 0)
        ErrorBuilder("histogram2d").invalid_argument("bins must be > 0");
    OpScopeFull scope{"histogram2d", a->device(), a->dtype(), a->shape()};

    const auto ca = to_cpu(a);
    const auto cb = to_cpu(b);
    const std::size_t n = shape_numel(a->shape());
    const double step_a = (hi_a - lo_a) / static_cast<double>(bins_a);
    const double step_b = (hi_b - lo_b) / static_cast<double>(bins_b);

    Shape counts_shape{bins_a, bins_b};
    auto counts = allocate_cpu(counts_shape, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);

    for (std::size_t i = 0; i < n; ++i) {
        const double va = read_double(ca, i, a->dtype());
        const double vb = read_double(cb, i, b->dtype());
        if (va < lo_a || va > hi_a)
            continue;
        if (vb < lo_b || vb > hi_b)
            continue;
        std::int64_t ia = static_cast<std::int64_t>((va - lo_a) / step_a);
        std::int64_t ib = static_cast<std::int64_t>((vb - lo_b) / step_b);
        // Clamp to last bin: a value exactly equal to the right boundary would
        // otherwise map to an out-of-range index one past the last bin.
        if (ia >= bins_a)
            ia = bins_a - 1;
        if (ib >= bins_b)
            ib = bins_b - 1;
        dst[ia * bins_b + ib] += 1.0;
    }

    if (density) {
        // Normalise so that integrating over all cells gives 1:
        // density[i,j] = count[i,j] / (step_a * step_b * N).
        const double area = step_a * step_b * n;
        for (std::size_t i = 0; i < (std::size_t)(bins_a * bins_b); ++i)
            dst[i] /= area;
    }

    Device out_dev = pick_out_device(a->device(), Dtype::F64);
    auto counts_storage = to_device_storage(std::move(counts), out_dev, counts_shape);
    auto counts_t = fresh(std::move(counts_storage), std::move(counts_shape), Dtype::F64, out_dev);
    // See ``histogram_op``: recorded before the edges are built, because
    // ``on_op_io`` writes to the most recently entered node and
    // ``build_edges`` traces ops of its own.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a, b}, counts_t);

    // Build independent edge arrays for each axis, then pack them into a
    // single concatenated 1-D edge tensor to match the declared return layout.
    auto ea = build_edges(lo_a, hi_a, bins_a);
    auto eb = build_edges(lo_b, hi_b, bins_b);
    Shape edge_shape{bins_a + 1 + bins_b + 1};
    auto edges = allocate_cpu(edge_shape, Dtype::F64);
    auto* edst = reinterpret_cast<double*>(edges.ptr.get());
    std::memcpy(edst, storage_cpu(ea->storage()).ptr.get(), (bins_a + 1) * sizeof(double));
    std::memcpy(edst + (bins_a + 1), storage_cpu(eb->storage()).ptr.get(),
                (bins_b + 1) * sizeof(double));
    auto edges_storage = to_device_storage(std::move(edges), out_dev, edge_shape);
    auto edges_t = fresh(std::move(edges_storage), std::move(edge_shape), Dtype::F64, out_dev);
    return {counts_t, edges_t};
}

// N-dimensional histogram from the rows of a 2-D input matrix of shape (N, D).
// Each row `a[i, :]` is a D-dimensional data point.  The output counts tensor
// has shape equal to `bins` (one entry per dimension).  The output edge tensor
// is a 1-D concatenation of per-dimension edge arrays (each of length bins[d]+1).
//
// The flat index for a data point is computed using a pre-computed row-major
// stride vector so that multi-dimensional bin addressing does not require
// nested indexing.  Points outside any dimension's range are silently skipped.
// Values exactly on the right boundary are clamped to the last bin.
std::vector<TensorImplPtr> histogramdd_op(const TensorImplPtr& a,
                                          std::vector<std::int64_t> bins,
                                          std::vector<std::pair<double, double>> ranges,
                                          bool density) {
    Validator::input(a, "histogramdd.a").non_null();
    if (a->shape().size() != 2)
        ErrorBuilder("histogramdd").fail("input must be 2-D (N, D)");
    const std::int64_t N = a->shape()[0];
    const std::int64_t D = a->shape()[1];
    if ((std::int64_t)bins.size() != D || (std::int64_t)ranges.size() != D)
        ErrorBuilder("histogramdd").fail("bins/ranges length must equal D");
    OpScopeFull scope{"histogramdd", a->device(), a->dtype(), a->shape()};

    const auto ca = to_cpu(a);

    const std::vector<double> step = histogramdd_steps(bins, ranges);

    Shape counts_shape(bins.begin(), bins.end());
    auto counts = allocate_cpu(counts_shape, Dtype::F64);
    auto* dst = reinterpret_cast<double*>(counts.ptr.get());
    std::memset(dst, 0, counts.nbytes);

    // Row-major strides over the counts tensor so that the flat index for a
    // multi-dimensional bin address can be accumulated in a single pass.
    Stride stride(D);
    if (D > 0) {
        stride.back() = 1;
        for (std::ptrdiff_t d = (std::ptrdiff_t)D - 2; d >= 0; --d)
            stride[d] = stride[d + 1] * bins[d + 1];
    }

    for (std::int64_t i = 0; i < N; ++i) {
        std::size_t flat = 0;
        bool inside = true;
        for (std::int64_t d = 0; d < D; ++d) {
            const double v = read_double(ca, i * D + d, a->dtype());
            if (v < ranges[d].first || v > ranges[d].second) {
                inside = false;
                break;
            }
            std::int64_t bd = static_cast<std::int64_t>((v - ranges[d].first) / step[d]);
            // Clamp to the last bin to handle values at exactly the right edge.
            if (bd >= bins[d])
                bd = bins[d] - 1;
            flat += static_cast<std::size_t>(bd) * static_cast<std::size_t>(stride[d]);
        }
        if (inside)
            dst[flat] += 1.0;
    }

    if (density) {
        // Each cell is normalised by (cell_volume * N) so the histogram is a
        // probability density: integrating over all cells gives 1.
        double cell_volume = 1.0;
        for (auto s : step)
            cell_volume *= s;
        const double total = cell_volume * static_cast<double>(N);
        const std::size_t total_cells = shape_numel(counts_shape);
        for (std::size_t i = 0; i < total_cells; ++i)
            dst[i] /= total;
    }

    Device out_dev = pick_out_device(a->device(), Dtype::F64);
    auto counts_storage = to_device_storage(std::move(counts), out_dev, counts_shape);
    auto counts_t = fresh(std::move(counts_storage), std::move(counts_shape), Dtype::F64, out_dev);
    // See ``histogram_op``: recorded before the edges are built, because
    // ``on_op_io`` writes to the most recently entered node and
    // ``build_edges`` traces ops of its own.
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({a}, counts_t);

    // Concatenate all per-dimension edge arrays into a single 1-D tensor.
    // The total length is sum(bins[d]+1) for d in [0, D).
    std::int64_t edge_total = 0;
    for (auto b_ : bins)
        edge_total += (b_ + 1);
    Shape edge_shape{edge_total};
    auto edges = allocate_cpu(edge_shape, Dtype::F64);
    auto* edst = reinterpret_cast<double*>(edges.ptr.get());
    std::int64_t off = 0;
    for (std::int64_t d = 0; d < D; ++d) {
        const double lo = ranges[d].first;
        const double hi = ranges[d].second;
        for (std::int64_t i = 0; i <= bins[d]; ++i)
            edst[off + i] = lo + static_cast<double>(i) * step[d];
        // Pin the last edge exactly to `hi` to avoid drift from repeated addition.
        edst[off + bins[d]] = hi;
        off += bins[d] + 1;
    }
    auto edges_storage = to_device_storage(std::move(edges), out_dev, edge_shape);
    auto edges_t = fresh(std::move(edges_storage), std::move(edge_shape), Dtype::F64, out_dev);
    return {counts_t, edges_t};
}

}  // namespace lucid
