// lucid/_C/ops/composite/Search.cpp
//
// ``searchsorted`` and ``bucketize`` via O(k log N) binary search in O(k)
// memory: host binary search on CPU and an ``mx::fast::metal_kernel`` on
// Metal (zero host syncs, LCD-581).

#include "Search.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <type_traits>
#include <vector>

#include <mlx/fast.h>
#include <mlx/ops.h>

#include "../../backend/Dispatcher.h"
#include "../../backend/gpu/BinaryKernelLoad.h"
#include "../../backend/gpu/MlxBridge.h"
#include "../../compile/Tracer.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Half.h"
#include "../../core/Helpers.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../../kernel/BinaryKernel.h"

namespace lucid {

namespace {

std::string format_shape(const Shape& sh) {
    std::string out = "[";
    for (std::size_t i = 0; i < sh.size(); ++i) {
        if (i > 0)
            out += ", ";
        out += std::to_string(sh[i]);
    }
    out += "]";
    return out;
}

template <typename T>
std::int64_t bsearch_one(const T* seq, std::int64_t n, T v, bool right) {
    std::int64_t lo = 0;
    std::int64_t hi = n;
    if (right) {
        while (lo < hi) {
            const std::int64_t mid = lo + ((hi - lo) >> 1);
            if (!(v < seq[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
    } else {
        while (lo < hi) {
            const std::int64_t mid = lo + ((hi - lo) >> 1);
            if (!(v <= seq[mid]))
                lo = mid + 1;
            else
                hi = mid;
        }
    }
    return lo;
}

template <typename T>
void bsearch_rows(const T* seq,
                  const T* vals,
                  std::int64_t* out,
                  std::size_t outer,
                  std::int64_t n_seq,
                  std::size_t n_val,
                  bool seq_1d,
                  bool right) {
    for (std::size_t o = 0; o < outer; ++o) {
        const T* row_seq = seq_1d ? seq : (seq + o * static_cast<std::size_t>(n_seq));
        const T* row_val = vals + o * n_val;
        std::int64_t* row_out = out + o * n_val;
        for (std::size_t j = 0; j < n_val; ++j)
            row_out[j] = bsearch_one(row_seq, n_seq, row_val[j], right);
    }
}

void dispatch_bsearch(Dtype dt,
                      const CpuStorage& seq_s,
                      const CpuStorage& val_s,
                      std::int64_t* out,
                      std::size_t outer,
                      std::int64_t n_seq,
                      std::size_t n_val,
                      bool seq_1d,
                      bool right) {
    switch (dt) {
    case Dtype::F16:
    case Dtype::BF16: {
        const bool bf = dt == Dtype::BF16;
        const std::size_t seq_total = (seq_1d ? 1 : outer) * static_cast<std::size_t>(n_seq);
        const std::size_t val_total = outer * n_val;
        std::vector<float> wide_seq(seq_total), wide_val(val_total);
        backend::detail::half_or_bfloat16_to_float_bulk(
            reinterpret_cast<const std::uint16_t*>(seq_s.ptr.get()), wide_seq.data(), seq_total,
            bf);
        backend::detail::half_or_bfloat16_to_float_bulk(
            reinterpret_cast<const std::uint16_t*>(val_s.ptr.get()), wide_val.data(), val_total,
            bf);
        bsearch_rows(wide_seq.data(), wide_val.data(), out, outer, n_seq, n_val, seq_1d, right);
        return;
    }
    case Dtype::F32:
        bsearch_rows(reinterpret_cast<const float*>(seq_s.ptr.get()),
                     reinterpret_cast<const float*>(val_s.ptr.get()), out, outer, n_seq, n_val,
                     seq_1d, right);
        return;
    case Dtype::F64:
        bsearch_rows(reinterpret_cast<const double*>(seq_s.ptr.get()),
                     reinterpret_cast<const double*>(val_s.ptr.get()), out, outer, n_seq, n_val,
                     seq_1d, right);
        return;
    case Dtype::Bool:
        bsearch_rows(reinterpret_cast<const std::uint8_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::uint8_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I8:
        bsearch_rows(reinterpret_cast<const std::int8_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int8_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I16:
        bsearch_rows(reinterpret_cast<const std::int16_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int16_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I32:
        bsearch_rows(reinterpret_cast<const std::int32_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int32_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    case Dtype::I64:
        bsearch_rows(reinterpret_cast<const std::int64_t*>(seq_s.ptr.get()),
                     reinterpret_cast<const std::int64_t*>(val_s.ptr.get()), out, outer, n_seq,
                     n_val, seq_1d, right);
        return;
    default:
        ErrorBuilder("searchsorted").not_implemented("complex dtype is not supported");
    }
}

void check_search_shapes(const Shape& ss, const Shape& vs, bool& seq_1d) {
    if (ss.empty())
        ErrorBuilder("searchsorted").invalid_argument("sorted_sequence cannot be 0-D");
    seq_1d = ss.size() == 1;
    if (seq_1d)
        return;
    bool match = vs.size() == ss.size();
    for (std::size_t i = 0; match && i + 1 < ss.size(); ++i)
        match = ss[i] == vs[i];
    if (!match)
        ErrorBuilder("searchsorted")
            .invalid_argument("boundaries tensor should be 1 dimension or the first N-1 "
                              "dimensions of boundaries tensor and input value tensor must match, "
                              "got boundaries " +
                              format_shape(ss) + " and input value " + format_shape(vs));
}

const char* searchsorted_head_src() {
    return R"(
        uint64_t elem = uint64_t(thread_position_in_grid.y) * uint64_t(threads_per_grid.x) +
                        uint64_t(thread_position_in_grid.x);
        if (threads_per_grid.y > 1) {
            uint64_t n_total = 1;
            for (int d = 0; d < val_ndim; ++d)
                n_total *= uint64_t(uint(val_shape[d]));
            if (elem >= n_total)
                return;
        }
        int64_t n_seq = int64_t(uint(seq_shape[seq_ndim - 1]));
        uint64_t row_offset = 0;
        if (!Seq1D) {
            uint64_t n_val = uint64_t(uint(val_shape[val_ndim - 1]));
            row_offset = (elem / n_val) * uint64_t(n_seq);
        }
        T v = val[elem];
        int64_t lo = 0;
        int64_t hi = n_seq;
    )";
}

const char* searchsorted_float_loop_src() {
    return R"(
        float vf = static_cast<float>(v);
        bool nan_v = (as_type<uint>(vf) & 0x7fffffffu) > 0x7f800000u;
        while (lo < hi) {
            int64_t mid = lo + ((hi - lo) >> 1);
            float sf = static_cast<float>(seq[row_offset + uint64_t(mid)]);
            bool nan_s = (as_type<uint>(sf) & 0x7fffffffu) > 0x7f800000u;
            if (nan_v || nan_s || (Right ? (vf >= sf) : (vf > sf)))
                lo = mid + 1;
            else
                hi = mid;
        }
        out[elem] = lo;
    )";
}

const char* searchsorted_int_loop_src() {
    return R"(
        while (lo < hi) {
            int64_t mid = lo + ((hi - lo) >> 1);
            T s = seq[row_offset + uint64_t(mid)];
            if (Right ? (v >= s) : (v > s))
                lo = mid + 1;
            else
                hi = mid;
        }
        out[elem] = lo;
    )";
}

GpuStorage searchsorted_gpu(const GpuStorage& seq_s,
                            const GpuStorage& val_s,
                            const Shape& seq_shape,
                            const Shape& val_shape,
                            Dtype dt,
                            bool seq_1d,
                            bool right) {
    namespace mx = ::mlx::core;
    const auto mlx_val_shape = gpu::to_mlx_shape(val_shape);
    const std::size_t total_val = shape_numel(val_shape);
    const std::int64_t n_seq = seq_shape.back();
    if (total_val == 0 || n_seq == 0)
        return gpu::wrap_mlx_array(mx::zeros(mlx_val_shape, mx::int64), Dtype::I64);

    const mx::Shape k_shape = mlx_val_shape.empty() ? mx::Shape{1} : mlx_val_shape;
    const mx::array& seq_arr = *seq_s.arr;
    const mx::array val_arr = mlx_val_shape.empty() ? mx::reshape(*val_s.arr, k_shape) : *val_s.arr;
    static const auto k_float = mx::fast::metal_kernel(
        "lucid_searchsorted_f", {"seq", "val"}, {"out"},
        std::string(searchsorted_head_src()) + searchsorted_float_loop_src(), "",
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false, mx::CompileOptions{});
    static const auto k_int = mx::fast::metal_kernel(
        "lucid_searchsorted_i", {"seq", "val"}, {"out"},
        std::string(searchsorted_head_src()) + searchsorted_int_loop_src(), "",
        /*ensure_row_contiguous=*/true, /*atomic_outputs=*/false, mx::CompileOptions{});
    const auto& kernel = is_floating_point(dt) ? k_float : k_int;
    const auto launch = backend::metal_launch_1d(total_val);
    auto outs = kernel({seq_arr, val_arr}, {k_shape}, {mx::int64}, launch.grid, launch.threadgroup,
                       {{"T", seq_arr.dtype()}, {"Right", right}, {"Seq1D", seq_1d}}, std::nullopt,
                       false, {});
    return gpu::wrap_mlx_array(mlx_val_shape.empty() ? mx::reshape(outs[0], mlx_val_shape)
                                                     : std::move(outs[0]),
                               Dtype::I64);
}

CpuStorage searchsorted_cpu(
    const TensorImplPtr& seq, const TensorImplPtr& val, Dtype dt, bool seq_1d, bool right) {
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    const Shape& vs = val->shape();
    CpuStorage seq_cpu = be.to_cpu(seq->storage(), seq->shape());
    CpuStorage val_cpu = be.to_cpu(val->storage(), vs);
    CpuStorage out_cpu = helpers::allocate_cpu(vs, Dtype::I64);

    const std::int64_t n_seq = seq->shape().back();
    const std::size_t total_val = shape_numel(vs);
    const std::size_t n_val =
        seq_1d ? total_val : (vs.empty() ? 1 : static_cast<std::size_t>(vs.back()));
    const std::size_t outer = seq_1d ? 1 : (n_val == 0 ? 0 : total_val / n_val);

    if (total_val > 0) {
        dispatch_bsearch(dt, seq_cpu, val_cpu, reinterpret_cast<std::int64_t*>(out_cpu.ptr.get()),
                         outer, n_seq, n_val, seq_1d, right);
    }
    return out_cpu;
}

}  // namespace

TensorImplPtr
searchsorted_op(const TensorImplPtr& sorted_1d, const TensorImplPtr& values, bool right) {
    Validator::pair(sorted_1d, values, "searchsorted").both_non_null().same_dtype();
    const auto [seq, val] = detail::align_binary_devices(sorted_1d, values, "searchsorted");
    const Dtype dt = seq->dtype();
    if (is_complex(dt))
        ErrorBuilder("searchsorted").not_implemented("complex dtype is not supported");
    bool seq_1d = true;
    check_search_shapes(seq->shape(), val->shape(), seq_1d);

    const Device dv = val->device();
    const Shape& vs = val->shape();
    OpScopeFull scope{"searchsorted", dv, Dtype::I64, vs};
    scope.set_attr("right", right ? std::int64_t{1} : std::int64_t{0});

    Storage out_s =
        (seq->is_phantom() || val->is_phantom())
            ? Storage{PhantomStorage{0, Dtype::I64, dv}}
            : (dv == Device::GPU ? Storage{searchsorted_gpu(std::get<GpuStorage>(seq->storage()),
                                                            std::get<GpuStorage>(val->storage()),
                                                            seq->shape(), vs, dt, seq_1d, right)}
                                 : Storage{searchsorted_cpu(seq, val, dt, seq_1d, right)});
    auto out = helpers::fresh(std::move(out_s), vs, Dtype::I64, dv);
    if (auto* trc = ::lucid::compile::current_tracer())
        trc->on_op_io({seq, val}, out);
    return out;
}

TensorImplPtr
bucketize_op(const TensorImplPtr& values, const TensorImplPtr& boundaries, bool right) {
    Validator::pair(values, boundaries, "bucketize").both_non_null().same_dtype();
    const auto [val, bnd] = detail::align_binary_devices(values, boundaries, "bucketize");
    if (bnd->shape().size() != 1)
        ErrorBuilder("bucketize")
            .invalid_argument("boundaries tensor must be 1 dimension, but got dim(" +
                              std::to_string(bnd->shape().size()) + ")");
    return searchsorted_op(bnd, val, right);
}

}  // namespace lucid
