// lucid/_C/backend/gpu/methods/Factory.cpp
// Out-of-line GpuBackend method definitions for the Factory family.
#include "../GpuBackend.h"

namespace lucid {
namespace backend {

Storage GpuBackend::from_cpu(CpuStorage cpu, const Shape& shape) {
    return Storage{gpu::upload_cpu_to_gpu(cpu, shape)};
}

Storage GpuBackend::zeros(const Shape& shape, Dtype dt) {
    auto arr = ::mlx::core::zeros(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
    return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
}

Storage GpuBackend::ones(const Shape& shape, Dtype dt) {
    auto arr = ::mlx::core::ones(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
    return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
}

Storage
GpuBackend::bernoulli_mask(double keep_prob, const Shape& shape, Dtype dt, std::uint64_t key_seed) {
    // A dropout mask is 1/(1-p) and 0, and neither survives an integer
    // dtype — the scale truncates to 1 and the op silently becomes a
    // random zeroing with no compensation.  The CPU has no integer
    // Bernoulli and said so; this cast made one, so `dropout` ran on
    // Metal and raised on the CPU for the same call.  The reference
    // refuses integer input here too.
    if (!is_floating_point(dt))
        ErrorBuilder("gpu_backend::bernoulli_mask")
            .not_implemented("a Bernoulli mask needs a floating dtype (F16/F32)");
    auto key = ::mlx::core::random::key(key_seed);
    auto mask = ::mlx::core::random::bernoulli(::mlx::core::array(static_cast<float>(keep_prob)),
                                               gpu::to_mlx_shape(shape), key);
    return Storage{gpu::wrap_mlx_array(::mlx::core::astype(mask, gpu::to_mlx_dtype(dt)), dt)};
}

Storage GpuBackend::random_uniform(
    const Shape& shape, double lo, double hi, Dtype dt, std::uint64_t key_seed) {
    auto key = ::mlx::core::random::key(key_seed);
    if (dt == Dtype::C64) {
        // MLX draws real floats only.  The reference draws each part
        // uniformly, so draw a trailing pair and read it as one value.
        auto pairs_shape = gpu::to_mlx_shape(shape);
        pairs_shape.push_back(2);
        auto pairs = ::mlx::core::random::uniform(static_cast<float>(lo), static_cast<float>(hi),
                                                  pairs_shape, ::mlx::core::float32, key);
        const float lo_f = static_cast<float>(lo);
        const float hi_f = static_cast<float>(hi);
        if (lo_f < hi_f) {
            pairs = ::mlx::core::minimum(
                pairs, ::mlx::core::array(std::nextafter(hi_f, lo_f), ::mlx::core::float32));
        }
        auto z = ::mlx::core::reshape(::mlx::core::view(pairs, ::mlx::core::complex64),
                                      gpu::to_mlx_shape(shape));
        return Storage{gpu::wrap_mlx_array(std::move(z), dt)};
    }
    const auto mdt = gpu::to_mlx_dtype(dt);
    auto arr = ::mlx::core::random::uniform(static_cast<float>(lo), static_cast<float>(hi),
                                            gpu::to_mlx_shape(shape), mdt, key);
    if (is_half_float(dt)) {
        const bool is_bf16 = (dt == Dtype::BF16);
        auto narrow = [is_bf16](float v) {
            return is_bf16 ? detail::float_to_bfloat16_bits(v) : detail::float_to_half_bits(v);
        };
        auto widen = [is_bf16](std::uint16_t b) {
            return is_bf16 ? detail::bfloat16_bits_to_float(b) : detail::half_bits_to_float(b);
        };
        const float lo_h = widen(narrow(static_cast<float>(lo)));
        const float hi_h = widen(narrow(static_cast<float>(hi)));
        if (lo_h < hi_h) {
            const std::uint16_t hi_bits = narrow(static_cast<float>(hi));
            const std::uint16_t max_bits =
                (hi_h > 0.0f) ? static_cast<std::uint16_t>(hi_bits - 1u)
                              : (hi_h == 0.0f ? (is_bf16 ? 0x8080u : 0x8001u)
                                              : static_cast<std::uint16_t>(hi_bits + 1u));
            arr = ::mlx::core::minimum(arr, ::mlx::core::array(widen(max_bits), mdt));
        }
    } else if (dt == Dtype::F32) {
        const float lo_f = static_cast<float>(lo);
        const float hi_f = static_cast<float>(hi);
        if (lo_f < hi_f) {
            arr = ::mlx::core::minimum(arr, ::mlx::core::array(std::nextafter(hi_f, lo_f), mdt));
        }
    }
    return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
}

Storage GpuBackend::random_normal(
    const Shape& shape, double mean, double std, Dtype dt, std::uint64_t key_seed) {
    auto key = ::mlx::core::random::key(key_seed);
    auto arr = ::mlx::core::random::normal(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt),
                                           static_cast<float>(mean), static_cast<float>(std), key);
    return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
}

Storage
GpuBackend::random_bernoulli(const Shape& shape, double p, Dtype dt, std::uint64_t key_seed) {
    auto key = ::mlx::core::random::key(key_seed);
    auto arr = ::mlx::core::random::bernoulli(::mlx::core::array(static_cast<float>(p)),
                                              gpu::to_mlx_shape(shape), key);
    return Storage{gpu::wrap_mlx_array(::mlx::core::astype(arr, gpu::to_mlx_dtype(dt)), dt)};
}

Storage GpuBackend::random_randint(
    const Shape& shape, std::int64_t low, std::int64_t high, Dtype dt, std::uint64_t key_seed) {
    const auto ulow = static_cast<std::uint64_t>(low);
    const auto span = static_cast<std::uint64_t>(high) - ulow;
    auto pairs_shape = gpu::to_mlx_shape(shape);
    pairs_shape.push_back(2);
    auto words = ::mlx::core::random::bits(pairs_shape, 4, ::mlx::core::random::key(key_seed));
    auto u64 = ::mlx::core::reshape(::mlx::core::view(words, ::mlx::core::uint64),
                                    gpu::to_mlx_shape(shape));
    auto sum = ::mlx::core::add(::mlx::core::array(ulow),
                                ::mlx::core::remainder(u64, ::mlx::core::array(span)));
    auto i64 = ::mlx::core::view(sum, ::mlx::core::int64);
    return Storage{gpu::wrap_mlx_array(::mlx::core::astype(i64, gpu::to_mlx_dtype(dt)), dt)};
}

Storage GpuBackend::clone(const Storage& src, const Shape&, Dtype dt) {
    const auto& gs = std::get<GpuStorage>(src);
    auto arr = ::mlx::core::copy(::mlx::core::contiguous(*gs.arr));
    return Storage{gpu::wrap_mlx_array(std::move(arr), dt)};
}

Storage GpuBackend::contiguous(
    const Storage& src, const Shape& shape, const Stride&, std::size_t, bool, Dtype dt) {
    return clone(src, shape, dt);
}

std::vector<Storage>
GpuBackend::meshgrid(const std::vector<Storage>& xs, const Shape&, Dtype dt, bool indexing_xy) {
    std::vector<::mlx::core::array> arrays;
    arrays.reserve(xs.size());
    for (const auto& x : xs)
        arrays.push_back(*std::get<GpuStorage>(x).arr);
    auto out = ::mlx::core::meshgrid(arrays, false, indexing_xy ? "xy" : "ij");
    std::vector<Storage> result;
    result.reserve(out.size());
    for (auto& arr : out) {
        auto contiguous = ::mlx::core::contiguous(arr);
        result.push_back(Storage{gpu::wrap_mlx_array(std::move(contiguous), dt)});
    }
    return result;
}

Storage GpuBackend::astype(const Storage& a, const Shape& shape, Dtype /*src_dt*/, Dtype dst_dt) {
    const auto& ga = std::get<GpuStorage>(a);
    auto result = gpu_convert(*ga.arr, dst_dt);
    return Storage{gpu::wrap_mlx_array(::mlx::core::contiguous(result), dst_dt)};
}

CpuStorage GpuBackend::to_cpu(const Storage& a, const Shape& shape) {
    return gpu::download_gpu_to_cpu(std::get<GpuStorage>(a), shape);
}

Storage GpuBackend::full(const Shape& shape, Dtype dt, double fill_value) {
    auto ones_arr = ::mlx::core::ones(gpu::to_mlx_shape(shape), gpu::to_mlx_dtype(dt));
    ::mlx::core::array scalar = [&]() -> ::mlx::core::array {
        switch (dt) {
        case Dtype::F16:
            // 3.3 AMP support: F16 fill values arise in Conv/Linear/Matmul
            // autocast scopes when Python scalar promotions touch a cast
            // tensor (e.g. BatchNorm's ``(1-eff) * running_mean`` builds an
            // F16 scalar tensor under autocast(F16)).  Build via an
            // F32 array then cast — ``mlx::core::array`` ctor doesn't
            // accept a float16 scalar directly, but ``astype`` from F32
            // is fast and matches the dtype of ``ones_arr``.
            return ::mlx::core::astype(::mlx::core::array(static_cast<float>(fill_value)),
                                       ::mlx::core::float16);
        case Dtype::BF16:
            return ::mlx::core::astype(::mlx::core::array(static_cast<float>(fill_value)),
                                       ::mlx::core::bfloat16);
        case Dtype::F32:
            return ::mlx::core::array(static_cast<float>(fill_value));
        case Dtype::F64:
            return ::mlx::core::array(fill_value, gpu::to_mlx_dtype(dt));
        case Dtype::I32:
            return ::mlx::core::array(static_cast<int32_t>(fill_value));
        case Dtype::I64:
            return ::mlx::core::array(static_cast<int64_t>(fill_value));
        case Dtype::I8:
            return ::mlx::core::array(static_cast<int8_t>(fill_value));
        case Dtype::I16:
            return ::mlx::core::array(static_cast<int16_t>(fill_value));
        case Dtype::Bool:
            return ::mlx::core::array(fill_value != 0.0);
        case Dtype::C64:
            // Real-only fill: imag = 0.  For arbitrary complex fill use
            // ``complex_combine`` after the fact.
            return ::mlx::core::array(std::complex<float>(static_cast<float>(fill_value), 0.0f));
        default:
            ErrorBuilder("gpu_backend::full").not_implemented("dtype not supported");
        }
    }();
    auto out = ::mlx::core::multiply(ones_arr, scalar);
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::eye(std::int64_t N, std::int64_t M, std::int64_t k, Dtype dt) {
    // MLX builds `eye` with a scatter, and Metal's scatter has no
    // int64 kernel.  Every entry is 0 or 1, so building at int32 and
    // widening is exact — an int64 identity holds nothing an int32
    // one does not.  ``wrap_mlx_array`` does the widening, because it
    // converts whenever the array's dtype and the requested one
    // differ.  The scatter has no complex kernel either, and the same
    // holds there: a complex identity is a float32 one with zero
    // imaginary parts.
    const Dtype build = dt == Dtype::I64 ? Dtype::I32 : dt == Dtype::C64 ? Dtype::F32 : dt;
    auto out = eye_or_empty(static_cast<int>(N), static_cast<int>(M), static_cast<int>(k),
                            gpu::to_mlx_dtype(build));
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::diag(
    const Storage& v, const Shape& v_shape, std::int64_t k, Dtype dt, Shape& out_shape) {
    const auto& gv = std::get<GpuStorage>(v);
    if (!gv.arr)
        ErrorBuilder("gpu_backend::diag").fail("null GPU array");
    auto out = ::mlx::core::diag(*gv.arr, static_cast<int>(k));
    out_shape.clear();
    for (auto d : out.shape())
        out_shape.push_back(static_cast<std::int64_t>(d));
    (void)v_shape;
    (void)dt;
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::tri(const Storage& input, const Shape& shape, Dtype dt, int k, bool upper) {
    const auto& ga = std::get<GpuStorage>(input);
    auto out = upper ? ::mlx::core::triu(*ga.arr, k) : ::mlx::core::tril(*ga.arr, k);
    (void)shape;
    return Storage{gpu::wrap_mlx_array(std::move(out), dt)};
}

Storage GpuBackend::to_shared_storage(const Storage& src, const Shape& shape) {
    if (storage_is_metal_shared(src))
        return src;

    const std::byte* src_ptr = nullptr;
    std::size_t src_bytes = 0;
    Dtype src_dtype = Dtype::F32;

    if (storage_is_cpu(src)) {
        const auto& cs = storage_cpu(src);
        src_ptr = cs.ptr.get();
        src_bytes = cs.nbytes;
        src_dtype = cs.dtype;
    } else {
        const auto& gs = storage_gpu(src);
        if (!gs.arr)
            return src;
        gs.arr->eval();
        MemoryTracker::track_host_sync(gs.nbytes);
        src_ptr = reinterpret_cast<const std::byte*>(gs.arr->data<std::uint8_t>());
        src_bytes = gs.nbytes;
        src_dtype = gs.dtype;
    }

    if (!src_ptr || src_bytes == 0)
        return src;

    auto owned = gpu::make_metal_shared(src_bytes);
    if (!owned.buf.cpu_ptr)
        return src;

    std::memcpy(owned.buf.cpu_ptr, src_ptr, src_bytes);

    SharedStorage ss;
    ss.cpu_ptr = owned.buf.cpu_ptr;
    ss.mtl_handle = owned.buf.mtl_handle;
    ss.nbytes = src_bytes;
    ss.dtype = src_dtype;
    ss.owner = std::move(owned.owner);
    return Storage{std::move(ss)};
}

Storage GpuBackend::run_custom_metal_kernel(const std::string& kernel_source,
                                            const std::string& function_name,
                                            const std::vector<Storage>& inputs,
                                            const Shape& output_shape,
                                            Dtype output_dtype,
                                            const std::array<std::size_t, 3>& grid,
                                            const std::array<std::size_t, 3>& threads) {
    auto k = gpu::compile_metal_kernel(kernel_source, function_name);
    if (!k.is_valid()) {
        ErrorBuilder("GpuBackend::run_custom_metal_kernel")
            .fail("kernel compilation failed for function '" + function_name + "'");
    }
    gpu::KernelLaunchConfig cfg{grid, threads};
    return gpu::run_metal_kernel(k, inputs, output_shape, output_dtype, cfg);
}

}  // namespace backend
}  // namespace lucid
