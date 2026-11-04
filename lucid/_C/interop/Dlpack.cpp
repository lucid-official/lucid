// lucid/_C/interop/Dlpack.cpp — kDLMetal DLPack producer + consumer.
//
// See Dlpack.h for why the Metal dialect exists next to the NumPy-backed
// CPU one.

#include "Dlpack.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mlx/allocator.h>
#include <mlx/array.h>

#include "../backend/gpu/MlxBridge.h"
#include "../backend/gpu/mps/MpsBridge.h"
#include "../core/PhantomMode.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"

namespace lucid::interop {

namespace {

// What the exported block owns.
//
// Holding the source ``mlx::core::array`` is what keeps the exported
// ``MTLBuffer`` alive: MLX frees a buffer when its last array reference
// dies, and a consumer is entitled to read the capsule long after the
// Lucid tensor it came from has gone out of scope.  The shape and stride
// vectors live here for the same reason — ``DLTensor`` stores bare
// pointers into them.
struct ExportContext {
    std::shared_ptr<::mlx::core::array> arr;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    DLManagedTensor managed{};
};

void destroy_export(DLManagedTensor* self) {
    if (self == nullptr)
        return;
    delete static_cast<ExportContext*>(self->manager_ctx);
}

// Lucid dtype -> DLPack (code, bits).
//
// F64 is absent on purpose rather than by omission: Metal has no
// double, so a float64 tensor is never GPU-resident and there would be
// no buffer to point a consumer at.
DLDataType to_dl_dtype(Dtype dt) {
    switch (dt) {
    case Dtype::Bool:
        return {kDLBool, 8, 1};
    case Dtype::I8:
        return {kDLInt, 8, 1};
    case Dtype::I16:
        return {kDLInt, 16, 1};
    case Dtype::I32:
        return {kDLInt, 32, 1};
    case Dtype::I64:
        return {kDLInt, 64, 1};
    case Dtype::F16:
        return {kDLFloat, 16, 1};
    case Dtype::BF16:
        return {kDLBfloat, 16, 1};
    case Dtype::F32:
        return {kDLFloat, 32, 1};
    case Dtype::C64:
        return {kDLComplex, 64, 1};
    default:
        throw std::invalid_argument(
            "metal_to_dlpack: dtype " + std::string(dtype_name(dt)) +
            " has no DLPack spelling on Metal (float64 in particular does not "
            "exist there, so a GPU tensor never carries it)");
    }
}

// The inverse, for the import direction.
Dtype from_dl_dtype(const DLDataType& dl) {
    if (dl.lanes != 1)
        throw std::invalid_argument("dlpack_to_metal: vector lanes are not supported (lanes=" +
                                    std::to_string(dl.lanes) + ")");
    switch (dl.code) {
    case kDLBool:
        if (dl.bits == 8)
            return Dtype::Bool;
        break;
    case kDLInt:
        switch (dl.bits) {
        case 8:
            return Dtype::I8;
        case 16:
            return Dtype::I16;
        case 32:
            return Dtype::I32;
        case 64:
            return Dtype::I64;
        default:
            break;
        }
        break;
    case kDLFloat:
        switch (dl.bits) {
        case 16:
            return Dtype::F16;
        case 32:
            return Dtype::F32;
        default:
            break;
        }
        break;
    case kDLBfloat:
        if (dl.bits == 16)
            return Dtype::BF16;
        break;
    case kDLComplex:
        if (dl.bits == 64)
            return Dtype::C64;
        break;
    default:
        break;
    }
    throw std::invalid_argument(
        "dlpack_to_metal: no Lucid dtype for DLPack (code=" + std::to_string(dl.code) +
        ", bits=" + std::to_string(dl.bits) + ")");
}

}  // namespace

DLManagedTensor* metal_to_dlpack(const TensorImplPtr& impl) {
    if (!impl)
        throw std::invalid_argument("metal_to_dlpack: null tensor");
    if (impl->device() != Device::GPU)
        throw std::invalid_argument(
            "metal_to_dlpack: tensor is on the CPU — the host path goes through "
            "NumPy's DLPack, not this one");
    if (impl->is_phantom())
        refuse_phantom_read("to_dlpack");

    const auto& gs = std::get<GpuStorage>(impl->storage());
    if (!gs.arr)
        throw std::runtime_error("metal_to_dlpack: GPU storage carries no MLX array");

    // Element type first: a rejected dtype should not have forced an
    // evaluation of the graph behind ``impl``.
    const DLDataType dl_dtype = to_dl_dtype(impl->dtype());

    // Materialises the array and blocks until its producing command
    // buffer completes, so the handle below refers to finished data.
    const auto view = lucid::gpu::mps::array_to_buffer(*gs.arr);

    auto ctx = std::make_unique<ExportContext>();
    ctx->arr = gs.arr;

    const Shape& shape = impl->shape();
    ctx->shape.assign(shape.begin(), shape.end());
    // A metal tensor is never a view, so the layout is always row-major
    // packed and the strides are derivable.  They are emitted explicitly
    // rather than left null because MLX emits them too, and a consumer
    // that trusts a null to mean "packed" is one we would rather not
    // depend on.
    ctx->strides.resize(ctx->shape.size());
    std::int64_t acc = 1;
    for (std::size_t i = ctx->shape.size(); i-- > 0;) {
        ctx->strides[i] = acc;
        acc *= ctx->shape[i];
    }

    DLManagedTensor& mt = ctx->managed;
    mt.dl_tensor.data = view.mtl_buffer;
    mt.dl_tensor.device = DLDevice{kDLMetal, 0};
    mt.dl_tensor.ndim = static_cast<std::int32_t>(ctx->shape.size());
    mt.dl_tensor.dtype = dl_dtype;
    mt.dl_tensor.shape = ctx->shape.empty() ? nullptr : ctx->shape.data();
    mt.dl_tensor.strides = ctx->strides.empty() ? nullptr : ctx->strides.data();
    mt.dl_tensor.byte_offset = static_cast<std::uint64_t>(view.offset_bytes);
    mt.manager_ctx = ctx.get();
    mt.deleter = &destroy_export;

    return &ctx.release()->managed;
}

TensorImplPtr dlpack_to_metal(DLManagedTensor* managed) {
    if (managed == nullptr)
        throw std::invalid_argument("dlpack_to_metal: null managed tensor");

    // Any rejection below still has to release the producer, or importing
    // a tensor Lucid cannot represent would leak it.
    auto release = [managed]() {
        if (managed->deleter != nullptr)
            managed->deleter(managed);
    };

    // ``t`` lives inside ``managed``, and ``release()`` runs the producer's
    // deleter, which may free it — so a refusal reads what its message
    // needs *before* releasing.  (The device and offset messages read ``t``
    // after the release and printed freed memory.)
    const DLTensor& t = managed->dl_tensor;
    if (t.device.device_type != kDLMetal) {
        const auto device_type = static_cast<int>(t.device.device_type);
        release();
        throw std::invalid_argument(
            "dlpack_to_metal: expected a Metal (device type 8) capsule, got device type " +
            std::to_string(device_type) +
            " — host capsules are imported through lucid.from_dlpack's NumPy path");
    }
    if (t.ndim < 0) {
        release();
        throw std::invalid_argument("dlpack_to_metal: negative ndim");
    }

    Dtype dtype;
    try {
        dtype = from_dl_dtype(t.dtype);
    } catch (...) {
        release();
        throw;
    }

    std::vector<int> dims(static_cast<std::size_t>(t.ndim));
    std::int64_t count = 1;
    for (std::int32_t i = 0; i < t.ndim; ++i) {
        const std::int64_t d = t.shape[i];
        if (d < 0) {
            release();
            throw std::invalid_argument("dlpack_to_metal: negative dimension");
        }
        dims[static_cast<std::size_t>(i)] = static_cast<int>(d);
        count *= d;
    }

    // Only packed row-major imports.  A strided capsule would need a
    // gather to become a Lucid tensor, and silently copying under an API
    // whose entire purpose is not copying would be the wrong trade.
    if (t.strides != nullptr) {
        std::int64_t acc = 1;
        for (std::int32_t i = t.ndim; i-- > 0;) {
            if (t.strides[i] != acc) {
                release();
                throw std::invalid_argument(
                    "dlpack_to_metal: only row-major packed capsules can be adopted "
                    "without a copy; this one is strided");
            }
            acc *= t.shape[i];
        }
    }

    // A capsule's first element lives at ``data + byte_offset``; the array
    // below is built from ``data`` alone.  Adopting an offset capsule
    // without reading it therefore returns a tensor shifted by
    // ``byte_offset / itemsize`` elements, with no error — the packed
    // check above does not catch it, because a leading-axis slice of a
    // contiguous buffer is still row-major packed and merely starts
    // later.
    //
    // Refused rather than honoured because honouring it needs a slice
    // MLX does not expose here: ``buffer_to_array`` takes the same
    // position for the same reason.  A refusal is recoverable and a
    // silently shifted tensor is not.
    if (t.byte_offset != 0) {
        const std::uint64_t byte_offset = t.byte_offset;
        release();
        throw std::invalid_argument(
            "dlpack_to_metal: this capsule's data begins at byte_offset " +
            std::to_string(byte_offset) +
            ", and adopting it without a copy would read from the start of the "
            "buffer instead — offset views are not supported yet. Make the "
            "producer hand over a capsule whose data begins at its buffer, or "
            "copy it across the host");
    }

    ::mlx::core::Shape mlx_shape(dims.begin(), dims.end());
    ::mlx::core::Dtype mlx_dtype = lucid::gpu::to_mlx_dtype(dtype);

    // The producer's block is kept alive by the array's deleter, so the
    // buffer outlives every Lucid view of it — the DLPack contract in
    // both directions.
    ::mlx::core::array arr(::mlx::core::allocator::Buffer(t.data), std::move(mlx_shape), mlx_dtype,
                           [managed](::mlx::core::allocator::Buffer) {
                               if (managed->deleter != nullptr)
                                   managed->deleter(managed);
                           });

    GpuStorage gs;
    gs.arr = std::make_shared<::mlx::core::array>(std::move(arr));
    gs.dtype = dtype;
    gs.nbytes = static_cast<std::size_t>(count) * dtype_size(dtype);

    Shape shape;
    shape.assign(dims.begin(), dims.end());
    return std::make_shared<TensorImpl>(Storage{std::move(gs)}, shape, dtype, Device::GPU, false);
}

}  // namespace lucid::interop
