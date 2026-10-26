// lucid/_C/ops/bfunc/Floordiv.h
//
// The floor-division family: floor division (``//``), the floored
// remainder (``%``) and the truncated remainder (``fmod``).  One kernel
// per op computes each element exactly, in the operands' dtype, on both
// streams (backend/DivMod.h has the rule).  They used to divide in float,
// which rounded every integer past 2**24 on Metal and 2**53 on the CPU and
// every float quotient past 2**24.
//
// An integer zero divisor anywhere in ``b`` raises ``RuntimeError:
// ZeroDivisionError`` on the CPU, as the reference's CPU does.  On Metal it
// is not checked (no device sync), as on the reference's accelerator: a
// zero divisor gives ``a // 0 == -1`` and ``a % 0 == fmod(a, 0) == a``, with
// no error.  Floats follow IEEE.  Bool and complex operands are refused.

#pragma once

#include <utility>

#include "../../api.h"
#include "../../core/AmpPolicy.h"
#include "../../core/OpSchema.h"
#include "../../core/Storage.h"
#include "../../core/fwd.h"
#include "_BinaryOp.h"

namespace lucid {

// ``floor(a / b)``.  Piecewise constant: both gradients are zero.
class LUCID_API FloordivBackward : public BinaryOp<FloordivBackward> {
public:
    static const OpSchema schema_v1;
    static constexpr bool kSavesInputs = false;

    static CpuStorage
    cpu_kernel(const CpuStorage& a, const CpuStorage& b, const Shape& shape, Dtype dt);
    static GpuStorage
    gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape& shape, Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// ``a - trunc(a / b) * b``, with the sign of ``a``.  ``da = g``,
// ``db = -g * trunc(a / b)``.
class LUCID_API FmodBackward : public BinaryOp<FmodBackward> {
public:
    static const OpSchema schema_v1;
    // ``da = g`` reads neither operand; ``db`` reads both.
    static constexpr GradReads<2> kGradReads{{{false, false}, {true, true}}};

    static CpuStorage
    cpu_kernel(const CpuStorage& a, const CpuStorage& b, const Shape& shape, Dtype dt);
    static GpuStorage
    gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape& shape, Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// ``a - floor(a / b) * b``, with the sign of ``b``.  ``da = g``,
// ``db = -g * (a // b)``.
class LUCID_API RemainderBackward : public BinaryOp<RemainderBackward> {
public:
    static const OpSchema schema_v1;
    // ``da = g`` reads neither operand; ``db`` reads both.
    static constexpr GradReads<2> kGradReads{{{false, false}, {true, true}}};

    static CpuStorage
    cpu_kernel(const CpuStorage& a, const CpuStorage& b, const Shape& shape, Dtype dt);
    static GpuStorage
    gpu_kernel(const GpuStorage& a, const GpuStorage& b, const Shape& shape, Dtype dt);

    std::pair<Storage, Storage> grad_formula(const Storage& grad_out);
    std::pair<TensorImplPtr, TensorImplPtr> grad_formula_impl(const TensorImplPtr& grad_out,
                                                              const TensorImplPtr& a,
                                                              const TensorImplPtr& b);
};

// Element-wise floor division, broadcast, in the operands' dtype.  A
// float16 / bfloat16 tensor with a 0-d float32 operand computes in float32
// and rounds once (ops/bfunc/_Opmath.h).
LUCID_API TensorImplPtr floordiv_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise C remainder: the result has the sign of ``a``.
LUCID_API TensorImplPtr fmod_op(const TensorImplPtr& a, const TensorImplPtr& b);

// Element-wise Python remainder: the result has the sign of ``b``.
LUCID_API TensorImplPtr remainder_op(const TensorImplPtr& a, const TensorImplPtr& b);

}  // namespace lucid
