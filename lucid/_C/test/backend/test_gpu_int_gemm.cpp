// lucid/_C/test/backend/test_gpu_int_gemm.cpp
//
// LCD-1067: the GPU integer matrix product (backend/gpu/IntGemm.h) is exact at
// the dtype's width.  MLX refuses an integer ``matmul`` and the engine used to
// widen to float32, multiply and round back, which lost the low bits past 2^24
// and *saturated* an overflow.  MLX's integer ``multiply`` and ``sum`` wrap, so
// the product is a broadcast multiply and a sum over ``K``.  The wrap and
// exactness cases every integer product must answer are shared with the CPU one
// (helpers/int_gemm_cases.h); these add what is the GPU form's own:
//
//   - the blocked path (rows, then ``K``) gives the answer of the one-block one;
//   - batches, a broadcast batch, a transposed view and ``tensordot`` agree with
//     a plain triple loop on the host;
//   - an empty extent keeps its shape, and an empty sum is zeros.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/array.h>
#include <mlx/ops.h>
#include <mlx/transforms.h>

#include "../helpers/int_gemm_cases.h"
#include "backend/gpu/IntGemm.h"

namespace mx = ::mlx::core;
namespace gpu = lucid::gpu;
namespace cases = lucid::test::int_gemm;

namespace {

template <typename T>
struct DtypeOf;
template <>
struct DtypeOf<std::int8_t> {
    static mx::Dtype get() { return mx::int8; }
};
template <>
struct DtypeOf<std::int16_t> {
    static mx::Dtype get() { return mx::int16; }
};
template <>
struct DtypeOf<std::int32_t> {
    static mx::Dtype get() { return mx::int32; }
};
template <>
struct DtypeOf<std::int64_t> {
    static mx::Dtype get() { return mx::int64; }
};

using Dims = std::vector<mx::ShapeElem>;

template <typename T>
mx::array to_array(const std::vector<T>& v, const Dims& dims) {
    return mx::array(v.begin(), mx::Shape(dims.begin(), dims.end()), DtypeOf<T>::get());
}

// ``out`` as host values, row-major, whatever view MLX returned.
template <typename T>
std::vector<T> to_host(const mx::array& out) {
    const mx::array flat = mx::contiguous(out);
    mx::eval(flat);
    return std::vector<T>(flat.data<T>(), flat.data<T>() + flat.size());
}

// ``A @ B`` for row-major ``M x K`` and ``K x N``.
auto gpu_product = [](const auto& a, const auto& b, std::size_t M, std::size_t K, std::size_t N) {
    using T = typename std::decay_t<decltype(a)>::value_type;
    const Dims a_dims = {static_cast<mx::ShapeElem>(M), static_cast<mx::ShapeElem>(K)};
    const Dims b_dims = {static_cast<mx::ShapeElem>(K), static_cast<mx::ShapeElem>(N)};
    return to_host<T>(gpu::int_matmul(to_array(a, a_dims), to_array(b, b_dims)));
};

template <typename T>
void expect_blocked_matches(std::size_t batch, std::size_t M, std::size_t N, std::size_t K) {
    const auto a = cases::filled<T>(batch * M * K, 21 + M);
    const auto b = cases::filled<T>(batch * K * N, 22 + N);
    const auto want = cases::reference(a, b, batch, M, N, K);
    const Dims a_dims = {static_cast<mx::ShapeElem>(batch), static_cast<mx::ShapeElem>(M),
                         static_cast<mx::ShapeElem>(K)};
    const Dims b_dims = {static_cast<mx::ShapeElem>(batch), static_cast<mx::ShapeElem>(K),
                         static_cast<mx::ShapeElem>(N)};
    const auto ma = to_array(a, a_dims);
    const auto mb = to_array(b, b_dims);
    // The default block, then one that splits the rows, then ones that split K.
    for (std::int64_t block :
         {gpu::kIntGemmChunkElems, std::int64_t{400}, std::int64_t{40}, std::int64_t{1}}) {
        const auto got = gpu::int_matmul(ma, mb, block);
        EXPECT_EQ(got.dtype(), DtypeOf<T>::get());
        EXPECT_EQ(to_host<T>(got), want) << "block=" << block;
    }
}

}  // namespace

TEST(GpuIntGemm, WrapsAtTheOperandWidthWhereAFloatRoundTripSaturated) {
    cases::expect_overflow_wraps(gpu_product);
}

TEST(GpuIntGemm, IsExactWhereAFloatIsNot) {
    cases::expect_exact_where_a_float_is_not(gpu_product);
}

TEST(GpuIntGemm, TheBlockedPathMatchesATripleLoopAtEveryWidth) {
    // 165 products a row, so the blocks of 400 / 40 / 1 split the rows, then K.
    expect_blocked_matches<std::int8_t>(1, 7, 5, 33);
    expect_blocked_matches<std::int16_t>(1, 7, 5, 33);
    expect_blocked_matches<std::int32_t>(1, 7, 5, 33);
    expect_blocked_matches<std::int64_t>(1, 7, 5, 33);
    expect_blocked_matches<std::int32_t>(3, 4, 6, 9);
}

TEST(GpuIntGemm, ABroadcastBatchAndATransposedViewAreRead) {
    const std::size_t M = 4, K = 6, N = 5;
    // a: [3, M, K] against b: [2, 1, K, N] -> batch [2, 3].
    const auto a = cases::filled<std::int32_t>(3 * M * K, 31);
    const auto b = cases::filled<std::int32_t>(2 * K * N, 32);
    const auto got = gpu::int_matmul(to_array(a, {3, 4, 6}), to_array(b, {2, 1, 6, 5}));
    EXPECT_EQ(got.shape(), (mx::Shape{2, 3, 4, 5}));
    std::vector<std::int32_t> want;
    for (std::size_t bb = 0; bb < 2; ++bb) {
        const std::vector<std::int32_t> b_one(b.begin() + bb * K * N, b.begin() + (bb + 1) * K * N);
        for (std::size_t ba = 0; ba < 3; ++ba) {
            const std::vector<std::int32_t> a_one(a.begin() + ba * M * K,
                                                  a.begin() + (ba + 1) * M * K);
            const auto one = cases::reference(a_one, b_one, 1, M, N, K);
            want.insert(want.end(), one.begin(), one.end());
        }
    }
    EXPECT_EQ(to_host<std::int32_t>(got), want);

    // A transposed operand: stored [K, M], read as its swapaxes.
    const auto a2 = cases::filled<std::int32_t>(M * K, 33);
    const auto b2 = cases::filled<std::int32_t>(K * N, 34);
    std::vector<std::int32_t> a2_t(a2.size());
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k)
            a2_t[k * M + i] = a2[i * K + k];
    const auto view = mx::swapaxes(to_array(a2_t, {6, 4}), -2, -1);
    EXPECT_EQ(to_host<std::int32_t>(gpu::int_matmul(view, to_array(b2, {6, 5}))),
              cases::reference(a2, b2, 1, M, N, K));
}

TEST(GpuIntGemm, EmptyExtentsKeepTheirShapeAndAnEmptySumIsZeros) {
    const auto empty_k = gpu::int_matmul(to_array(std::vector<std::int32_t>{}, {3, 0}),
                                         to_array(std::vector<std::int32_t>{}, {0, 4}));
    EXPECT_EQ(empty_k.shape(), (mx::Shape{3, 4}));
    EXPECT_EQ(to_host<std::int32_t>(empty_k), std::vector<std::int32_t>(12, 0));

    const auto empty_m = gpu::int_matmul(to_array(std::vector<std::int8_t>{}, {0, 3}),
                                         to_array(std::vector<std::int8_t>(12, 1), {3, 4}));
    EXPECT_EQ(empty_m.shape(), (mx::Shape{0, 4}));
    EXPECT_EQ(empty_m.dtype(), mx::int8);

    const auto empty_n = gpu::int_matmul(to_array(std::vector<std::int64_t>(6, 1), {2, 3}),
                                         to_array(std::vector<std::int64_t>{}, {3, 0}));
    EXPECT_EQ(empty_n.shape(), (mx::Shape{2, 0}));

    const auto empty_batch = gpu::int_matmul(to_array(std::vector<std::int16_t>{}, {0, 2, 3}),
                                             to_array(std::vector<std::int16_t>{}, {0, 3, 2}));
    EXPECT_EQ(empty_batch.shape(), (mx::Shape{0, 2, 2}));
}

TEST(GpuIntGemm, TensordotContractsInTheGivenAxisOrder) {
    // a: [2, 3, 4], b: [4, 3, 5]; a's axes (1, 2) pair with b's axes (1, 0).
    const std::size_t A0 = 2, A1 = 3, A2 = 4, B2 = 5;
    const auto a = cases::filled<std::int32_t>(A0 * A1 * A2, 41);
    const auto b = cases::filled<std::int32_t>(A2 * A1 * B2, 42);
    std::vector<std::int32_t> want(A0 * B2);
    for (std::size_t i = 0; i < A0; ++i) {
        for (std::size_t j = 0; j < B2; ++j) {
            std::uint32_t acc = 0;
            for (std::size_t p = 0; p < A1; ++p)
                for (std::size_t q = 0; q < A2; ++q)
                    acc += static_cast<std::uint32_t>(a[(i * A1 + p) * A2 + q]) *
                           static_cast<std::uint32_t>(b[(q * A1 + p) * B2 + j]);
            want[i * B2 + j] = static_cast<std::int32_t>(acc);
        }
    }
    const auto ma = to_array(a, {2, 3, 4});
    const auto mb = to_array(b, {4, 3, 5});
    const auto got = gpu::int_tensordot(ma, mb, {1, 2}, {1, 0});
    EXPECT_EQ(got.shape(), (mx::Shape{2, 5}));
    EXPECT_EQ(to_host<std::int32_t>(got), want);
    // A negative axis counts from the end.
    EXPECT_EQ(to_host<std::int32_t>(gpu::int_tensordot(ma, mb, {-2, -1}, {-2, -3})), want);
    // Every axis contracted leaves a scalar.
    const auto all = gpu::int_tensordot(ma, ma, {0, 1, 2}, {0, 1, 2});
    EXPECT_EQ(all.shape(), mx::Shape{});
}
