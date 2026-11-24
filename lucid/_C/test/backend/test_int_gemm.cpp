// lucid/_C/test/backend/test_int_gemm.cpp
//
// The CPU integer matrix product (backend/cpu/IntGemm.h) is exact at the
// dtype's width (LCD-1067).  Accelerate has no integer GEMM, and the product
// used to be taken in double and rounded back: a dot product past 2^53 lost its
// low bits before the wrap.  The wrap and exactness cases every integer product
// must answer are shared with the GPU one (helpers/int_gemm_cases.h); these add
// what is the kernel's own:
//
//   - transposed operands, batches and the parallel path agree with a plain
//     triple loop, at every width;
//   - an empty extent is answered without reading an operand;
//   - the dtype dispatch takes the signed integers and refuses the rest.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

#include "../../backend/cpu/IntGemm.h"
#include "../helpers/int_gemm_cases.h"

namespace cpu = lucid::backend::cpu;
namespace cases = lucid::test::int_gemm;

namespace {

// ``A @ B`` for row-major ``M x K`` and ``K x N``.
auto cpu_product = [](const auto& a, const auto& b, std::size_t M, std::size_t K, std::size_t N) {
    using T = typename std::decay_t<decltype(a)>::value_type;
    std::vector<T> out(M * N);
    cpu::int_gemm<T>(false, false, 1, M, N, K, a.data(), b.data(), out.data());
    return out;
};

// ``a`` of ``batch x R x C`` as ``batch x C x R``.
template <typename T>
std::vector<T>
transposed(const std::vector<T>& a, std::size_t batch, std::size_t R, std::size_t C) {
    std::vector<T> out(a.size());
    for (std::size_t bi = 0; bi < batch; ++bi)
        for (std::size_t r = 0; r < R; ++r)
            for (std::size_t c = 0; c < C; ++c)
                out[(bi * C + c) * R + r] = a[(bi * R + r) * C + c];
    return out;
}

// The kernel and the triple loop on random ``T`` operands.
template <typename T>
void expect_matches_the_triple_loop(std::size_t M, std::size_t K, std::size_t N, int seed) {
    const auto a = cases::filled<T>(M * K, seed);
    const auto b = cases::filled<T>(K * N, seed + 1);
    EXPECT_EQ(cpu_product(a, b, M, K, N), cases::reference(a, b, 1, M, N, K));
}

}  // namespace

TEST(IntGemm, WrapsAtTheOperandWidth) {
    cases::expect_overflow_wraps(cpu_product);
}

TEST(IntGemm, IsExactWhereADoubleIsNot) {
    cases::expect_exact_where_a_float_is_not(cpu_product);
}

TEST(IntGemm, MatchesATripleLoopForEveryTransposition) {
    const std::size_t batch = 3, M = 5, N = 7, K = 11;
    const auto a = cases::filled<std::int32_t>(batch * M * K, 1);
    const auto b = cases::filled<std::int32_t>(batch * K * N, 2);
    const auto want = cases::reference<std::int32_t>(a, b, batch, M, N, K);
    const auto a_t = transposed(a, batch, M, K);
    const auto b_t = transposed(b, batch, K, N);
    for (bool trans_a : {false, true}) {
        for (bool trans_b : {false, true}) {
            std::vector<std::int32_t> got(batch * M * N);
            cpu::int_gemm<std::int32_t>(trans_a, trans_b, batch, M, N, K,
                                        (trans_a ? a_t : a).data(), (trans_b ? b_t : b).data(),
                                        got.data());
            EXPECT_EQ(got, want) << "trans_a=" << trans_a << " trans_b=" << trans_b;
        }
    }
}

TEST(IntGemm, MatchesATripleLoopAtEveryWidthAndThroughTheParallelPath) {
    // Enough rows that parallel_for splits them across tasks.
    expect_matches_the_triple_loop<std::int8_t>(1024, 16, 16, 3);
    expect_matches_the_triple_loop<std::int16_t>(1024, 16, 16, 5);
    expect_matches_the_triple_loop<std::int32_t>(1024, 16, 16, 7);
    expect_matches_the_triple_loop<std::int64_t>(1024, 16, 16, 9);
}

TEST(IntGemm, ATransposeFlagOnAUnitExtentReadsTheOperandAsStored) {
    // A ``[K, M]`` operand with ``M == 1`` or ``K == 1``, and a ``[N, K]`` one with
    // ``N == 1`` or ``K == 1``, hold the same flat sequence as the matrix they
    // transpose to, so the kernel copies nothing and the flag must change nothing.
    struct Extents {
        std::size_t M, N, K;
    };
    for (const Extents e : {Extents{4, 5, 1}, Extents{1, 1, 7}, Extents{1, 1, 1}, Extents{3, 1, 1},
                            Extents{1, 5, 1}, Extents{1, 4, 6}, Extents{5, 1, 6}}) {
        for (std::size_t batch : {1, 2}) {
            const auto a = cases::filled<std::int32_t>(batch * e.M * e.K, 51 + e.M);
            const auto b = cases::filled<std::int32_t>(batch * e.K * e.N, 52 + e.N);
            const auto want = cases::reference(a, b, batch, e.M, e.N, e.K);
            for (bool trans_a : {false, true}) {
                for (bool trans_b : {false, true}) {
                    // A flag is only a no-op where its operand has a unit extent.
                    if ((trans_a && e.M != 1 && e.K != 1) || (trans_b && e.N != 1 && e.K != 1))
                        continue;
                    std::vector<std::int32_t> got(batch * e.M * e.N);
                    cpu::int_gemm<std::int32_t>(trans_a, trans_b, batch, e.M, e.N, e.K, a.data(),
                                                b.data(), got.data());
                    EXPECT_EQ(got, want)
                        << "M=" << e.M << " N=" << e.N << " K=" << e.K << " batch=" << batch
                        << " trans_a=" << trans_a << " trans_b=" << trans_b;
                }
            }
        }
    }
}

TEST(IntGemm, VectorShapesNeedNoTransposedCopy) {
    // A column of B (N == 1) and a row of A (M == 1) read the same flat sequence
    // as their transposes: the matrix-vector and vector-matrix products.
    expect_matches_the_triple_loop<std::int32_t>(6, 9, 1, 11);
    expect_matches_the_triple_loop<std::int32_t>(1, 9, 6, 13);
}

TEST(IntGemm, EmptyExtentsAreAnsweredWithoutReadingAnOperand) {
    // K == 0 is an empty sum: zeros, with the operands never read.
    std::vector<std::int32_t> out(6, 99);
    cpu::int_gemm<std::int32_t>(false, false, 1, 2, 3, 0, nullptr, nullptr, out.data());
    for (std::int32_t v : out)
        EXPECT_EQ(v, 0);
    // An empty output is left alone.
    std::vector<std::int32_t> sentinel(4, 99);
    cpu::int_gemm<std::int32_t>(false, false, 1, 0, 4, 3, nullptr, nullptr, sentinel.data());
    cpu::int_gemm<std::int32_t>(false, false, 0, 2, 2, 3, nullptr, nullptr, sentinel.data());
    for (std::int32_t v : sentinel)
        EXPECT_EQ(v, 99);
}

TEST(IntGemm, DtypeDispatchTakesTheSignedIntegersOnly) {
    const std::vector<std::int16_t> a = {3, 4};
    const std::vector<std::int16_t> b = {5, 6};
    std::vector<std::int16_t> out(1, 0);
    const auto* ap = reinterpret_cast<const std::byte*>(a.data());
    const auto* bp = reinterpret_cast<const std::byte*>(b.data());
    auto* op = reinterpret_cast<std::byte*>(out.data());
    EXPECT_TRUE(cpu::int_gemm_by_dtype(lucid::Dtype::I16, false, false, 1, 1, 1, 2, ap, bp, op));
    EXPECT_EQ(out[0], 3 * 5 + 4 * 6);
    for (auto refused : {lucid::Dtype::Bool, lucid::Dtype::F32, lucid::Dtype::F64,
                         lucid::Dtype::C64, lucid::Dtype::F16}) {
        out[0] = 7;
        EXPECT_FALSE(cpu::int_gemm_by_dtype(refused, false, false, 1, 1, 1, 2, ap, bp, op));
        EXPECT_EQ(out[0], 7);
    }
}
