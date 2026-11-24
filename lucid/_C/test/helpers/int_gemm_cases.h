// lucid/_C/test/helpers/int_gemm_cases.h
//
// What every integer matrix product must answer, for the CPU kernel
// (backend/cpu/IntGemm.h) and the GPU one (backend/gpu/IntGemm.h) alike (LCD-1067).
//
// A product wraps at the operand width, two's complement, and is exact where a
// float32 (24 bits) or a double (53 bits) is not.  Both streams once took the
// product through a floating GEMM and cast it back, so the two failed in
// different ways: the CPU dropped low bits past 2^53, and the GPU also
// *saturated* an overflow (an int8 ``[[100, 100], [100, 100]]`` squared came out
// 127, not 32).  The expected constants are Python integers reduced modulo
// 2^width.
//
// A case is run through ``product(a, b, M, K, N)``: a generic callable taking
// row-major ``std::vector<T>`` operands of ``M x K`` and ``K x N`` and returning
// the ``M x N`` product as a ``std::vector<T>``.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>

namespace lucid::test::int_gemm {

// A deterministic stream of values spanning the whole range of ``T``.
template <typename T>
std::vector<T> filled(std::size_t count, std::uint64_t seed) {
    std::vector<T> v(count);
    std::uint64_t state = seed;
    for (auto& x : v) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        x = static_cast<T>(state >> 24);
    }
    return v;
}

// The product of ``batch`` row-major ``M x K`` and ``K x N`` matrices as a plain
// triple loop in 64-bit unsigned arithmetic, cut to ``T``: no transposition, no
// blocking and no parallel dispatch to share a mistake with the kernel it checks.
template <typename T>
std::vector<T> reference(const std::vector<T>& a,
                         const std::vector<T>& b,
                         std::size_t batch,
                         std::size_t M,
                         std::size_t N,
                         std::size_t K) {
    std::vector<T> out(batch * M * N);
    for (std::size_t bi = 0; bi < batch; ++bi) {
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < N; ++j) {
                std::uint64_t acc = 0;
                for (std::size_t k = 0; k < K; ++k) {
                    acc += static_cast<std::uint64_t>(
                               static_cast<std::int64_t>(a[(bi * M + i) * K + k])) *
                           static_cast<std::uint64_t>(
                               static_cast<std::int64_t>(b[(bi * K + k) * N + j]));
                }
                out[(bi * M + i) * N + j] = static_cast<T>(acc);
            }
        }
    }
    return out;
}

// ``[[v, v], [v, v]]`` squared is ``2 * v * v`` in every cell, reduced to the
// dtype, and the extremes of the narrow dtypes wrap to the values below.
template <typename Product>
void expect_overflow_wraps(Product&& product) {
    using V8 = std::vector<std::int8_t>;
    using V16 = std::vector<std::int16_t>;
    using V32 = std::vector<std::int32_t>;
    EXPECT_EQ(product(V8(4, 100), V8(4, 100), 2, 2, 2)[0], 32);
    EXPECT_EQ(product(V16(4, 30000), V16(4, 30000), 2, 2, 2)[0], -11776);
    EXPECT_EQ(product(V32(4, std::int32_t{1} << 30), V32(4, std::int32_t{1} << 30), 2, 2, 2)[0], 0);
    // int8 extremes: 3 * (-128)^2 is 0, 1000 * 127^2 is -24.
    EXPECT_EQ(product(V8(3, -128), V8(3, -128), 1, 3, 1)[0], 0);
    EXPECT_EQ(product(V8(1000, 127), V8(1000, 127), 1, 1000, 1)[0], -24);
    // int16 extremes: 7 * 32767^2 is 7.
    EXPECT_EQ(product(V16(7, 32767), V16(7, 32767), 1, 7, 1)[0], 7);
}

// The first integers a float and a double cannot hold, and the low bits a double
// drops before the wrap runs.
template <typename Product>
void expect_exact_where_a_float_is_not(Product&& product) {
    using V32 = std::vector<std::int32_t>;
    using V64 = std::vector<std::int64_t>;
    // 2^24 + 1 is the first integer a float cannot hold.
    EXPECT_EQ(product(V32{16777217}, V32{1}, 1, 1, 1)[0], 16777217);
    // 2^53 + 1 is the first integer a double cannot hold.
    const std::int64_t past = (std::int64_t{1} << 53) + 1;
    EXPECT_EQ(product(V64{past}, V64{1}, 1, 1, 1)[0], past);
    // (2^62 + 1) * 3 + 7 * 5 modulo 2^64.
    EXPECT_EQ(product(V64{(std::int64_t{1} << 62) + 1, 7}, V64{3, 5}, 1, 2, 1)[0],
              std::int64_t{-4611686018427387866});
    // (-2^63)^2 + (2^63 - 1)^2 is 1 modulo 2^64.
    EXPECT_EQ(product(V64{INT64_MIN, INT64_MAX}, V64{INT64_MIN, INT64_MAX}, 1, 2, 1)[0], 1);
    // 2147483647^2 + 2147483647 * (-2147483648) modulo 2^32.
    EXPECT_EQ(product(V32{INT32_MAX, INT32_MAX}, V32{INT32_MAX, INT32_MIN}, 1, 2, 1)[0],
              -2147483647);
}

}  // namespace lucid::test::int_gemm
