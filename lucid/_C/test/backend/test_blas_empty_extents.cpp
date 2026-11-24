// lucid/_C/test/backend/test_blas_empty_extents.cpp
//
// The BLAS wrappers answer an empty extent themselves.  Handed to
// Accelerate, a zero extent arrives as a zero leading dimension, and
// Accelerate's error handler exits the process with status 255 — an
// attention with no keys killed the interpreter that way.  These pin the
// arithmetic the wrappers substitute, which is BLAS's own:
//
//   - an output with no elements is left alone;
//   - an empty contraction leaves ``C = beta * C``, and with ``beta == 0``
//     ``C`` is overwritten with zeros, never read — a NaN in it must not
//     survive as ``0 * NaN``;
//   - the leading dimension is respected: padding between rows is not
//     written.

#include <cmath>
#include <complex>
#include <limits>
#include <vector>

#include <gtest/gtest.h>

#include "../../backend/cpu/Blas.h"

namespace cpu = lucid::backend::cpu;

namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN();
constexpr float kPad = -7.0f;  // a value no case should produce

}  // namespace

TEST(BlasEmptyExtents, EmptyContractionWithBetaZeroWritesZerosOverNaN) {
    // 3 x 4 output, ldc = 4, K = 0 — the shape of an attention with no keys.
    std::vector<float> C(12, kNan);
    cpu::sgemm(false, false, 3, 4, 0, 1.0f, nullptr, 0, nullptr, 4, 0.0f, C.data(), 4);
    for (float v : C)
        EXPECT_EQ(v, 0.0f);
}

TEST(BlasEmptyExtents, EmptyContractionScalesByBeta) {
    std::vector<float> C(6, 1.5f);
    cpu::sgemm(false, true, 2, 3, 0, 1.0f, nullptr, 0, nullptr, 0, 2.0f, C.data(), 3);
    for (float v : C)
        EXPECT_EQ(v, 3.0f);

    std::vector<double> D(6, 1.5);
    cpu::dgemm(true, false, 2, 3, 0, 1.0, nullptr, 2, nullptr, 3, 1.0, D.data(), 3);
    for (double v : D)
        EXPECT_EQ(v, 1.5);  // beta == 1: an accumulation of nothing
}

TEST(BlasEmptyExtents, EmptyContractionRespectsLeadingDimension) {
    // 2 x 2 block inside rows of 3: the third column is padding.
    std::vector<float> C = {1.0f, 1.0f, kPad, 1.0f, 1.0f, kPad};
    cpu::sgemm(false, false, 2, 2, 0, 1.0f, nullptr, 0, nullptr, 2, 0.0f, C.data(), 3);
    EXPECT_EQ(C, (std::vector<float>{0.0f, 0.0f, kPad, 0.0f, 0.0f, kPad}));
}

TEST(BlasEmptyExtents, EmptyOutputIsNotTouched) {
    std::vector<float> sentinel(4, kPad);
    // M == 0 and N == 0, each with a zero leading dimension, as a row-major
    // caller passes them.
    cpu::sgemm(false, false, 0, 4, 3, 1.0f, nullptr, 3, nullptr, 4, 0.0f, sentinel.data(), 4);
    cpu::sgemm(false, false, 3, 0, 4, 1.0f, nullptr, 4, nullptr, 0, 0.0f, sentinel.data(), 0);
    std::vector<double> dsentinel(4, kPad);
    cpu::dgemm(false, false, 0, 0, 0, 1.0, nullptr, 0, nullptr, 0, 0.0, dsentinel.data(), 0);
    for (float v : sentinel)
        EXPECT_EQ(v, kPad);
    for (double v : dsentinel)
        EXPECT_EQ(v, kPad);
}

TEST(BlasEmptyExtents, NonEmptyProductIsUnchanged) {
    // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    const std::vector<float> A = {1, 2, 3, 4};
    const std::vector<float> B = {5, 6, 7, 8};
    std::vector<float> C(4, kNan);
    cpu::sgemm(false, false, 2, 2, 2, 1.0f, A.data(), 2, B.data(), 2, 0.0f, C.data(), 2);
    EXPECT_EQ(C, (std::vector<float>{19, 22, 43, 50}));
}

TEST(BlasEmptyExtents, GemvEmptyReductionScalesOutput) {
    // A is 3 x 0: y (length 3) = beta * y.
    std::vector<float> y(3, kNan);
    cpu::sgemv(false, 3, 0, 1.0f, nullptr, 0, nullptr, 1, 0.0f, y.data(), 1);
    for (float v : y)
        EXPECT_EQ(v, 0.0f);

    // Transposed, A is 0 x 2: y (length 2) = 2 * y, at stride 2.
    std::vector<double> z = {1.0, kPad, 1.0, kPad};
    cpu::dgemv(true, 0, 2, 1.0, nullptr, 2, nullptr, 1, 2.0, z.data(), 2);
    EXPECT_EQ(z, (std::vector<double>{2.0, kPad, 2.0, kPad}));
}

TEST(BlasEmptyExtents, GemvEmptyOutputIsNotTouched) {
    std::vector<float> y(2, kPad);
    cpu::sgemv(false, 0, 3, 1.0f, nullptr, 3, nullptr, 1, 0.0f, y.data(), 1);
    cpu::sgemv(true, 3, 0, 1.0f, nullptr, 0, nullptr, 1, 0.0f, y.data(), 1);
    for (float v : y)
        EXPECT_EQ(v, kPad);
}

TEST(BlasEmptyExtents, AxpyOfNothing) {
    std::vector<float> y(2, kPad);
    cpu::saxpy(0, 2.0f, nullptr, y.data());
    std::vector<double> z(2, kPad);
    cpu::daxpy(0, 2.0, nullptr, z.data());
    EXPECT_EQ(y, (std::vector<float>(2, kPad)));
    EXPECT_EQ(z, (std::vector<double>(2, kPad)));
}

TEST(BlasEmptyExtents, TrsmWithNoRowsOrNoRightHandSidesIsNotTouched) {
    // A 3 x 3 triangle against a 3 x 0 right-hand side arrives with
    // ldb = 0, and a 0 x 0 triangle with lda = 0 — each one an exit(255)
    // if it reached Accelerate.
    const std::vector<float> A = {1, 2, 3, 0, 4, 5, 0, 0, 6};
    std::vector<float> sentinel(2, kPad);
    cpu::strsm(true, false, 3, 0, A.data(), 3, sentinel.data(), 0);
    cpu::strsm(false, true, 0, 2, nullptr, 0, sentinel.data(), 2);
    std::vector<double> dsentinel(2, kPad);
    cpu::dtrsm(true, false, 0, 0, nullptr, 0, dsentinel.data(), 0);
    for (float v : sentinel)
        EXPECT_EQ(v, kPad);
    for (double v : dsentinel)
        EXPECT_EQ(v, kPad);
}

TEST(BlasEmptyExtents, TrsmSolvesBothTrianglesInPlace) {
    // Upper [[2, 1], [0, 4]] x = [4, 8]  ->  x = [1, 2], with two right-hand
    // sides in a row-major 2 x 2 B (the second column doubled).  The lower
    // triangle holds a value that must not be read.
    const std::vector<float> U = {2, 1, kNan, 4};
    std::vector<float> B = {4, 8, 8, 16};
    cpu::strsm(true, false, 2, 2, U.data(), 2, B.data(), 2);
    EXPECT_EQ(B, (std::vector<float>{1, 2, 2, 4}));

    // Lower unit [[1, 0], [3, 1]] with a diagonal that must not be read:
    // x0 = 1, x1 = 5 - 3 * 1 = 2.
    const std::vector<double> L = {kNan, kNan, 3, kNan};
    std::vector<double> b = {1, 5};
    cpu::dtrsm(false, true, 2, 1, L.data(), 2, b.data(), 1);
    EXPECT_EQ(b, (std::vector<double>{1, 2}));
}

TEST(BlasEmptyExtents, TrsmOnASingularTriangleGivesTheIeeeResult) {
    // [[1, 2], [0, 0]] x = [1, 1]: back substitution divides 1 by the zero
    // pivot (+inf), then 1 - 2 * inf = -inf.  No refusal, no partial result
    // — the reference's answer for a singular triangle.
    const std::vector<float> U = {1, 2, 0, 0};
    std::vector<float> b = {1, 1};
    cpu::strsm(true, false, 2, 1, U.data(), 2, b.data(), 1);
    EXPECT_TRUE(std::isinf(b[0]) && b[0] < 0);
    EXPECT_TRUE(std::isinf(b[1]) && b[1] > 0);

    // 0 / 0 is NaN.
    std::vector<double> z = {1, 0};
    const std::vector<double> Ud = {1, 2, 0, 0};
    cpu::dtrsm(true, false, 2, 1, Ud.data(), 2, z.data(), 1);
    EXPECT_TRUE(std::isnan(z[1]));
}

TEST(BlasComplex, EmptyContractionWithBetaZeroWritesZerosOverNaN) {
    const std::complex<float> nan{kNan, kNan};
    std::vector<std::complex<float>> C(12, nan);
    cpu::cgemm(false, false, 3, 4, 0, {1.0f, 0.0f}, nullptr, 0, nullptr, 4, {0.0f, 0.0f}, C.data(),
               4);
    for (const auto& v : C)
        EXPECT_EQ(v, (std::complex<float>{0.0f, 0.0f}));

    std::vector<std::complex<double>> D(6, {1.5, 0.0});
    cpu::zgemm(false, false, 2, 3, 0, {1.0, 0.0}, nullptr, 0, nullptr, 3, {0.0, 1.0}, D.data(), 3);
    for (const auto& v : D)
        EXPECT_EQ(v, (std::complex<double>{0.0, 1.5}));  // beta = i scales the old C
}

TEST(BlasComplex, GemmIsAPlainProductNotAConjugateOne) {
    // [[i, 1]] (1 x 2) times [[i], [2]] (2 x 1) = i*i + 1*2 = 1.  A conjugate
    // product would give (-i)*i + 2 = 3.
    const std::vector<std::complex<double>> A = {{0, 1}, {1, 0}};
    const std::vector<std::complex<double>> B = {{0, 1}, {2, 0}};
    std::vector<std::complex<double>> C(1);
    cpu::zgemm(false, false, 1, 1, 2, {1.0, 0.0}, A.data(), 2, B.data(), 1, {0.0, 0.0}, C.data(),
               1);
    EXPECT_EQ(C[0], (std::complex<double>{1.0, 0.0}));
    // The transpose flag is a plain transpose too: A^T (2 x 1) as a 2 x 1 operand.
    std::vector<std::complex<double>> D(1);
    cpu::zgemm(true, false, 1, 1, 2, {1.0, 0.0}, A.data(), 1, B.data(), 1, {0.0, 0.0}, D.data(), 1);
    EXPECT_EQ(D[0], (std::complex<double>{1.0, 0.0}));
}

TEST(BlasComplex, TrsmWithNoRowsOrNoRightHandSidesIsNotTouched) {
    const std::vector<std::complex<float>> A(9, {1.0f, 1.0f});
    std::vector<std::complex<float>> sentinel(2, {kPad, kPad});
    cpu::ctrsm(true, false, 3, 0, A.data(), 3, sentinel.data(), 0);
    cpu::ctrsm(false, true, 0, 2, nullptr, 0, sentinel.data(), 2);
    std::vector<std::complex<double>> dsentinel(2, {kPad, kPad});
    cpu::ztrsm(true, false, 0, 0, nullptr, 0, dsentinel.data(), 0);
    for (const auto& v : sentinel)
        EXPECT_EQ(v, (std::complex<float>{kPad, kPad}));
    for (const auto& v : dsentinel)
        EXPECT_EQ(v, (std::complex<double>{kPad, kPad}));
}

TEST(BlasComplex, TrsmSolvesBothTrianglesInPlace) {
    // Upper [[2i, 1], [x, 4]] z = [2i * (1 + i) + 2, 8]: back substitution
    // gives z1 = 2, z0 = (b0 - 1 * 2) / (2i) = 1 + i.  The strict lower entry
    // is NaN and must not be read.
    const std::complex<double> nan{kNan, kNan};
    const std::vector<std::complex<double>> U = {{0, 2}, {1, 0}, nan, {4, 0}};
    std::vector<std::complex<double>> b = {std::complex<double>{0, 2} * std::complex<double>{1, 1} +
                                               std::complex<double>{2, 0},
                                           {8, 0}};
    cpu::ztrsm(true, false, 2, 1, U.data(), 2, b.data(), 1);
    EXPECT_NEAR(b[0].real(), 1.0, 1e-12);
    EXPECT_NEAR(b[0].imag(), 1.0, 1e-12);
    EXPECT_NEAR(b[1].real(), 2.0, 1e-12);
    EXPECT_NEAR(b[1].imag(), 0.0, 1e-12);

    // Lower unit [[?, 0], [3i, ?]]: z0 = 1 + i, z1 = b1 - 3i * z0.
    const std::vector<std::complex<float>> L = {{kNan, kNan}, {kNan, kNan}, {0, 3}, {kNan, kNan}};
    std::vector<std::complex<float>> c = {{1, 1}, {5, 0}};
    cpu::ctrsm(false, true, 2, 1, L.data(), 2, c.data(), 1);
    EXPECT_EQ(c[0], (std::complex<float>{1, 1}));
    const std::complex<float> want1 =
        std::complex<float>{5, 0} - std::complex<float>{0, 3} * std::complex<float>{1, 1};
    EXPECT_EQ(c[1], want1);
}
