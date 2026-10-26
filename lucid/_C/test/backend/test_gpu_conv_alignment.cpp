// Tests that no Metal convolution reaches MLX with channels MLX would pad.
//
// MLX pads unaligned convolution channels to a multiple of 16 inside its
// Metal kernel and fills the padding from a zero scalar it frees before the
// GPU reads it, so a buffer claimed in between turns the padding into
// garbage.  The GPU backend aligns the channels itself
// (``GpuBackend::gpu_align_conv_channels``); these tests walk the MLX graph a
// convolution builds and check every ``Convolution`` node gets aligned
// operands.  Values are checked against the CPU by the Python suite
// (``test_conv_channel_alignment.py``); what matters here is the invariant,
// which a race cannot hide.

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <mlx/array.h>
#include <mlx/primitives.h>

#include "backend/Dispatcher.h"
#include "backend/IBackend.h"
#include "backend/gpu/BinaryKernelLoad.h"
#include "backend/gpu/DivMod.h"
#include "backend/gpu/Pow.h"
#include "core/Storage.h"

using namespace lucid;

namespace {

// Every ``Convolution`` node reachable from ``roots``.
std::vector<mlx::core::array> convolutions(const std::vector<Storage>& roots) {
    std::vector<mlx::core::array> found;
    std::unordered_set<std::uintptr_t> seen;
    std::function<void(const mlx::core::array&)> walk = [&](const mlx::core::array& a) {
        if (!seen.insert(a.id()).second)
            return;
        if (a.has_primitive() && std::string(a.primitive().name()) == "Convolution")
            found.push_back(a);
        for (const auto& in : a.inputs())
            walk(in);
    };
    for (const auto& s : roots)
        if (const auto* g = std::get_if<GpuStorage>(&s); g && g->arr)
            walk(*g->arr);
    return found;
}

// The operand shapes MLX's own padding checks (``conv.cpp``): the input's
// channels are its last axis, the weight's output channels its first.
void expect_aligned(const std::vector<mlx::core::array>& convs, int rank) {
    ASSERT_FALSE(convs.empty()) << "no Convolution node in the graph";
    for (const auto& c : convs) {
        const auto& in = c.inputs()[0];
        const auto& wt = c.inputs()[1];
        const int C = static_cast<int>(in.shape().back());
        const int O = static_cast<int>(wt.shape()[0]);
        if (rank == 2)
            EXPECT_TRUE(C <= 4 || C % 16 == 0) << "input channels " << C;
        else
            EXPECT_EQ(C % 16, 0) << "input channels " << C;
        EXPECT_TRUE(O <= 16 || O % 16 == 0) << "output channels " << O;
    }
}

struct Case {
    int rank;
    int Cin;
    int Cout;
    int size;
};

void check(const Case& c) {
    auto& be = backend::Dispatcher::for_device(Device::GPU);
    const int B = 2;
    const int k = 3;
    Shape xs{B, c.Cin};
    Shape ws{c.Cout, c.Cin};
    Shape os{B, c.Cout};
    int S[3] = {1, 1, 1};
    int K[3] = {1, 1, 1};
    int O[3] = {1, 1, 1};
    for (int i = 0; i < c.rank; ++i) {
        xs.push_back(c.size);
        ws.push_back(k);
        os.push_back(c.size);  // stride 1, pad 1, kernel 3
        S[i] = c.size;
        K[i] = k;
        O[i] = c.size;
    }
    auto x = be.ones(xs, Dtype::F32);
    auto w = be.ones(ws, Dtype::F32);
    auto b = be.zeros(Shape{c.Cout}, Dtype::F32);
    backend::IBackend::ConvNdOpts opts{c.rank, 1, {1, 1, 1}, {1, 1, 1}, {1, 1, 1}};

    auto y =
        be.conv_nd_forward(x, w, b, B, c.Cin, c.Cout, c.Cin, c.Cout, S, K, O, opts, os, Dtype::F32);
    expect_aligned(convolutions({y}), c.rank);

    auto g = be.ones(os, Dtype::F32);
    auto grads =
        be.conv_nd_backward(g, x, w, B, c.Cin, c.Cout, c.Cin, c.Cout, S, K, O, opts, Dtype::F32);
    expect_aligned(convolutions(grads), c.rank);
}

}  // namespace

// The ResNet-on-MNIST block that exposed it: 8 -> 8, 3x3, stride 1.
TEST(GpuConvAlignment, Conv2dEightToEight) {
    check({2, 8, 8, 16});
}

TEST(GpuConvAlignment, Conv2dUnalignedIntoAligned) {
    check({2, 12, 32, 16});
}

// Three-channel video input, unaligned output channels: MLX would pad both.
TEST(GpuConvAlignment, Conv3dRgbIntoTwentyFour) {
    check({3, 3, 24, 6});
}

// LCD-591: 64-bit elementwise Metal launch helper index math past INT32_MAX.
TEST(GpuConvAlignment, MetalLaunch1DCoversPastInt32Max) {
    constexpr std::size_t two_pow_31 = 1ULL << 31;
    for (std::size_t k : {0ULL, 1ULL, 7ULL, 255ULL, 256ULL, 1000003ULL, (1ULL << 31) + 19ULL}) {
        const std::size_t n = two_pow_31 + k;
        const auto launch = backend::metal_launch_1d(n);
        const auto [gx, gy, gz] = launch.grid;
        const auto [tx, ty, tz] = launch.threadgroup;
        EXPECT_GT(gx, 0);
        EXPECT_GT(gy, 1);
        EXPECT_EQ(gz, 1);
        EXPECT_EQ(tx, 256);
        EXPECT_EQ(ty, 1);
        EXPECT_EQ(tz, 1);

        const auto ugx = static_cast<std::uint32_t>(gx);
        const auto ugy = static_cast<std::uint32_t>(gy);
        EXPECT_GE(static_cast<std::uint64_t>(ugx) * static_cast<std::uint64_t>(ugy), n);
        EXPECT_EQ(backend::metal_linear_elem(0u, 0u, ugx), 0u);
        EXPECT_EQ(backend::metal_linear_elem(ugx - 1u, 0u, ugx), ugx - 1u);
        EXPECT_EQ(backend::metal_linear_elem(0u, 1u, ugx), ugx);

        const auto last_y = static_cast<std::uint32_t>((n - 1) / ugx);
        const auto last_x = static_cast<std::uint32_t>((n - 1) % ugx);
        EXPECT_LT(last_y, ugy);
        EXPECT_LT(last_x, ugx);
        EXPECT_EQ(backend::metal_linear_elem(last_x, last_y, ugx), n - 1);

        const auto oob_y = static_cast<std::uint32_t>(n / ugx);
        const auto oob_x = static_cast<std::uint32_t>(n % ugx);
        EXPECT_EQ(backend::metal_linear_elem(oob_x, oob_y, ugx), n);
    }
}

// LCD-591: synthetic small-memory check of the 2-D multi-row shader path
// (``threads_per_grid.y > 1`` with tail threads that must early-return).
TEST(GpuConvAlignment, MetalLaunch1DSyntheticMultiRowGuardsTailThreads) {
    namespace mx = ::mlx::core;
    const std::size_t n = 10;
    const auto launch = backend::metal_launch_1d(n, /*max_tg=*/4, /*max_grid_x=*/4);
    const auto [gx, gy, gz] = launch.grid;
    EXPECT_EQ(gx, 4);
    EXPECT_EQ(gy, 3);
    EXPECT_EQ(gz, 1);

    const mx::Shape shape{2, 5};
    auto a = mx::reshape(mx::arange(1, 11, mx::float32), shape);
    auto b = mx::full({1}, 2.0f, mx::float32);

    auto contig_out = backend::gpu_pow_detail::pow_kernel(false)({a, b}, {shape}, {mx::float32},
                                                                 launch.grid, launch.threadgroup,
                                                                 {{"T", mx::float32},
                                                                  {"AScalar", false},
                                                                  {"BScalar", true},
                                                                  {"FixedY", false},
                                                                  {"YBits", 0},
                                                                  {"YNeg", false}},
                                                                 std::nullopt, false, {})[0];
    contig_out.eval();
    const float* cp = contig_out.data<float>();
    for (int i = 0; i < 10; ++i) {
        const float v = static_cast<float>(i + 1);
        EXPECT_FLOAT_EQ(cp[i], v * v);
    }

    auto strided_out = backend::gpu_divmod_detail::divmod_kernel(true, true)(
        {a, mx::broadcast_to(b, shape)}, {shape}, {mx::float32}, launch.grid, launch.threadgroup,
        {{"T", mx::float32}, {"Mode", 0}}, std::nullopt, false, {})[0];
    strided_out.eval();
    const float* sp = strided_out.data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(sp[i], std::floor(static_cast<float>(i + 1) / 2.0f));
    }
}
