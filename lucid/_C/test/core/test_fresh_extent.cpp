// lucid/_C/test/core/test_fresh_extent.cpp
//
// LCD-1072: ``helpers::fresh`` is the one function a forward builds its result
// tensor through, so it is where a GPU storage of another extent than its shape
// is refused.  LCD-1062 made the check in the three kernel families only; an op
// with a forward of its own (bitwise, compare, linalg, a loss, a norm) built its
// result by hand and skipped it, so the shifts — the ops LCD-1062 found broken —
// were protected by the compile fix and not by the guard.  What is held here is
// the boundary, not the ops: any GPU storage handed to ``fresh``, whoever calls.

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "../../backend/Dispatcher.h"
#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Helpers.h"
#include "../../core/PhantomMode.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"
#include "../../core/TensorImpl.h"
#include "tensor_factory.h"

using namespace lucid;

namespace {

// A GPU storage that records ``nbytes`` and holds no array: the check reads the
// byte count alone, and a refused storage never reaches the tensor.
Storage gpu_holding(std::size_t nbytes, Dtype dtype) {
    GpuStorage g;
    g.nbytes = nbytes;
    g.dtype = dtype;
    return Storage{std::move(g)};
}

// The refusal ``fresh`` raises for ``s``, or empty when it builds the tensor.
std::string refusal(Storage s, const Shape& shape, Dtype dtype) {
    try {
        helpers::fresh(std::move(s), shape, dtype, Device::GPU);
    } catch (const LucidError& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST(FreshExtentTest, RefusesAGpuStorageOfAnotherExtent) {
    // The wrong buffer LCD-1062 found: one element under an empty shape.
    EXPECT_THROW(helpers::fresh(gpu_holding(4, Dtype::I32), {0}, Dtype::I32, Device::GPU),
                 LucidError);
    EXPECT_NE(refusal(gpu_holding(8, Dtype::F32), {3}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu_holding(16, Dtype::F32), {3}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu_holding(12, Dtype::F16), {3, 4}, Dtype::F16), "");
}

TEST(FreshExtentTest, NamesTheOpItIsBuildingTheResultOf) {
    ErrorContextGuard op("bitwise_left_shift");
    const std::string msg = refusal(gpu_holding(4, Dtype::I32), {0}, Dtype::I32);
    EXPECT_EQ(msg.rfind("bitwise_left_shift: ", 0), 0u) << msg;
    EXPECT_NE(msg.find("holds 4 bytes"), std::string::npos) << msg;
    EXPECT_NE(msg.find("[0]"), std::string::npos) << msg;
}

TEST(FreshExtentTest, BuildsAGpuResultOfExactlyTheShapesBytes) {
    if (!test::gpu_available())
        GTEST_SKIP() << "no Metal device";
    auto& gpu = backend::Dispatcher::for_device(Device::GPU);
    for (const Shape& shape : {Shape{}, Shape{5}, Shape{3, 4}, Shape{0}, Shape{2, 0, 3}}) {
        auto t = helpers::fresh(gpu.zeros(shape, Dtype::F32), shape, Dtype::F32, Device::GPU);
        ASSERT_NE(t, nullptr);
        EXPECT_EQ(t->shape(), shape);
        EXPECT_EQ(t->allocation_nbytes(), t->nbytes()) << shape_to_string(shape);
    }
}

TEST(FreshExtentTest, RefusesARealGpuArrayLabelledWithAnotherShape) {
    if (!test::gpu_available())
        GTEST_SKIP() << "no Metal device";
    auto& gpu = backend::Dispatcher::for_device(Device::GPU);
    // The array is real and complete; the shape it is wrapped as is not its own.
    EXPECT_NE(refusal(gpu.zeros({4}, Dtype::F32), {0}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu.zeros({0}, Dtype::F32), {1}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu.zeros({3}, Dtype::F32), {3}, Dtype::F16), "");
}

TEST(FreshExtentTest, LeavesACpuStorageAlone) {
    // A CPU buffer is sized by the shape when it is allocated, and only a GPU
    // result is held to the byte count.
    auto t = helpers::fresh(Storage{helpers::allocate_cpu({2, 3}, Dtype::F32)}, {2, 3}, Dtype::F32,
                            Device::CPU);
    ASSERT_NE(t, nullptr);
    EXPECT_EQ(t->nbytes(), 6 * sizeof(float));
}

TEST(FreshExtentTest, LeavesAPhantomStorageAlone) {
    // A phantom has no buffer to compare; it is sized by the shape it is made for.
    PhantomModeGuard on(true);
    auto t = helpers::fresh(phantom_storage({3, 4}, Dtype::F32, Device::GPU), {3, 4}, Dtype::F32,
                            Device::GPU);
    ASSERT_NE(t, nullptr);
    EXPECT_TRUE(t->is_phantom());
}

TEST(FreshExtentTest, MarksAGradientLeafOnlyWhenAskedTo) {
    auto plain = helpers::fresh(Storage{helpers::allocate_cpu({2}, Dtype::F32)}, {2}, Dtype::F32,
                                Device::CPU);
    auto leaf = helpers::fresh(Storage{helpers::allocate_cpu({2}, Dtype::F32)}, {2}, Dtype::F32,
                               Device::CPU, true);
    EXPECT_FALSE(plain->requires_grad());
    EXPECT_TRUE(leaf->requires_grad());
}
