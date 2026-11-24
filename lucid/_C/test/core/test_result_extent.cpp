// lucid/_C/test/core/test_result_extent.cpp
//
// LCD-1062: a kernel family refuses a GPU result whose byte count is not its
// shape's element count times the dtype's size.  A shapeless-compiled
// ``maximum`` of ``(0,)`` and ``(1,)`` had returned one element under the label
// ``(0,)``; nothing compared the two, so the tensor was built and the in-place
// form failed much later in ``copy_from``.  ``require_result_extent`` reads only
// the byte count the storage records, so a GpuStorage with no array is enough to
// exercise it.

#include <string>
#include <utility>

#include <gtest/gtest.h>

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ResultExtent.h"
#include "../../core/Shape.h"
#include "../../core/Storage.h"

using namespace lucid;

namespace {

Storage gpu_holding(std::size_t nbytes, Dtype dtype) {
    GpuStorage g;
    g.nbytes = nbytes;
    g.dtype = dtype;
    return Storage{std::move(g)};
}

// The message of the refusal ``require_result_extent`` raises, or empty when it
// accepts the result.
std::string refusal(const Storage& s, const Shape& shape, Dtype dtype) {
    try {
        require_result_extent(s, shape, dtype, "maximum");
    } catch (const LucidError& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST(ResultExtentTest, AcceptsAResultOfExactlyTheShapesBytes) {
    EXPECT_EQ(refusal(gpu_holding(12 * 4, Dtype::F32), {3, 4}, Dtype::F32), "");
    EXPECT_EQ(refusal(gpu_holding(7 * 2, Dtype::F16), {7}, Dtype::F16), "");
    EXPECT_EQ(refusal(gpu_holding(5, Dtype::Bool), {5}, Dtype::Bool), "");
}

TEST(ResultExtentTest, AScalarShapeTakesOneElement) {
    EXPECT_EQ(refusal(gpu_holding(4, Dtype::F32), {}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu_holding(0, Dtype::F32), {}, Dtype::F32), "");
}

TEST(ResultExtentTest, AnEmptyShapeHoldsNoBytes) {
    EXPECT_EQ(refusal(gpu_holding(0, Dtype::F32), {0}, Dtype::F32), "");
    EXPECT_EQ(refusal(gpu_holding(0, Dtype::F32), {3, 0, 4}, Dtype::F32), "");
}

TEST(ResultExtentTest, RefusesOneElementUnderAnEmptyShape) {
    // What the shapeless trace returned for ``maximum((0,), (1,))``.
    const std::string msg = refusal(gpu_holding(4, Dtype::F32), {0}, Dtype::F32);
    EXPECT_NE(msg.find("holds 4 bytes"), std::string::npos) << msg;
    EXPECT_NE(msg.find("(0)"), std::string::npos) << msg;
    EXPECT_NE(msg.find("takes 0"), std::string::npos) << msg;
    EXPECT_THROW(require_result_extent(gpu_holding(4, Dtype::F32), {0}, Dtype::F32, "maximum"),
                 LucidError);
}

TEST(ResultExtentTest, RefusesTooFewAndTooManyBytes) {
    EXPECT_NE(refusal(gpu_holding(8, Dtype::F32), {3}, Dtype::F32), "");
    EXPECT_NE(refusal(gpu_holding(16, Dtype::F32), {3}, Dtype::F32), "");
    // The count is of elements, whatever their width: 12 float32 bytes are 6
    // float16 elements, not 12.
    EXPECT_NE(refusal(gpu_holding(12, Dtype::F16), {3, 4}, Dtype::F16), "");
}

TEST(ResultExtentTest, NamesTheShapeAndTheDtype) {
    const std::string msg = refusal(gpu_holding(4, Dtype::F32), {2, 0}, Dtype::F32);
    EXPECT_NE(msg.find("(2, 0)"), std::string::npos) << msg;
    EXPECT_NE(msg.find(std::string(dtype_name(Dtype::F32))), std::string::npos) << msg;
}

TEST(ResultExtentTest, LeavesACpuStorageAlone) {
    // A CPU buffer is sized by the shape when it is allocated, and a view's
    // storage legitimately differs from its shape: only a GPU result is held to
    // the byte count.
    Storage cpu{CpuStorage{}};
    EXPECT_EQ(refusal(cpu, {3, 4}, Dtype::F32), "");
}
