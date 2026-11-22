// lucid/_C/test/core/test_pointer_offset.cpp
// LCD-1047: ``offset_or_null`` is the one place a pointer into a tensor's buffer is offset where the
// buffer may be empty, and an empty tensor's buffer is a null pointer, which no non-zero offset may
// be applied to.

#include <cstddef>

#include <gtest/gtest.h>

#include "../../core/Allocator.h"
#include "../../core/Dtype.h"
#include "../../core/PointerOffset.h"
#include "../../core/Storage.h"

using namespace lucid;

static_assert(offset_or_null<int>(nullptr, 5) == nullptr,
              "a null base stays null whatever the offset");

TEST(PointerOffsetTest, ANullBaseStaysNull) {
    float* null_base = nullptr;
    EXPECT_EQ(offset_or_null(null_base, 7), nullptr);
    EXPECT_EQ(offset_or_null(null_base, std::size_t{17}), nullptr);
    EXPECT_EQ(offset_or_null(null_base, -3), nullptr);
}

TEST(PointerOffsetTest, ALiveBaseMovesExactlyAsPlainArithmeticDoes) {
    float buffer[8] = {};
    EXPECT_EQ(offset_or_null(buffer, 0), buffer);
    EXPECT_EQ(offset_or_null(buffer, 3), buffer + 3);
    EXPECT_EQ(offset_or_null(buffer, std::size_t{5}), buffer + 5);
    EXPECT_EQ(offset_or_null(buffer + 5, -2), buffer + 3);
}

TEST(PointerOffsetTest, ByteBuffersAndConstPointersKeepTheirType) {
    std::byte bytes[16] = {};
    const std::byte* first = offset_or_null(static_cast<const std::byte*>(bytes), 8);
    EXPECT_EQ(first, bytes + 8);
    std::byte* none = offset_or_null(static_cast<std::byte*>(nullptr), 8);
    EXPECT_EQ(none, nullptr);
}

TEST(PointerOffsetTest, AnEmptyAllocationHasNoBufferToOffset) {
    // allocate_aligned_bytes(0) is the null pointer every empty tensor's offset meets.
    const auto empty = allocate_aligned_bytes(0, Device::CPU);
    ASSERT_EQ(empty.get(), nullptr);
    EXPECT_EQ(offset_or_null(empty.get(), 17), nullptr);
}
