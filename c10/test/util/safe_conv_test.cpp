#include <c10/util/safe_conv.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace c10 {
namespace {

TEST(safeConvTest, InRangePassesThrough) {
  EXPECT_EQ(safe_conv<int32_t>(int64_t{5}), 5);
  EXPECT_EQ(
      safe_conv<int32_t>(int64_t{std::numeric_limits<int32_t>::max()}),
      std::numeric_limits<int32_t>::max());
  EXPECT_EQ(
      safe_conv<int32_t>(int64_t{std::numeric_limits<int32_t>::min()}),
      std::numeric_limits<int32_t>::min());
}

TEST(safeConvTest, SameTypeIsNoOp) {
  EXPECT_EQ(safe_conv<int32_t>(int32_t{42}), 42);
  EXPECT_EQ(
      safe_conv<uint64_t>(std::numeric_limits<uint64_t>::max()),
      std::numeric_limits<uint64_t>::max());
}

TEST(safeConvTest, OverflowHighThrows) {
  constexpr int64_t tooBig = int64_t{std::numeric_limits<int32_t>::max()} + 1;
  EXPECT_THROW(safe_conv<int32_t>(tooBig), std::runtime_error);
}

TEST(safeConvTest, OverflowLowThrows) {
  constexpr int64_t tooSmall = int64_t{std::numeric_limits<int32_t>::min()} - 1;
  EXPECT_THROW(safe_conv<int32_t>(tooSmall), std::runtime_error);
}

// The key differentiator from wrapping_convert: a negative value converted to an
// unsigned type is REJECTED, never wrapped to a large positive value.
TEST(safeConvTest, NegativeToUnsignedThrows) {
  EXPECT_THROW(safe_conv<uint8_t>(int32_t{-1}), std::runtime_error);
  EXPECT_THROW(safe_conv<uint32_t>(int64_t{-1}), std::runtime_error);
}

TEST(safeConvTest, UnsignedToUnsignedBoundary) {
  EXPECT_EQ(safe_conv<uint32_t>(int64_t{0}), 0u);
  EXPECT_EQ(
      safe_conv<uint32_t>(int64_t{std::numeric_limits<uint32_t>::max()}),
      std::numeric_limits<uint32_t>::max());
  constexpr int64_t tooBig = int64_t{std::numeric_limits<uint32_t>::max()} + 1;
  EXPECT_THROW(safe_conv<uint32_t>(tooBig), std::runtime_error);
}

TEST(safeConvTest, UnsignedToSignedOverflowThrows) {
  EXPECT_EQ(safe_conv<int8_t>(uint32_t{100}), 100);
  EXPECT_THROW(safe_conv<int8_t>(uint32_t{200}), std::runtime_error);
}

} // namespace
} // namespace c10
