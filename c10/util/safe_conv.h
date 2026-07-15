#pragma once

#include <c10/macros/Macros.h>

#include <type_traits>
#include <utility>

#if !defined(__cpp_lib_integer_comparison_functions)
#include <c10/util/TypeSafeSignMath.h>
#endif

namespace c10 {

namespace detail {
// Defined out-of-line in safe_conv.cpp (part of //c10/util:base) so the throw is
// not inlined at every call site (avoids code-size bloat), and so this header
// stays lightweight.
[[noreturn]] C10_API void report_narrowing_overflow(const char* name);

#if !defined(__cpp_lib_integer_comparison_functions)
// Pre-C++20 fallback for std::in_range (e.g. the ExecuTorch C++17 header
// mirror): true iff `from` is representable in `To`, with no signed->unsigned
// wraparound.
template <typename To, typename From>
C10_HOST_DEVICE constexpr bool in_range(From f) {
  return !c10::less_than_lowest<To>(f) && !c10::greater_than_max<To>(f);
}
#endif

#if defined(__cpp_concepts)
template <typename T>
concept SafeConvIntegral = std::is_integral_v<T> && !std::is_same_v<T, bool>;
#endif
} // namespace detail

// Strict, range-checked integer narrowing conversion.
//
// Unlike c10::wrapping_convert (the renamed c10::checked_convert), this rejects
// ANY value not representable in To -- there is no signed->unsigned
// two's-complement wraparound. Use this wherever a narrowing truncation would be
// a bug; use wrapping_convert only where modular/wrapping semantics are intended.
//
// Usable in both host and device code: on host it throws via
// report_narrowing_overflow; in CUDA/HIP device code (which cannot throw) it
// traps via CUDA_KERNEL_ASSERT_PRINTF, printing name if one is provided.
#if defined(__cpp_concepts)
template <detail::SafeConvIntegral To, detail::SafeConvIntegral From>
#else
template <typename To, typename From>
#endif
C10_HOST_DEVICE To safe_conv(From f, const char* name = nullptr) {
#if !defined(__cpp_concepts)
  static_assert(
      std::is_integral_v<To> && !std::is_same_v<To, bool>,
      "safe_conv requires an integral, non-bool destination type");
  static_assert(
      std::is_integral_v<From> && !std::is_same_v<From, bool>,
      "safe_conv requires an integral, non-bool source type");
#endif
#if defined(__cpp_lib_integer_comparison_functions)
  const bool representable = std::in_range<To>(f);
#else
  const bool representable = detail::in_range<To>(f);
#endif
  if (!representable) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // Device code cannot throw; trap on overflow instead, printing name if set.
    CUDA_KERNEL_ASSERT_PRINTF(
        false,
        "value cannot be safely converted without overflow: %s",
        name != nullptr ? name : "<unknown>");
#else
    detail::report_narrowing_overflow(name);
#endif
  }
  return static_cast<To>(f);
}

} // namespace c10
