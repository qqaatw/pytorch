#pragma once

#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/TypeCast.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

struct HeaderOnlySymbolicOpaque;

// This layout is shared across the stable ABI. Numeric values are stored
// inline; symbolic pointers refer to objects owned by libtorch.
struct HeaderOnlyScalarData {
  enum class Kind : uint32_t {
    Int,
    UInt,
    Double,
    Bool,
    Complex,
    SymInt,
    SymFloat,
    SymBool,
  };

  static constexpr uint32_t OwnsSymbolic = 1;
  Kind kind = Kind::Int;
  uint32_t flags = 0;
  union Payload {
    int64_t integer;
    uint64_t unsigned_integer;
    double floating;
    struct {
      double real;
      double imag;
    } complex;
    // Owning records relinquish this handle when unboxed. Borrowed records
    // never permit ownership transfer, even though this slot is mutable.
    mutable HeaderOnlySymbolicOpaque* symbolic;
  } value{};
  // Null for borrowed storage; otherwise releases storage in its allocating
  // module.
  void (*destroy)(const HeaderOnlyScalarData*) = nullptr;
};

static_assert(std::is_standard_layout_v<HeaderOnlyScalarData>);
static_assert(std::is_trivially_copyable_v<HeaderOnlyScalarData>);

HIDDEN_NAMESPACE_BEGIN(torch, headeronly)

class HeaderOnlyScalar {
 public:
  using Kind = HeaderOnlyScalarData::Kind;

  HeaderOnlyScalar() = default;

  HeaderOnlyScalar(const HeaderOnlyScalar&) = default;
  HeaderOnlyScalar& operator=(const HeaderOnlyScalar&) = default;

  HeaderOnlyScalar(HeaderOnlyScalar&& other) noexcept
      : data_(other.data_), symbolic_(std::move(other.symbolic_)) {
    other.data_ = {};
  }

  HeaderOnlyScalar& operator=(HeaderOnlyScalar&& other) noexcept {
    if (this != &other) {
      data_ = other.data_;
      symbolic_ = std::move(other.symbolic_);
      other.data_ = {};
    }
    return *this;
  }

  template <typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
  HeaderOnlyScalar(T value) {
    if constexpr (std::is_same_v<T, bool>) {
      data_.kind = Kind::Bool;
      data_.value.integer = value;
    } else if constexpr (std::is_unsigned_v<T>) {
      if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        data_.kind = Kind::UInt;
        data_.value.unsigned_integer = value;
      } else {
        data_.value.integer = static_cast<int64_t>(value);
      }
    } else {
      data_.value.integer = value;
    }
  }

  HeaderOnlyScalar(double value) {
    data_.kind = Kind::Double;
    data_.value.floating = value;
  }

#define HEADER_ONLY_SCALAR_FLOAT_CTOR(type, name) \
  HeaderOnlyScalar(type value) : HeaderOnlyScalar(static_cast<double>(value)) {}

  HEADER_ONLY_SCALAR_FLOAT_CTOR(float, Float)
  HEADER_ONLY_SCALAR_FLOAT_CTOR(c10::Half, Half)
  HEADER_ONLY_SCALAR_FLOAT_CTOR(c10::BFloat16, BFloat16)
  AT_FORALL_FLOAT8_TYPES(HEADER_ONLY_SCALAR_FLOAT_CTOR)
#undef HEADER_ONLY_SCALAR_FLOAT_CTOR

  template <typename T>
  HeaderOnlyScalar(c10::complex<T> value) {
    data_.kind = Kind::Complex;
    data_.value.complex = {
        static_cast<double>(value.real()), static_cast<double>(value.imag())};
  }

  static HeaderOnlyScalar fromSymbolic(
      Kind kind,
      std::shared_ptr<HeaderOnlySymbolicOpaque> owner) {
    STD_TORCH_CHECK(
        kind == Kind::SymInt || kind == Kind::SymFloat || kind == Kind::SymBool,
        "Expected a symbolic Scalar kind");
    STD_TORCH_CHECK(owner != nullptr, "Expected a non-null symbolic Scalar");
    HeaderOnlyScalar result;
    result.data_.kind = kind;
    result.data_.value.symbolic = owner.get();
    result.symbolic_ = std::move(owner);
    return result;
  }

  static HeaderOnlyScalar fromData(const HeaderOnlyScalarData& data) {
    STD_TORCH_CHECK(
        data.kind <= Kind::Complex, "Symbolic Scalar data requires an owner");
    HeaderOnlyScalar result;
    result.data_ = data;
    result.data_.flags = 0;
    result.data_.destroy = nullptr;
    return result;
  }

  const HeaderOnlyScalarData& data() const {
    return data_;
  }

  const std::shared_ptr<HeaderOnlySymbolicOpaque>& symbolicOwner() const {
    return symbolic_;
  }

  bool isSymInt() const {
    return data_.kind == Kind::SymInt;
  }

  bool isSymFloat() const {
    return data_.kind == Kind::SymFloat;
  }

  bool isSymBool() const {
    return data_.kind == Kind::SymBool;
  }

  bool isSymbolic() const {
    return isSymInt() || isSymFloat() || isSymBool();
  }

  bool isBoolean() const {
    return data_.kind == Kind::Bool || isSymBool();
  }

  bool isIntegral(bool includeBool) const {
    return data_.kind == Kind::Int || data_.kind == Kind::UInt || isSymInt() ||
        (includeBool && isBoolean());
  }

  bool isFloatingPoint() const {
    return data_.kind == Kind::Double || isSymFloat();
  }

  bool isComplex() const {
    return data_.kind == Kind::Complex;
  }

  bool isUnsigned() const {
    return data_.kind == Kind::UInt ||
        (data_.kind == Kind::Int && data_.value.integer >= 0);
  }

  ScalarType type() const {
    switch (data_.kind) {
      case Kind::Int:
      case Kind::SymInt:
        return ScalarType::Long;
      case Kind::UInt:
        return ScalarType::UInt64;
      case Kind::Double:
      case Kind::SymFloat:
        return ScalarType::Double;
      case Kind::Bool:
      case Kind::SymBool:
        return ScalarType::Bool;
      case Kind::Complex:
        return ScalarType::ComplexDouble;
    }
    STD_TORCH_CHECK(false, "Unknown Scalar kind");
  }

  template <typename T>
  T to() const {
    switch (data_.kind) {
      case Kind::Int:
        return unsafe_wrapping_convert<T>(data_.value.integer, "Scalar");
      case Kind::UInt:
        return checked_convert<T>(data_.value.unsigned_integer, "Scalar");
      case Kind::Double:
        return checked_convert<T>(data_.value.floating, "Scalar");
      case Kind::Bool:
        return checked_convert<T>(
            static_cast<bool>(data_.value.integer), "Scalar");
      case Kind::Complex:
        return checked_convert<T>(
            c10::complex<double>(
                data_.value.complex.real, data_.value.complex.imag),
            "Scalar");
      default:
        STD_TORCH_CHECK(
            false, "Symbolic Scalars cannot be converted to concrete values");
    }
  }

#define HEADER_ONLY_SCALAR_ACCESSOR(type, name) \
  type to##name() const {                       \
    return to<type>();                          \
  }

  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(HEADER_ONLY_SCALAR_ACCESSOR)
  HEADER_ONLY_SCALAR_ACCESSOR(uint16_t, UInt16)
  HEADER_ONLY_SCALAR_ACCESSOR(uint32_t, UInt32)
  HEADER_ONLY_SCALAR_ACCESSOR(uint64_t, UInt64)
  HEADER_ONLY_SCALAR_ACCESSOR(c10::complex<c10::BFloat16>, BComplex32)
#undef HEADER_ONLY_SCALAR_ACCESSOR

 private:
  HeaderOnlyScalarData data_;
  std::shared_ptr<HeaderOnlySymbolicOpaque> symbolic_;
};

HIDDEN_NAMESPACE_END(torch, headeronly)
