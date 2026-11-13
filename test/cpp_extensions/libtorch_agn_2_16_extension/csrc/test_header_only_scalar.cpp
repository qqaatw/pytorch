#include <array>
#include <optional>
#include <tuple>
#include <vector>

#include <torch/csrc/stable/library.h>
#include <torch/headeronly/core/Scalar.h>
#include <torch/csrc/stable/tensor.h>

using Scalar = torch::headeronly::HeaderOnlyScalar;
using torch::stable::Tensor;

template <typename T, typename = void>
struct CanPackScalar : std::false_type {};

template <typename T>
struct CanPackScalar<
    T,
    std::void_t<decltype(torch::stable::detail::from(std::declval<T>()))>>
    : std::true_type {};

static_assert(!std::is_copy_constructible_v<HeaderOnlyScalarData>);
static_assert(!std::is_copy_assignable_v<HeaderOnlyScalarData>);
static_assert(std::is_copy_constructible_v<Scalar>);
static_assert(std::is_copy_assignable_v<Scalar>);
static_assert(std::is_nothrow_move_constructible_v<Scalar>);
static_assert(std::is_nothrow_move_assignable_v<Scalar>);
static_assert(CanPackScalar<const Scalar&>::value);
static_assert(!CanPackScalar<Scalar&&>::value);
static_assert(!CanPackScalar<std::optional<Scalar>&&>::value);
static_assert(!CanPackScalar<std::vector<Scalar>&&>::value);

Scalar my_scalar_identity(Scalar value) {
  return value;
}

Scalar my_scalar_copy_move(Scalar value) {
  Scalar copy = value;
  Scalar& alias = copy;
  copy = alias;
  copy = std::move(alias);
  Scalar assigned(value);
  assigned = copy;
  Scalar moved(std::move(assigned));
  STD_TORCH_CHECK(assigned.isIntegral(false) && assigned.toLong() == 0, "Move must clear the source");
  assigned = value;
  assigned = std::move(moved);
  STD_TORCH_CHECK(moved.isIntegral(false) && moved.toLong() == 0, "Move assignment must clear the source");
  value = Scalar();
  copy = Scalar();
  return assigned;
}

Scalar my_scalar_assign(Scalar source, Scalar destination, bool move) {
  if (move) {
    destination = std::move(source);
    STD_TORCH_CHECK(source.isIntegral(false) && source.toLong() == 0,
                    "Move assignment must clear the source");
  } else {
    destination = source;
  }
  return destination;
}

std::tuple<bool, bool, bool> my_scalar_symbolic_kind(Scalar value) {
  return {value.isSymInt(), value.isSymFloat(), value.isSymBool()};
}

Scalar my_scalar_dispatch(Scalar value) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(value)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      C10_STRINGIZE(STABLE_LIB_NAME) "::my_scalar_identity",
      "",
      stack.data(),
      TORCH_ABI_VERSION));
  return torch::stable::detail::to<Scalar>(stack[0]);
}

bool my_scalar_borrowed(Scalar value) {
  const auto packed = torch::stable::detail::from(value);
  const auto* data = torch::stable::detail::to<const HeaderOnlyScalarData*>(packed);
  return data == &value.data() && data->destroy == nullptr &&
      !(data->flags & HeaderOnlyScalarData::OwnsSymbolic);
}

Scalar my_scalar_transfer(Scalar value) {
  STD_TORCH_CHECK(value.isSymbolic(), "Expected a symbolic Scalar");
  auto* original = value.data().value.symbolic;
  bool rejects_borrow = false;
  try {
    auto invalid = torch::stable::detail::to<Scalar>(
        torch::stable::detail::from(value));
  } catch (const std::runtime_error& error) {
    rejects_borrow = std::string(error.what()).find("borrowed") != std::string::npos;
  }
  STD_TORCH_CHECK(rejects_borrow, "Borrowed symbolic records must reject adoption");
  STD_TORCH_CHECK(value.data().value.symbolic == original,
                  "Rejected adoption must preserve the borrowed handle");

  std::array<StableIValue, 1> stack{torch::stable::detail::from(value)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      C10_STRINGIZE(STABLE_LIB_NAME) "::my_scalar_identity",
      "",
      stack.data(),
      TORCH_ABI_VERSION));
  auto* data = torch::stable::detail::to<HeaderOnlyScalarData*>(stack[0]);
  // Keep the record alive to check that its handle can only be adopted once.
  auto storage = std::unique_ptr<HeaderOnlyScalarData, void (*)(const HeaderOnlyScalarData*)>(
      data, data->destroy);
  data->destroy = nullptr;
  Scalar adopted = torch::stable::detail::to<Scalar>(stack[0]);
  STD_TORCH_CHECK(data->value.symbolic == nullptr, "Adoption must clear the record handle");
  Scalar copy = adopted;
  STD_TORCH_CHECK(copy.data().value.symbolic == adopted.data().value.symbolic,
                  "Scalar copies must retain the same symbolic handle");
  adopted = Scalar();

  bool rejects_second = false;
  try {
    auto invalid = torch::stable::detail::to<Scalar>(stack[0]);
  } catch (const std::runtime_error& error) {
    rejects_second = std::string(error.what()).find("already transferred") != std::string::npos;
  }
  STD_TORCH_CHECK(rejects_second, "Symbolic records must reject a second adoption");
  return copy;
}

void my_scalar_throw(Scalar value) {
  STD_TORCH_CHECK(value.isSymbolic(), "Expected a symbolic Scalar");
  STD_TORCH_CHECK(false, "test scalar failure");
}

bool my_scalar_numeric_types() {
#define CHECK_SCALAR_TYPE(type, name)                                    \
  {                                                                     \
    const type value = c10::convert<type>(1.0);                          \
    const Scalar scalar(value);                                         \
    STD_TORCH_CHECK(                                                     \
        scalar.toDouble() == 1.0 &&                                      \
            c10::convert<double>(scalar.to<type>()) == 1.0,               \
        "Scalar conversion failed for ", #name);                        \
  }
  AT_FORALL_SCALAR_TYPES_WITH_COMPLEX(CHECK_SCALAR_TYPE)
  CHECK_SCALAR_TYPE(uint16_t, UInt16)
  CHECK_SCALAR_TYPE(uint32_t, UInt32)
  CHECK_SCALAR_TYPE(uint64_t, UInt64)
  CHECK_SCALAR_TYPE(c10::complex<c10::BFloat16>, BComplex32)
#undef CHECK_SCALAR_TYPE
  return true;
}

std::vector<Scalar> my_scalar_list(std::vector<Scalar> values) {
  return values;
}

std::optional<Scalar> my_scalar_optional(std::optional<Scalar> value) {
  return value;
}

std::tuple<bool, bool, bool, bool> my_scalar_kind(Scalar value) {
  return {
      value.isBoolean(),
      value.isIntegral(false),
      value.isFloatingPoint(),
      value.isComplex()};
}

Scalar my_scalar_from_int(int64_t value) {
  return Scalar(value);
}

Scalar my_scalar_from_uint(int64_t high, int64_t low) {
  return Scalar((static_cast<uint64_t>(high) << 32) | static_cast<uint64_t>(low));
}

Scalar my_scalar_from_double(double value) {
  return Scalar(value);
}

Scalar my_scalar_from_bool(bool value) {
  return Scalar(value);
}

Scalar my_scalar_from_complex(double real, double imag) {
  return Scalar(c10::complex<double>(real, imag));
}

std::tuple<int64_t, double, bool> my_scalar_to_real(Scalar value) {
  return {value.toLong(), value.toDouble(), value.toBool()};
}

std::tuple<double, double> my_scalar_to_complex(Scalar value) {
  auto result = value.toComplexDouble();
  return {result.real(), result.imag()};
}

Tensor my_scalar_add(Tensor self, Tensor other, Scalar alpha) {
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(self),
      torch::stable::detail::from(other),
      torch::stable::detail::from(alpha)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::add", "Tensor", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Tensor>(stack[0]);
}

Scalar my_scalar_item(Tensor self) {
  std::array<StableIValue, 1> stack{torch::stable::detail::from(self)};
  STABLE_TORCH_ERROR_CODE_CHECK(torch_call_dispatcher(
      "aten::item", "", stack.data(), TORCH_ABI_VERSION));
  return torch::stable::detail::to<Scalar>(stack[0]);
}

STABLE_TORCH_LIBRARY_FRAGMENT(STABLE_LIB_NAME, m) {
  m.def("my_scalar_identity(Scalar value) -> Scalar");
  m.def("my_scalar_assign(Scalar source, Scalar destination, bool move) -> Scalar");
  m.def("my_scalar_copy_move(Scalar value) -> Scalar");
  m.def("my_scalar_symbolic_kind(Scalar value) -> (bool, bool, bool)");
  m.def("my_scalar_dispatch(Scalar value) -> Scalar");
  m.def("my_scalar_borrowed(Scalar value) -> bool");
  m.def("my_scalar_transfer(Scalar value) -> Scalar");
  m.def("my_scalar_throw(Scalar value) -> ()");
  m.def("my_scalar_numeric_types() -> bool");
  m.def("my_scalar_list(Scalar[] values) -> Scalar[]");
  m.def("my_scalar_optional(Scalar? value) -> Scalar?");
  m.def("my_scalar_kind(Scalar value) -> (bool, bool, bool, bool)");
  m.def("my_scalar_from_int(int value) -> Scalar");
  m.def("my_scalar_from_uint(int high, int low) -> Scalar");
  m.def("my_scalar_from_double(float value) -> Scalar");
  m.def("my_scalar_from_bool(bool value) -> Scalar");
  m.def("my_scalar_from_complex(float real, float imag) -> Scalar");
  m.def("my_scalar_to_real(Scalar value) -> (int, float, bool)");
  m.def("my_scalar_to_complex(Scalar value) -> (float, float)");
  m.def("my_scalar_add(Tensor self, Tensor other, Scalar alpha) -> Tensor");
  m.def("my_scalar_item(Tensor self) -> Scalar");
}

STABLE_TORCH_LIBRARY_IMPL(STABLE_LIB_NAME, CompositeExplicitAutograd, m) {
  m.impl("my_scalar_identity", TORCH_BOX(&my_scalar_identity));
  m.impl("my_scalar_assign", TORCH_BOX(&my_scalar_assign));
  m.impl("my_scalar_copy_move", TORCH_BOX(&my_scalar_copy_move));
  m.impl("my_scalar_symbolic_kind", TORCH_BOX(&my_scalar_symbolic_kind));
  m.impl("my_scalar_dispatch", TORCH_BOX(&my_scalar_dispatch));
  m.impl("my_scalar_borrowed", TORCH_BOX(&my_scalar_borrowed));
  m.impl("my_scalar_transfer", TORCH_BOX(&my_scalar_transfer));
  m.impl("my_scalar_throw", TORCH_BOX(&my_scalar_throw));
  m.impl("my_scalar_numeric_types", TORCH_BOX(&my_scalar_numeric_types));
  m.impl("my_scalar_list", TORCH_BOX(&my_scalar_list));
  m.impl("my_scalar_optional", TORCH_BOX(&my_scalar_optional));
  m.impl("my_scalar_kind", TORCH_BOX(&my_scalar_kind));
  m.impl("my_scalar_from_int", TORCH_BOX(&my_scalar_from_int));
  m.impl("my_scalar_from_uint", TORCH_BOX(&my_scalar_from_uint));
  m.impl("my_scalar_from_double", TORCH_BOX(&my_scalar_from_double));
  m.impl("my_scalar_from_bool", TORCH_BOX(&my_scalar_from_bool));
  m.impl("my_scalar_from_complex", TORCH_BOX(&my_scalar_from_complex));
  m.impl("my_scalar_to_real", TORCH_BOX(&my_scalar_to_real));
  m.impl("my_scalar_to_complex", TORCH_BOX(&my_scalar_to_complex));
  m.impl("my_scalar_add", TORCH_BOX(&my_scalar_add));
  m.impl("my_scalar_item", TORCH_BOX(&my_scalar_item));
}
