// lucid/_C/test/core/test_schema_bool_refusal.cpp
// Tests for the dtype refusal an OpSchema declares and SchemaGuard enforces.
//
// The file kept the name it had when the refusal was a bool's alone (LCD-1020);
// it covers every dtype now (LCD-1026).

#include <array>
#include <concepts>
#include <stdexcept>
#include <string>
#include <type_traits>

#include <gtest/gtest.h>

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/OpSchema.h"
#include "../../core/SchemaGuard.h"

using namespace lucid;

namespace {

constexpr std::array<Dtype, 11> kAll = {
    Dtype::Bool, Dtype::I8,  Dtype::I16, Dtype::I32,  Dtype::I64,  Dtype::F16,
    Dtype::F32,  Dtype::F64, Dtype::C64, Dtype::BF16, Dtype::C128,
};

constexpr std::array<Dtype, 4> kSigned = {Dtype::I8, Dtype::I16, Dtype::I32, Dtype::I64};

constexpr std::array<Dtype, 10> kNonBool = {
    Dtype::I8,  Dtype::I16, Dtype::I32, Dtype::I64,  Dtype::F16,
    Dtype::F32, Dtype::F64, Dtype::C64, Dtype::BF16, Dtype::C128,
};

// Restores the process-wide determinism flag whatever the test does to it.
class DeterminismRestore {
public:
    DeterminismRestore() : was_(Determinism::is_enabled()) {}
    ~DeterminismRestore() { Determinism::set_enabled(was_); }

private:
    bool was_;
};

const OpSchema kPlain{"plain_op", 1, AmpPolicy::KeepInput, true};
const OpSchema kKernel = OpSchema{"kernel_op", 1, AmpPolicy::KeepInput, true}.refusing_bool(
    DtypeRefusal::Kernel, CastHint::Number);
const OpSchema kGuard = OpSchema{"guard_op", 1, AmpPolicy::KeepInput, true}.refusing_bool(
    DtypeRefusal::Guard, CastHint::Floating);
const OpSchema kIntegers = OpSchema{"int_op", 1, AmpPolicy::KeepInput, true}.refusing(
    kSignedIntegers, DtypeRefusal::Kernel);
const OpSchema kNarrow = OpSchema{"narrow_op", 1, AmpPolicy::KeepInput, true}.refusing(
    dtype_bit(Dtype::Bool) | kNarrowSigned, DtypeRefusal::Guard);

std::string what_of_construction(const OpSchema& schema, Dtype dtype) {
    try {
        SchemaGuard guard(schema, dtype, Device::CPU);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

// -- compile-time contract ---------------------------------------------------

constexpr OpSchema kBase{"op", 1, AmpPolicy::KeepInput, true};
constexpr OpSchema kInts = kBase.refusing(kSignedIntegers, DtypeRefusal::Guard);

static_assert(kInts.refuses(Dtype::I8) && kInts.refuses(Dtype::I16) && kInts.refuses(Dtype::I32) &&
              kInts.refuses(Dtype::I64));
static_assert(!kInts.refuses(Dtype::Bool) && !kInts.refuses(Dtype::F32) &&
              !kInts.refuses(Dtype::C64));
static_assert(!kBase.refuses(Dtype::I32), "an undeclared schema refuses nothing");

// The same class and hint may add dtypes ...
constexpr OpSchema kBoolAndInts =
    OpSchema{"op", 1, AmpPolicy::KeepInput, true}
        .refusing_bool(DtypeRefusal::Kernel, CastHint::Floating)
        .refusing(kSignedIntegers, DtypeRefusal::Kernel, CastHint::Floating);
static_assert(kBoolAndInts.refuses(Dtype::Bool) && kBoolAndInts.refuses(Dtype::I64));

// ... a different class or hint is not a constant expression, so a declaration
// that is one fails to compile, and one that is not fails at load.
template <DtypeRefusal Kind, CastHint Hint>
concept CanRefuseAgain = requires {
    typename std::integral_constant<DtypeMask,
                                    kInts.refusing(dtype_bit(Dtype::Bool), Kind, Hint).refused>;
};
static_assert(CanRefuseAgain<DtypeRefusal::Guard, CastHint::Floating>);
static_assert(!CanRefuseAgain<DtypeRefusal::Kernel, CastHint::Floating>);
static_assert(!CanRefuseAgain<DtypeRefusal::Guard, CastHint::Number>);

// One bit per dtype code, all distinct, all inside the 16-bit word.
constexpr DtypeMask all_bits() {
    DtypeMask seen = 0;
    for (const Dtype dt : kAll) {
        if (seen & dtype_bit(dt))
            return 0;
        seen = static_cast<DtypeMask>(seen | dtype_bit(dt));
    }
    return seen;
}
static_assert(all_bits() == 0x07FF, "11 dtypes, codes 0..10, one bit each");
static_assert(kSignedIntegers == (dtype_bit(Dtype::I8) | dtype_bit(Dtype::I16) |
                                  dtype_bit(Dtype::I32) | dtype_bit(Dtype::I64)));
static_assert((kNarrowSigned & dtype_bit(Dtype::I64)) == 0);
static_assert(kBoolAndSignedIntegers == (kSignedIntegers | dtype_bit(Dtype::Bool)));

}  // namespace

TEST(SchemaDtypeRefusal, DefaultsToNone) {
    EXPECT_EQ(kPlain.refusal, DtypeRefusal::None);
    EXPECT_EQ(kPlain.refused, DtypeMask{0});
    EXPECT_EQ(kPlain.cast, CastHint::Number);
    for (const Dtype dtype : kAll)
        EXPECT_FALSE(kPlain.refuses(dtype)) << dtype_name(dtype);
}

TEST(SchemaDtypeRefusal, DeclaringItDoesNotTouchTheRestOfTheSchema) {
    EXPECT_EQ(kKernel.name, "kernel_op");
    EXPECT_EQ(kKernel.version, 1);
    EXPECT_EQ(kKernel.declared_amp_policy, AmpPolicy::KeepInput);
    EXPECT_TRUE(kKernel.deterministic);
    EXPECT_EQ(kKernel.refusal, DtypeRefusal::Kernel);
    EXPECT_EQ(kGuard.refusal, DtypeRefusal::Guard);
    EXPECT_EQ(kGuard.cast, CastHint::Floating);
}

TEST(SchemaDtypeRefusal, TheFingerprintIgnoresIt) {
    // A refusal changes what the op accepts, not what a saved graph meant by it.
    const OpSchema twin{"kernel_op", 1, AmpPolicy::KeepInput, true};
    EXPECT_EQ(schema_hash(kKernel), schema_hash(twin));
    EXPECT_EQ(schema_hash(kIntegers),
              schema_hash(OpSchema{"int_op", 1, AmpPolicy::KeepInput, true}));
}

TEST(SchemaDtypeRefusal, RefusingBoolIsTheBoolBitOnly) {
    for (const Dtype dtype : kAll)
        EXPECT_EQ(kKernel.refuses(dtype), dtype == Dtype::Bool) << dtype_name(dtype);
    EXPECT_EQ(kKernel.refused, dtype_bit(Dtype::Bool));
}

TEST(SchemaDtypeRefusal, EveryDeclaredWidthIsRefusedAndNoOtherDtype) {
    for (const Dtype dtype : kAll) {
        const bool signed_int =
            dtype == Dtype::I8 || dtype == Dtype::I16 || dtype == Dtype::I32 || dtype == Dtype::I64;
        EXPECT_EQ(kIntegers.refuses(dtype), signed_int) << dtype_name(dtype);
    }
    for (const Dtype dtype : kSigned) {
        EXPECT_THROW(SchemaGuard(kIntegers, dtype, Device::CPU), NotImplementedError)
            << dtype_name(dtype);
        EXPECT_THROW(SchemaGuard(kIntegers, dtype, Device::GPU), NotImplementedError)
            << dtype_name(dtype);
    }
    // A bool and the floats are not in the integer mask: they stay accepted.
    for (const Dtype dtype :
         {Dtype::Bool, Dtype::F16, Dtype::F32, Dtype::F64, Dtype::BF16, Dtype::C64, Dtype::C128})
        EXPECT_NO_THROW(SchemaGuard(kIntegers, dtype, Device::CPU)) << dtype_name(dtype);
}

TEST(SchemaDtypeRefusal, ANarrowMaskLeavesInt64Accepted) {
    for (const Dtype dtype : {Dtype::Bool, Dtype::I8, Dtype::I16, Dtype::I32})
        EXPECT_THROW(SchemaGuard(kNarrow, dtype, Device::CPU), DtypeMismatch) << dtype_name(dtype);
    EXPECT_NO_THROW(SchemaGuard(kNarrow, Dtype::I64, Device::CPU));
    EXPECT_NO_THROW(SchemaGuard(kNarrow, Dtype::F32, Device::CPU));
}

TEST(SchemaDtypeRefusal, KernelRefusalIsNotImplementedError) {
    EXPECT_THROW(SchemaGuard(kKernel, Dtype::Bool, Device::CPU), NotImplementedError);
    EXPECT_THROW(SchemaGuard(kKernel, Dtype::Bool, Device::GPU), NotImplementedError);
}

TEST(SchemaDtypeRefusal, GuardRefusalIsDtypeMismatch) {
    EXPECT_THROW(SchemaGuard(kGuard, Dtype::Bool, Device::CPU), DtypeMismatch);
    EXPECT_THROW(SchemaGuard(kGuard, Dtype::Bool, Device::GPU), DtypeMismatch);
}

TEST(SchemaDtypeRefusal, TheTwoClassesDoNotCross) {
    EXPECT_THROW(
        {
            try {
                SchemaGuard guard(kIntegers, Dtype::I32, Device::CPU);
            } catch (const DtypeMismatch&) {
                FAIL() << "a kernel refusal must not be a DtypeMismatch";
            }
        },
        NotImplementedError);
    EXPECT_THROW(
        {
            try {
                SchemaGuard guard(kNarrow, Dtype::I32, Device::CPU);
            } catch (const NotImplementedError&) {
                FAIL() << "a guard refusal must not be a NotImplementedError";
            }
        },
        DtypeMismatch);
}

TEST(SchemaDtypeRefusal, EveryOtherDtypePasses) {
    for (const Dtype dtype : kNonBool) {
        EXPECT_NO_THROW(SchemaGuard(kKernel, dtype, Device::CPU)) << dtype_name(dtype);
        EXPECT_NO_THROW(SchemaGuard(kGuard, dtype, Device::CPU)) << dtype_name(dtype);
    }
}

TEST(SchemaDtypeRefusal, AnUnflaggedSchemaStillAcceptsEverything) {
    for (const Dtype dtype : kAll)
        EXPECT_NO_THROW(SchemaGuard(kPlain, dtype, Device::CPU)) << dtype_name(dtype);
}

TEST(SchemaDtypeRefusal, TheMessageNamesTheOpTheDtypeAndTheCast) {
    const std::string kernel = what_of_construction(kKernel, Dtype::Bool);
    EXPECT_NE(kernel.find("kernel_op does not accept bool"), std::string::npos) << kernel;
    EXPECT_NE(kernel.find("cast to an integer or floating dtype, e.g. x.long() or x.float()"),
              std::string::npos)
        << kernel;

    const std::string guard = what_of_construction(kGuard, Dtype::Bool);
    EXPECT_NE(guard.find("guard_op does not accept bool"), std::string::npos) << guard;
    EXPECT_NE(guard.find("cast to a floating dtype, e.g. x.float()"), std::string::npos) << guard;

    // The dtype is spelled as the engine spells it everywhere else.
    for (const Dtype dtype : kSigned) {
        const std::string what = what_of_construction(kIntegers, dtype);
        const std::string want = "int_op does not accept " + std::string(dtype_name(dtype)) +
                                 " — cast to a floating dtype, e.g. x.float()";
        EXPECT_NE(what.find(want), std::string::npos) << what;
    }
}

TEST(SchemaDtypeRefusal, AGuardMessageCarriesTheOpAndTheExpectedDtype) {
    // ``DtypeMismatch("a floating dtype", "int32", "<op>: <msg>")``; a bool is
    // told "a non-bool dtype" because an op that refuses only a bool takes ints.
    const std::string integer = what_of_construction(kNarrow, Dtype::I32);
    EXPECT_NE(integer.find("narrow_op: narrow_op does not accept int32"), std::string::npos)
        << integer;
    EXPECT_NE(integer.find("a floating dtype"), std::string::npos) << integer;
    EXPECT_NE(integer.find("int32"), std::string::npos) << integer;

    const std::string boolean = what_of_construction(kGuard, Dtype::Bool);
    EXPECT_NE(boolean.find("a non-bool dtype"), std::string::npos) << boolean;
}

TEST(SchemaDtypeRefusal, TheHintFollowsTheCast) {
    EXPECT_EQ(cast_hint_text(CastHint::Number),
              "an integer or floating dtype, e.g. x.long() or x.float()");
    EXPECT_EQ(cast_hint_text(CastHint::Floating), "a floating dtype, e.g. x.float()");
    EXPECT_EQ(cast_hint_text(CastHint::Integer), "an integer dtype, e.g. x.long()");
}

TEST(SchemaDtypeRefusal, ThrowHelperIsCallableWithoutASchema) {
    // A composite with no schema of its own refuses through this.
    EXPECT_THROW(
        throw_dtype_refusal("composite", Dtype::I16, DtypeRefusal::Kernel, CastHint::Floating),
        NotImplementedError);
    EXPECT_THROW(
        throw_dtype_refusal("composite", Dtype::Bool, DtypeRefusal::Guard, CastHint::Floating),
        DtypeMismatch);
}

TEST(SchemaDtypeRefusal, CheckSchemaDtypeIsTheGuardsRefusalForAnotherOperand) {
    EXPECT_THROW(check_schema_dtype(kIntegers, Dtype::I8), NotImplementedError);
    EXPECT_THROW(check_schema_dtype(kNarrow, Dtype::I16), DtypeMismatch);
    EXPECT_NO_THROW(check_schema_dtype(kNarrow, Dtype::I64));
    EXPECT_NO_THROW(check_schema_dtype(kPlain, Dtype::I8));
    // The same message as the guard's, whichever entry built it.
    for (const Dtype dtype : kSigned) {
        std::string via_check;
        try {
            check_schema_dtype(kIntegers, dtype);
        } catch (const std::exception& e) {
            via_check = e.what();
        }
        EXPECT_EQ(via_check, what_of_construction(kIntegers, dtype)) << dtype_name(dtype);
    }
}

TEST(SchemaDtypeRefusal, ARefusalIsOneClassAndOneHint) {
    // The constexpr rule above, at run time: a second call may add dtypes but
    // not change the class or the hint.
    EXPECT_NO_THROW(kInts.refusing(dtype_bit(Dtype::Bool), DtypeRefusal::Guard));
    EXPECT_THROW(kInts.refusing(dtype_bit(Dtype::Bool), DtypeRefusal::Kernel), std::logic_error);
    EXPECT_THROW(kInts.refusing(dtype_bit(Dtype::Bool), DtypeRefusal::Guard, CastHint::Number),
                 std::logic_error);
    EXPECT_THROW(kBase.refusing(0, DtypeRefusal::Kernel), std::logic_error);
    EXPECT_THROW(kBase.refusing(kSignedIntegers, DtypeRefusal::None), std::logic_error);

    const OpSchema widened = kInts.refusing(dtype_bit(Dtype::Bool), DtypeRefusal::Guard);
    EXPECT_TRUE(widened.refuses(Dtype::Bool));
    EXPECT_TRUE(widened.refuses(Dtype::I32));
}

TEST(SchemaDtypeRefusal, RefusalComesBeforeTheDeterminismGate) {
    // The first statement of the constructor: a dtype is turned away as that
    // dtype, not as a non-deterministic op, when both would apply.
    const DeterminismRestore restore;
    Determinism::set_enabled(true);
    const OpSchema nondeterministic =
        OpSchema{"nd_op", 1, AmpPolicy::KeepInput, false, "atomics"}.refusing(
            kBoolAndSignedIntegers, DtypeRefusal::Kernel, CastHint::Number);
    EXPECT_THROW(SchemaGuard(nondeterministic, Dtype::Bool, Device::CPU), NotImplementedError);
    EXPECT_THROW(SchemaGuard(nondeterministic, Dtype::I32, Device::CPU), NotImplementedError);
    EXPECT_THROW(SchemaGuard(nondeterministic, Dtype::F32, Device::CPU), LucidError);
}
