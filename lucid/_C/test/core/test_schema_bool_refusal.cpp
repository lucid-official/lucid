// lucid/_C/test/core/test_schema_bool_refusal.cpp
// Tests for the bool refusal an OpSchema declares and SchemaGuard enforces.

#include <array>
#include <string>

#include <gtest/gtest.h>

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/OpSchema.h"
#include "../../core/SchemaGuard.h"

using namespace lucid;

namespace {

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
    BoolRefusal::Kernel, BoolCast::Number);
const OpSchema kGuard = OpSchema{"guard_op", 1, AmpPolicy::KeepInput, true}.refusing_bool(
    BoolRefusal::Guard, BoolCast::Floating);

std::string what_of_construction(const OpSchema& schema, Dtype dtype) {
    try {
        SchemaGuard guard(schema, dtype, Device::CPU);
    } catch (const std::exception& e) {
        return e.what();
    }
    return {};
}

}  // namespace

TEST(SchemaBoolRefusal, DefaultsToNone) {
    EXPECT_EQ(kPlain.bool_refusal, BoolRefusal::None);
    EXPECT_EQ(kPlain.bool_cast, BoolCast::Number);
}

TEST(SchemaBoolRefusal, DeclaringItDoesNotTouchTheRestOfTheSchema) {
    EXPECT_EQ(kKernel.name, "kernel_op");
    EXPECT_EQ(kKernel.version, 1);
    EXPECT_EQ(kKernel.declared_amp_policy, AmpPolicy::KeepInput);
    EXPECT_TRUE(kKernel.deterministic);
    EXPECT_EQ(kKernel.bool_refusal, BoolRefusal::Kernel);
    EXPECT_EQ(kGuard.bool_refusal, BoolRefusal::Guard);
    EXPECT_EQ(kGuard.bool_cast, BoolCast::Floating);
}

TEST(SchemaBoolRefusal, TheFingerprintIgnoresIt) {
    // A refusal changes what the op accepts, not what a saved graph meant by it.
    const OpSchema twin{"kernel_op", 1, AmpPolicy::KeepInput, true};
    EXPECT_EQ(schema_hash(kKernel), schema_hash(twin));
}

TEST(SchemaBoolRefusal, KernelRefusalIsNotImplementedError) {
    EXPECT_THROW(SchemaGuard(kKernel, Dtype::Bool, Device::CPU), NotImplementedError);
    EXPECT_THROW(SchemaGuard(kKernel, Dtype::Bool, Device::GPU), NotImplementedError);
}

TEST(SchemaBoolRefusal, GuardRefusalIsDtypeMismatch) {
    EXPECT_THROW(SchemaGuard(kGuard, Dtype::Bool, Device::CPU), DtypeMismatch);
    EXPECT_THROW(SchemaGuard(kGuard, Dtype::Bool, Device::GPU), DtypeMismatch);
}

TEST(SchemaBoolRefusal, TheTwoClassesDoNotCross) {
    EXPECT_THROW(
        {
            try {
                SchemaGuard guard(kKernel, Dtype::Bool, Device::CPU);
            } catch (const DtypeMismatch&) {
                FAIL() << "a kernel refusal must not be a DtypeMismatch";
            }
        },
        NotImplementedError);
    EXPECT_THROW(
        {
            try {
                SchemaGuard guard(kGuard, Dtype::Bool, Device::CPU);
            } catch (const NotImplementedError&) {
                FAIL() << "a guard refusal must not be a NotImplementedError";
            }
        },
        DtypeMismatch);
}

TEST(SchemaBoolRefusal, EveryOtherDtypePasses) {
    for (const Dtype dtype : kNonBool) {
        EXPECT_NO_THROW(SchemaGuard(kKernel, dtype, Device::CPU)) << dtype_name(dtype);
        EXPECT_NO_THROW(SchemaGuard(kGuard, dtype, Device::CPU)) << dtype_name(dtype);
    }
}

TEST(SchemaBoolRefusal, AnUnflaggedSchemaStillAcceptsBool) {
    EXPECT_NO_THROW(SchemaGuard(kPlain, Dtype::Bool, Device::CPU));
}

TEST(SchemaBoolRefusal, TheMessageNamesTheOpAndTheCast) {
    const std::string kernel = what_of_construction(kKernel, Dtype::Bool);
    EXPECT_NE(kernel.find("kernel_op does not accept bool"), std::string::npos) << kernel;
    EXPECT_NE(kernel.find("cast to an integer or floating dtype, e.g. x.long() or x.float()"),
              std::string::npos)
        << kernel;

    const std::string guard = what_of_construction(kGuard, Dtype::Bool);
    EXPECT_NE(guard.find("guard_op does not accept bool"), std::string::npos) << guard;
    EXPECT_NE(guard.find("cast to a floating dtype, e.g. x.float()"), std::string::npos) << guard;
}

TEST(SchemaBoolRefusal, TheHintFollowsTheCast) {
    EXPECT_EQ(bool_cast_hint(BoolCast::Number),
              "an integer or floating dtype, e.g. x.long() or x.float()");
    EXPECT_EQ(bool_cast_hint(BoolCast::Floating), "a floating dtype, e.g. x.float()");
    EXPECT_EQ(bool_cast_hint(BoolCast::Integer), "an integer dtype, e.g. x.long()");
}

TEST(SchemaBoolRefusal, ThrowHelperIsCallableWithoutASchema) {
    // A composite with no schema of its own (``frac``) refuses through this.
    EXPECT_THROW(throw_bool_refusal("composite", BoolRefusal::Kernel, BoolCast::Floating),
                 NotImplementedError);
    EXPECT_THROW(throw_bool_refusal("composite", BoolRefusal::Guard, BoolCast::Floating),
                 DtypeMismatch);
}

TEST(SchemaBoolRefusal, RefusalComesBeforeTheDeterminismGate) {
    // The first statement of the constructor: a bool is turned away as a bool,
    // not as a non-deterministic op, when both would apply.
    const DeterminismRestore restore;
    Determinism::set_enabled(true);
    const OpSchema nondeterministic =
        OpSchema{"nd_op", 1, AmpPolicy::KeepInput, false, "atomics"}.refusing_bool(
            BoolRefusal::Kernel, BoolCast::Number);
    EXPECT_THROW(SchemaGuard(nondeterministic, Dtype::Bool, Device::CPU), NotImplementedError);
    EXPECT_THROW(SchemaGuard(nondeterministic, Dtype::F32, Device::CPU), LucidError);
}
