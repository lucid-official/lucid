// lucid/_C/test/core/test_phantom.cpp
// Phantom tensors: under PhantomMode every factory and op returns a real
// TensorImpl with a PhantomStorage — the op's own shape and dtype, no
// allocation, no kernel — and a read of its values raises.

#include <thread>

#include <gtest/gtest.h>

#include "../../backend/Dispatcher.h"
#include "../../core/Error.h"
#include "../../core/MemoryStats.h"
#include "../../core/PhantomMode.h"
#include "../../ops/bfunc/Add.h"
#include "../../ops/bfunc/Matmul.h"
#include "../../ops/ufunc/Exponential.h"
#include "../../ops/ufunc/Reductions.h"
#include "../../ops/utils/View.h"
#include "../../random/Random.h"
#include "tensor_factory.h"

using namespace lucid;
using namespace lucid::test;

namespace {

std::size_t cpu_allocs() {
    return MemoryTracker::get_stats(Device::CPU).alloc_count;
}

}  // namespace

TEST(PhantomMode, OffByDefaultAndScoped) {
    EXPECT_FALSE(PhantomMode::is_enabled());
    {
        PhantomModeGuard on(true);
        EXPECT_TRUE(PhantomMode::is_enabled());
        {
            PhantomModeGuard off(false);
            EXPECT_FALSE(PhantomMode::is_enabled());
        }
        EXPECT_TRUE(PhantomMode::is_enabled());
    }
    EXPECT_FALSE(PhantomMode::is_enabled());
}

TEST(PhantomMode, GuardRestoresOnThrow) {
    try {
        PhantomModeGuard on(true);
        throw LucidError("boom");
    } catch (const LucidError&) {
    }
    EXPECT_FALSE(PhantomMode::is_enabled());
}

TEST(PhantomMode, OtherThreadsUnaffected) {
    PhantomModeGuard on(true);
    bool other_phantom = true;
    bool other_mode = true;
    std::thread worker([&] {
        other_mode = PhantomMode::is_enabled();
        other_phantom = cpu_zeros({3})->is_phantom();
    });
    worker.join();
    EXPECT_FALSE(other_mode);
    EXPECT_FALSE(other_phantom);
    EXPECT_TRUE(cpu_zeros({3})->is_phantom());
}

TEST(PhantomStorageTest, VariantIndexAndSize) {
    const Storage s = phantom_storage({2, 3}, Dtype::F32, Device::GPU);
    EXPECT_TRUE(storage_is_phantom(s));
    EXPECT_FALSE(storage_is_cpu(s));
    EXPECT_EQ(s.index(), 3u);
    EXPECT_EQ(storage_nbytes(s), 6u * 4u);
    EXPECT_EQ(storage_dtype(s), Dtype::F32);
    // A phantom has no kind to tell its device by: it records one.
    EXPECT_EQ(storage_device(s), Device::GPU);
    EXPECT_EQ(storage_device(phantom_storage({1}, Dtype::F32, Device::CPU)), Device::CPU);
    EXPECT_EQ(storage_device(Storage{CpuStorage{}}), Device::CPU);
}

TEST(PhantomStorageTest, RefusesADtypeItsDeviceCannotHold) {
    EXPECT_THROW(phantom_storage({2, 3}, Dtype::F64, Device::GPU), NotImplementedError);
    EXPECT_THROW(phantom_storage({2, 3}, Dtype::C128, Device::GPU), NotImplementedError);
    EXPECT_EQ(storage_dtype(phantom_storage({2, 3}, Dtype::F64, Device::CPU)), Dtype::F64);
    EXPECT_EQ(storage_dtype(phantom_storage({2, 3}, Dtype::C128, Device::CPU)), Dtype::C128);
    PhantomModeGuard on(true);
    EXPECT_THROW(backend::Dispatcher::for_device(Device::GPU).zeros({2, 3}, Dtype::F64),
                 NotImplementedError);
    EXPECT_THROW(backend::Dispatcher::for_device(Device::GPU).zeros({2, 3}, Dtype::C128),
                 NotImplementedError);
}

// A node can keep one of these with no tensor to adopt it, so the phantom has to
// say that an index is an integer: sort's backward reads the dtype back.
TEST(PhantomBackendTest, AnIndexIsAnIntegerWhateverItIndexes) {
    PhantomModeGuard on(true);
    const Shape shape{2, 3};
    const Storage x = phantom_storage(shape, Dtype::F32, Device::CPU);
    auto& be = backend::Dispatcher::for_device(Device::CPU);
    EXPECT_EQ(storage_dtype(be.argsort(x, shape, 0, Dtype::F32)), Dtype::I64);
    EXPECT_EQ(storage_dtype(be.arg_reduce_index(x, shape, 0, false, Dtype::F32, false)),
              Dtype::I64);
    const auto [values, indices] = be.sort_select(x, shape, shape, 0, Dtype::F32, false);
    EXPECT_EQ(storage_dtype(values), Dtype::F32);
    EXPECT_EQ(storage_dtype(indices), Dtype::I64);
    const auto pooled = be.max_pool_nd_forward(x, shape, shape, {}, Dtype::F32);
    EXPECT_EQ(storage_dtype(pooled[0]), Dtype::F32);
    EXPECT_EQ(storage_dtype(pooled[1]), Dtype::I32);
    for (const auto& factors :
         {be.linalg_lu_factor(x, shape, Dtype::F32), be.linalg_ldl_factor(x, shape, Dtype::F32)}) {
        EXPECT_EQ(storage_dtype(factors[0]), Dtype::F32);
        EXPECT_EQ(storage_dtype(factors[1]), Dtype::I32);
        EXPECT_EQ(storage_dtype(factors[2]), Dtype::I32);
    }
}

TEST(PhantomFactories, ShapeDtypeNoAllocation) {
    PhantomModeGuard on(true);
    const auto before = cpu_allocs();
    auto z = zeros_op({4, 5}, Dtype::F32, Device::CPU, /*requires_grad=*/true);
    auto f = cpu_full({2}, 3.0, Dtype::F64);
    auto r = randn_op({3, 3}, Dtype::F32, Device::CPU);
    auto a = arange_op(0.0, 10.0, 2.0, Dtype::I64, Device::CPU, false);
    EXPECT_EQ(cpu_allocs(), before);
    for (const auto& t : {z, f, r, a})
        EXPECT_TRUE(t->is_phantom());
    EXPECT_EQ(z->shape(), (Shape{4, 5}));
    EXPECT_TRUE(z->requires_grad());
    EXPECT_EQ(f->dtype(), Dtype::F64);
    EXPECT_EQ(a->shape(), (Shape{5}));
    EXPECT_EQ(a->dtype(), Dtype::I64);
    // The adopting tensor sizes the storage the phantom backend handed out.
    EXPECT_EQ(storage_nbytes(z->storage()), z->nbytes());
}

TEST(PhantomOps, KernelBasesKeepShapeAndDtype) {
    PhantomModeGuard on(true);
    const auto before = cpu_allocs();
    auto x = cpu_ones({2, 1, 4});
    auto y = cpu_ones({3, 1});
    auto s = add_op(x, y);
    EXPECT_EQ(s->shape(), (Shape{2, 3, 4}));
    auto e = exp_op(s);
    EXPECT_EQ(e->shape(), (Shape{2, 3, 4}));
    auto r = sum_op(e, {1}, /*keepdims=*/false);
    EXPECT_EQ(r->shape(), (Shape{2, 4}));
    auto m = matmul_op(cpu_ones({5, 2}), r);
    EXPECT_EQ(m->shape(), (Shape{5, 4}));
    auto v = reshape_op(m, {-1});
    EXPECT_EQ(v->shape(), (Shape{20}));
    for (const auto& t : {s, e, r, m, v})
        EXPECT_TRUE(t->is_phantom());
    EXPECT_EQ(cpu_allocs(), before);
}

TEST(PhantomOps, PhantomInputRefusedOutsideTheMode) {
    TensorImplPtr p;
    {
        PhantomModeGuard on(true);
        p = cpu_ones({3, 2});
    }
    auto real = cpu_ones({3, 2});
    EXPECT_FALSE(real->is_phantom());
    EXPECT_THROW(add_op(p, real), LucidError);
    EXPECT_THROW(exp_op(p), LucidError);
    EXPECT_THROW(sum_op(p, {0}, false), LucidError);
    EXPECT_THROW(TensorImpl::make_view(p, {2, 3}, contiguous_stride({2, 3}, 4)), LucidError);
    EXPECT_THROW(p->storage(), LucidError);
}

TEST(PhantomWrites, OneRuleForEveryWrite) {
    const Storage ghost = phantom_storage({3}, Dtype::F32, Device::CPU);
    const Storage real = CpuStorage{};
    EXPECT_TRUE(absorbs_phantom_write(ghost, real, "w"));
    EXPECT_TRUE(absorbs_phantom_write(ghost, ghost, "w"));
    EXPECT_FALSE(absorbs_phantom_write(real, real, "w"));
    EXPECT_THROW(absorbs_phantom_write(real, ghost, "w"), LucidError);
}

TEST(PhantomWrites, RealLeafRefusesAPhantomFirstGradient) {
    auto w = zeros_op({3}, Dtype::F32, Device::CPU, /*requires_grad=*/true);
    const Storage ghost = phantom_storage({3}, Dtype::F32, Device::CPU);
    EXPECT_THROW(w->absorbs_phantom_gradient(ghost), LucidError);
    EXPECT_FALSE(w->grad_storage().has_value());
}

TEST(PhantomWrites, PhantomTargetTakesAnyWrite) {
    TensorImplPtr p;
    {
        PhantomModeGuard on(true);
        p = cpu_zeros({3});
    }
    auto real = cpu_ones({3});
    const auto v = p->version();
    p->copy_from(*real);
    EXPECT_TRUE(p->is_phantom());
    EXPECT_GT(p->version(), v);
}

TEST(PhantomWrites, RealTargetRefusesPhantomSource) {
    TensorImplPtr p;
    {
        PhantomModeGuard on(true);
        p = cpu_zeros({3});
    }
    auto real = cpu_ones({3});
    EXPECT_THROW(real->copy_from(*p), LucidError);
    EXPECT_FALSE(real->is_phantom());
}

TEST(PhantomViews, ViewIsPhantomOfItsShape) {
    PhantomModeGuard on(true);
    auto base = cpu_zeros({4, 6});
    auto view = TensorImpl::make_view(base, {6, 4}, contiguous_stride({6, 4}, 4));
    EXPECT_TRUE(view->is_phantom());
    EXPECT_EQ(view->shape(), (Shape{6, 4}));
    EXPECT_FALSE(view->is_aliased());
}

TEST(PhantomDispatch, GpuDeviceGetsAPhantomToo) {
    PhantomModeGuard on(true);
    auto g = zeros_op({2, 2}, Dtype::F32, Device::GPU, false);
    EXPECT_TRUE(g->is_phantom());
    EXPECT_EQ(g->device(), Device::GPU);
    EXPECT_EQ(storage_device(g->raw_storage()), Device::GPU);
    auto h = add_op(g, g);
    EXPECT_TRUE(h->is_phantom());
    EXPECT_EQ(h->device(), Device::GPU);
    EXPECT_NO_THROW(h->eval());
}
