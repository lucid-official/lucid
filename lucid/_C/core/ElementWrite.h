// lucid/_C/core/ElementWrite.h
//
// An engine writer that updates a tensor its caller handed it — an
// optimizer step, BatchNorm's running statistics — and where that tensor's
// elements are.
//
// Such a writer takes a buffer as the tensor's elements: the CPU loops walk
// ``numel`` of them from the buffer's first byte in row-major order, and a
// Metal writer replaces the whole array.  A tensor whose elements sit
// anywhere else had the wrong ones written.  A column view had the row at
// its buffer's start written and its own elements left as they were, a
// transpose had each value paired with another element, and a Metal view of
// part of a tensor was refused (LCD-476 for the optimizers, LCD-491 for
// BatchNorm).  ``write_elements`` is where that is decided, once: the
// writer it runs may assume the dense layout.

#pragma once

#include <cstring>
#include <memory>
#include <utility>
#include <variant>

#include "Allocator.h"
#include "Device.h"
#include "PhantomMode.h"
#include "Storage.h"
#include "TensorImpl.h"

namespace lucid {

// Whether ``t``'s own buffer, from its first byte, is its elements in
// row-major order: a contiguous CPU tensor at offset 0 (a run at the start
// of a larger buffer included), or a Metal tensor that reads all of its
// family's root.
inline bool buffer_is_elements(const TensorImpl& t) {
    if (t.storage_offset() != 0 || !t.is_contiguous())
        return false;
    return t.device() == Device::CPU || t.is_dense();
}

// A dense copy of ``t``'s elements in a buffer of its own, for a writer to
// update in ``t``'s place.
inline std::shared_ptr<TensorImpl> dense_elements(const TensorImpl& t) {
    const Storage& elems = t.storage();
    // An MLX array is a value: a writer replaces the copy's array and leaves
    // the one it was read from as it was.
    if (const auto* g = std::get_if<GpuStorage>(&elems))
        return std::make_shared<TensorImpl>(Storage{*g}, t.shape(), t.dtype(), t.device(), false);
    // The CPU loops write the buffer they are handed, so it must be its own:
    // ``storage()`` hands out a view's cached packing.
    const auto& src = std::get<CpuStorage>(elems);
    CpuStorage own;
    own.dtype = t.dtype();
    own.nbytes = t.nbytes();
    own.ptr = allocate_aligned_bytes(own.nbytes, Device::CPU);
    if (own.nbytes > 0)
        std::memcpy(own.ptr.get(), src.ptr.get(), own.nbytes);
    return std::make_shared<TensorImpl>(Storage{std::move(own)}, t.shape(), t.dtype(), t.device(),
                                        false);
}

// Runs ``write`` on a tensor that holds ``t``'s elements densely from its
// buffer's first byte, and leaves what it wrote in ``t``'s own elements.
//
// That tensor is ``t`` itself when its buffer is its elements
// (``buffer_is_elements``).  Its version is then ``write``'s to bump, as
// for any write it makes in place: an optimizer step bumps it, and a
// running statistic whose storage slot is replaced does not, as in the
// reference.  Otherwise it is a dense copy, written back with ``copy_from``:
// through ``t``'s offset and strides on the CPU, into its part of the
// family's root on Metal.  That bumps the version, which the tensors sharing
// the buffer read their cached elements by.  A tensor whose elements overlap
// (an expanded view) is refused at that write-back, after ``write`` has run,
// so anything else ``write`` changed has already changed.
//
// ``src`` is what the writer writes from (the gradient, the fresh running
// statistic) and ``op`` names it for an error.  The phantom write rule
// (core/PhantomMode.h) is held here, before ``write`` runs: a phantom ``t``
// takes the write without running the writer, which reads and writes values,
// and its version moves as for any write; a real ``t`` refuses a phantom
// ``src``, which BatchNorm's slot replacement would otherwise have taken in
// silence.
template <class Write>
void write_elements(std::shared_ptr<TensorImpl>& t,
                    const Storage& src,
                    const char* op,
                    Write&& write) {
    t->check_inference_mutation(op);
    if (absorbs_phantom_write(t->raw_storage(), src, op)) {
        t->bump_version();
        return;
    }
    if (buffer_is_elements(*t)) {
        write(t);
        return;
    }
    auto dense = dense_elements(*t);
    write(dense);
    t->copy_from(*dense);
}

// For a writer whose source is never a phantom — a compiled step writing its
// executable's own output.  A phantom ``t`` still takes the write unrun.
template <class Write>
void write_elements(std::shared_ptr<TensorImpl>& t, Write&& write) {
    static const Storage kRealSource{};
    write_elements(t, kRealSource, "write_elements", std::forward<Write>(write));
}

}  // namespace lucid
