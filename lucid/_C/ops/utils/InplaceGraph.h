// lucid/_C/ops/utils/InplaceGraph.h
//
// What an in-place op has to do to the autograd graph, in one place.
//
// There are two in-place families — the unary one in ``ufunc/Inplace.cpp``
// and the binary one in ``bfunc/Inplace.cpp`` — and they had drifted:
// the unary side learned to adopt the forward's grad_fn and the binary
// side never did, so ``x.mul_(y)`` returned the gradient of whatever
// produced ``x`` and ``y`` received none at all.  Sharing the rules is
// what keeps the two from disagreeing again.

#pragma once

#include <utility>

#include "../../autograd/TensorHooks.h"
#include "../../backend/Dispatcher.h"
#include "../../core/AmpPolicy.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/GradMode.h"
#include "../../core/TensorImpl.h"
#include "../../kernel/NaryKernel.h"
#include "View.h"

namespace lucid::inplace {

// Refuse to mutate a leaf that requires grad.
//
// A leaf is where gradient accumulates, and an in-place write moves it
// without leaving anything behind that says so.  Nor is there a right
// number to return: after the write ``x`` *is* the result, so "the
// gradient with respect to x" names two different tensors depending on
// when it is asked.  The reference refuses the same setup for the same
// reason.  Callers that mean to overwrite say so with ``no_grad``, which
// is what every optimiser step already does.
//
// ``is_leaf()``, not ``!grad_fn()``: a leaf that has taken part in a
// forward carries its gradient accumulator as ``grad_fn``, so the old test
// stopped refusing the moment a parameter was first used — every in-place
// op on a trained parameter then went through, and the ones that rebind
// turned the parameter into a non-leaf that never received ``.grad``.
inline void refuse_on_leaf(const TensorImplPtr& a, const char* name) {
    if (GradMode::is_enabled() && a->requires_grad() && a->is_leaf())
        ErrorBuilder(name).fail("a leaf tensor that requires grad cannot be modified in place — "
                                "wrap the call in no_grad, or use the out-of-place form");
    if (GradMode::is_enabled() && a->is_custom_function_output())
        ErrorBuilder(name).fail(
            "Output of a custom Function is a view and is being modified in place — custom "
            "Function output views cannot be modified in place; clone() first");
}

// Whether the op about to run records a node — and so keeps a handle on
// ``a`` and on ``other``, the second operand of a binary op.  ``other`` is
// what makes ``buf.mul_(v)`` record a node when only ``v`` requires grad.
inline bool records_graph(const TensorImplPtr& a, const TensorImplPtr& other) {
    return GradMode::is_enabled() && (a->requires_grad() || (other && other->requires_grad()));
}

// A stand-in for ``a`` to run the forward against.
//
// A node keeps a handle on each tensor it was given, and graph-mode
// backward reads its values and its graph position through that handle,
// not through the Storage it saved by value.  Handed ``a`` itself, the node
// read ``a`` as the write left it — the new values, and the new grad_fn,
// which is the node itself:
//
//     sin_                ->  cos(sin(x)) instead of cos(x)
//     buf.mul_(v)         ->  d/dv = buf * v instead of buf
//
// in graph mode only, since eager ``backward()`` reads the Storage.  So
// whenever the op records a node, the node is handed a tensor of its own:
// one that holds the values ``a`` holds now and sits where ``a`` sits now.
// ``a`` not requiring grad is no exception — the other operand of a binary
// op may, and its node saves this one.  Being its own tensor, it also keeps
// its own version count, which is what lets the write move ``a``'s
// (:func:`adopt_graph_position`) without the node refusing itself.
//
// The snapshot shares the buffer rather than copying it, and the caller's
// assignment replaces ``a``'s *slot* rather than the buffer, so the
// original values stay alive and unmutated for as long as the node needs
// them.  No data is copied, and when no node is recorded — ``no_grad``, or
// no operand requiring grad — nothing is allocated either.
//
// A write that lands in the buffer (``TensorImpl::write_lands_in_buffer``:
// a CPU tensor with live views, or one read from ``.grad``) is the
// exception: a node holding the buffer would read the new values back, and
// the holder would stop the write besides — ``p.grad.add_(w)`` with ``w``
// requiring grad was refused as a gradient something else reads.  Its
// snapshot is a copy.
inline TensorImplPtr snapshot(const TensorImplPtr& a, const TensorImplPtr& other = nullptr) {
    if (!records_graph(a, other))
        return a;
    Storage storage = a->write_lands_in_buffer() ? backend::Dispatcher::for_device(a->device())
                                                       .clone(a->storage(), a->shape(), a->dtype())
                                                 : a->storage();
    auto source = std::make_shared<TensorImpl>(std::move(storage), a->shape(), a->dtype(),
                                               a->device(), a->requires_grad());
    if (a->requires_grad()) {
        source->set_grad_fn(a->grad_fn());
        source->set_grad_output_nr(a->grad_output_nr());
    }
    return source;
}

// Re-derive the other members of ``a``'s view family after a write through
// ``a``.
//
// Every member reads one run of the buffer, as ``a`` does, so the write
// changed exactly the part of each member the two runs share.  Autograd has
// to say so, or the member sends its gradient to the values the write
// replaced:
//
// * a member the write missed keeps its values, and so its graph;
// * a member the write covered reads only ``a``'s values now — a reshape of
//   ``a`` when the runs are the same, a run of ``a`` otherwise — and when
//   ``a`` does not require grad it is cut from the graph, since those are a
//   constant's values;
// * a member the write reached into has ``a``'s part spliced into its old
//   graph by a CopySlices node, as the reference does for the base of a
//   written slice.  When ``a`` does not require grad, that part just stops
//   passing gradient back.
//
// A leaf that requires grad keeps its place and only its values move: the
// write was into its base or beside it, since ``write_through`` refuses one
// through a view of it while autograd records.
LUCID_API void rebase_views(const TensorImplPtr& a);

// Move ``out``'s place in the autograd graph onto ``a``.
//
// ``fwd_fn`` builds a differentiable ``out`` whose grad_fn knows how to
// undo this op.  Taking only its storage kept the new numbers and threw
// the derivative away, so ``a`` still sat where it was before the call
// and reported the gradient of whatever produced it:
//
//     y = x * 1.0; y.exp_(); y.sum().backward()  ->  dx = 1
//                                     reference  ->  dx = exp(x)
//
// Silent, in both families: the value was right and only the derivative
// was not, so a model using one trained on a wrong gradient with nothing
// to show for it.  The views of ``a`` read the values it now holds, so
// they move with it.
//
// The version moves too, as for any write.  A node that saved ``a`` before
// the write holds a tensor whose values and grad_fn are no longer the ones
// it saw; with the count left alone nothing told it so.  Eager backward of
// an engine node got away with that, reading the Storage it had saved, but
// graph-mode backward read the written values, and a custom ``Function``
// reads its saved tensors as they are, with the count the only sign of the
// write: ``y = Fn.apply(x); y.mul_(2)`` gave the gradient of the written
// ``y`` on both devices.  The op's own node is not among those that
// refuse: it was handed a snapshot (:func:`snapshot`), a tensor of its own
// whose count no write to ``a`` reaches.
//
// The hooks registered on ``a`` stay on its old slot, as the reference's do:
// they were asked about the values before the write.  ``retain_grad`` follows
// the tensor to its new slot.
inline bool adopt_graph_position(const TensorImplPtr& a, const TensorImplPtr& out) {
    if (!out->requires_grad() && !out->grad_fn())
        return false;
    // A leaf ``out`` — ``copy_`` from a parameter — has no grad_fn until
    // something asks for its AccumulateGrad, and ``a`` needs one to point at.
    auto fn = out->grad_fn() ? out->grad_fn() : detail::ensure_grad_fn(out);
    a->set_requires_grad(true);
    a->set_grad_fn(std::move(fn));
    a->set_grad_output_nr(out->grad_output_nr());
    // ``a`` is an op's output now, whatever it was before.  A factory
    // tensor starts out a leaf, and one left marked so looked, to the next
    // write into its family, like a parameter's view — which was refused.
    a->set_leaf(false);
    if (a->retains_grad())
        retain_grad(a);
    if (a->is_aliased())
        rebase_views(a);
    a->bump_version();
    return true;
}

// What to do when there was no graph position to adopt.
//
// Nothing to adopt means the op is not differentiable — ``ceil``,
// ``floor``, ``round`` and ``sign`` all end the graph.  ``a``'s contents
// no longer depend on what they were, so leaving its old grad_fn in place
// answers with the gradient of whatever produced ``a``, unchanged.  Cut
// it, which is the out-of-place convention — and cut its views with it,
// since they read the same constant values.
//
// Only under an active GradMode: inside ``no_grad`` the write is an
// ordinary mutation of a tensor belonging to a graph built earlier, and
// severing it there would lose a chain the caller means to keep — the
// version counter guards that case, and it still runs.
inline void detach_and_bump(const TensorImplPtr& a) {
    if (GradMode::is_enabled()) {
        if (a->grad_fn()) {
            a->set_grad_fn(nullptr);
            a->set_grad_output_nr(0);
            a->set_requires_grad(false);
        }
        if (a->is_aliased())
            rebase_views(a);
    }
    a->bump_version();
}

// Make ``out``, an in-place op's result, the values of ``a``; return ``a``.
//
// Every engine in-place op's result lands here (through :func:`run`), so
// what holds for all of them is held once:
//
// * the shape does not change — the live views of ``a`` read it;
// * nor does the dtype.  ``out`` is cast to ``a``'s dtype when the
//   reference's in-place rule allows it (``lucid._promotion.can_cast``: a
//   write may narrow within a kind but never move down bool < integer <
//   floating < complex), and the op is refused otherwise, ``a`` untouched.
//   Taking ``out``'s dtype instead, ``int64.exp_()`` turned the tensor into
//   float32 where the reference raises;
// * ``a`` takes ``out``'s place in the graph (:func:`adopt_graph_position`),
//   or leaves it when ``out`` has none (:func:`detach_and_bump`).
TensorImplPtr write_result(const TensorImplPtr& a, TensorImplPtr out, const char* name);

// Run an in-place op: ``forward()`` computes the result from a stand-in for
// ``a`` (:func:`snapshot`), and :func:`write_result` makes it ``a``'s values.
//
// Autocast does not apply to an in-place op, as the reference's does not,
// so the op computes in ``a``'s own dtype.  The forward used to take the
// op's AMP policy as an out-of-place call does, and the write then gave
// ``a`` the result's dtype: under ``autocast(metal, float16)``,
// ``x.mul_(y)`` of two float32 tensors left ``x`` float16 (an optimiser's
// parameter among them), and ``x.exp_()`` of a float16 ``x`` left it
// float32.  Casting the result back keeps the dtype but not the values: the
// float32 product would be rounded through float16 first, where the
// reference computes it in float32.
template <typename Fn>
TensorImplPtr run(const TensorImplPtr& a, const char* name, Fn&& forward) {
    const amp::AutocastGuard autocast_off(Dtype::F32, a->device(), /*enabled=*/false);
    return write_result(a, std::forward<Fn>(forward)(), name);
}

}  // namespace lucid::inplace
