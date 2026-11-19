// lucid/_C/compile/MpsBuilder.h
//
// Lowering of a :class:`TraceGraph` into an MPSGraph executable.
//
// The builder walks the recorded op DAG in dispatch order:
//
//   1. Allocate an empty :class:`MPSGraph`.
//   2. For each external feed (an input pointer the tracer did *not*
//      see produced by any earlier op — typically a model parameter
//      or the user's input tensor), create an MPSGraph placeholder
//      sized to the feed's shape and dtype.
//   3. For each :class:`OpNode` in dispatch order, look up an
//      :class:`OpEmitter` by op name and ask it to extend the graph.
//      A missing emitter or an emitter returning ``nullptr`` aborts
//      the build — the caller treats the trace signature as
//      eager-only.
//   4. Compile the assembled graph for the trace's device, producing
//      an :class:`MPSGraphExecutable`.
//   5. Wrap the executable + feed-order / target-order ids in a
//      :class:`CompiledExecutable` (allocated on the heap; caller
//      must release via :func:`destroy_executable` or hand it to
//      :class:`ExecutableCache`).
//
// Header is pure C++ — :class:`CompiledExecutable` is opaque (Obj-C
// types stay in :file:`MpsBuilder.mm` / :file:`CompiledExecutable.mm`).

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../api.h"
#include "../core/fwd.h"  // TensorImplPtr
#include "CompiledExecutable.h"
#include "TraceIR.h"

namespace lucid::compile {

// Metal operation class, label, incoming operation indices and physical output types.
// Missing shape denotes unknown rank. An empty shape denotes a scalar.
using EmittedTensor = std::pair<unsigned, std::optional<std::vector<std::int64_t>>>;
using EmittedOperation =
    std::tuple<std::string, std::string, std::vector<std::uint32_t>, std::vector<EmittedTensor>>;

class BuilderContext;  // forward decl — defined in OpEmitters/OpEmitter.h

// Lower ``graph`` into an MPSGraph executable.
//
// Parameters
// ----------
// graph : const TraceGraph&
//     The trace IR returned by ``Tracer::graph()``.
// external_feeds : const std::unordered_map<TensorId, TensorImplPtr>&
//     The trace's external feeds (model parameters + user inputs),
//     mapped by the :type:`TensorId` minted during
//     :func:`Tracer::on_op_io`.  Each entry becomes one MPSGraph
//     placeholder, and the corresponding ``TensorImplPtr`` carries
//     the runtime data that will be bound to that placeholder.
// error_msg : std::string*, optional
//     If non-null, populated with a human-readable reason whenever
//     the build aborts (unsupported op, mixed-device trace, dtype
//     mismatch, …).
//
// Returns
// -------
// CompiledExecutable*
//     Heap-allocated executable handle on success, or ``nullptr`` on
//     any abort condition (caller falls back to eager dispatch).
//     Caller owns the returned pointer.
//
// Notes
// -----
// Phase 1.2 step 1 supports only Linear forward in the emitter
// registry; everything else returns ``nullptr`` cleanly.  The op gap
// protocol (per-op spike → expand coverage incrementally) is the
// canonical way new families come online.
LUCID_API CompiledExecutable*
compile_trace(const TraceGraph& graph,
              const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
              std::string* error_msg = nullptr,
              bool dynamic_batch = false,
              const std::vector<TensorId>& param_ids = {},
              // Explicit graph output ids.  When non-empty, the builder uses
              // these as the target tensors instead of auto-detecting unconsumed
              // outputs.  Lets the Python compile harness pass exactly the
              // user's return-value tensors so that Python-discarded
              // intermediates (e.g. RNN's ``h_n`` when user does
              // ``out, _ = rnn(x)``) don't leak into the executable's output
              // list.  Empty (default) preserves the auto-detect behaviour.
              const std::vector<TensorId>& explicit_outputs = {});

// Phase 1.3: forward + backward in one executable.
//
// Walks the trace exactly like :func:`compile_trace`, then calls
// MPSGraph's autograd builder
// ``[graph gradientForPrimaryTensor:loss withTensors:params name:]``
// to derive the gradient of the loss with respect to each parameter,
// and appends those gradient tensors to the executable's target list.
//
// Parameters
// ----------
// graph : const TraceGraph&
//     The recorded forward op DAG.
// external_feeds : const std::unordered_map<TensorId, TensorImplPtr>&
//     External-feed map (parameters + user inputs).  Must contain an
//     entry for every id in ``param_ids``.
// loss_id : TensorId
//     Trace id of the scalar loss tensor.  Must be produced by some
//     op inside ``graph`` (not an external feed).
// param_ids : const std::vector<TensorId>&
//     Trace ids of the model parameters with respect to which the
//     gradient is taken.  Must all be external feeds.
// error_msg : std::string*, optional
//     Reason on abort (unsupported op, missing param feed, …).
//
// Returns
// -------
// CompiledExecutable*
//     Owns the compiled executable.  ``output_ids`` = ``{loss_id}``;
//     ``grad_output_ids`` = freshly-minted ids for each
//     ``param_ids[i]``'s gradient (same order as ``param_ids``).
//     ``output_shapes`` / ``output_dtypes`` carry the loss meta
//     followed by per-gradient meta (matches the layout the runtime
//     uses when binding output buffers).  Returns ``nullptr`` on
//     abort.
LUCID_API CompiledExecutable*
compile_trace_with_backward(const TraceGraph& graph,
                            const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                            TensorId loss_id,
                            const std::vector<TensorId>& param_ids,
                            std::string* error_msg = nullptr,
                            bool dynamic_batch = false,
                            const std::vector<TensorId>& extra_output_ids = {});

// ── Fused training step compile (Phase 1.7) ─────────────────────────

// Optimizer spec passed to :func:`compile_fused_training_step`.
//
// The C++ side directly emits MPSGraph ops for the update rule, using
// the auto-derived gradient tensors + a small set of state buffer
// feeds (m / v for Adam-family, momentum for SGD).  Per-step scalars
// (bias-correction factors for Adam) are passed as additional external
// feeds at run time so that one cached executable covers the whole
// training loop without recompiles.
struct LUCID_API OptimizerSpec {
    // Enum values intentionally mirror Python ``OptimizerKind`` in
    // :file:`lucid/compile/_optim_spec.py` so the spec can be
    // serialised across the binding boundary as a single int.
    //
    // SGD / ADAM / ADAMW are the kinds the C++
    // :func:`compile_fused_training_step` currently emits via
    // hardcoded ``emit_sgd_update`` / ``emit_adam_update`` helpers.
    // The reserved values (RMSPROP / ADAGRAD / ADADELTA / ADAMAX /
    // NADAM) are recognised so the Python spec can round-trip
    // through the binding without losing information, but
    // ``compile_fused_training_step`` currently rejects them — the
    // production fused-step path uses :func:`compile_generic_fused_step`
    // which traces the update math from Python and so supports all 8
    // kinds today.  Adding C++ ``emit_<kind>_update`` helpers for the
    // 5 reserved kinds is a tracked follow-up.
    enum class Kind {
        SGD = 0,
        ADAM = 1,
        ADAMW = 2,
        RMSPROP = 3,
        ADAGRAD = 4,
        ADADELTA = 5,
        ADAMAX = 6,
        NADAM = 7,
    };
    Kind kind = Kind::SGD;
    // SGD hyperparameters (also used as Adam-family wd / lr).
    double lr = 0.0;
    double momentum = 0.0;
    double dampening = 0.0;
    double weight_decay = 0.0;
    bool nesterov = false;
    // Adam-family hyperparameters.
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps = 1e-8;
};

// Compile a single MPSGraph executable that runs the full training
// step in one dispatch:
//
//   loss = loss_fn(model(x), target)
//   grad_param[i] = ∂loss / ∂param[i]   ← derived by MPSGraph
//   new_param[i], new_state[i, k] = opt_update(param[i], grad_param[i],
//                                              state[i, k], ...)
//
// Output layout (in this order):
//   1. loss            ← single scalar/tensor produced by the forward.
//   2. new_param[i]    ← one per ``param_ids[i]``.
//   3. new_state[i, k] ← per (param, state-slot-index) — exactly the
//                        ``state_buf_ids_per_param`` layout flattened
//                        row-major (i, k).
//
// At run time the caller uses :func:`run_executable_inplace` with the
// output_targets = [loss_scratch, params (in order), state_bufs
// (flattened)] so the new values overwrite the existing storage.  The
// loss is allocated fresh and returned to Python for logging.
//
// Parameters
// ----------
// graph : const TraceGraph&
//     Forward trace (model + loss).  Same shape as
//     :func:`compile_trace_with_backward`.
// external_feeds : ...
//     id → impl map for every external feed in ``graph``.  Must
//     include every parameter id, every state buffer id, and every
//     per-step scalar id.
// loss_id, param_ids :
//     Forward-graph ids for the scalar loss and the parameters with
//     respect to which the gradient is taken.
// opt_spec :
//     Hyperparameters + kind of the optimizer.
// state_buf_ids_per_param :
//     For each param i, the trace ids of its state buffers (in the
//     order each optimizer expects: SGD = [momentum] or empty, Adam =
//     [m, v]).  All must already be in ``external_feeds``.
// scalar_input_ids :
//     Per-step scalar feed ids in the order each optimizer expects
//     (Adam / AdamW: [bias1, bias2]; SGD: empty).  These are read at
//     run time as 0-D float feeds.
// error_msg : std::string*, optional
//     Populated on failure.
//
// Returns
// -------
// CompiledExecutable*
//     Owns the compiled executable.  ``output_ids`` = ``{loss_id}``;
//     ``grad_output_ids`` carries one id per (new_param, new_state)
//     output in the order described above.  Returns nullptr on abort.
LUCID_API CompiledExecutable*
compile_fused_training_step(const TraceGraph& graph,
                            const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                            TensorId loss_id,
                            const std::vector<TensorId>& param_ids,
                            const OptimizerSpec& opt_spec,
                            const std::vector<std::vector<TensorId>>& state_buf_ids_per_param,
                            const std::vector<TensorId>& scalar_input_ids,
                            std::string* error_msg = nullptr);

// ── Generic fused step (Phase 1.8) ──────────────────────────────────

// Like :func:`compile_fused_training_step` but the optimizer update
// math is captured in the same :class:`TraceGraph` as the forward
// pass — no per-optimizer C++ emit branch.  Each ``ghost_grad_id``
// is a tensor placeholder that the Python wrapper reserved as the
// "grad input" of the optimizer update; this function binds those
// placeholders to MPSGraph-derived gradient tensors after the forward
// has been emitted, then continues emitting the remaining (optimizer)
// ops which now resolve their grad reads correctly.
//
// Why this exists
// ---------------
// The hardcoded path supports SGD / Adam / AdamW.  Adding RMSprop /
// Adagrad / Adadelta / Adamax / NAdam (and any future optimizer) via
// the hardcoded path would duplicate the math already expressed by
// each :class:`compile_optimizer` subclass.  This function lets the
// Python wrapper reuse that math directly — trace it once into the
// same graph, then bind the ghost grads here.
//
// Output layout (in order):
//   1. loss
//   2. one entry per ``output_target_ids[i]`` (the trace ids of the
//      new_param / new_state tensors the optimizer wrote into).
//
// At run time, callers use :func:`run_executable_inplace` with
// output_targets = [loss_scratch, param_0, param_1, ...,
// state_buf_0_0, state_buf_0_1, ...].
LUCID_API CompiledExecutable*
compile_generic_fused_step(const TraceGraph& graph,
                           const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
                           TensorId loss_id,
                           const std::vector<TensorId>& param_ids,
                           const std::vector<TensorId>& ghost_grad_ids,
                           const std::vector<TensorId>& output_target_ids,
                           std::string* error_msg = nullptr);

// ── Stateful-variables variant of compile_generic_fused_step ────────
//
// Same lifecycle and contract as :func:`compile_generic_fused_step` —
// the trace covers forward + loss + (ghost-grad → opt update) — but a
// designated subset of the *external feeds* is promoted to MPSGraph
// **stateful variables** instead of placeholders, and the matching
// subset of the trace's *output targets* is bound to ``assignVariable``
// operations instead of executable graph outputs.
//
// Per-step semantics (when paired with ``run_executable_inplace``):
//
//   * The transient feeds (model input ``x``, loss target, per-step
//     scalars) remain regular placeholders — same I/O as before.
//   * Each variable feed initialises its variable from the current
//     :class:`MTLBuffer` contents of the matching Lucid Tensor at
//     *compile* time.  Later steps reuse the variable's persistent
//     internal storage; the executable's input list excludes them
//     entirely.
//   * Each variable write target is emitted as an ``assignVariable``
//     op (added to ``targetOperations``).  Immediately after the
//     assignment, a ``readVariable`` op is included as a regular
//     ``targetTensor`` so the new value can be flushed back to the
//     Lucid Tensor's MTLBuffer on each call — keeping
//     ``model.fc1.weight``-style reads up to date between steps.
//
// Why this wins
// -------------
// The original ``run_executable_inplace`` path allocates one fresh
// ``MTLBuffer`` per output target (``newBufferWithLength:``,
// ~5-30μs × N), wraps it as ``MPSGraphTensorData``, runs the
// executable, then swaps the Lucid Tensor's MLX array onto the
// fresh buffer.  Variables eliminate the per-step buffer
// allocation: the readback writes directly into the existing Lucid
// Tensor buffer, and the assignment writes happen *inside* the
// variable's persistent storage with no host-visible buffer at all.
//
// Parameters
// ----------
// graph, external_feeds, loss_id, param_ids, ghost_grad_ids, output_target_ids
//     Same semantics as :func:`compile_generic_fused_step`.
// variable_pairs : vector<pair<TensorId, TensorId>>
//     Each pair is ``(feed_id, write_id)``.  ``feed_id`` must be in
//     ``external_feeds``; ``write_id`` must appear in
//     ``output_target_ids`` (or be the trace output bound to a
//     compile-time-known write slot for this variable).  The feed's
//     initial values are snapshot from the Lucid Tensor's current
//     MTLBuffer at compile time.  Empty vector → equivalent to
//     :func:`compile_generic_fused_step`.
// error_msg : optional
//     Abort reason on failure.
//
// Returns
// -------
// CompiledExecutable*
//     Owns the compiled executable.  ``input_ids`` excludes every
//     ``feed_id`` in ``variable_pairs``; ``grad_output_ids`` excludes
//     every ``write_id`` and replaces them with the corresponding
//     ``readVariable`` outputs (so :func:`run_executable_inplace`
//     can still bind the Lucid Tensor buffers as targets).
//     ``nullptr`` on abort.
LUCID_API CompiledExecutable* compile_generic_fused_step_with_vars(
    const TraceGraph& graph,
    const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
    TensorId loss_id,
    const std::vector<TensorId>& param_ids,
    const std::vector<TensorId>& ghost_grad_ids,
    const std::vector<TensorId>& output_target_ids,
    const std::vector<std::pair<TensorId, TensorId>>& variable_pairs,
    std::string* error_msg = nullptr);

// ────────────────────────────────────────────────────────────────────
// Class-form facade (Phase B of the compile OOP refactor)
// ────────────────────────────────────────────────────────────────────
//
// `MpsBuilder` is the OOP shape behind the 5 ``compile_*`` free
// functions above.  Each free function above is now a thin
// forwarder that constructs a transient :class:`MpsBuilder` and
// dispatches to the matching method.  The free-function signatures
// are preserved for binding stability — Python (and the cache) still
// import them by name.
//
// **Lifetime.** One :class:`MpsBuilder` per compile call.  The class
// owns:
//   * Reference to the :class:`TraceGraph` and ``external_feeds`` map.
//   * The in-progress ``MPSGraph*`` (held as ``void*`` in this pure-C++
//     header; the ``.mm`` re-casts via ``__bridge``).
//   * The :class:`BuilderContext` that emitters write into.
//   * A scratch ``std::string* error_msg`` pointer for failure
//     reporting; populated on failure return.
//
// **Construction.** The constructor takes the same three arguments
// shared by every public method: ``graph``, ``external_feeds``, and
// the optional ``error_msg``.  It does *not* do any MPSGraph work —
// the MPSGraph is created lazily inside each ``compile_<mode>``
// method.  This keeps construction lightweight (the free-function
// forwarders create an instance even when validation rejects the
// call) and avoids leaking an MPSGraph allocation on early-exit
// paths.
//
// **Why a class rather than 5 free functions.**  The 5 functions
// shared ~60% scaffolding expressed as copy-paste (device check,
// emitter precheck, placeholder creation, emit loop, target
// construction, MPSGraph compile).  Grouping them as methods on a
// shared class lets future shared-scaffolding refactors land in one
// place rather than in 5 lock-stepped diffs.  The 5 methods
// intentionally remain as distinct entry points rather than virtual
// dispatch — the divergence between forward-only / forward-bwd /
// fused-step / generic-fused-step / variables is finite-state, not
// hierarchical.
class LUCID_API MpsBuilder {
public:
    MpsBuilder(const TraceGraph& graph,
               const std::unordered_map<TensorId, TensorImplPtr>& external_feeds,
               std::string* error_msg = nullptr,
               std::vector<EmittedOperation>* emitted = nullptr);
    ~MpsBuilder();

    MpsBuilder(const MpsBuilder&) = delete;
    MpsBuilder& operator=(const MpsBuilder&) = delete;

    // Forward-only compile.  See :func:`compile_trace` for semantics.
    CompiledExecutable* compile_trace(bool dynamic_batch = false,
                                      const std::vector<TensorId>& param_ids = {},
                                      const std::vector<TensorId>& explicit_outputs = {});

    // Forward + backward.  See :func:`compile_trace_with_backward`.
    // ``extra_output_ids`` are trace ids of explicit non-gradient outputs
    // (e.g. BN running-stat EMA new_rm/new_rv) that the executable must also
    // produce + return after [loss, *grads], so the runtime can write them back.
    CompiledExecutable*
    compile_trace_with_backward(TensorId loss_id,
                                const std::vector<TensorId>& param_ids,
                                bool dynamic_batch = false,
                                const std::vector<TensorId>& extra_output_ids = {});

    // Forward + bwd + hardcoded SGD/Adam.  See
    // :func:`compile_fused_training_step`.
    CompiledExecutable*
    compile_fused_training_step(TensorId loss_id,
                                const std::vector<TensorId>& param_ids,
                                const OptimizerSpec& opt_spec,
                                const std::vector<std::vector<TensorId>>& state_buf_ids_per_param,
                                const std::vector<TensorId>& scalar_input_ids);

    // Forward + bwd + ghost-grad optimizer (any kind).  See
    // :func:`compile_generic_fused_step`.
    CompiledExecutable* compile_generic_fused_step(TensorId loss_id,
                                                   const std::vector<TensorId>& param_ids,
                                                   const std::vector<TensorId>& ghost_grad_ids,
                                                   const std::vector<TensorId>& output_target_ids);

    // As above + MPSGraph variables.  See
    // :func:`compile_generic_fused_step_with_vars`.
    CompiledExecutable* compile_generic_fused_step_with_vars(
        TensorId loss_id,
        const std::vector<TensorId>& param_ids,
        const std::vector<TensorId>& ghost_grad_ids,
        const std::vector<TensorId>& output_target_ids,
        const std::vector<std::pair<TensorId, TensorId>>& variable_pairs);

private:
    // Members.  ``graph_`` and ``external_feeds_`` are references —
    // the caller owns them for the duration of the MpsBuilder
    // lifetime (one compile call).  ``error_msg_`` is an out-param
    // pointer also owned by the caller.
    const TraceGraph& graph_;
    const std::unordered_map<TensorId, TensorImplPtr>& external_feeds_;
    std::string* error_msg_;
    std::vector<EmittedOperation>* emitted_;
};

}  // namespace lucid::compile
