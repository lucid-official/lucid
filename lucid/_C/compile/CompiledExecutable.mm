// lucid/_C/compile/CompiledExecutable.mm
//
// Objective-C++ side of the opaque :class:`CompiledExecutable` handle.
// Defines the actual class (with an ARC-managed ``MPSGraphExecutable*``
// member) and the free-function helpers declared in
// :file:`CompiledExecutable.h`.

#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#import <objc/runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <mlx/array.h>

#include "../backend/gpu/MlxBridge.h"
#include "../backend/gpu/mps/MpsBridge.h"
#include "../core/Determinism.h"
#include "../core/ElementWrite.h"
#include "../core/Storage.h"
#include "../core/TensorImpl.h"
#include "CompiledExecutable.h"
#include "FirstRunWatchdog.h"
#include "MpsDtype.h"
#include "RngFeeds.h"

// The first-run ledger of one MPSGraphExecutable (FirstRunWatchdog.h), hung on it as an
// associated object.  Every relabelled or cached CompiledExecutable shares the
// executable, so they share the ledger too, without a new field in the class below (it
// is defined twice, here and in MpsBuilder.mm, and the two must agree).
@interface LucidFirstRunLedgerBox : NSObject {
@public
    std::shared_ptr<lucid::compile::FirstRunLedger> ledger;
}
@end

@implementation LucidFirstRunLedgerBox
@end

namespace lucid::compile {

namespace {
// Its address is the association key.
const char kFirstRunLedgerKey = 0;
}  // namespace

// Class definition (visible only inside .mm callers).  Holds the
// ARC-strong reference to the MPSGraphExecutable plus the I/O ordering
// plan minted by :class:`MpsBuilder`.
class CompiledExecutable {
public:
    // The shape each feed placeholder was declared at when it differs from the logical
    // ``input_shapes`` (a rank-5+ feed read only through views, ``canonical_feed_shapes``):
    // the caller's buffer is bound at it.  Empty = every slot is bound at ``input_shapes``;
    // otherwise one entry per slot, empty for a slot bound at its logical shape.
    std::vector<Shape> feed_target_shapes;
    MPSGraphExecutable* executable = nil;  // ARC strong

    // When the executable was compiled from a graph that contains
    // ``MPSGraphVariable``-backed tensors (``variableWithData:`` /
    // ``assignVariable:``), the variable's internal storage belongs to
    // the source MPSGraph object, not to the executable.  Releasing the
    // graph (e.g. at the end of the compile autoreleasepool) frees the
    // variable's buffer; subsequent ``runWithMTLCommandQueue:`` calls
    // then crash inside ``GPU::VarHandleOpHandler::encodeOp`` when the
    // executor tries to dereference the dangling variable handle.  We
    // retain the source graph here for variable-bearing compiles so the
    // variable storage stays live for the executable's lifetime.  For
    // pure-forward / placeholder-only compiles this field is nil.
    void* source_graph = nullptr;  // __bridge_retained MPSGraph*

    // Trace ids for inputs (feed order) and outputs (target order).
    std::vector<TensorId> input_ids;
    std::vector<TensorId> output_ids;

    // Shape + dtype per input/output, parallel to the id vectors.
    // Used to build MPSGraphTensorData at run time.
    std::vector<Shape> input_shapes;
    std::vector<Dtype> input_dtypes;
    std::vector<Shape> output_shapes;
    std::vector<Dtype> output_dtypes;

    Device device = Device::GPU;

    // Phase 1.3: optional backward outputs.  When non-empty, the
    // executable also produces ``len(grad_output_ids)`` gradient
    // tensors after the forward outputs, one per parameter passed to
    // :func:`compile_trace_with_backward`.  Empty for pure-forward
    // executables.  ``grad_output_ids[i]`` is the TraceId minted for
    // the gradient of ``param_ids[i]``; the corresponding shape and
    // dtype are appended to ``output_shapes`` / ``output_dtypes``
    // immediately after the loss tensor's entry.
    std::vector<TensorId> grad_output_ids;

    // Phase 1.6: when ``true`` the executable was compiled with a
    // symbolic leading dim on every non-parameter feed.  At run time
    // we replace the recorded shape's first axis with the actual
    // batch size pulled off the call-site input, both for the input
    // MPSGraphTensorData binding and for the output buffer allocation.
    bool dynamic_batch = false;
    // Set of feed-slot indices (into ``input_ids``) that were
    // declared parameters and therefore keep their static shape even
    // in dynamic-batch mode.
    std::unordered_set<std::size_t> static_feed_slots;
    // Bind order against ``executable.feedTensors``: position ``k`` of the
    // ``inputsArray`` handed to MPSGraph carries Lucid feed slot
    // ``feed_order[k]``.  Empty = the two orders coincide, which is the
    // case for a freshly compiled executable.  A package round-trip does
    // NOT preserve the order (the graph is compiled from an NSDictionary
    // of feeds, so the order was never ours to choose), which is why
    // ``load_executable`` re-derives this from the placeholder names.
    std::vector<std::size_t> feed_order;

    // Run-time ObjC object caches populated lazily on the first execution:
    //   * ``cached_input_ns_shapes`` / ``cached_output_ns_shapes``: static
    //     ``NSArray<NSNumber*>*`` per input/output slot so ``shape_to_nsarray``
    //     does not allocate 100+ NSNumber arrays per step.
    //   * ``cached_feed_td`` / ``cached_feed_buf`` / ``cached_feed_arr``:
    //     per-MPS-feed-position ``MPSGraphTensorData*`` cache keyed on the live
    //     ``mlx::core::array`` and its ``MTLBuffer`` pointer so static
    //     parameters and constants reuse their ``MPSGraphTensorData*`` with
    //     zero Objective-C allocations on subsequent steps.
    NSMutableArray<NSArray<NSNumber*>*>* cached_input_ns_shapes = nil;
    NSMutableArray<NSArray<NSNumber*>*>* cached_output_ns_shapes = nil;
    NSMutableArray<id>* cached_feed_td = nil;
    std::vector<const void*> cached_feed_buf;
    std::vector<std::weak_ptr<::mlx::core::array>> cached_feed_arr;

    ~CompiledExecutable() {
        // Release the variable-storage-owning MPSGraph if we retained
        // one during compile (``compile_generic_fused_step_with_vars``).
        // ARC handles ``executable``.
        if (source_graph != nullptr) {
            @autoreleasepool {
                MPSGraph* g = (__bridge_transfer MPSGraph*)source_graph;
                (void)g;
            }
            source_graph = nullptr;
        }
    }
};

std::size_t executable_num_inputs(const CompiledExecutable* exe) {
    return exe ? exe->input_ids.size() : 0;
}

std::size_t executable_num_outputs(const CompiledExecutable* exe) {
    return exe ? exe->output_ids.size() : 0;
}

std::vector<TensorId> executable_input_ids(const CompiledExecutable* exe) {
    return exe ? exe->input_ids : std::vector<TensorId>{};
}

std::vector<TensorId> executable_output_ids(const CompiledExecutable* exe) {
    return exe ? exe->output_ids : std::vector<TensorId>{};
}

std::vector<TensorId> executable_grad_output_ids(const CompiledExecutable* exe) {
    return exe ? exe->grad_output_ids : std::vector<TensorId>{};
}

std::vector<Shape> executable_input_shapes(const CompiledExecutable* exe) {
    return exe ? exe->input_shapes : std::vector<Shape>{};
}

std::vector<Dtype> executable_input_dtypes(const CompiledExecutable* exe) {
    return exe ? exe->input_dtypes : std::vector<Dtype>{};
}

std::vector<std::size_t> executable_feed_order(const CompiledExecutable* exe) {
    return exe ? exe->feed_order : std::vector<std::size_t>{};
}

FirstRunState executable_first_run_state(const CompiledExecutable* exe) {
    if (exe == nullptr || exe->executable == nil)
        return FirstRunState::Cold;
    @autoreleasepool {
        LucidFirstRunLedgerBox* box =
            objc_getAssociatedObject(exe->executable, &kFirstRunLedgerKey);
        return box == nil ? FirstRunState::Cold : box->ledger->summary();
    }
}

void destroy_executable(CompiledExecutable* exe) {
    // ``delete`` invokes the C++ destructor which releases the
    // ARC-strong reference to ``executable``.  Safe with ``nullptr``.
    delete exe;
}

CompiledExecutable* relabel_executable(const CompiledExecutable* exe,
                                       std::vector<TensorId> input_ids,
                                       std::vector<TensorId> output_ids) {
    if (exe == nullptr)
        return nullptr;
    if (input_ids.size() != exe->input_ids.size() || output_ids.size() != exe->output_ids.size())
        return nullptr;
    auto* copy = new CompiledExecutable();
    copy->executable = exe->executable;
    if (exe->source_graph != nullptr) {
        @autoreleasepool {
            MPSGraph* g = (__bridge MPSGraph*)exe->source_graph;
            copy->source_graph = (__bridge_retained void*)g;
        }
    }
    copy->input_ids = std::move(input_ids);
    copy->output_ids = std::move(output_ids);
    copy->input_shapes = exe->input_shapes;
    copy->input_dtypes = exe->input_dtypes;
    copy->output_shapes = exe->output_shapes;
    copy->output_dtypes = exe->output_dtypes;
    copy->device = exe->device;
    copy->grad_output_ids = exe->grad_output_ids;
    copy->dynamic_batch = exe->dynamic_batch;
    copy->static_feed_slots = exe->static_feed_slots;
    copy->feed_order = exe->feed_order;
    copy->feed_target_shapes = exe->feed_target_shapes;
    return copy;
}

// Helpers shared with Linear emitter / MpsBuilder (defined here so the
// .mm can use them; not exported to other TUs).
namespace detail {

inline NSArray<NSNumber*>* shape_to_nsarray(const Shape& shape) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:shape.size()];
    for (std::int64_t d : shape)
        [out addObject:[NSNumber numberWithLongLong:d]];
    return out;
}

inline std::size_t shape_nbytes(const Shape& shape, Dtype dt) {
    std::size_t n = 1;
    for (std::int64_t d : shape)
        n *= static_cast<std::size_t>(d);
    // ``dtype_size`` rather than a two-case guess.  The guess was "2 for
    // F16, else 4", which under-allocates every eight-byte dtype: an
    // ``argmax`` output (I64) or a complex one got half the buffer it
    // needed, and MPSGraph answers that with a failed assertion inside
    // MPSNDArray — a process abort, not an exception anything can catch.
    return n * dtype_size(dt);
}

inline std::string shape_str(const Shape& shape) {
    std::string out = "(";
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i != 0)
            out += ", ";
        out += std::to_string(shape[i]);
    }
    if (shape.size() == 1)
        out += ",";
    out += ")";
    return out;
}

// Check one feed slot against the contract recorded at trace time and
// return the shape it should be bound with.
//
// This has to happen while the feed is still a ``TensorImpl`` we can
// reason about.  ``MPSGraphTensorData initWithMTLBuffer:shape:dataType:``
// takes the recorded shape on faith, so a feed handed to the wrong slot
// reaches MetalPerformanceShaders as a buffer too small for the shape it
// was labelled with, and MPS *aborts the process* ("MPSNDArray ... buffer
// is not large enough") or walks off the end of the allocation.  Neither
// is catchable from Python, so every mismatch has to become an exception
// here.
inline Shape validate_feed(const CompiledExecutable* exe,
                           std::size_t i,
                           const TensorImplPtr& impl,
                           const char* fn) {
    const std::string at = std::string(fn) + ": feed slot " + std::to_string(i);
    if (!impl)
        throw std::invalid_argument(at + " is null");
    if (impl->device() != Device::GPU)
        throw std::invalid_argument(at + " must be on Device::GPU — every input feed of a "
                                         "compiled executable lives on the GPU");
    if (impl->dtype() != exe->input_dtypes[i])
        throw std::invalid_argument(at + " expects dtype " +
                                    std::string(dtype_name(exe->input_dtypes[i])) + ", got " +
                                    std::string(dtype_name(impl->dtype())));

    const Shape& want = exe->input_shapes[i];
    const Shape& got = impl->shape();
    const bool is_static_slot = exe->static_feed_slots.find(i) != exe->static_feed_slots.end();
    const bool batch_varies = exe->dynamic_batch && !is_static_slot && !want.empty();

    if (!batch_varies) {
        if (got != want)
            throw std::invalid_argument(at + " expects shape " + shape_str(want) + ", got " +
                                        shape_str(got));
        return want;
    }
    // Dynamic-batch, non-parameter slot: the leading axis is symbolic,
    // every trailing axis was frozen at trace time.
    if (got.size() != want.size())
        throw std::invalid_argument(at + " expects rank " + std::to_string(want.size()) +
                                    " (traced shape " + shape_str(want) +
                                    ", leading axis free), got rank " + std::to_string(got.size()) +
                                    " (shape " + shape_str(got) + ")");
    for (std::size_t d = 1; d < want.size(); ++d) {
        if (got[d] != want[d])
            throw std::invalid_argument(at + " expects shape " + shape_str(want) +
                                        " with only the leading axis free, got " + shape_str(got));
    }
    return got;
}

// Recover the mapping from MPSGraph's feed positions back to Lucid's
// feed-slot order.
//
// ``runWithMTLCommandQueue:inputsArray:`` is positional against
// ``executable.feedTensors``, and that order is not ours to pick: the
// graph is compiled from an *NSDictionary* of feeds, and a
// ``.mpsgraphpackage`` round-trip reorders it again.  For a freshly
// compiled executable the two orders happen to agree, which is why the
// in-process path has always worked; a reloaded one silently permutes,
// and binding a feed under the wrong slot's shape aborts inside MPS.
//
// Every placeholder is named ``feed_<tid>`` by the builder, so the
// permutation is recoverable exactly.  Returns ``feed_order[k]`` = the
// Lucid slot belonging at MPS feed position ``k``, or an empty vector
// when the names do not resolve (caller then falls back to positional
// binding, i.e. the historical behaviour).
inline std::vector<std::size_t> derive_feed_order(MPSGraphExecutable* exec,
                                                  const std::vector<TensorId>& input_ids) {
    std::vector<std::size_t> order;
    if (exec == nil)
        return order;
    NSArray<MPSGraphTensor*>* feeds = exec.feedTensors;
    if (feeds == nil || feeds.count != input_ids.size())
        return order;

    std::unordered_map<std::int64_t, std::size_t> slot_of;
    slot_of.reserve(input_ids.size());
    for (std::size_t i = 0; i < input_ids.size(); ++i)
        slot_of.emplace(input_ids[i].v, i);

    order.reserve(input_ids.size());
    for (MPSGraphTensor* t in feeds) {
        // The builder's ``name:`` argument names the placeholder
        // *operation*; ``MPSGraphTensor`` itself carries no name.
        MPSGraphOperation* op = t.operation;
        NSString* nm = (op == nil) ? nil : op.name;
        const char* c = (nm == nil) ? nullptr : nm.UTF8String;
        if (c == nullptr)
            return {};
        // MPSGraph may decorate the name it was given, so look the
        // marker up rather than requiring it at offset 0.
        const char* marker = std::strstr(c, "feed_");
        if (marker == nullptr)
            return {};
        char* endp = nullptr;
        const long long tid = std::strtoll(marker + 5, &endp, 10);
        if (endp == marker + 5)
            return {};
        const auto it = slot_of.find(static_cast<std::int64_t>(tid));
        if (it == slot_of.end())
            return {};
        order.push_back(it->second);
    }
    return order;
}

// Names MPSGraph reports for the executable's feed tensors, in its own
// order.  Diagnostic only — :func:`derive_feed_order` is what the run
// path uses.
inline std::vector<std::string> feed_tensor_names(MPSGraphExecutable* exec) {
    std::vector<std::string> out;
    if (exec == nil)
        return out;
    NSArray<MPSGraphTensor*>* feeds = exec.feedTensors;
    if (feeds == nil)
        return out;
    for (MPSGraphTensor* t in feeds) {
        MPSGraphOperation* op = t.operation;
        NSString* nm = (op == nil) ? nil : op.name;
        out.emplace_back((nm == nil) ? "<unnamed>" : nm.UTF8String);
    }
    return out;
}

// The first-run ledger of ``exec``, made on first use.  The reference is into the
// association's box, which lives as long as ``exec``: valid for any call that holds it.
// An association on nil is a no-op, so a nil ``exec`` would be handed a box that dies
// with this call.
inline const std::shared_ptr<FirstRunLedger>& ledger_of(MPSGraphExecutable* exec) {
    if (exec == nil)
        throw std::invalid_argument("first-run ledger: the executable has no MPSGraphExecutable");
    static std::mutex creation_mutex;
    LucidFirstRunLedgerBox* box = objc_getAssociatedObject(exec, &kFirstRunLedgerKey);
    if (box == nil) {
        const std::lock_guard<std::mutex> lock(creation_mutex);
        box = objc_getAssociatedObject(exec, &kFirstRunLedgerKey);
        if (box == nil) {
            box = [[LucidFirstRunLedgerBox alloc] init];
            box->ledger = std::make_shared<FirstRunLedger>();
            objc_setAssociatedObject(exec, &kFirstRunLedgerKey, box,
                                     OBJC_ASSOCIATION_RETAIN_NONATOMIC);
        }
    }
    return box->ledger;
}

// What MPSGraph specializes an executable for: nothing varies on a static one, and a
// dynamic-batch one specializes once per batch size (the leading axis of every feed that
// is not a parameter).  Never ``FirstRunLedger::kStaticSignature`` for the latter.
inline std::uint64_t first_run_signature(const CompiledExecutable* exe,
                                         const std::vector<TensorImplPtr>& feeds) {
    if (!exe->dynamic_batch)
        return FirstRunLedger::kStaticSignature;
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    for (std::size_t i = 0; i < feeds.size(); ++i) {
        if (exe->static_feed_slots.find(i) != exe->static_feed_slots.end() || !feeds[i] ||
            feeds[i]->shape().empty())
            continue;
        hash = (hash ^ static_cast<std::uint64_t>(feeds[i]->shape()[0])) * 0x100000001b3ULL;
    }
    return hash | 1;
}

// Whether ``signature`` has nothing left to prove, or the guard is off: only then may a
// result slot be bound to a live buffer of the caller's (``LUCID_COMPILE_ALIAS_INPLACE``).
// A first run given up on can still finish later and would write into it.
inline bool first_run_settled(CompiledExecutable* exe, std::uint64_t signature) {
    return !FirstRunWatchdog::instance().engaged() ||
           ledger_of(exe->executable)->state(signature) == FirstRunState::Warm;
}

// One synchronous run on this thread.  MPSGraph reports a failure by throwing an
// Objective-C exception, which pybind11 can only call "Caught an unknown
// exception!"; this hands Python the reason instead.
inline void run_now(MPSGraphExecutable* exe,
                    id<MTLCommandQueue> queue,
                    NSArray<MPSGraphTensorData*>* feeds,
                    NSArray<MPSGraphTensorData*>* results,
                    const char* who) {
    MPSGraphExecutableExecutionDescriptor* desc =
        [[MPSGraphExecutableExecutionDescriptor alloc] init];
    desc.waitUntilCompleted = YES;
    @try {
        (void)[exe runWithMTLCommandQueue:queue
                              inputsArray:feeds
                             resultsArray:results
                      executionDescriptor:desc];
    } @catch (NSException* e) {
        throw std::runtime_error(std::string(who) + ": MPSGraph failed: " +
                                 (e.reason ? e.reason.UTF8String : e.name.UTF8String));
    }
}

// Run ``exe`` synchronously.  Every synchronous run goes through here.
//
// MPSGraph specializes the executable inside its first run, with no bound of its own,
// so a signature that has not finished one runs on a worker under the first-run budget
// (FirstRunWatchdog.h).  After that, and with the guard off, the run is on the caller's
// thread as it always was; the check costs one ledger lookup.
inline void run_or_throw(MPSGraphExecutable* exe,
                         id<MTLCommandQueue> queue,
                         NSArray<MPSGraphTensorData*>* feeds,
                         NSArray<MPSGraphTensorData*>* results,
                         const char* who,
                         std::uint64_t signature,
                         const FirstRunPoll& poll) {
    FirstRunWatchdog& watchdog = FirstRunWatchdog::instance();
    if (!watchdog.engaged()) {
        run_now(exe, queue, feeds, results, who);
        return;
    }
    const std::shared_ptr<FirstRunLedger>& ledger = ledger_of(exe);
    if (ledger->state(signature) == FirstRunState::Warm) {
        run_now(exe, queue, feeds, results, who);
        return;
    }
    // What the worker holds are ARC references to the executable, the queue and the
    // buffers: a worker given up on outlives this call, and the CompiledExecutable and
    // the caller's tensors may not.
    watchdog.run(
        ledger, signature, who,
        [exe, queue, feeds, results, who] {
            @autoreleasepool {
                run_now(exe, queue, feeds, results, who);
            }
        },
        poll);
}

// The target's own MTLBuffer, for ``LUCID_COMPILE_ALIAS_INPLACE`` to bind as
// its result slot, or nil when the mode is off.  Only a target whose buffer
// is its elements has one: a Metal view of part of a tensor reads a packed
// copy of them, so it gets a fresh buffer and the swap.
inline id<MTLBuffer> alias_buffer(bool alias_mode, const TensorImpl& target) {
    if (!alias_mode || !buffer_is_elements(target))
        return nil;
    const auto& gs = std::get<GpuStorage>(target.storage());
    return (__bridge id<MTLBuffer>)lucid::gpu::mps::array_to_buffer(*gs.arr).mtl_buffer;
}

// The shape slot ``i``'s buffer is bound at: the one its placeholder was declared at
// (``feed_target_shapes``, a rank-5+ feed read only through views) or its logical shape.
inline const Shape& bound_input_shape(const CompiledExecutable* exe, std::size_t i) {
    if (i < exe->feed_target_shapes.size() && !exe->feed_target_shapes[i].empty())
        return exe->feed_target_shapes[i];
    return exe->input_shapes[i];
}

// The target shapes of the executable: ``targetTensors`` for one compiled in this
// process, and for a package-loaded one (which has none) what
// ``getOutputTypesWithDevice:`` reports for the input shapes it will run.  Nil
// when neither is available.
inline NSArray<NSArray<NSNumber*>*>* executable_target_shapes(CompiledExecutable* exe) {
    const std::size_t n_out = exe->output_shapes.size();
    NSMutableArray<NSArray<NSNumber*>*>* shapes = [NSMutableArray arrayWithCapacity:n_out];
    NSArray<MPSGraphTensor*>* targets = exe->executable.targetTensors;
    if (targets != nil) {
        if (targets.count != n_out)
            return nil;
        for (MPSGraphTensor* t in targets)
            [shapes addObject:t.shape != nil ? t.shape : @[]];
        return shapes;
    }
    const std::size_t n_in = exe->input_shapes.size();
    NSMutableArray<MPSGraphShapedType*>* in_types = [NSMutableArray arrayWithCapacity:n_in];
    for (std::size_t k = 0; k < n_in; ++k) {
        const std::size_t i = exe->feed_order.empty() ? k : exe->feed_order[k];
        [in_types addObject:[[MPSGraphShapedType alloc]
                                initWithShape:shape_to_nsarray(bound_input_shape(exe, i))
                                     dataType:mps_dtype_or_throw(exe->input_dtypes[i])]];
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
    NSArray<MPSGraphShapedType*>* outs =
        [exe->executable getOutputTypesWithDevice:[MPSGraphDevice deviceWithMTLDevice:device]
                                       inputTypes:in_types
                            compilationDescriptor:nil];
    if (outs == nil || outs.count != n_out)
        return nil;
    for (MPSGraphShapedType* t in outs)
        [shapes addObject:t.shape != nil ? t.shape : @[]];
    return shapes;
}

// The shapes the executable's results are bound at: its own target shapes, which
// may differ from the logical ``output_shapes`` in rank but never in numel.  A
// rank-5+ output is named at rank 4 (``canonical_output`` in MpsBuilder.mm) so
// its copy kernel avoids MPSGraph's slice loop, and a keepdim reduce emits
// ``(1, 1)`` where the recorded loss is ``(1,)``.  A result bound at a rank
// other than its target's forces an MPSNDArrayIdentity reshape that asserts
// under F16 (or reads the buffer wrongly), so the ``MPSGraphTensorData`` is
// always created at this shape and only the host wrap uses the logical one.
// Nil when the target shapes are unknown, symbolic (a dynamic batch keeps its
// targets as they are) or disagree with the logical numel: the caller then
// binds every result at its logical shape.
inline NSArray<NSArray<NSNumber*>*>* natural_result_shapes(CompiledExecutable* exe) {
    NSArray<NSArray<NSNumber*>*>* shapes = executable_target_shapes(exe);
    if (shapes == nil)
        return nil;
    for (std::size_t j = 0; j < shapes.count; ++j) {
        std::int64_t natural_numel = 1;
        for (NSNumber* d in shapes[j]) {
            if (d.longLongValue < 0)
                return nil;
            natural_numel *= d.longLongValue;
        }
        std::int64_t logical_numel = 1;
        for (std::int64_t d : exe->output_shapes[j])
            logical_numel *= d;
        if (natural_numel != logical_numel)
            return nil;
    }
    return shapes;
}

// The leading axis of the first entry of ``shapes`` that is not a parameter, or -1.
inline std::int64_t first_batch_axis(const CompiledExecutable* exe,
                                     const std::vector<Shape>& shapes) {
    for (std::size_t i = 0; i < shapes.size(); ++i) {
        if (exe->static_feed_slots.find(i) == exe->static_feed_slots.end() && !shapes[i].empty())
            return shapes[i][0];
    }
    return -1;
}

// The shape each output takes for this call: ``exe->output_shapes``, except that on a
// dynamic-batch executable (Phase 1.6) an output whose leading axis is the *traced* batch
// size (the first non-parameter input's recorded one) is batch-shaped and takes this call's
// batch size, the leading axis of the first non-parameter feed.  Scalar reductions over
// the batch stay 0-D, and outputs unrelated to the batch stay static.
inline std::vector<Shape> output_shapes_for_call(const CompiledExecutable* exe,
                                                 const std::vector<TensorImplPtr>& feeds) {
    std::vector<Shape> shapes = exe->output_shapes;
    if (!exe->dynamic_batch)
        return shapes;
    std::vector<Shape> fed;
    fed.reserve(feeds.size());
    for (const auto& impl : feeds)
        fed.push_back(impl ? impl->shape() : Shape{});
    const std::int64_t batch_size = first_batch_axis(exe, fed);
    const std::int64_t traced_batch_size = first_batch_axis(exe, exe->input_shapes);
    if (batch_size < 0 || traced_batch_size < 0)
        return shapes;
    for (Shape& s : shapes) {
        if (!s.empty() && s[0] == traced_batch_size)
            s[0] = batch_size;
    }
    return shapes;
}

// Lazily populate the per-executable ``NSArray<NSNumber*>*`` shape tables and
// feed ``MPSGraphTensorData*`` cache slots.
inline void ensure_runtime_caches(CompiledExecutable* exe) {
    const std::size_t n_in = exe->input_shapes.size();
    const std::size_t n_out = exe->output_shapes.size();
    if (exe->cached_input_ns_shapes == nil || exe->cached_input_ns_shapes.count != n_in) {
        NSMutableArray<NSArray<NSNumber*>*>* in_ns = [NSMutableArray arrayWithCapacity:n_in];
        for (std::size_t i = 0; i < n_in; ++i)
            [in_ns addObject:shape_to_nsarray(bound_input_shape(exe, i))];
        exe->cached_input_ns_shapes = in_ns;
    }
    if (exe->cached_output_ns_shapes == nil || exe->cached_output_ns_shapes.count != n_out) {
        NSMutableArray<NSArray<NSNumber*>*>* out_ns = [NSMutableArray arrayWithCapacity:n_out];
        NSArray<NSArray<NSNumber*>*>* natural = natural_result_shapes(exe);
        for (std::size_t j = 0; j < n_out; ++j)
            [out_ns
                addObject:natural != nil ? natural[j] : shape_to_nsarray(exe->output_shapes[j])];
        exe->cached_output_ns_shapes = out_ns;
    }
    if (exe->cached_feed_td == nil || exe->cached_feed_td.count != n_in) {
        NSMutableArray<id>* td_cache = [NSMutableArray arrayWithCapacity:n_in];
        for (std::size_t k = 0; k < n_in; ++k)
            [td_cache addObject:[NSNull null]];
        exe->cached_feed_td = td_cache;
        exe->cached_feed_buf.assign(n_in, nullptr);
        exe->cached_feed_arr.assign(n_in, std::weak_ptr<::mlx::core::array>{});
    }
}

// Build the positional ``MPSGraphTensorData*`` array for ``input_feeds``,
// reusing cached ``MPSGraphTensorData*`` objects for static-shape feeds whose
// underlying MLX array and ``MTLBuffer`` have not changed since the previous
// execution.
inline NSMutableArray<MPSGraphTensorData*>* build_feeds_array(
    CompiledExecutable* exe, const std::vector<TensorImplPtr>& input_feeds, const char* who) {
    // Every run fills and reads ``exe``'s feed caches here, with the GIL
    // released by the bindings, so concurrent runs serialise on it (LCD-655).
    static std::recursive_mutex feeds_mutex;
    const std::lock_guard<std::recursive_mutex> feeds_lock(feeds_mutex);
    ensure_runtime_caches(exe);
    const std::size_t n_in = input_feeds.size();
    NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray arrayWithCapacity:n_in];
    for (std::size_t k = 0; k < n_in; ++k) {
        // ``feeds[k]`` is matched positionally against
        // ``executable.feedTensors[k]``, which is not necessarily
        // our slot k — see ``detail::derive_feed_order``.
        const std::size_t i = exe->feed_order.empty() ? k : exe->feed_order[k];
        const auto& impl = input_feeds[i];
        // Dtype / shape are checked before the buffer is labelled —
        // see :func:`detail::validate_feed` for why a mismatch may
        // not be allowed to reach MPSGraph.  The returned shape is
        // the trace-time one, except on a dynamic-batch slot where
        // the call-site leading axis wins.
        const Shape feed_shape = validate_feed(exe, i, impl, who);

        const auto& gs = std::get<GpuStorage>(impl->storage());
        if (!gs.arr)
            throw std::runtime_error(std::string(who) + ": input feed GpuStorage has no MLX array");

        lucid::gpu::mps::BufferView view = lucid::gpu::mps::array_to_buffer(*gs.arr);
        const bool static_shape = (feed_shape == exe->input_shapes[i]);
        if (static_shape && exe->cached_feed_buf[k] == view.mtl_buffer &&
            !exe->cached_feed_arr[k].expired() && exe->cached_feed_arr[k].lock() == gs.arr) {
            id cached = exe->cached_feed_td[k];
            if (cached != [NSNull null]) {
                [feeds addObject:(MPSGraphTensorData*)cached];
                continue;
            }
        }

        id<MTLBuffer> in_buf = (__bridge id<MTLBuffer>)view.mtl_buffer;
        NSArray<NSNumber*>* ns_shape =
            static_shape ? exe->cached_input_ns_shapes[i] : shape_to_nsarray(feed_shape);
        MPSDataType ns_dt = mps_dtype_or_throw(exe->input_dtypes[i]);
        MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:in_buf
                                                                         shape:ns_shape
                                                                      dataType:ns_dt];
        if (static_shape) {
            exe->cached_feed_td[k] = td;
            exe->cached_feed_buf[k] = view.mtl_buffer;
            exe->cached_feed_arr[k] = gs.arr;
        }
        [feeds addObject:td];
    }
    return feeds;
}

}  // namespace detail

std::vector<std::string> executable_feed_names(const CompiledExecutable* exe) {
    if (exe == nullptr)
        return {};
    @autoreleasepool {
        return detail::feed_tensor_names(exe->executable);
    }
}

// Run a compiled executable.  Feed-order inputs and target-order
// outputs are paired against the executable's recorded ordering.
// Output ``TensorImpl`` objects are freshly allocated GPU buffers
// produced by the executable; their MLX-array wrappers reuse the
// output ``MTLBuffer`` (no copy).
//
// Defined here rather than in MpsBuilder.mm because all the
// MPSGraph-side machinery is in this translation unit already.
LUCID_API std::vector<TensorImplPtr> run_executable(CompiledExecutable* exe,
                                                    const std::vector<TensorImplPtr>& traced_feeds,
                                                    const FirstRunPoll& poll) {
    // Draws the trace made are made again — see RngFeeds.h.
    const std::vector<TensorImplPtr> input_feeds = redraw_rng_feeds(traced_feeds);
    if (exe == nullptr)
        throw std::invalid_argument("run_executable: null executable");
    if (input_feeds.size() != exe->input_ids.size())
        throw std::invalid_argument(
            "run_executable: input count mismatch (expected feed-order list of "
            "length matching executable_num_inputs)");

    // Total output count = forward outputs + grad outputs.  The output_*
    // vectors (shapes/dtypes) are sized to the combined total; output_ids
    // only enumerates the forward portion, grad_output_ids the backward.
    const std::size_t total_outs = exe->output_ids.size() + exe->grad_output_ids.size();
    if (exe->output_shapes.size() != total_outs || exe->output_dtypes.size() != total_outs) {
        throw std::runtime_error("run_executable: output_shapes/dtypes vector mismatch "
                                 "(internal bug — builder did not populate consistently)");
    }

    id<MTLDevice> device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lucid::gpu::mps::shared_mtl_queue();
    const std::uint64_t signature = detail::first_run_signature(exe, input_feeds);

    std::vector<TensorImplPtr> outputs;
    outputs.reserve(exe->output_ids.size());

    @autoreleasepool {
        NSMutableArray<MPSGraphTensorData*>* feeds =
            detail::build_feeds_array(exe, input_feeds, "run_executable");

        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithCapacity:total_outs];
        std::vector<id<MTLBuffer>> out_bufs;
        out_bufs.reserve(total_outs);

        const std::vector<Shape> realized_output_shapes =
            detail::output_shapes_for_call(exe, input_feeds);

        for (std::size_t j = 0; j < total_outs; ++j) {
            const std::size_t nbytes =
                detail::shape_nbytes(realized_output_shapes[j], exe->output_dtypes[j]);
            id<MTLBuffer> out_buf = [device newBufferWithLength:nbytes
                                                        options:MTLResourceStorageModeShared];
            if (!out_buf)
                throw std::runtime_error("run_executable: MTLBuffer allocation for output failed");
            out_bufs.push_back(out_buf);

            NSArray<NSNumber*>* ns_shape =
                (realized_output_shapes[j] == exe->output_shapes[j])
                    ? exe->cached_output_ns_shapes[j]
                    : detail::shape_to_nsarray(realized_output_shapes[j]);
            MPSDataType ns_dt = mps_dtype_or_throw(exe->output_dtypes[j]);
            MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                                             shape:ns_shape
                                                                          dataType:ns_dt];
            [results addObject:td];
        }

        // Default = **synchronous** ``runWithMTLCommandQueue:`` path.
        //
        // The async ``encodeToCommandBuffer:`` + ``commit`` path is
        // ~100-200μs cheaper per call on M-series and was the prior
        // default, but it is **CORRECTNESS-UNSAFE under many
        // consecutive calls** — see
        // ``obsidian/engine/engine-compile-output-buffer-pool-
        // exhaustion-2026-05-26.md`` for the full reproducer.  After
        // N consecutive ``cm(x)`` calls without intermediate
        // materialisation (N≈14 for a GPT-2 transformer-block-sized
        // output, ~50 for a CIFAR ResNet block, ~100 with periodic
        // ``.sum().item()`` force-evals interleaved), the returned
        // output's MTLBuffer is silently overwritten to zeros — the
        // computation completed correctly but the buffer lifecycle
        // got tangled in MLX's async tracker.
        //
        // The bug surfaced as a P0 priority while investigating the
        // bench harness's parity divergence on gpt2_block @ BS=128.
        // Switching to the sync path eliminates all reproductions
        // (every N up to 100, every workload tested) at the cost of
        // ~100-200μs per call — acceptable when the alternative is
        // silent wrong outputs.
        //
        // ``LUCID_COMPILE_ASYNC=1`` opts back into the async path for
        // users who (a) know their workload doesn't hit the
        // threshold or (b) wrap every compile call with explicit
        // ``.sum().item()`` materialisation.  The sync default
        // protects the common case.
        static const bool force_async = []() {
            const char* s = std::getenv("LUCID_COMPILE_ASYNC");
            return s && std::string(s) == "1";
        }();
        if (force_async) {
            MPSGraphExecutableExecutionDescriptor* desc =
                [[MPSGraphExecutableExecutionDescriptor alloc] init];
            desc.waitUntilCompleted = NO;
            MPSCommandBuffer* mps_cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            (void)[exe->executable encodeToCommandBuffer:mps_cb
                                             inputsArray:feeds
                                            resultsArray:results
                                     executionDescriptor:desc];
            [mps_cb commit];
        } else {
            detail::run_or_throw(exe->executable, queue, feeds, results, "run_executable",
                                 signature, poll);
        }

        // Wrap each output MTLBuffer back into a GpuStorage-backed
        // TensorImpl.  ``__bridge_retained`` transfers the strong ref
        // into the MLX deleter.  Phase 1.6 dynamic-batch: use the
        // per-call ``realized_output_shapes`` (leading axis rewritten
        // to the current BS) instead of the trace-time shape — failing
        // to do so leaves the wrapped Tensor reporting the trace BS
        // while the MTLBuffer holds run-time-BS data, truncating the
        // user's view of the result.
        for (std::size_t j = 0; j < total_outs; ++j) {
            void* raw = (__bridge_retained void*)out_bufs[j];
            const Shape& wrap_shape = realized_output_shapes[j];
            std::vector<int> mlx_shape(wrap_shape.begin(), wrap_shape.end());
            ::mlx::core::array arr =
                lucid::gpu::mps::buffer_to_array(raw, std::move(mlx_shape), exe->output_dtypes[j]);
            // The storage carries its own dtype and byte count, separate
            // from the tensor's.  Left at their defaults (F32, 0) every
            // integer or bool output claimed to be float32 to any reader
            // that asks the storage — ``.numpy()`` returned an int64
            // ``x * 2 + 1`` as float32 bit patterns — and ``nbytes == 0``
            // made every compiled output look non-dense.
            GpuStorage gs;
            gs.dtype = exe->output_dtypes[j];
            gs.nbytes = detail::shape_nbytes(wrap_shape, exe->output_dtypes[j]);
            gs.arr = std::make_shared<::mlx::core::array>(std::move(arr));
            outputs.push_back(std::make_shared<TensorImpl>(
                Storage{std::move(gs)}, wrap_shape, exe->output_dtypes[j], Device::GPU, false));
        }
    }

    return outputs;
}

// Swap-buffer variant: instead of allocating fresh MTLBuffers per
// output and returning new TensorImpls, allocate one fresh
// MTLBuffer per output target and then *replace* the target's
// ``GpuStorage::arr`` pointer with a new MLX array wrapping the
// fresh buffer.  No ``mlx::copy`` runs — the target's old MLX
// array (and the buffer it owned) is released, and the new buffer
// takes over.  Conceptually equivalent to ``target.copy_(new_t)``
// but free of the per-output MLX evaluation sync.
//
// Why not bind the target's existing buffer as the output target?
// MPSGraph's executable doesn't guarantee read-before-write
// ordering when the same MTLBuffer appears in both the inputs and
// results arrays.  Optimizer-step graphs frequently have the
// momentum / m / v state on both sides of the dataflow (read old,
// write new) — aliasing the buffer corrupts the read.  The
// swap-buffer path sidesteps the hazard by always producing
// outputs in a fresh allocation, then replacing the target's
// pointer atomically once the executable has finished.
//
// Contract:
//   * Each target must be on Device::GPU with a GpuStorage.
//   * Shape + dtype must match ``exe->output_shapes[j]`` /
//     ``output_dtypes[j]`` exactly; we do NOT broadcast.  Every target
//     is checked before the run, so a mismatch leaves all of them as
//     they were.
//   * A target whose buffer is its elements has its ``GpuStorage::arr``
//     replaced by the fresh array; any external references to the
//     previous ``arr`` remain valid (MLX's own refcounting keeps the
//     old buffer alive until they drop).  A Metal view of part of a
//     tensor has the fresh values written into its part of the
//     family's root instead (``write_elements``).
//   * Each target's version moves so autograd's mutation tracker sees
//     the write.
LUCID_API std::vector<GpuStorage>
run_executable_inplace_outputs(CompiledExecutable* exe,
                               const std::vector<TensorImplPtr>& traced_feeds,
                               const std::vector<TensorImplPtr>& output_targets,
                               const FirstRunPoll& poll) {
    const std::vector<TensorImplPtr> input_feeds = redraw_rng_feeds(traced_feeds);
    if (exe == nullptr)
        throw std::invalid_argument("run_executable_inplace: null executable");
    if (input_feeds.size() != exe->input_ids.size())
        throw std::invalid_argument("run_executable_inplace: input count mismatch");
    const std::size_t total_outs = exe->output_ids.size() + exe->grad_output_ids.size();
    if (output_targets.size() != total_outs)
        throw std::invalid_argument(
            "run_executable_inplace: output_targets count mismatch (expected " +
            std::to_string(total_outs) + ", got " + std::to_string(output_targets.size()) + ")");
    if (exe->dynamic_batch)
        throw std::runtime_error("run_executable_inplace: dynamic-batch executables are not "
                                 "supported (output shapes resolved at run time, no stable "
                                 "target buffers).");

    id<MTLDevice> device = (__bridge id<MTLDevice>)lucid::gpu::mps::shared_mtl_device();
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)lucid::gpu::mps::shared_mtl_queue();

    // Pre-validate every target's shape/dtype — fails the whole call
    // early instead of mid-MPS-dispatch.
    for (std::size_t j = 0; j < total_outs; ++j) {
        const auto& target = output_targets[j];
        if (!target)
            throw std::invalid_argument("run_executable_inplace: null output target at slot " +
                                        std::to_string(j));
        if (target->device() != Device::GPU)
            throw std::invalid_argument(
                "run_executable_inplace: output target not on GPU at slot " + std::to_string(j));
        if (target->shape() != exe->output_shapes[j])
            throw std::invalid_argument("run_executable_inplace: output target shape mismatch at "
                                        "slot " +
                                        std::to_string(j));
        if (target->dtype() != exe->output_dtypes[j])
            throw std::invalid_argument("run_executable_inplace: output target dtype mismatch at "
                                        "slot " +
                                        std::to_string(j));
    }

    std::vector<id<MTLBuffer>> fresh_out_bufs;
    fresh_out_bufs.reserve(total_outs);
    std::vector<GpuStorage> fresh_outputs(total_outs);

    @autoreleasepool {
        NSMutableArray<MPSGraphTensorData*>* feeds =
            detail::build_feeds_array(exe, input_feeds, "run_executable_inplace");

        // ``LUCID_COMPILE_ALIAS_INPLACE=1``: diagnostic — bind the
        // target's existing MTLBuffer directly as the output result.
        // Re-validated 2026-05-25 on macOS 26 SDK: **read-write hazard
        // confirmed** — Adam-step trace produces zero loss / divergence
        // when input and output share a buffer.  Path is retained as
        // an opt-in diagnostic for future SDK versions but MUST stay
        // off by default.  The production fix for the per-step
        // ``newBufferWithLength`` cost is MPSGraph stateful variables
        // (``assignVariable``), not aliasing — variables move state
        // inside the executable and remove the param/state I/O
        // entirely.  See ``obsidian/engine/engine-compile-stateful-variables.md``.
        static const bool alias_requested = []() {
            const char* s = std::getenv("LUCID_COMPILE_ALIAS_INPLACE");
            return s && std::string(s) == "1";
        }();
        // Not on a first run: one given up on may finish after the caller has updated
        // the target some other way, and would then write over it.  Dynamic-batch
        // executables are refused above, so the signature is the static one.
        const std::uint64_t signature = FirstRunLedger::kStaticSignature;
        const bool alias_mode = alias_requested && detail::first_run_settled(exe, signature);

        // Destination ndarrays are built at the executable's NATURAL output
        // shape (``natural_result_shapes``, cached), not
        // the logical ``output_shapes``.  A keepdim loss reduce emits e.g.
        // ``(1,1)`` while the recorded loss shape is the squeezed ``(1,)``, and
        // a rank-5+ output is named at rank 4; writing either into a destination
        // of another rank forces an MPSNDArrayIdentity reshape that ASSERTS under
        // F16 (``New rank: 2 should match destination ... : 1``).  Same numel =>
        // same buffer bytes, so the post-run swap into the logical-shape
        // ``output_target`` stays valid.
        NSMutableArray<MPSGraphTensorData*>* results =
            [NSMutableArray arrayWithCapacity:total_outs];
        for (std::size_t j = 0; j < total_outs; ++j) {
            NSArray<NSNumber*>* ns_shape = exe->cached_output_ns_shapes[j];
            MPSDataType ns_dt = mps_dtype_or_throw(exe->output_dtypes[j]);

            id<MTLBuffer> out_buf = detail::alias_buffer(alias_mode, *output_targets[j]);
            if (out_buf != nil) {
                // The target's existing MTLBuffer is the result slot.  No
                // allocation, no swap.
                fresh_out_bufs.push_back(nil);  // sentinel: no swap needed
            } else {
                const std::size_t nbytes =
                    detail::shape_nbytes(exe->output_shapes[j], exe->output_dtypes[j]);
                out_buf = [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
                if (!out_buf)
                    throw std::runtime_error("run_executable_inplace: MTLBuffer allocation failed");
                fresh_out_bufs.push_back(out_buf);
            }

            MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:out_buf
                                                                             shape:ns_shape
                                                                          dataType:ns_dt];
            [results addObject:td];
        }

        // Synchronous ``runWithMTLCommandQueue:`` — the freshly-
        // allocated output MTLBuffer (esp. the 0-D loss scalar) is
        // wrapped as an MLX leaf array right after this call, and
        // ``loss.item()`` in user code reads its contents directly.
        // The async ``encodeToCommandBuffer:`` path (used in
        // ``run_executable`` for inference) breaks training because
        // MLX has no way to track the pending MPSGraph write on a
        // brand-new buffer; reads come back as zeros.  Forcing
        // ``waitUntilCompleted = YES`` here keeps the swap-buffer
        // dance correct.  Cost: ~30-100μs/step for the host sync;
        // still cheap relative to the executable's GPU work for
        // anything bigger than a toy MLP.
        detail::run_or_throw(exe->executable, queue, feeds, results, "run_executable_inplace",
                             signature, poll);

        // Wrap each fresh MTLBuffer as a leaf MLX array while still
        // inside the autoreleasepool.  The actual target storage swap +
        // version bump happens in :func:`swap_executable_inplace_targets`
        // (with the Python GIL re-acquired when called from Python so
        // concurrent readers of Metal view-family siblings never race
        // the root replacement or cache invalidation).
        for (std::size_t j = 0; j < total_outs; ++j) {
            id<MTLBuffer> buf = fresh_out_bufs[j];
            if (buf != nil) {
                void* raw = (__bridge_retained void*)buf;
                const Shape& out_shape = exe->output_shapes[j];
                const Dtype out_dtype = exe->output_dtypes[j];
                std::vector<int> mlx_shape(out_shape.begin(), out_shape.end());
                ::mlx::core::array fresh =
                    lucid::gpu::mps::buffer_to_array(raw, std::move(mlx_shape), out_dtype);
                fresh_outputs[j].dtype = out_dtype;
                fresh_outputs[j].nbytes = detail::shape_nbytes(out_shape, out_dtype);
                fresh_outputs[j].arr = std::make_shared<::mlx::core::array>(std::move(fresh));
            }
        }
    }

    return fresh_outputs;
}

LUCID_API void swap_executable_inplace_targets(const std::vector<TensorImplPtr>& output_targets,
                                               std::vector<GpuStorage>& fresh_outputs) {
    if (output_targets.size() != fresh_outputs.size())
        throw std::invalid_argument(
            "swap_executable_inplace_targets: target/output count mismatch");
    for (std::size_t j = 0; j < output_targets.size(); ++j) {
        TensorImplPtr target = output_targets[j];
        auto& fresh = fresh_outputs[j].arr;
        if (!fresh) {
            // Alias mode: the executable wrote the target's own buffer.
            target->bump_version();
            continue;
        }
        // Over a Metal shared buffer, the values go into the buffer, or another
        // tensor over it never sees them.
        if (target->shared_buffer() != nullptr) {
            target->copy_from(TensorImpl(Storage{std::move(fresh_outputs[j])}, target->shape(),
                                         target->dtype(), Device::GPU, false));
            continue;
        }
        // The fresh array is the target's elements, dense.  A target whose
        // buffer is its elements takes it as its array; a Metal view of part
        // of a tensor has it written into its part of the family's root, as
        // the eager optimizer step does (LCD-497).  A target whose elements
        // overlap is refused there, after the targets ahead of it took
        // theirs, as in the eager step.
        write_elements(target, [&](TensorImplPtr& elems) {
            std::get<GpuStorage>(elems->mutable_storage()).arr = std::move(fresh);
            elems->bump_version();
        });
    }
}

LUCID_API void run_executable_inplace(CompiledExecutable* exe,
                                      const std::vector<TensorImplPtr>& traced_feeds,
                                      const std::vector<TensorImplPtr>& output_targets,
                                      const FirstRunPoll& poll) {
    std::vector<GpuStorage> fresh_outputs =
        run_executable_inplace_outputs(exe, traced_feeds, output_targets, poll);
    swap_executable_inplace_targets(output_targets, fresh_outputs);
}

// ── Disk cache implementation ───────────────────────────────────────
//
// Binary metadata format (.meta sidecar), all little-endian:
//
//   byte  0:        magic = 'L', 'M', 'P', 'S' (4 bytes)
//   byte  4:        format version = 2 (1 byte; a version-1 file has no trailer)
//   byte  5:        device (1 byte; 0=CPU, 1=GPU)
//   byte  6:        dynamic_batch (1 byte; 0=false, 1=true)
//   byte  7:        padding (1 byte, zero)
//   then, in this order:
//     vec<int64> input_ids        (u32 count + count × i64)
//     vec<int64> output_ids
//     vec<int64> grad_output_ids
//     vec<vec<int64>> input_shapes   (u32 count + count × (u32 + count × i64))
//     vec<vec<int64>> output_shapes
//     vec<u8> input_dtypes           (u32 count + count × u8)
//     vec<u8> output_dtypes
//     vec<u64> static_feed_slots     (u32 count + count × u64)
//     vec<vec<int64>> feed_target_shapes   (version 2: the shape each feed placeholder
//                                           was declared at; an empty entry, or no
//                                           entries, means the slot's input_shape)
//
// Hand-rolled to avoid pulling in protobuf / json deps.  Bumping
// the version byte lets future Lucid releases reject incompatible
// caches without crashing.

namespace {
constexpr uint8_t kMetaFormatVersion = 2;
constexpr uint8_t kMetaFirstVersion = 1;
constexpr char kMetaMagic[4] = {'L', 'M', 'P', 'S'};

template <typename T>
inline void write_pod(std::vector<uint8_t>& out, T v) {
    const uint8_t* p = reinterpret_cast<const uint8_t*>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}

template <typename T>
inline void write_vec(std::vector<uint8_t>& out, const std::vector<T>& v) {
    write_pod<uint32_t>(out, static_cast<uint32_t>(v.size()));
    for (const T& x : v)
        write_pod<T>(out, x);
}

inline void write_vec_vec_i64(std::vector<uint8_t>& out, const std::vector<Shape>& vv) {
    write_pod<uint32_t>(out, static_cast<uint32_t>(vv.size()));
    for (const auto& v : vv) {
        write_pod<uint32_t>(out, static_cast<uint32_t>(v.size()));
        for (std::int64_t x : v)
            write_pod<std::int64_t>(out, x);
    }
}

template <typename T>
inline bool read_pod(const uint8_t*& p, const uint8_t* end, T& out) {
    if (end - p < (ptrdiff_t)sizeof(T))
        return false;
    std::memcpy(&out, p, sizeof(T));
    p += sizeof(T);
    return true;
}

template <typename T>
inline bool read_vec(const uint8_t*& p, const uint8_t* end, std::vector<T>& v) {
    uint32_t n;
    if (!read_pod(p, end, n))
        return false;
    v.resize(n);
    for (uint32_t i = 0; i < n; ++i)
        if (!read_pod(p, end, v[i]))
            return false;
    return true;
}

inline bool read_vec_vec_i64(const uint8_t*& p, const uint8_t* end, std::vector<Shape>& vv) {
    uint32_t n;
    if (!read_pod(p, end, n))
        return false;
    vv.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t m;
        if (!read_pod(p, end, m))
            return false;
        vv[i].resize(m);
        for (uint32_t j = 0; j < m; ++j)
            if (!read_pod(p, end, vv[i][j]))
                return false;
    }
    return true;
}

// The version-2 trailer: the shape each feed placeholder was declared at.  A version-1
// file has none (every slot is bound at its input shape).  The error text, or nullptr.
inline const char* read_feed_target_shapes(uint8_t version,
                                           const uint8_t*& p,
                                           const uint8_t* end,
                                           std::size_t n_inputs,
                                           std::vector<Shape>& out) {
    if (version < 2)
        return nullptr;
    if (!read_vec_vec_i64(p, end, out))
        return "truncated feed_target_shapes";
    if (!out.empty() && out.size() != n_inputs)
        return "load_executable: feed_target_shapes does not match the inputs";
    return nullptr;
}

// Whether every feed of ``exe`` is bound at the shape and dtype of the first: the one case a
// reloaded package keeps its feed order, because there is no order to lose.
inline bool feeds_share_shape_and_dtype(const CompiledExecutable& exe) {
    for (std::size_t i = 1; i < exe.input_ids.size(); ++i) {
        if (detail::bound_input_shape(&exe, i) != detail::bound_input_shape(&exe, 0) ||
            exe.input_dtypes[i] != exe.input_dtypes[0])
            return false;
    }
    return true;
}

inline bool atomic_write(const std::string& path, const std::vector<uint8_t>& data) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f)
            return false;
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
        if (!f)
            return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

inline bool read_all(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f)
        return false;
    std::streamsize size = f.tellg();
    if (size < 0)
        return false;
    f.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));
    f.read(reinterpret_cast<char*>(out.data()), size);
    return static_cast<std::streamsize>(out.size()) == size;
}
}  // namespace

bool save_executable(const CompiledExecutable* exe, const std::string& path) {
    if (exe == nullptr || exe->executable == nil)
        return false;

    @autoreleasepool {
        const std::string pkg_path = path + ".mpsgraphpackage";
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pkg_path.c_str()]];
        // Apple's ``serializeToMPSGraphPackageAtURL:descriptor:``
        // returns ``void`` and offers no error signal (macOS 14+,
        // SDK header confirms).  Wrap in ``@try`` so a fatal ObjC
        // exception (e.g. read-only filesystem) doesn't propagate
        // into C++ unwinding.  If the file fails to materialise,
        // the meta write below or the subsequent ``load_executable``
        // will catch it as a cache miss.
        @try {
            [exe->executable serializeToMPSGraphPackageAtURL:url descriptor:nil];
        } @catch (NSException* e) {
            (void)e;
            return false;
        }

        // Pack metadata.
        std::vector<uint8_t> meta;
        meta.reserve(1024);
        meta.insert(meta.end(), kMetaMagic, kMetaMagic + 4);
        write_pod<uint8_t>(meta, kMetaFormatVersion);
        write_pod<uint8_t>(meta, static_cast<uint8_t>(exe->device));
        write_pod<uint8_t>(meta, exe->dynamic_batch ? 1 : 0);
        write_pod<uint8_t>(meta, 0);  // padding
        // input_ids/output_ids/grad_output_ids are vector<TraceId>;
        // serialise the underlying int64 values.
        auto to_int64 = [](const std::vector<TensorId>& v) {
            std::vector<std::int64_t> out;
            out.reserve(v.size());
            for (const auto& id : v)
                out.push_back(id.v);
            return out;
        };
        write_vec<std::int64_t>(meta, to_int64(exe->input_ids));
        write_vec<std::int64_t>(meta, to_int64(exe->output_ids));
        write_vec<std::int64_t>(meta, to_int64(exe->grad_output_ids));
        write_vec_vec_i64(meta, exe->input_shapes);
        write_vec_vec_i64(meta, exe->output_shapes);
        // Dtypes encoded as uint8 (matches the Dtype enum's
        // underlying width — all entries fit).
        std::vector<uint8_t> in_dt(exe->input_dtypes.size());
        for (std::size_t i = 0; i < in_dt.size(); ++i)
            in_dt[i] = static_cast<uint8_t>(exe->input_dtypes[i]);
        write_vec<uint8_t>(meta, in_dt);
        std::vector<uint8_t> out_dt(exe->output_dtypes.size());
        for (std::size_t i = 0; i < out_dt.size(); ++i)
            out_dt[i] = static_cast<uint8_t>(exe->output_dtypes[i]);
        write_vec<uint8_t>(meta, out_dt);
        std::vector<uint64_t> static_slots(exe->static_feed_slots.begin(),
                                           exe->static_feed_slots.end());
        write_vec<uint64_t>(meta, static_slots);
        write_vec_vec_i64(meta, exe->feed_target_shapes);

        const std::string meta_path = path + ".meta";
        return atomic_write(meta_path, meta);
    }
}

CompiledExecutable* load_executable(const std::string& path, std::string* error_msg) {
    auto fail = [&](std::string msg) -> CompiledExecutable* {
        if (error_msg)
            *error_msg = std::move(msg);
        return nullptr;
    };

    const std::string meta_path = path + ".meta";
    std::vector<uint8_t> meta;
    if (!read_all(meta_path, meta))
        return fail("load_executable: missing or unreadable .meta sidecar");

    const uint8_t* p = meta.data();
    const uint8_t* end = p + meta.size();

    // Magic + version + flags
    if ((end - p) < 8)
        return fail("load_executable: truncated header");
    if (std::memcmp(p, kMetaMagic, 4) != 0)
        return fail("load_executable: bad magic — not a Lucid meta file");
    p += 4;
    uint8_t version, device_byte, dynamic_byte, pad;
    if (!read_pod(p, end, version))
        return fail("truncated version");
    if (version < kMetaFirstVersion || version > kMetaFormatVersion)
        return fail("load_executable: meta format version mismatch");
    if (!read_pod(p, end, device_byte))
        return fail("truncated device");
    if (!read_pod(p, end, dynamic_byte))
        return fail("truncated dynamic flag");
    if (!read_pod(p, end, pad))
        return fail("truncated padding");

    std::vector<std::int64_t> input_ids, output_ids, grad_output_ids;
    std::vector<Shape> input_shapes, output_shapes;
    std::vector<uint8_t> input_dt_raw, output_dt_raw;
    std::vector<uint64_t> static_slots;
    std::vector<Shape> feed_target_shapes;

    if (!read_vec(p, end, input_ids))
        return fail("truncated input_ids");
    if (!read_vec(p, end, output_ids))
        return fail("truncated output_ids");
    if (!read_vec(p, end, grad_output_ids))
        return fail("truncated grad_output_ids");
    if (!read_vec_vec_i64(p, end, input_shapes))
        return fail("truncated input_shapes");
    if (!read_vec_vec_i64(p, end, output_shapes))
        return fail("truncated output_shapes");
    if (!read_vec(p, end, input_dt_raw))
        return fail("truncated input_dtypes");
    if (!read_vec(p, end, output_dt_raw))
        return fail("truncated output_dtypes");
    if (!read_vec(p, end, static_slots))
        return fail("truncated static_feed_slots");
    if (const char* bad =
            read_feed_target_shapes(version, p, end, input_shapes.size(), feed_target_shapes);
        bad != nullptr)
        return fail(bad);

    @autoreleasepool {
        const std::string pkg_path = path + ".mpsgraphpackage";
        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:pkg_path.c_str()]];
        // A half-precision executable keeps to the GPU when it is loaded, as
        // when it was built (make_compile_descriptor): the default would place
        // it across devices again, the Neural Engine included, where float16
        // graphs come back wrong on M4-class hardware.
        const auto is_half = [](std::uint8_t raw) {
            const auto dt = static_cast<Dtype>(raw);
            return dt == Dtype::F16 || dt == Dtype::BF16;
        };
        const bool half = std::any_of(input_dt_raw.begin(), input_dt_raw.end(), is_half) ||
                          std::any_of(output_dt_raw.begin(), output_dt_raw.end(), is_half);
        // Always a descriptor, never nil: macOS 15's MPSGraph asserts on a nil
        // one here (``-[MPSGraphExecutable commonPreInitWithDescriptor:]``)
        // and takes the process down.  Later releases read nil as the
        // defaults, which a fresh descriptor also holds.
        MPSGraphCompilationDescriptor* cdesc = [[MPSGraphCompilationDescriptor alloc] init];
        if (::lucid::Determinism::is_enabled() || half)
            cdesc.optimizationLevel = MPSGraphOptimizationLevel0;
        MPSGraphExecutable* exec = [[MPSGraphExecutable alloc] initWithMPSGraphPackageAtURL:url
                                                                      compilationDescriptor:cdesc];
        if (exec == nil)
            return fail("load_executable: MPSGraphExecutable load returned nil");

        auto* result = new CompiledExecutable();
        result->executable = exec;
        result->device = static_cast<Device>(device_byte);
        result->dynamic_batch = (dynamic_byte != 0);
        // input_ids et al. are vector<int64_t> from the on-disk file;
        // wrap each into the strong-typed TraceId.
        auto wrap_ids = [](std::vector<std::int64_t>&& v) {
            std::vector<TensorId> out;
            out.reserve(v.size());
            for (std::int64_t x : v)
                out.push_back(TensorId{x});
            return out;
        };
        result->input_ids = wrap_ids(std::move(input_ids));
        result->output_ids = wrap_ids(std::move(output_ids));
        result->grad_output_ids = wrap_ids(std::move(grad_output_ids));
        result->input_shapes = std::move(input_shapes);
        result->feed_target_shapes = std::move(feed_target_shapes);
        result->output_shapes = std::move(output_shapes);
        result->input_dtypes.resize(input_dt_raw.size());
        for (std::size_t i = 0; i < input_dt_raw.size(); ++i)
            result->input_dtypes[i] = static_cast<Dtype>(input_dt_raw[i]);
        result->output_dtypes.resize(output_dt_raw.size());
        for (std::size_t i = 0; i < output_dt_raw.size(); ++i)
            result->output_dtypes[i] = static_cast<Dtype>(output_dt_raw[i]);
        for (uint64_t s : static_slots)
            result->static_feed_slots.insert(static_cast<std::size_t>(s));
        // Feed order after a package round-trip.
        //
        // ``executable.feedTensors`` is nil on a package-loaded
        // executable (it is populated only for one compiled in this
        // process), so the name-based recovery below yields nothing and
        // there is no way to *ask* the artifact what order it wants.
        // Measured behaviour: the order survives intact as long as every
        // feed has the same shape and dtype, and is silently rearranged
        // once they differ — at which point ``run_executable`` labels a
        // buffer with another slot's shape and MetalPerformanceShaders
        // aborts the process.
        //
        // Reproducing whatever ordering the package applies would mean
        // guessing at an undocumented Apple internal, so this refuses the
        // artifacts it cannot bind safely instead.  A refusal is a cache
        // miss for the disk cache (recompile, correct but slower) and a
        // clear error for ``load_compiled`` — never a crash.
        result->feed_order = detail::derive_feed_order(exec, result->input_ids);
        if (result->feed_order.empty() && result->input_ids.size() > 1) {
            if (!feeds_share_shape_and_dtype(*result)) {
                delete result;
                return fail("load_executable: this artifact has feeds of differing "
                            "shape/dtype, and a reloaded MPSGraph package does not "
                            "expose the feed order it expects — binding it would "
                            "abort inside MetalPerformanceShaders.  Recompile from "
                            "the source trace instead of loading this artifact.");
            }
        }
        return result;
    }
}

}  // namespace lucid::compile
