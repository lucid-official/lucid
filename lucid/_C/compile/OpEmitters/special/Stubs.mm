// lucid/_C/compile/OpEmitters/misc/Stubs.mm
//
// Eager-fallback markers — :class:`StubEmitter` + the registration
// list.  Each registered name resolves through :func:`find_emitter`
// and returns nullptr from ``emit()``, which signals the
// :class:`MpsBuilder` to abort the compile (clean fallback to eager
// dispatch).
//
// The categorisation comments group the stubs by *why* they can't be
// real-emit — see ``obsidian/`` notes for the long-form rationale.
//
//   - linalg long tail       : need iterative algorithms (Jacobi /
//                              Householder / Gram-Schmidt) MPSGraph
//                              doesn't expose
//   - multi-output / dynamic : :class:`OpNode` IR currently models
//                              one output per node
//   - FFT                    : MPSGraph *does* expose FFT
//                              (``fastFourierTransformWithTensor:`` and
//                              the Hermitean pair).  What is missing is
//                              a complex dtype on this path at all:
//                              ``MpsBuilder``'s ``to_mps_dtype`` has no
//                              C64 / C128 case and throws
//   - histogram              : no native histogram primitive
//   - embedding_bag          : dynamic per-bag offsets need a
//                              segment-reduce
//   - complex                : no complex dtype on the compile path —
//                              complex is one interleaved storage, not
//                              two, so the blocker is the missing
//                              ``to_mps_dtype`` case rather than the
//                              backing layout
//   - factory header rows    : ``arange`` / ``eye`` / etc. — the
//                              builder already factory-skips empty
//                              inputs, but registering keeps
//                              :func:`find_emitter` non-null for
//                              every traceable op name

#include <memory>
#include <string>
#include <string_view>

#include "../OpEmitter.h"

namespace lucid::compile {

namespace {

class StubEmitter final : public OpEmitter {
public:
    explicit StubEmitter(std::string name) : name_(std::move(name)) {}
    std::string_view op_name() const override { return name_; }
    bool emit(BuilderContext&, const OpNode&) override { return false; }

private:
    std::string name_;
};

struct StubsRegistrar {
    StubsRegistrar() {
        for (
            const char* name : {
                // ── linalg long tail (iterative algorithms) ─────────
                // ``cholesky`` / ``qr`` / ``eigh`` / ``svd`` / ``pinv`` /
                // ``solve_triangular`` moved to real-emit
                // (OpEmitters/linalg/Decompositions.mm).
                "eig",
                "solve",
                // ``erfinv`` moved to real-emit (OpEmitters/elementwise/Math.mm)
                // via Giles' polynomial approximation.
                // ── multi-output / dynamic shape ────────────────────
                // ``split`` / ``split_at`` / ``topk`` moved to real-emit
                // (OpEmitters/shape/Split.mm) via manual multi-output
                // bind through BuilderContext::bind.  ``nonzero`` and
                // ``unique`` remain stubbed (dynamic output shape).
                "nonzero",
                "unique",
                // ── FFT (no MPSGraph primitive) ─────────────────────
                "fftn",
                "ifftn",
                "rfftn",
                "irfftn",
                // ── histogram (no MPSGraph primitive) ───────────────
                "histogram",
                "histogram2d",
                "histogramdd",
                // ``max_pool3d`` / ``avg_pool3d`` moved to real-emit
                // (OpEmitters/nn/Pool.mm): the SDK's 4-D pooling takes
                // the rank-5 volume once a length-1 spatial axis is
                // reshaped in front of it.  ``conv_transpose3d`` moved
                // to real-emit (OpEmitters/nn/Conv.mm): the SDK ships
                // no ``convolutionTranspose3D``, but a transposed
                // convolution is the data gradient of a convolution,
                // and ``convolution3DDataGradient`` is exposed.  The
                // 3-D resamplers moved too (OpEmitters/nn/Spatial.mm):
                // resampling is separable, so depth rides along as
                // channels through the 2-D ``resizeTensor`` and is then
                // its own gather — the reason filed here said ``resize``
                // would blend across the folded channels, which a 2-D
                // resize does not do.
                // ── spatial sampling / segment-reduce (dyn. gather) ─
                // ``rotate`` is emittable — nearest sampling on a grid
                // the angle and centre fix at build time — but nothing
                // in ``lucid/`` calls ``engine.nn.rotate``, so it stays
                // here rather than becoming an emitter no trace reaches.
                "rotate",
                "embedding_bag",
                // ── host-precomputed factory headers ────────────────
                // (already factory-skipped by the builder when inputs
                // are empty; stubbed for find_emitter completeness).
                // ``zeros`` / ``ones`` / ``arange`` / ``linspace`` / ``meshgrid`` / ``eye`` moved to real-emit
                // (OpEmitters/special/Factory.mm) — ``zeros`` / ``ones``
                // appear mid-trace in RNN-style code (default ``hx =
                // zeros``); ``arange`` backs strided slicing (``x[..., ::2]``
                // lowers to gather over an arange index), so baking it as a
                // constant unblocks e.g. RoFormer's interleaved RoPE.
                "logspace",
                "empty",
                // ``rand`` / ``uniform`` / ``randn`` / ``normal`` /
                // ``randint`` / ``bernoulli`` moved to real-emit in
                // OpEmitters/special/Random.mm (deterministic-per-
                // executable via the stateless MPSGraph RNG path).
            }) {
            register_emitter(std::make_unique<StubEmitter>(std::string(name)));
        }
    }
};

[[maybe_unused]] static const StubsRegistrar g_stubs_registrar;

}  // namespace

}  // namespace lucid::compile
