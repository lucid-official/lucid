// lucid/_C/compile/RngFeeds.h
//
// Random draws inside a traced function, drawn again on every run.
//
// A compiled executable is a fixed graph.  An RNG op lowered into it with
// a seed draws the same values on every call: a diffusion model's noise
// and timesteps, a stochastic-depth mask, a VAE's reparameterisation
// sample — all frozen at their trace-time values, and training on them
// raises no error.  Instead the tracer turns each draw from the default
// generator into an ordinary feed and remembers how it was made, and
// :func:`run_executable` makes it again, through the same eager op, before
// binding the feeds.  Drawn in trace order, the values are exactly the
// ones eager would have produced from the same generator state, and the
// generator advances as it would have.
//
// The recipe lives beside the traced tensor, not in the executable: the
// executable only sees a placeholder, and one executable is shared by
// every trace with the same structure — a constant feed of the same shape
// must not start drawing.  The tensor is the pinned feed every entry
// point hands back to :func:`run_executable`, so each of them re-draws
// without knowing about it.

#pragma once

#include <cstdint>
#include <vector>

#include "../api.h"
#include "../core/Device.h"
#include "../core/Dtype.h"
#include "../core/Shape.h"
#include "../core/fwd.h"

namespace lucid::compile {

// How a traced draw was made.  ``a`` / ``b`` are low / high (uniform),
// mean / std (normal), p (bernoulli) or keep probability / scale (a
// dropout mask, drawn as dropout draws it and scaled by 1/keep);
// ``lo`` / ``hi`` bound randint.
struct RngRecipe {
    enum class Kind : std::uint8_t { Uniform, Normal, Randint, Bernoulli, DropoutMask };
    Kind kind = Kind::Uniform;
    double a = 0.0;

    double b = 0.0;
    std::int64_t lo = 0;
    std::int64_t hi = 0;
    Shape shape;
    Dtype dtype = Dtype::F32;
    Device device = Device::CPU;
};

// Remember that ``out`` — a tensor drawn while a tracer was recording —
// is to be drawn again from ``recipe`` whenever it is fed.
LUCID_API void note_rng_feed(const TensorImplPtr& out, RngRecipe recipe);

// ``feeds`` with every noted draw replaced by a fresh one, drawn in the
// order the trace drew them.  Returns ``feeds`` itself when none is.
LUCID_API std::vector<TensorImplPtr> redraw_rng_feeds(const std::vector<TensorImplPtr>& feeds);

}  // namespace lucid::compile
