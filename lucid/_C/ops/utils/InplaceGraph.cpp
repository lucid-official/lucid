// lucid/_C/ops/utils/InplaceGraph.cpp
//
// Re-deriving a view family's graph after a write through one member —
// the half of InplaceGraph.h that needs a backward node of its own.

#include "InplaceGraph.h"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "../../autograd/Helpers.h"
#include "../../autograd/Node.h"
#include "../../core/Allocator.h"
#include "../../core/Error.h"
#include "../../core/Shape.h"
#include "../../kernel/BinaryKernel.h"
#include "../complex/Complex.h"
#include "../complex/Real.h"
#include "../composite/Indexing.h"
#include "../gfunc/Gfunc.h"
#include "../ufunc/Astype.h"
#include "Concat.h"
#include "Select.h"

namespace lucid::inplace {

namespace {

Dtype lane_dtype(Dtype dt) {
    return is_complex(dt) ? real_lane_of(dt) : dt;
}

Shape lane_shape(const Shape& shape, Dtype dt) {
    if (!is_complex(dt))
        return shape;
    Shape s = shape;
    s.push_back(2);
    return s;
}

Stride lane_stride(const Stride& stride, Dtype dt) {
    if (!is_complex(dt))
        return stride;
    Stride s = stride;
    s.push_back(static_cast<std::int64_t>(dtype_size(real_lane_of(dt))));
    return s;
}

TensorImplPtr to_lane_tensor(const TensorImplPtr& t, bool is_cplx) {
    return is_cplx ? view_as_real_op(t) : t;
}

TensorImplPtr from_lane_tensor(const TensorImplPtr& t, bool is_cplx) {
    return (t && is_cplx) ? view_as_complex_op(t) : t;
}

Storage to_lane_storage(Storage s, const Shape& lane_shp, Dtype lane_dt, Device dev, bool is_cplx) {
    if (!is_cplx)
        return s;
    NoGradGuard no_grad;
    Shape cplx_shp(lane_shp.begin(), lane_shp.end() - 1);
    auto impl = std::make_shared<TensorImpl>(std::move(s), std::move(cplx_shp),
                                             complex_for(lane_dt), dev, false);
    return view_as_real_op(impl)->storage();
}

Storage
from_lane_storage(Storage s, const Shape& lane_shp, Dtype lane_dt, Device dev, bool is_cplx) {
    if (!is_cplx)
        return s;
    NoGradGuard no_grad;
    auto impl = std::make_shared<TensorImpl>(std::move(s), lane_shp, lane_dt, dev, false);
    return view_as_complex_op(impl)->storage();
}

// A CPU buffer of ``nbytes`` zero bytes.
CpuStorage zeroed_cpu(std::size_t nbytes, Dtype dtype) {
    CpuStorage s;
    s.ptr = allocate_aligned_bytes(nbytes);
    s.nbytes = nbytes;
    s.dtype = dtype;
    if (nbytes > 0)
        std::memset(s.ptr.get(), 0, nbytes);
    return s;
}

// A 1-D int64 tensor holding ``values`` on ``device``.
TensorImplPtr index_tensor(const std::vector<std::int64_t>& values, Device device) {
    const Shape shape{static_cast<std::int64_t>(values.size())};
    CpuStorage s = zeroed_cpu(values.size() * sizeof(std::int64_t), Dtype::I64);
    if (!values.empty())
        std::memcpy(s.ptr.get(), values.data(), s.nbytes);
    Storage st = device == Device::CPU
                     ? Storage{std::move(s)}
                     : backend::Dispatcher::for_device(device).from_cpu(std::move(s), shape);
    return std::make_shared<TensorImpl>(std::move(st), shape, Dtype::I64, device, false);
}

// ``out[src[k]] += g[dst[k]]`` for one floating element type.
template <typename T>
void accumulate(const std::byte* g,
                std::byte* out,
                const std::vector<std::int64_t>& dst,
                const std::vector<std::int64_t>& src) {
    const auto* gp = reinterpret_cast<const T*>(g);
    auto* op = reinterpret_cast<T*>(out);
    for (std::size_t k = 0; k < dst.size(); ++k)
        op[src[k]] += gp[dst[k]];
}

// The gradient of a tensor after a write into some of its elements.
//
// The forward this stands for copies elements of ``src`` over elements of
// ``base``, both counted in row-major order:
//
//     out = base
//     out.flat[dst_begin : dst_begin + len] = src.flat[src_begin : src_begin + len]   (runs)
//     out.flat[dst_idx[k]] = src.flat[src_idx[k]]                                     (positions)
//
// the reference's CopySlices.  Two contiguous members meet in a run; a
// strided one — a transpose, a column — meets another at scattered
// positions, which the second form spells out.  The gradient of a copy does
// not depend on what was copied, so nothing is saved but where the elements
// sit — no tensor, and so nothing that holds the buffer and would stop the
// next write to it.  Either input may be absent: ``base`` when the write
// covered the member, ``src`` when what was written is a constant.  Its
// edge is then empty and it is given no gradient.
class CopySlicesBackward : public Node {
public:
    CopySlicesBackward(Shape out_shape,
                       Shape src_shape,
                       std::int64_t dst_begin,
                       std::int64_t src_begin,
                       std::int64_t len,
                       Dtype out_dtype,
                       Dtype src_dtype,
                       Device device)
        : out_shape_(lane_shape(out_shape, out_dtype)),
          src_shape_(lane_shape(src_shape, src_dtype)),
          dst_begin_(dst_begin),
          src_begin_(src_begin),
          len_(len),
          dtype_(lane_dtype(src_dtype)),
          out_complex_(is_complex(out_dtype)),
          src_complex_(is_complex(src_dtype)),
          device_(device) {}

    CopySlicesBackward(Shape out_shape,
                       Shape src_shape,
                       std::vector<std::int64_t> dst_idx,
                       std::vector<std::int64_t> src_idx,
                       Dtype out_dtype,
                       Dtype src_dtype,
                       Device device)
        : out_shape_(lane_shape(out_shape, out_dtype)),
          src_shape_(lane_shape(src_shape, src_dtype)),
          dtype_(lane_dtype(src_dtype)),
          out_complex_(is_complex(out_dtype)),
          src_complex_(is_complex(src_dtype)),
          device_(device),
          indexed_(true),
          dst_idx_(std::move(dst_idx)),
          src_idx_(std::move(src_idx)) {
        // A member that repeats an element (an expanded view) reads one
        // source element at several positions; their gradients add.
        std::vector<std::int64_t> sorted(src_idx_);
        std::sort(sorted.begin(), sorted.end());
        injective_ = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
    }

    // ``base`` gets the output's gradient with the written elements zeroed —
    // they no longer depend on it — and ``src`` gets theirs, placed where
    // they were read from.  A live edge always gets a buffer: the engine
    // hands an empty one to the next node as it is.
    std::vector<Storage> apply(Storage grad_out) override {
        grad_out = to_lane_storage(std::move(grad_out), out_shape_, dtype_, device_, out_complex_);
        std::vector<Storage> grads = indexed_ ? apply_indexed(grad_out) : apply_run(grad_out);
        if (next_edges_[0].node)
            grads[0] =
                from_lane_storage(std::move(grads[0]), out_shape_, dtype_, device_, out_complex_);
        if (next_edges_[1].node)
            grads[1] =
                from_lane_storage(std::move(grads[1]), src_shape_, dtype_, device_, src_complex_);
        return grads;
    }

    // The same two gradients, built from ops so they carry a graph of their
    // own.
    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr& grad_out) override {
        const TensorImplPtr g = to_lane_tensor(grad_out, out_complex_);
        std::vector<TensorImplPtr> grads = indexed_ ? graph_indexed(g) : graph_run(g);
        grads[0] = from_lane_tensor(grads[0], out_complex_);
        grads[1] = from_lane_tensor(grads[1], src_complex_);
        return grads;
    }

    std::string node_name() const override { return "CopySlices"; }

private:
    std::vector<Storage> apply_run(const Storage& grad_out) const {
        auto& be = backend::Dispatcher::for_device(device_);
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const Shape flat{n};
        const Storage flat_grad =
            out_shape_ == flat ? grad_out : be.reshape(grad_out, out_shape_, flat, dtype_);
        std::vector<Storage> grads(2);
        if (next_edges_[0].node) {
            std::vector<Storage> pieces;
            std::vector<Shape> shapes;
            const std::int64_t tail = n - dst_begin_ - len_;
            if (dst_begin_ > 0) {
                pieces.push_back(be.slice_axis(flat_grad, flat, Shape{dst_begin_}, 0, 0, dtype_));
                shapes.push_back(Shape{dst_begin_});
            }
            pieces.push_back(make_zero_storage(Shape{len_}, dtype_, device_));
            shapes.push_back(Shape{len_});
            if (tail > 0) {
                pieces.push_back(
                    be.slice_axis(flat_grad, flat, Shape{tail}, 0, dst_begin_ + len_, dtype_));
                shapes.push_back(Shape{tail});
            }
            Storage flat_g0 = pieces.size() == 1 ? std::move(pieces[0])
                                                 : be.concatenate(pieces, shapes, 0, dtype_);
            grads[0] = out_shape_ == flat ? std::move(flat_g0)
                                          : be.reshape(flat_g0, flat, out_shape_, dtype_);
        }
        if (next_edges_[1].node) {
            const auto src_n = static_cast<std::int64_t>(shape_numel(src_shape_));
            const Shape src_flat{src_n};
            Storage run = len_ == n
                              ? flat_grad
                              : be.slice_axis(flat_grad, flat, Shape{len_}, 0, dst_begin_, dtype_);
            Storage flat_g1 = len_ == src_n ? std::move(run)
                                            : be.insert_axis_slice(run, Shape{len_}, src_flat, 0,
                                                                   src_begin_, dtype_);
            grads[1] = src_shape_ == src_flat ? std::move(flat_g1)
                                              : be.reshape(flat_g1, src_flat, src_shape_, dtype_);
        }
        return grads;
    }

    std::vector<TensorImplPtr> graph_run(const TensorImplPtr& grad_out) const {
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const auto flat = reshape_op(grad_out, {n});
        const auto zeros = [&](std::int64_t count) {
            return zeros_op(Shape{count}, grad_out->dtype(), grad_out->device());
        };
        const auto join = [](const std::vector<TensorImplPtr>& pieces) {
            return pieces.size() == 1 ? pieces[0] : concatenate_op(pieces, 0);
        };
        std::vector<TensorImplPtr> grads(2);
        if (next_edges_[0].node) {
            std::vector<TensorImplPtr> pieces;
            const std::int64_t tail = n - dst_begin_ - len_;
            if (dst_begin_ > 0)
                pieces.push_back(narrow_op(flat, 0, 0, dst_begin_));
            pieces.push_back(zeros(len_));
            if (tail > 0)
                pieces.push_back(narrow_op(flat, 0, dst_begin_ + len_, tail));
            grads[0] = reshape_op(join(pieces), out_shape_);
        }
        if (next_edges_[1].node) {
            std::vector<TensorImplPtr> pieces;
            const std::int64_t after =
                static_cast<std::int64_t>(shape_numel(src_shape_)) - src_begin_ - len_;
            if (src_begin_ > 0)
                pieces.push_back(zeros(src_begin_));
            pieces.push_back(narrow_op(flat, 0, dst_begin_, len_));
            if (after > 0)
                pieces.push_back(zeros(after));
            grads[1] = reshape_op(join(pieces), src_shape_);
        }
        return grads;
    }
    std::vector<Storage> apply_indexed(const Storage& grad_out) const {
        const auto* g = std::get_if<CpuStorage>(&grad_out);
        if (!g ||
            (!injective_ && dtype_ != Dtype::F32 && dtype_ != Dtype::F64 && dtype_ != Dtype::C64)) {
            NoGradGuard no_grad;
            const auto out = graph_indexed(
                std::make_shared<TensorImpl>(grad_out, out_shape_, dtype_, device_, false));
            std::vector<Storage> grads(2);
            if (out[0])
                grads[0] = out[0]->storage();
            if (out[1])
                grads[1] = out[1]->storage();
            return grads;
        }
        const std::size_t item = dtype_size(dtype_);
        std::vector<Storage> grads(2);
        if (next_edges_[0].node) {
            CpuStorage base = zeroed_cpu(shape_numel(out_shape_) * item, dtype_);
            if (base.nbytes > 0)
                std::memcpy(base.ptr.get(), g->ptr.get(), base.nbytes);
            for (const auto j : dst_idx_)
                std::memset(base.ptr.get() + j * item, 0, item);
            grads[0] = Storage{std::move(base)};
        }
        if (next_edges_[1].node) {
            grads[1] = Storage{scatter_src_grad(*g, item)};
        }
        return grads;
    }

    CpuStorage scatter_src_grad(const CpuStorage& g, std::size_t item) const {
        CpuStorage src = zeroed_cpu(shape_numel(src_shape_) * item, dtype_);
        if (injective_) {
            for (std::size_t k = 0; k < dst_idx_.size(); ++k)
                std::memcpy(src.ptr.get() + src_idx_[k] * item, g.ptr.get() + dst_idx_[k] * item,
                            item);
        } else if (dtype_ == Dtype::F32) {
            accumulate<float>(g.ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
        } else if (dtype_ == Dtype::F64) {
            accumulate<double>(g.ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
        } else {
            accumulate<std::complex<float>>(g.ptr.get(), src.ptr.get(), dst_idx_, src_idx_);
        }
        return src;
    }

    std::vector<TensorImplPtr> graph_indexed(const TensorImplPtr& grad_out) const {
        const auto n = static_cast<std::int64_t>(shape_numel(out_shape_));
        const auto flat = reshape_op(grad_out, {n});
        const auto dst = index_tensor(dst_idx_, grad_out->device());
        const auto k = static_cast<std::int64_t>(dst_idx_.size());
        std::vector<TensorImplPtr> grads(2);
        if (next_edges_[0].node)
            grads[0] = reshape_op(
                scatter_op(flat, 0, dst, zeros_op(Shape{k}, grad_out->dtype(), grad_out->device())),
                out_shape_);
        if (next_edges_[1].node) {
            const auto src_n = static_cast<std::int64_t>(shape_numel(src_shape_));
            const auto picked = gather_op(flat, dst, 0);
            const auto zeros = zeros_op(Shape{src_n}, grad_out->dtype(), grad_out->device());
            grads[1] = reshape_op(
                scatter_add_op(zeros, index_tensor(src_idx_, grad_out->device()), picked, 0),
                src_shape_);
        }
        return grads;
    }

    Shape out_shape_;
    Shape src_shape_;
    std::int64_t dst_begin_ = 0;
    std::int64_t src_begin_ = 0;
    std::int64_t len_ = 0;
    Dtype dtype_;
    bool out_complex_ = false;
    bool src_complex_ = false;
    Device device_;
    bool indexed_ = false;
    bool injective_ = true;
    std::vector<std::int64_t> dst_idx_;
    std::vector<std::int64_t> src_idx_;
};

class InvalidatedCustomViewNode : public Node {
public:
    void validate_versions() override { fail(); }

    std::vector<Storage> apply(Storage) override { fail(); }

    std::vector<TensorImplPtr> apply_for_graph(const TensorImplPtr&) override { fail(); }

    std::string node_name() const override { return "InvalidatedCustomView"; }

private:
    [[noreturn]] static void fail() {
        ErrorBuilder("backward")
            .fail("Output of a custom Function is a view and its base or another view of its "
                  "base has been modified in place; this view is the output of a function that "
                  "returns a view of its input, so its gradient cannot be computed after the "
                  "base is modified in place");
    }
};

// Where a tensor's elements sit, in its own row-major order: one run from
// ``first`` when the tensor is contiguous, one address per element
// otherwise.  A CPU tensor's addresses are its buffer's; a Metal family
// member's count from its family's identity (TensorImpl::family_key) as if
// that were the root's first byte, so members of one family compare as CPU
// views of one buffer do.  Addresses are integers — a Metal family's
// identity is no buffer to point into — and ``first`` is 0 for any other
// tensor.
struct Footprint {
    std::uintptr_t first = 0;
    std::size_t n = 0;
    std::size_t item = 0;
    std::vector<std::uintptr_t> addr;

    bool run() const { return addr.empty(); }
    std::uintptr_t at(std::size_t i) const { return run() ? first + i * item : addr[i]; }
};

Footprint footprint(const TensorImpl& t) {
    Footprint f;
    auto base = reinterpret_cast<std::uintptr_t>(t.family_key());
    if (const auto* cpu = std::get_if<CpuStorage>(&t.raw_storage()))
        base = reinterpret_cast<std::uintptr_t>(cpu->ptr.get());
    if (base == 0)
        return f;
    const Dtype ldt = lane_dtype(t.dtype());
    const Shape shape = lane_shape(t.shape(), t.dtype());
    const Stride stride = lane_stride(t.stride(), t.dtype());
    f.first = base + t.storage_offset();
    f.n = shape_numel(shape);
    f.item = dtype_size(ldt);
    if (t.is_contiguous() || f.n == 0)
        return f;
    f.addr.resize(f.n);
    std::vector<std::int64_t> idx(shape.size(), 0);
    for (std::size_t i = 0; i < f.n; ++i) {
        std::int64_t off = 0;
        for (std::size_t d = 0; d < shape.size(); ++d)
            off += idx[d] * stride[d];
        // Modular: a negative offset steps back from ``first``.
        f.addr[i] = f.first + static_cast<std::uintptr_t>(off);
        for (std::size_t d = shape.size(); d-- > 0;) {
            if (++idx[d] < shape[d])
                break;
            idx[d] = 0;
        }
    }
    return f;
}

// Take ``m`` off the graph: the values it reads are a constant's now.
void cut(const TensorImplPtr& m) {
    if (!m->grad_fn())
        return;
    m->set_grad_fn(nullptr);
    m->set_grad_output_nr(0);
    m->set_requires_grad(false);
}

// Give ``m`` the graph ``node`` describes: its old graph as the base (unless
// the write covered it) and ``a``'s as the source (unless ``a`` holds a
// constant).
void attach(const TensorImplPtr& m,
            const TensorImplPtr& a,
            bool covered,
            bool differentiable,
            std::shared_ptr<CopySlicesBackward> node) {
    auto base = covered ? nullptr : detail::ensure_grad_fn(m);
    auto src = differentiable ? detail::ensure_grad_fn(a) : nullptr;
    if (!base && !src) {
        cut(m);
        return;
    }
    const std::uint32_t base_nr = base ? m->grad_output_nr() : 0;
    const std::uint32_t src_nr = src ? a->grad_output_nr() : 0;
    node->set_next_edges({Edge(std::move(base), base_nr), Edge(std::move(src), src_nr)});
    m->set_grad_fn(std::move(node));
    m->set_grad_output_nr(0);
    m->set_requires_grad(true);
    m->set_leaf(false);
}

bool is_identity_mapping(const std::vector<std::int64_t>& src,
                         std::size_t fm_n,
                         std::size_t fa_n,
                         bool covered,
                         bool same_dtype) {
    if (!covered || fm_n != fa_n || !same_dtype)
        return false;
    for (std::size_t k = 0; k < src.size(); ++k)
        if (src[k] != static_cast<std::int64_t>(k))
            return false;
    return true;
}

void finish_rebased_member(const TensorImplPtr& m,
                           const TensorImplPtr& a,
                           bool covered,
                           bool same,
                           bool differentiable,
                           std::shared_ptr<CopySlicesBackward> node) {
    if (m->is_custom_function_output()) {
        m->set_grad_fn(std::make_shared<InvalidatedCustomViewNode>());
        m->set_grad_output_nr(0);
        m->set_requires_grad(true);
        m->set_leaf(false);
        if (m->retains_grad())
            retain_grad(m);
        return;
    }
    if (covered && !differentiable) {
        cut(m);
        return;
    }
    if (same) {
        // The same elements in the same order: a reshape of ``a``.
        kernel::NaryKernel<ViewBackward, 1>::wire_autograd({a}, m, false);
    } else {
        attach(m, a, covered, differentiable, std::move(node));
    }
    // A member that asked to keep its gradient keeps the one of its new
    // place in the graph.
    if (m->retains_grad())
        retain_grad(m);
}

}  // namespace

void rebase_views(const TensorImplPtr& a) {
    // A write through a detached alias (``.data``) is untracked by design,
    // and a detached member takes no part in autograd either way.
    if (a->is_detached_alias())
        return;
    const Footprint fa = footprint(*a);
    if (!fa.first)
        return;
    const bool differentiable = a->requires_grad();

    // Which of ``a``'s elements sits at an address, if any.  A run answers
    // by arithmetic; a strided ``a`` builds its table once, on first need.
    std::unordered_map<std::uintptr_t, std::int64_t> where;
    const auto locate = [&](std::uintptr_t x) -> std::int64_t {
        if (fa.run()) {
            const auto lo = fa.first;
            if (x < lo || x >= lo + fa.n * fa.item || (x - lo) % fa.item != 0)
                return -1;
            return static_cast<std::int64_t>((x - lo) / fa.item);
        }
        if (where.empty()) {
            where.reserve(fa.n);
            for (std::size_t i = 0; i < fa.n; ++i)
                where.emplace(fa.addr[i], static_cast<std::int64_t>(i));
        }
        const auto it = where.find(x);
        return it == where.end() ? -1 : it->second;
    };

    for (const auto& m : a->live_views()) {
        // A leaf keeps its place: it is where gradient accumulates, and its
        // AccumulateGrad is a grad_fn this must not cut.
        if (!m->rederived_by_family_writes())
            continue;
        const Footprint fm = footprint(*m);
        if (!fm.first || fm.n == 0)
            continue;

        bool covered = false;
        bool same = false;
        const bool same_dtype = m->dtype() == a->dtype();
        std::shared_ptr<CopySlicesBackward> node;
        if (fa.run() && fm.run()) {
            // Two runs meet in a run.
            const std::uintptr_t lo = std::max(fa.first, fm.first);
            const std::uintptr_t hi =
                std::min(fa.first + fa.n * fa.item, fm.first + fm.n * fm.item);
            if (hi <= lo)
                continue;
            const auto item = static_cast<std::int64_t>(fa.item);
            const auto len = static_cast<std::int64_t>(hi - lo) / item;
            covered = len == static_cast<std::int64_t>(fm.n);
            same = covered && len == static_cast<std::int64_t>(fa.n) && same_dtype;
            if (!same)
                node = std::make_shared<CopySlicesBackward>(
                    m->shape(), a->shape(), static_cast<std::int64_t>(lo - fm.first) / item,
                    static_cast<std::int64_t>(lo - fa.first) / item, len, m->dtype(), a->dtype(),
                    a->device());
        } else {
            // A strided side meets the other at scattered positions.
            std::vector<std::int64_t> dst;
            std::vector<std::int64_t> src;
            for (std::size_t j = 0; j < fm.n; ++j)
                if (const auto i = locate(fm.at(j)); i >= 0) {
                    dst.push_back(static_cast<std::int64_t>(j));
                    src.push_back(i);
                }
            if (dst.empty())
                continue;
            covered = dst.size() == fm.n;
            same = is_identity_mapping(src, fm.n, fa.n, covered, same_dtype);
            if (!same)
                node = std::make_shared<CopySlicesBackward>(m->shape(), a->shape(), std::move(dst),
                                                            std::move(src), m->dtype(), a->dtype(),
                                                            a->device());
        }

        finish_rebased_member(m, a, covered, same, differentiable, std::move(node));
    }
}

namespace {

// A dtype's rung on the ladder an in-place write may not move down — the
// kinds ``lucid._promotion`` ranks the Python operands by.
int kind_rank(Dtype dt) {
    if (dt == Dtype::Bool)
        return 0;
    if (is_integral(dt))
        return 1;
    return is_complex(dt) ? 3 : 2;
}

// ``out`` at ``a``'s dtype, or the reference's refusal when the cast would
// move down a kind (the message ``lucid._dispatch`` gives a promoted
// operand pair).
TensorImplPtr at_dtype_of(const TensorImplPtr& a, const TensorImplPtr& out, const char* name) {
    const Dtype want = a->dtype();
    if (out->dtype() == want)
        return out;
    if (kind_rank(out->dtype()) > kind_rank(want))
        ErrorBuilder(name).fail("result type " + std::string(dtype_name(out->dtype())) +
                                " can't be cast to the desired output type " +
                                std::string(dtype_name(want)));
    return astype_op(out, want);
}

}  // namespace

TensorImplPtr write_result(const TensorImplPtr& a, TensorImplPtr out, const char* name) {
    if (out->shape() != a->shape())
        throw ShapeMismatch(a->shape(), out->shape(),
                            std::string(name) + " (in-place: shape changed)");
    out = at_dtype_of(a, out, name);
    // Into ``a``'s Metal shared buffer instead when it has one and no graph
    // was recorded — see TensorImpl::take_storage_from.
    a->take_storage_from(*out, name);
    a->set_device(out->device());
    // Either branch moves ``a``'s version, which is what tells a node that
    // saved ``a`` before this write that its values moved.  This op's own
    // node is not one of them: it holds the snapshot, whose count is its own.
    if (!adopt_graph_position(a, out))
        detach_and_bump(a);
    return a;
}

}  // namespace lucid::inplace
