// lucid/_C/ops/einops/Einsum.cpp
//
// Implementation of einsum_op: general tensor contraction.
//
// This is a pure-C++ implementation of Einstein summation that does NOT call
// a backend-level einsum kernel; instead it builds a sequence of
// mul + sum_op + permute calls so that autograd flows through each primitive.
//
// Algorithm (left-to-right pairwise contraction):
//   1. Parse the equation: strip whitespace, split at "->", split lhs at ",".
//      If no "->" is present, infer the output as the sorted set of labels
//      that appear exactly once across all input specs (numpy implicit form).
//   2. Validate that every input's label count matches its ndim, and that no
//      label has conflicting sizes across operands.
//   3. Single-operand shortcut: sum over labels not in rhs, permute to rhs order.
//   4. Multi-operand: for each successive pair (accumulated result, next operand):
//        a. drop_loners: sum away axes that are private to one operand and won't
//           appear in the output or future operands; this reduces memory traffic.
//        b. align_to_labels: unsqueeze + permute + broadcast both tensors to a
//           common axis order [keep..., contract...].
//        c. elementwise multiply.
//        d. sum over the contraction axes.
//   5. Final cleanup: sum any remaining surplus axes, permute to rhs order.
//
// Limitations:
//   - Ellipsis (...) patterns are not supported.
//   - All operands must be on the same device and share a dtype.

#include <algorithm>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "../../core/Dtype.h"
#include "../../core/Error.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/Profiler.h"
#include "../../core/Scope.h"
#include "../../core/TensorImpl.h"
#include "../../core/Validate.h"
#include "../bfunc/Matmul.h"
#include "../bfunc/Mul.h"
#include "../ufunc/Astype.h"
#include "../ufunc/Reductions.h"
#include "../ufunc/Transpose.h"
#include "../utils/Contiguous.h"
#include "../utils/Layout.h"
#include "../utils/Promote.h"
#include "../utils/View.h"
#include "Einops.h"

namespace lucid {

namespace {

// Remove all spaces and tabs from a string.
//
// einsum equations often contain spaces for readability ("i j, j k -> i k");
// strip_ws normalises them before any label-level parsing so the rest of the
// code can assume a compact equation with no whitespace.
std::string strip_ws(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s)
        if (c != ' ' && c != '\t')
            out.push_back(c);
    return out;
}

// Split s at every ',' character and return the resulting substrings.
//
// Used to separate the per-operand label strings on the lhs of the equation,
// e.g. "ij,jk" → ["ij", "jk"].  The splitting includes the degenerate case
// of a single operand (no comma), which yields a one-element result.
std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::size_t start = 0;
    for (std::size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            out.push_back(s.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

// Align tensor t (whose current axes are described by src_labels) to a
// specified target axis ordering by:
//   1. Inserting size-1 dimensions at the trailing end for any label present
//      in target_labels but absent from src_labels.  These become broadcast
//      dimensions that will be expanded in step 3.
//   2. Permuting src_labels to match the target order.  If the permutation is
//      the identity it is skipped to avoid a no-op op node.
//   3. Broadcasting to the full sizes given in the sizes map, expanding any
//      size-1 dimensions introduced in step 1.
//
// This function is the core mechanism that makes two tensors pointwise-
// compatible along a shared set of axes: after align_to_labels is called on
// both tensors with the same target_labels, they can be multiplied element-
// wise to accumulate the einsum product.
//
// The src_labels vector is passed by value so the function can track the
// evolving label list after each unsqueeze without mutating the caller's copy.
TensorImplPtr align_to_labels(const TensorImplPtr& t,
                              std::vector<std::string> src_labels,
                              const std::vector<std::string>& target_labels,
                              const std::map<std::string, std::int64_t>& sizes) {
    auto cur = t;

    // Step 1: append any missing target labels as trailing size-1 dimensions.
    for (const auto& c : target_labels) {
        if (std::find(src_labels.begin(), src_labels.end(), c) == src_labels.end()) {
            cur = unsqueeze_op(cur, static_cast<int>(cur->shape().size()));
            src_labels.push_back(c);
        }
    }

    // Step 2: permute src_labels into the target order.
    std::vector<int> perm(target_labels.size());
    for (std::size_t i = 0; i < target_labels.size(); ++i) {
        auto it = std::find(src_labels.begin(), src_labels.end(), target_labels[i]);
        perm[i] = static_cast<int>(it - src_labels.begin());
    }
    bool identity = true;
    for (std::size_t i = 0; i < perm.size(); ++i)
        if (perm[i] != static_cast<int>(i)) {
            identity = false;
            break;
        }
    if (!identity)
        cur = permute_op(cur, perm);

    // Step 3: broadcast to the full sizes (expands the size-1 dims from step 1).
    Shape target_shape;
    for (const auto& c : target_labels)
        target_shape.push_back(sizes.at(c));
    if (cur->shape() != target_shape)
        cur = broadcast_to_op(cur, target_shape);
    return cur;
}

Dtype common_dtype(Dtype a, Dtype b) {
    if (a == b)
        return a;
    const auto kind = [](Dtype d) {
        return is_complex(d) ? 3 : is_floating_point(d) ? 2 : is_integral(d) ? 1 : 0;
    };
    if (kind(a) != kind(b))
        return kind(a) > kind(b) ? a : b;
    if (dtype_size(a) != dtype_size(b))
        return dtype_size(a) > dtype_size(b) ? a : b;
    return Dtype::F32;
}

TensorImplPtr permute_to(const TensorImplPtr& t,
                         const std::vector<std::string>& src,
                         const std::vector<std::string>& dst) {
    std::vector<int> perm(dst.size());
    for (std::size_t i = 0; i < dst.size(); ++i) {
        auto it = std::find(src.begin(), src.end(), dst[i]);
        perm[i] = static_cast<int>(it - src.begin());
    }
    bool identity = true;
    for (std::size_t i = 0; i < perm.size(); ++i)
        if (perm[i] != static_cast<int>(i)) {
            identity = false;
            break;
        }
    return identity ? t : permute_op(t, perm);
}

TensorImplPtr contract_via_matmul(const TensorImplPtr& cur,
                                  std::vector<std::string>& cur_labels,
                                  const TensorImplPtr& right_pre,
                                  const std::vector<std::string>& right_labels,
                                  const std::set<std::string>& future_labels,
                                  const std::map<std::string, std::int64_t>& sizes) {
    std::set<std::string> cl(cur_labels.begin(), cur_labels.end());
    std::set<std::string> rl(right_labels.begin(), right_labels.end());
    std::vector<std::string> batch, lonly, ronly, contr;
    for (const auto& c : cl) {
        if (rl.count(c)) {
            if (future_labels.count(c))
                batch.push_back(c);
            else
                contr.push_back(c);
        } else {
            lonly.push_back(c);
        }
    }
    for (const auto& c : rl)
        if (!cl.count(c))
            ronly.push_back(c);
    std::sort(batch.begin(), batch.end());
    std::sort(lonly.begin(), lonly.end());
    std::sort(ronly.begin(), ronly.end());
    std::sort(contr.begin(), contr.end());

    std::vector<std::string> ldst = batch;
    ldst.insert(ldst.end(), lonly.begin(), lonly.end());
    ldst.insert(ldst.end(), contr.begin(), contr.end());
    std::vector<std::string> rdst = batch;
    rdst.insert(rdst.end(), contr.begin(), contr.end());
    rdst.insert(rdst.end(), ronly.begin(), ronly.end());

    auto lhs_t = permute_to(cur, cur_labels, ldst);
    auto rhs_t = permute_to(right_pre, right_labels, rdst);

    auto group_prod = [&](const std::vector<std::string>& g) {
        std::int64_t p = 1;
        for (const auto& c : g)
            p *= sizes.at(c);
        return p;
    };
    const std::int64_t in = group_prod(lonly);
    const std::int64_t kn = group_prod(contr);
    const std::int64_t nn = group_prod(ronly);

    Shape lshape, rshape;
    for (const auto& c : batch) {
        lshape.push_back(sizes.at(c));
        rshape.push_back(sizes.at(c));
    }
    lshape.push_back(in);
    lshape.push_back(kn);
    rshape.push_back(kn);
    rshape.push_back(nn);
    lhs_t = reshape_op(lhs_t, lshape);
    rhs_t = reshape_op(rhs_t, rshape);

    auto prod = matmul_op(lhs_t, rhs_t);

    Shape oshape;
    for (const auto& c : batch)
        oshape.push_back(sizes.at(c));
    for (const auto& c : lonly)
        oshape.push_back(sizes.at(c));
    for (const auto& c : ronly)
        oshape.push_back(sizes.at(c));

    cur_labels = batch;
    cur_labels.insert(cur_labels.end(), lonly.begin(), lonly.end());
    cur_labels.insert(cur_labels.end(), ronly.begin(), ronly.end());
    return reshape_op(prod, oshape);
}

}  // namespace

// Compute an Einstein summation over a list of operands.
//
// The full algorithm is described in the file-level comment.  Key invariant
// throughout the multi-operand loop: `cur` is the accumulated partial result
// of contracting operands[0..k-1], and `cur_labels` is its axis label list
// in the same order as the tensor dimensions.
TensorImplPtr einsum_op(const std::string& pattern, const std::vector<TensorImplPtr>& operands) {
    if (operands.empty())
        ErrorBuilder("einsum").fail("at least one operand required");
    if (pattern.find("...") != std::string::npos)
        ErrorBuilder("einsum").not_implemented(
            "ellipsis (...) patterns not yet supported in C++ engine");

    OpScopeFull scope{"einsum", operands[0]->device(), operands[0]->dtype(), operands[0]->shape()};

    // Parse equation: strip whitespace, separate lhs and rhs at "->".
    auto eq = strip_ws(pattern);
    std::string lhs, rhs;
    auto arrow = eq.find("->");
    if (arrow != std::string::npos) {
        lhs = eq.substr(0, arrow);
        rhs = eq.substr(arrow + 2);
    } else {
        // Implicit (no-arrow) form: the output is the sorted list of labels
        // that appear exactly once across all operand specs.  Labels that
        // appear more than once are summed over (contracted).  This matches
        // the numpy.einsum implicit-form convention.
        lhs = eq;
        std::map<char, int> counts;
        for (char c : lhs) {
            if (c == ',')
                continue;
            counts[c]++;
        }
        std::vector<char> singles;
        for (auto& [c, v] : counts)
            if (v == 1)
                singles.push_back(c);
        // Sort alphabetically, consistent with numpy's implicit-mode ordering.
        std::sort(singles.begin(), singles.end());
        rhs.assign(singles.begin(), singles.end());
    }

    auto in_specs = split_commas(lhs);
    if (in_specs.size() != operands.size())
        ErrorBuilder("einsum").fail("pattern operand count != operands.size()");

    // Build a global label-size map.  Raise immediately if any label appears
    // with two different sizes across operands (would indicate a shape mismatch
    // that could not produce a valid contraction).
    std::map<std::string, std::int64_t> sizes;
    for (std::size_t k = 0; k < operands.size(); ++k) {
        const auto& spec = in_specs[k];
        const auto& t = operands[k];
        if (spec.size() != t->shape().size())
            ErrorBuilder("einsum").fail("operand" + std::to_string(k) + " has " +
                                        std::to_string(t->shape().size()) + " axes but spec '" +
                                        spec + "' has " + std::to_string(spec.size()) + " labels");
        for (std::size_t i = 0; i < spec.size(); ++i) {
            std::string c(1, spec[i]);
            std::int64_t n = t->shape()[i];
            auto it = sizes.find(c);
            if (it == sizes.end())
                sizes[c] = n;
            else if (it->second != n)
                ErrorBuilder("einsum").fail("label '" + c + "' has conflicting sizes " +
                                            std::to_string(it->second) + " vs " +
                                            std::to_string(n));
        }
    }
    // All output labels must have been seen in the input specs.
    for (char c : rhs) {
        std::string s(1, c);
        if (sizes.find(s) == sizes.end())
            ErrorBuilder("einsum").fail("output label '" + s + "' not in inputs");
    }

    // --- Single-operand shortcut ---
    // For a single tensor (e.g. "ij->i"), just sum axes absent from rhs, then
    // permute.  No multiplication is needed.
    if (operands.size() == 1) {
        std::vector<std::string> cur_labels;
        for (char c : in_specs[0])
            cur_labels.push_back(std::string(1, c));
        auto cur = operands[0];

        std::set<std::string> rhs_set;
        for (char c : rhs)
            rhs_set.insert(std::string(1, c));

        // Collect indices of labels that should be summed out.
        std::vector<int> kill;
        for (std::size_t i = 0; i < cur_labels.size(); ++i)
            if (rhs_set.find(cur_labels[i]) == rhs_set.end())
                kill.push_back(static_cast<int>(i));
        if (!kill.empty()) {
            cur = sum_op(cur, kill, false);
            // Rebuild cur_labels to only contain the surviving (rhs) labels.
            std::vector<std::string> new_labels;
            for (std::size_t i = 0; i < cur_labels.size(); ++i)
                if (rhs_set.find(cur_labels[i]) != rhs_set.end())
                    new_labels.push_back(cur_labels[i]);
            cur_labels = std::move(new_labels);
        }

        // Permute surviving labels to the requested output order.
        std::vector<std::string> rhs_labels;
        for (char c : rhs)
            rhs_labels.push_back(std::string(1, c));
        if (cur_labels != rhs_labels) {
            std::vector<int> perm(rhs_labels.size());
            for (std::size_t i = 0; i < rhs_labels.size(); ++i) {
                auto it = std::find(cur_labels.begin(), cur_labels.end(), rhs_labels[i]);
                perm[i] = static_cast<int>(it - cur_labels.begin());
            }
            bool identity = true;
            for (std::size_t i = 0; i < perm.size(); ++i)
                if (perm[i] != static_cast<int>(i)) {
                    identity = false;
                    break;
                }
            if (!identity)
                cur = permute_op(cur, perm);
        }
        return cur;
    }

    // --- Multi-operand: left-to-right pairwise contraction ---
    // Initialise the accumulator from operands[0].
    auto cur = operands[0];
    std::vector<std::string> cur_labels;
    for (char c : in_specs[0])
        cur_labels.push_back(std::string(1, c));

    for (std::size_t k = 1; k < operands.size(); ++k) {
        const auto& right = operands[k];
        std::vector<std::string> right_labels;
        for (char c : in_specs[k])
            right_labels.push_back(std::string(1, c));

        // future_labels: labels needed by the output or by operands after k.
        // Used to decide whether a label appearing only in one side can be
        // summed away immediately (drop_loners) rather than being carried
        // through the broadcast.
        std::set<std::string> future_labels;
        for (char c : rhs)
            future_labels.insert(std::string(1, c));
        for (std::size_t j = k + 1; j < operands.size(); ++j)
            for (char c : in_specs[j])
                future_labels.insert(std::string(1, c));

        std::set<std::string> cur_set(cur_labels.begin(), cur_labels.end());
        std::set<std::string> right_set(right_labels.begin(), right_labels.end());
        // union_set holds every label present in either the left or the right tensor.
        std::set<std::string> union_set;
        union_set.insert(cur_set.begin(), cur_set.end());
        union_set.insert(right_set.begin(), right_set.end());

        // Categorise each label in the union into:
        //   keep     — appears in future_labels (needed later); survives this step.
        //   contract — shared between cur and right, not needed later; summed here.
        // Labels private to one side and not needed later are handled by
        // drop_loners before the broadcast.
        std::vector<std::string> keep, contract;
        for (auto& c : union_set) {
            bool inter = (cur_set.count(c) && right_set.count(c));
            if (future_labels.count(c))
                keep.push_back(c);
            else if (inter)
                contract.push_back(c);
        }
        // Sort to make the axis order deterministic across operand orderings.
        std::sort(keep.begin(), keep.end());
        std::sort(contract.begin(), contract.end());

        // Lay the common broadcast order as [keep..., contract...].
        // After mul_op the contract axes (at the trailing positions) are summed
        // out so the result carries only the keep axes.
        std::vector<std::string> order;
        order.insert(order.end(), keep.begin(), keep.end());
        order.insert(order.end(), contract.begin(), contract.end());

        // drop_loners: eagerly reduce axes that are private to one operand and
        // will not appear in future steps.  This shrinks intermediate tensors
        // and can significantly reduce memory traffic for large contractions.
        //
        // Concretely: a label is a "loner" if it is in `labels` but not in
        // `other_set` and not in `future_labels`.  Summing it out here is
        // mathematically equivalent to summing it at the end because it does
        // not participate in the mul with the other operand.
        auto drop_loners = [&](TensorImplPtr t, std::vector<std::string>& labels,
                               const std::set<std::string>& other_set) {
            std::vector<int> kill;
            std::vector<std::string> new_labels;
            for (std::size_t i = 0; i < labels.size(); ++i) {
                if (!future_labels.count(labels[i]) && !other_set.count(labels[i])) {
                    kill.push_back(static_cast<int>(i));
                } else {
                    new_labels.push_back(labels[i]);
                }
            }
            if (!kill.empty())
                t = sum_op(t, kill, false);
            labels = std::move(new_labels);
            return t;
        };
        // Drop loner axes from both sides before the broadcast.
        cur = drop_loners(cur, cur_labels, right_set);
        auto right_pre = drop_loners(right, right_labels, cur_set);
        Validator::pair(cur, right_pre, "einsum").same_device();

        if (!contract.empty()) {
            // Floating contractions route through matmul_op, which applies its
            // LOWER autocast policy and refuses mixed dtypes outside autocast
            // (LCD-575).  Non-floating contractions require matching dtypes and
            // stay on the exact mul+sum path.
            if (is_floating_point(cur->dtype()) && is_floating_point(right_pre->dtype())) {
                cur = contract_via_matmul(cur, cur_labels, right_pre, right_labels, future_labels,
                                          sizes);
                continue;
            }
            Validator::pair(cur, right_pre, "einsum").same_dtype();
        } else if (cur->dtype() != right_pre->dtype()) {
            // Non-contracted outer / Hadamard steps promote mixed dtypes, as
            // the reference framework's mul path does.
            const Dtype dt = common_dtype(cur->dtype(), right_pre->dtype());
            cur = astype_op(cur, dt);
            right_pre = astype_op(right_pre, dt);
        }

        // Align both tensors to the common [keep, contract] axis order via
        // unsqueeze + permute + broadcast, then multiply pointwise.
        cur = align_to_labels(cur, cur_labels, order, sizes);
        right_pre = align_to_labels(right_pre, right_labels, order, sizes);

        // Elementwise product: broadcasts the unsqueezed unit dims from
        // align_to_labels so every element of the product carries contributions
        // from both tensors at the correct indices.
        cur = mul_op(cur, right_pre);

        // Sum over the trailing contract axes to complete this pairwise step.
        // After this sum the result has axes [keep...] in alphabetical order.
        if (!contract.empty()) {
            std::vector<int> sum_axes;
            for (std::size_t i = keep.size(); i < order.size(); ++i)
                sum_axes.push_back(static_cast<int>(i));
            cur = sum_op(cur, sum_axes, false);
        }
        // Record that cur's axes are now exactly the keep labels.
        cur_labels.assign(order.begin(), order.begin() + keep.size());
    }

    // Final cleanup: sum any surplus labels that survived the pairwise loop
    // but are not requested in the rhs (e.g. labels private to the last
    // operand that were not dropped by drop_loners because of a future
    // reference that no longer exists after all operands are consumed).
    std::set<std::string> rhs_set;
    for (char c : rhs)
        rhs_set.insert(std::string(1, c));
    std::vector<int> kill;
    std::vector<std::string> new_cur_labels;
    for (std::size_t i = 0; i < cur_labels.size(); ++i) {
        if (rhs_set.find(cur_labels[i]) == rhs_set.end())
            kill.push_back(static_cast<int>(i));
        else
            new_cur_labels.push_back(cur_labels[i]);
    }
    if (!kill.empty()) {
        cur = sum_op(cur, kill, false);
        cur_labels = std::move(new_cur_labels);
    }

    // Final permutation: reorder the surviving axes to match the rhs.
    std::vector<std::string> rhs_labels;
    for (char c : rhs)
        rhs_labels.push_back(std::string(1, c));
    if (cur_labels != rhs_labels) {
        std::vector<int> perm(rhs_labels.size());
        for (std::size_t i = 0; i < rhs_labels.size(); ++i) {
            auto it = std::find(cur_labels.begin(), cur_labels.end(), rhs_labels[i]);
            if (it == cur_labels.end())
                ErrorBuilder("einsum").fail("rhs label '" + rhs_labels[i] +
                                            "' missing from cur_labels");
            perm[i] = static_cast<int>(it - cur_labels.begin());
        }
        bool identity = true;
        for (std::size_t i = 0; i < perm.size(); ++i)
            if (perm[i] != static_cast<int>(i)) {
                identity = false;
                break;
            }
        if (!identity)
            cur = permute_op(cur, perm);
    }
    return cur;
}

}  // namespace lucid
