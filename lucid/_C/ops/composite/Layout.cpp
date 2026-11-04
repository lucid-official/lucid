// lucid/_C/ops/composite/Layout.cpp
//
// ``movedim`` rebuilds a permutation: every axis listed in ``source`` is
// placed at the matching index in ``destination``; unmoved axes preserve
// their order and fill the gaps.  ``unflatten`` is a thin wrapper around
// ``reshape`` after validating the size product.

#include "Layout.h"

#include <algorithm>
#include <cstdint>

#include "../../autograd/Helpers.h"
#include "../../core/ErrorBuilder.h"
#include "../../core/TensorImpl.h"
#include "../ufunc/Transpose.h"
#include "../utils/View.h"

namespace lucid {

TensorImplPtr movedim_op(const TensorImplPtr& a,
                         const std::vector<int>& source,
                         const std::vector<int>& destination) {
    if (!a)
        ErrorBuilder("movedim").fail("null input");
    const int ndim = static_cast<int>(a->shape().size());
    if (source.size() != destination.size())
        ErrorBuilder("movedim").invalid_argument("source and destination size mismatch");

    auto wrap = [&](int v) { return wrap_dim(v, ndim, "movedim"); };
    std::vector<int> src;
    std::vector<int> dst;
    src.reserve(source.size());
    dst.reserve(destination.size());
    for (int s : source)
        src.push_back(wrap(s));
    for (int d : destination)
        dst.push_back(wrap(d));

    // Reject duplicates — they would make the permutation ill-defined.
    std::vector<int> sorted_src = src;
    std::sort(sorted_src.begin(), sorted_src.end());
    if (std::adjacent_find(sorted_src.begin(), sorted_src.end()) != sorted_src.end())
        ErrorBuilder("movedim").invalid_argument("source dims must be unique");
    std::vector<int> sorted_dst = dst;
    std::sort(sorted_dst.begin(), sorted_dst.end());
    if (std::adjacent_find(sorted_dst.begin(), sorted_dst.end()) != sorted_dst.end())
        ErrorBuilder("movedim").invalid_argument("destination dims must be unique");

    // A 0-d tensor moves its notional axis onto itself: the empty permutation.
    if (ndim == 0)
        return permute_op(a, {});

    // Axes not mentioned in ``source`` keep their relative order; they fill
    // the destination slots not claimed by ``dst``.
    std::vector<bool> moved(static_cast<std::size_t>(ndim), false);
    for (int s : src)
        moved[static_cast<std::size_t>(s)] = true;
    std::vector<int> rest;
    for (int i = 0; i < ndim; ++i)
        if (!moved[static_cast<std::size_t>(i)])
            rest.push_back(i);

    std::vector<int> perm(static_cast<std::size_t>(ndim), -1);
    for (std::size_t i = 0; i < src.size(); ++i)
        perm[static_cast<std::size_t>(dst[i])] = src[i];
    auto rest_it = rest.begin();
    for (int i = 0; i < ndim; ++i) {
        if (perm[static_cast<std::size_t>(i)] == -1)
            perm[static_cast<std::size_t>(i)] = *rest_it++;
    }
    return permute_op(a, perm);
}

TensorImplPtr unflatten_op(const TensorImplPtr& a, int dim, const Shape& sizes) {
    if (!a)
        ErrorBuilder("unflatten").fail("null input");
    const int ndim = static_cast<int>(a->shape().size());
    // Refused as a state error, as the reference refuses it: a 0-d tensor
    // has no dimension to split, whatever ``dim`` names.
    if (ndim == 0)
        ErrorBuilder("unflatten").fail("cannot unflatten a 0-d tensor: it has no dimension");
    const int d = wrap_dim(dim, ndim, "unflatten");
    if (sizes.empty())
        ErrorBuilder("unflatten").invalid_argument("sizes must be non-empty");
    const std::int64_t dim_len = a->shape()[static_cast<std::size_t>(d)];
    Shape resolved;
    resolved.reserve(sizes.size());
    int wildcard_pos = -1;
    std::int64_t known_product = 1;
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == -1) {
            if (wildcard_pos != -1)
                ErrorBuilder("unflatten").invalid_argument("only one -1 is allowed");
            wildcard_pos = static_cast<int>(i);
            resolved.push_back(0);
        } else if (sizes[i] < 0) {
            ErrorContextGuard ctx{"unflatten"};
            throw_negative_dim(sizes);
        } else {
            known_product *= sizes[i];
            resolved.push_back(sizes[i]);
        }
    }
    if (wildcard_pos != -1) {
        if (known_product == 0 || dim_len % known_product != 0)
            ErrorBuilder("unflatten").invalid_argument("sizes do not multiply to dim length");
        resolved[static_cast<std::size_t>(wildcard_pos)] = dim_len / known_product;
    } else if (known_product != dim_len) {
        ErrorBuilder("unflatten").invalid_argument("sizes do not multiply to dim length");
    }

    Shape new_shape;
    new_shape.reserve(static_cast<std::size_t>(ndim) - 1 + resolved.size());
    for (int i = 0; i < d; ++i)
        new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    for (auto s : resolved)
        new_shape.push_back(s);
    for (int i = d + 1; i < ndim; ++i)
        new_shape.push_back(a->shape()[static_cast<std::size_t>(i)]);
    return reshape_op(a, new_shape);
}

}  // namespace lucid
