// lucid/_C/backend/cpu/Pool.cpp
//
// Implements 1-D, 2-D, and 3-D MaxPool and AvgPool forward and backward
// kernels.  All kernels are instantiated via typed templates and exposed
// through thin f32/f64 entry points.
//
// MaxPool forward scans the receptive field of each output position and
// records the flat spatial index of the winner in argmax[].  Padding regions
// are excluded from consideration (they can never beat NEG_INF).  The backward
// simply scatter-adds g[o] to dx[argmax[o]].
//
// AvgPool forward sums the receptive field and divides by the full window area
// (count-include-pad semantics).  The backward divides g uniformly by the
// same area and scatter-adds to every input position in the window.

#include "Pool.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lucid::backend::cpu {

namespace {

template <typename T>
inline void update_max_pool_best(T v, int idx, T& best, int& best_idx) {
    if (best_idx < 0 || v > best || (std::isnan(v) && !std::isnan(best))) {
        best = v;
        best_idx = idx;
    }
}

template <typename T>
void max_pool_scatter_backward(
    const T* g, const std::int32_t* argmax, T* dx, int BC, int in_spatial, int out_spatial) {
    for (int bc = 0; bc < BC; ++bc) {
        const T* gb = g + bc * out_spatial;
        const auto* ab = argmax + bc * out_spatial;
        T* dxb = dx + bc * in_spatial;
        for (int o = 0; o < out_spatial; ++o) {
            const int idx = ab[o];
            if (idx >= 0)
                dxb[idx] += gb[o];
        }
    }
}

// 2-D max pooling forward typed implementation.
template <typename T>
// NaN wins the pooling window, exactly as it does in ``max``.
//
// ``best`` starts at -infinity and every comparison against a NaN is
// false, so a window that was entirely NaN pooled to *-inf* — not merely
// a wrong answer but an impossible one, since no input element was -inf.
// The reference pools it to nan.  Three sites: 1-D, 2-D and 3-D.
void max_pool2d_forward_typed(const T* x,
                              T* y,
                              std::int32_t* argmax,
                              int B,
                              int C,
                              int H,
                              int W,
                              int KH,
                              int KW,
                              int OH,
                              int OW,
                              int sh,
                              int sw,
                              int ph,
                              int pw) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * H * W;
            T* yb = y + (b * C + c) * OH * OW;
            auto* ab = argmax + (b * C + c) * OH * OW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    T best = NEG_INF;
                    int best_idx = -1;
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            const int idx = ih * W + iw;
                            update_max_pool_best(xb[idx], idx, best, best_idx);
                        }
                    }
                    yb[oh * OW + ow] = best;
                    ab[oh * OW + ow] = best_idx;
                }
            }
        }
    }
}

template <typename T>
void max_pool2d_backward_typed(
    const T* g, const std::int32_t* argmax, T* dx, int B, int C, int H, int W, int OH, int OW) {
    max_pool_scatter_backward(g, argmax, dx, B * C, H * W, OH * OW);
}

// Per-axis divisor contribution for the average-pool window at output index
// ``o``.
//
// The window end is clamped to ``S + pad`` — never past the right padding —
// so a ceil-mode overhang counts toward neither the sum nor the divisor.
// ``count_include_pad`` then selects between the padded extent and the number
// of real input elements the window actually covers.  For floor-mode pooling
// with ``count_include_pad`` this returns exactly ``K``, which is why every
// pre-existing call site keeps its previous divisor.
inline int avg_pool_span(int o, int S, int K, int stride, int pad, bool count_include_pad) {
    const int start = o * stride - pad;
    const int end = std::min(start + K, S + pad);
    if (count_include_pad)
        return end - start;
    const int lo = std::max(start, 0);
    const int hi = std::min(end, S);
    return hi > lo ? hi - lo : 0;
}

template <typename T>
void avg_pool2d_forward_typed(const T* x,
                              T* y,
                              int B,
                              int C,
                              int H,
                              int W,
                              int KH,
                              int KW,
                              int OH,
                              int OW,
                              int sh,
                              int sw,
                              int ph,
                              int pw,
                              bool count_include_pad) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * H * W;
            T* yb = y + (b * C + c) * OH * OW;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    T sum = T{};
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            sum += xb[ih * W + iw];
                        }
                    }
                    const int denom = avg_pool_span(oh, H, KH, sh, ph, count_include_pad) *
                                      avg_pool_span(ow, W, KW, sw, pw, count_include_pad);
                    yb[oh * OW + ow] = denom > 0 ? sum / static_cast<T>(denom) : T{};
                }
            }
        }
    }
}

template <typename T>
void avg_pool2d_backward_typed(const T* g,
                               T* dx,
                               int B,
                               int C,
                               int H,
                               int W,
                               int KH,
                               int KW,
                               int OH,
                               int OW,
                               int sh,
                               int sw,
                               int ph,
                               int pw,
                               bool count_include_pad) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OH * OW;
            T* dxb = dx + (b * C + c) * H * W;
            for (int oh = 0; oh < OH; ++oh) {
                for (int ow = 0; ow < OW; ++ow) {
                    const int denom = avg_pool_span(oh, H, KH, sh, ph, count_include_pad) *
                                      avg_pool_span(ow, W, KW, sw, pw, count_include_pad);
                    if (denom <= 0)
                        continue;
                    const T scaled = gb[oh * OW + ow] / static_cast<T>(denom);
                    for (int kh = 0; kh < KH; ++kh) {
                        const int ih = oh * sh - ph + kh;
                        if (ih < 0 || ih >= H)
                            continue;
                        for (int kw = 0; kw < KW; ++kw) {
                            const int iw = ow * sw - pw + kw;
                            if (iw < 0 || iw >= W)
                                continue;
                            dxb[ih * W + iw] += scaled;
                        }
                    }
                }
            }
        }
    }
}

template <typename T>
void max_pool1d_forward_typed(
    const T* x, T* y, std::int32_t* argmax, int B, int C, int L, int KL, int OL, int sl, int pl) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * L;
            T* yb = y + (b * C + c) * OL;
            auto* ab = argmax + (b * C + c) * OL;
            for (int ol = 0; ol < OL; ++ol) {
                T best = NEG_INF;
                int best_idx = -1;
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    update_max_pool_best(xb[il], il, best, best_idx);
                }
                yb[ol] = best;
                ab[ol] = best_idx;
            }
        }
    }
}

template <typename T>
void max_pool1d_backward_typed(
    const T* g, const std::int32_t* argmax, T* dx, int B, int C, int L, int OL) {
    max_pool_scatter_backward(g, argmax, dx, B * C, L, OL);
}

template <typename T>
void avg_pool1d_forward_typed(
    const T* x, T* y, int B, int C, int L, int KL, int OL, int sl, int pl, bool count_include_pad) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * L;
            T* yb = y + (b * C + c) * OL;
            for (int ol = 0; ol < OL; ++ol) {
                T sum = T{};
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    sum += xb[il];
                }
                const int denom = avg_pool_span(ol, L, KL, sl, pl, count_include_pad);
                yb[ol] = denom > 0 ? sum / static_cast<T>(denom) : T{};
            }
        }
    }
}

template <typename T>
void avg_pool1d_backward_typed(const T* g,
                               T* dx,
                               int B,
                               int C,
                               int L,
                               int KL,
                               int OL,
                               int sl,
                               int pl,
                               bool count_include_pad) {
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OL;
            T* dxb = dx + (b * C + c) * L;
            for (int ol = 0; ol < OL; ++ol) {
                const int denom = avg_pool_span(ol, L, KL, sl, pl, count_include_pad);
                if (denom <= 0)
                    continue;
                const T scaled = gb[ol] / static_cast<T>(denom);
                for (int kl = 0; kl < KL; ++kl) {
                    const int il = ol * sl - pl + kl;
                    if (il < 0 || il >= L)
                        continue;
                    dxb[il] += scaled;
                }
            }
        }
    }
}

template <typename T>
void max_pool3d_forward_typed(const T* x,
                              T* y,
                              std::int32_t* argmax,
                              int B,
                              int C,
                              int D,
                              int H,
                              int W,
                              int KD,
                              int KH,
                              int KW,
                              int OD,
                              int OH,
                              int OW,
                              int sd,
                              int sh,
                              int sw,
                              int pd,
                              int ph,
                              int pw) {
    constexpr T NEG_INF = -std::numeric_limits<T>::infinity();
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * D * HW;
            T* yb = y + (b * C + c) * OD * OHW;
            auto* ab = argmax + (b * C + c) * OD * OHW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        T best = NEG_INF;
                        int best_idx = -1;
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    const int idx = (id * H + ih) * W + iw;
                                    update_max_pool_best(xb[idx], idx, best, best_idx);
                                }
                            }
                        }
                        const int oidx = (od * OH + oh) * OW + ow;
                        yb[oidx] = best;
                        ab[oidx] = best_idx;
                    }
                }
            }
        }
    }
}

template <typename T>
void max_pool3d_backward_typed(const T* g,
                               const std::int32_t* argmax,
                               T* dx,
                               int B,
                               int C,
                               int D,
                               int H,
                               int W,
                               int OD,
                               int OH,
                               int OW) {
    max_pool_scatter_backward(g, argmax, dx, B * C, D * H * W, OD * OH * OW);
}

template <typename T>
void avg_pool3d_forward_typed(const T* x,
                              T* y,
                              int B,
                              int C,
                              int D,
                              int H,
                              int W,
                              int KD,
                              int KH,
                              int KW,
                              int OD,
                              int OH,
                              int OW,
                              int sd,
                              int sh,
                              int sw,
                              int pd,
                              int ph,
                              int pw,
                              bool count_include_pad) {
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* xb = x + (b * C + c) * D * HW;
            T* yb = y + (b * C + c) * OD * OHW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        T sum = T{};
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    sum += xb[(id * H + ih) * W + iw];
                                }
                            }
                        }
                        const int denom = avg_pool_span(od, D, KD, sd, pd, count_include_pad) *
                                          avg_pool_span(oh, H, KH, sh, ph, count_include_pad) *
                                          avg_pool_span(ow, W, KW, sw, pw, count_include_pad);
                        yb[(od * OH + oh) * OW + ow] =
                            denom > 0 ? sum / static_cast<T>(denom) : T{};
                    }
                }
            }
        }
    }
}

template <typename T>
void avg_pool3d_backward_typed(const T* g,
                               T* dx,
                               int B,
                               int C,
                               int D,
                               int H,
                               int W,
                               int KD,
                               int KH,
                               int KW,
                               int OD,
                               int OH,
                               int OW,
                               int sd,
                               int sh,
                               int sw,
                               int pd,
                               int ph,
                               int pw,
                               bool count_include_pad) {
    const int HW = H * W;
    const int OHW = OH * OW;
    for (int b = 0; b < B; ++b) {
        for (int c = 0; c < C; ++c) {
            const T* gb = g + (b * C + c) * OD * OHW;
            T* dxb = dx + (b * C + c) * D * HW;
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const int denom = avg_pool_span(od, D, KD, sd, pd, count_include_pad) *
                                          avg_pool_span(oh, H, KH, sh, ph, count_include_pad) *
                                          avg_pool_span(ow, W, KW, sw, pw, count_include_pad);
                        if (denom <= 0)
                            continue;
                        const T scaled = gb[(od * OH + oh) * OW + ow] / static_cast<T>(denom);
                        for (int kd = 0; kd < KD; ++kd) {
                            const int id = od * sd - pd + kd;
                            if (id < 0 || id >= D)
                                continue;
                            for (int kh = 0; kh < KH; ++kh) {
                                const int ih = oh * sh - ph + kh;
                                if (ih < 0 || ih >= H)
                                    continue;
                                for (int kw = 0; kw < KW; ++kw) {
                                    const int iw = ow * sw - pw + kw;
                                    if (iw < 0 || iw >= W)
                                        continue;
                                    dxb[(id * H + ih) * W + iw] += scaled;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

}  // namespace

void max_pool1d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl) {
    max_pool1d_forward_typed<float>(x, y, a, B, C, L, KL, OL, sl, pl);
}
void max_pool1d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl) {
    max_pool1d_forward_typed<double>(x, y, a, B, C, L, KL, OL, sl, pl);
}
void max_pool1d_backward_f32(
    const float* g, const std::int32_t* a, float* dx, int B, int C, int L, int OL) {
    max_pool1d_backward_typed<float>(g, a, dx, B, C, L, OL);
}
void max_pool1d_backward_f64(
    const double* g, const std::int32_t* a, double* dx, int B, int C, int L, int OL) {
    max_pool1d_backward_typed<double>(g, a, dx, B, C, L, OL);
}
void avg_pool1d_forward_f32(const float* x,
                            float* y,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl,
                            bool count_include_pad) {
    avg_pool1d_forward_typed<float>(x, y, B, C, L, KL, OL, sl, pl, count_include_pad);
}
void avg_pool1d_forward_f64(const double* x,
                            double* y,
                            int B,
                            int C,
                            int L,
                            int KL,
                            int OL,
                            int sl,
                            int pl,
                            bool count_include_pad) {
    avg_pool1d_forward_typed<double>(x, y, B, C, L, KL, OL, sl, pl, count_include_pad);
}
void avg_pool1d_backward_f32(const float* g,
                             float* dx,
                             int B,
                             int C,
                             int L,
                             int KL,
                             int OL,
                             int sl,
                             int pl,
                             bool count_include_pad) {
    avg_pool1d_backward_typed<float>(g, dx, B, C, L, KL, OL, sl, pl, count_include_pad);
}
void avg_pool1d_backward_f64(const double* g,
                             double* dx,
                             int B,
                             int C,
                             int L,
                             int KL,
                             int OL,
                             int sl,
                             int pl,
                             bool count_include_pad) {
    avg_pool1d_backward_typed<double>(g, dx, B, C, L, KL, OL, sl, pl, count_include_pad);
}

void max_pool3d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    max_pool3d_forward_typed<float>(x, y, a, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                    ph, pw);
}
void max_pool3d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw) {
    max_pool3d_forward_typed<double>(x, y, a, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw);
}
void max_pool3d_backward_f32(const float* g,
                             const std::int32_t* a,
                             float* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int OD,
                             int OH,
                             int OW) {
    max_pool3d_backward_typed<float>(g, a, dx, B, C, D, H, W, OD, OH, OW);
}
void max_pool3d_backward_f64(const double* g,
                             const std::int32_t* a,
                             double* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int OD,
                             int OH,
                             int OW) {
    max_pool3d_backward_typed<double>(g, a, dx, B, C, D, H, W, OD, OH, OW);
}
void avg_pool3d_forward_f32(const float* x,
                            float* y,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw,
                            bool count_include_pad) {
    avg_pool3d_forward_typed<float>(x, y, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd, ph,
                                    pw, count_include_pad);
}
void avg_pool3d_forward_f64(const double* x,
                            double* y,
                            int B,
                            int C,
                            int D,
                            int H,
                            int W,
                            int KD,
                            int KH,
                            int KW,
                            int OD,
                            int OH,
                            int OW,
                            int sd,
                            int sh,
                            int sw,
                            int pd,
                            int ph,
                            int pw,
                            bool count_include_pad) {
    avg_pool3d_forward_typed<double>(x, y, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw, count_include_pad);
}
void avg_pool3d_backward_f32(const float* g,
                             float* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int KD,
                             int KH,
                             int KW,
                             int OD,
                             int OH,
                             int OW,
                             int sd,
                             int sh,
                             int sw,
                             int pd,
                             int ph,
                             int pw,
                             bool count_include_pad) {
    avg_pool3d_backward_typed<float>(g, dx, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                     ph, pw, count_include_pad);
}
void avg_pool3d_backward_f64(const double* g,
                             double* dx,
                             int B,
                             int C,
                             int D,
                             int H,
                             int W,
                             int KD,
                             int KH,
                             int KW,
                             int OD,
                             int OH,
                             int OW,
                             int sd,
                             int sh,
                             int sw,
                             int pd,
                             int ph,
                             int pw,
                             bool count_include_pad) {
    avg_pool3d_backward_typed<double>(g, dx, B, C, D, H, W, KD, KH, KW, OD, OH, OW, sd, sh, sw, pd,
                                      ph, pw, count_include_pad);
}

void max_pool2d_forward_f32(const float* x,
                            float* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    max_pool2d_forward_typed<float>(x, y, a, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void max_pool2d_forward_f64(const double* x,
                            double* y,
                            std::int32_t* a,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw) {
    max_pool2d_forward_typed<double>(x, y, a, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw);
}
void max_pool2d_backward_f32(
    const float* g, const std::int32_t* a, float* dx, int B, int C, int H, int W, int OH, int OW) {
    max_pool2d_backward_typed<float>(g, a, dx, B, C, H, W, OH, OW);
}
void max_pool2d_backward_f64(const double* g,
                             const std::int32_t* a,
                             double* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int OH,
                             int OW) {
    max_pool2d_backward_typed<double>(g, a, dx, B, C, H, W, OH, OW);
}
void avg_pool2d_forward_f32(const float* x,
                            float* y,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw,
                            bool count_include_pad) {
    avg_pool2d_forward_typed<float>(x, y, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw,
                                    count_include_pad);
}
void avg_pool2d_forward_f64(const double* x,
                            double* y,
                            int B,
                            int C,
                            int H,
                            int W,
                            int KH,
                            int KW,
                            int OH,
                            int OW,
                            int sh,
                            int sw,
                            int ph,
                            int pw,
                            bool count_include_pad) {
    avg_pool2d_forward_typed<double>(x, y, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw,
                                     count_include_pad);
}
void avg_pool2d_backward_f32(const float* g,
                             float* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int KH,
                             int KW,
                             int OH,
                             int OW,
                             int sh,
                             int sw,
                             int ph,
                             int pw,
                             bool count_include_pad) {
    avg_pool2d_backward_typed<float>(g, dx, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw,
                                     count_include_pad);
}
void avg_pool2d_backward_f64(const double* g,
                             double* dx,
                             int B,
                             int C,
                             int H,
                             int W,
                             int KH,
                             int KW,
                             int OH,
                             int OW,
                             int sh,
                             int sw,
                             int ph,
                             int pw,
                             bool count_include_pad) {
    avg_pool2d_backward_typed<double>(g, dx, B, C, H, W, KH, KW, OH, OW, sh, sw, ph, pw,
                                      count_include_pad);
}

}  // namespace lucid::backend::cpu
