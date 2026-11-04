// lucid/_C/core/ErrorBuilder.cpp
//
// Implementation of ErrorContext and ErrorBuilder.
//
// The call-stack state is stored in a thread_local std::vector<std::string>.
// Using thread_local avoids any locking; each thread's stack is independent.
// The stack is intentionally simple (no capacity limit) because deeply nested
// op call chains are uncommon and each name is typically a short string.

#include "ErrorBuilder.h"

#include <cstdlib>
#include <string_view>
#include <vector>

namespace lucid {

namespace {

// Thread-local call stack of operation names.
thread_local std::vector<std::string> tl_stack_;

bool trace_enabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("LUCID_ERROR_TRACE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return enabled;
}

std::string canonical_op_name(std::string_view name) {
    if (const auto pos = name.rfind("::"); pos != std::string_view::npos)
        name = name.substr(pos + 2);
    if (name.starts_with("linalg_"))
        name = name.substr(7);
    if (name == "unfold_dim")
        return "unfold";
    if (name == "transpose_dim")
        return "transpose";
    if (name == "scatter_set")
        return "scatter";
    if (name == "from_string")
        return "device";
    return std::string(name);
}

std::string resolve_op_name(const std::string& op) {
    if (!tl_stack_.empty() &&
        (op.find("::") != std::string::npos || op == "unfold_dim" || op == "transpose_dim" ||
         op == "scatter_set" || op == "from_string" ||
         (op == "gather" &&
          (tl_stack_.front() == "take" || tl_stack_.front() == "index_select")))) {
        return canonical_op_name(tl_stack_.front());
    }
    return canonical_op_name(op);
}

}  // namespace

void ErrorContext::push(std::string op_name) {
    tl_stack_.emplace_back(std::move(op_name));
}

void ErrorContext::pop() {
    if (!tl_stack_.empty())
        tl_stack_.pop_back();
}

// Joins all entries on the stack with " > " as the separator.
// Returns an empty string when the stack is empty so that callers can
// omit the trace annotation entirely in the common no-context case.
std::string ErrorContext::trace() {
    if (tl_stack_.empty())
        return {};
    std::string out;
    for (std::size_t i = 0; i < tl_stack_.size(); ++i) {
        if (i)
            out += " > ";
        out += tl_stack_[i];
    }
    return out;
}

void ErrorContext::reset() {
    tl_stack_.clear();
}

// The op is the innermost frame — the op whose entry pushed it is the one
// about to build the tensor; a call from outside any op names none.
void throw_negative_dim(const Shape& shape) {
    std::string dims;
    std::int64_t bad = 0;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i)
            dims += ", ";
        dims += std::to_string(shape[i]);
        if (shape[i] < 0 && bad == 0)
            bad = shape[i];
    }
    std::string msg = tl_stack_.empty() ? std::string{} : tl_stack_.back() + ": ";
    msg += "Trying to create tensor with negative dimension " + std::to_string(bad) + ": [" + dims +
           "]";
    throw InvalidArgument(std::move(msg));
}

void throw_numel_overflow(const Shape& shape) {
    std::string dims;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i)
            dims += ", ";
        dims += std::to_string(shape[i]);
    }
    std::string msg = tl_stack_.empty() ? std::string{} : tl_stack_.back() + ": ";
    msg += "Storage size calculation overflowed with sizes=[" + dims +
           "] (a tensor holds at most " + std::to_string(kMaxNumel) +
           " elements, and its strides are 64-bit byte offsets)";
    throw InvalidArgument(std::move(msg));
}

void throw_extent_overflow(std::int64_t a, std::int64_t b) {
    std::string msg = tl_stack_.empty() ? std::string{} : tl_stack_.back() + ": ";
    msg += "Storage size calculation overflowed: an extent of " + std::to_string(a) + " * " +
           std::to_string(b) + " does not fit in 64 bits";
    throw InvalidArgument(std::move(msg));
}

ErrorContextGuard::ErrorContextGuard(std::string op_name) {
    ErrorContext::push(std::move(op_name));
}

ErrorContextGuard::~ErrorContextGuard() {
    ErrorContext::pop();
}

// Produces a string of the form "op_name: msg" (plus " [trace=A > B > C]" when
// LUCID_ERROR_TRACE is enabled).
std::string ErrorBuilder::format_with_context(const std::string& msg) const {
    std::string out = resolve_op_name(op_);
    out += ": ";
    out += msg;
    if (trace_enabled()) {
        const std::string trace = ErrorContext::trace();
        if (!trace.empty()) {
            out += " [trace=";
            out += trace;
            out += "]";
        }
    }
    return out;
}

void ErrorBuilder::fail(const std::string& msg) const {
    throw LucidError(format_with_context(msg));
}

void ErrorBuilder::not_implemented(const std::string& msg) const {
    throw NotImplementedError(format_with_context(msg));
}

void ErrorBuilder::index_error(const std::string& msg) const {
    throw IndexError(format_with_context(msg));
}

void ErrorBuilder::invalid_argument(const std::string& msg) const {
    throw InvalidArgument(format_with_context(msg));
}

void ErrorBuilder::linalg_error(const std::string& msg) const {
    throw LinAlgError(format_with_context(msg));
}

// For typed mismatch throws, the context string is built separately so the
// ShapeMismatch / DtypeMismatch / DeviceMismatch constructors receive a fully
// qualified context (op + detail + optional trace) as their context argument.
void ErrorBuilder::shape_mismatch(const Shape& expected,
                                  const Shape& got,
                                  const std::string& detail) const {
    std::string ctx = resolve_op_name(op_);
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    if (trace_enabled()) {
        const std::string trace = ErrorContext::trace();
        if (!trace.empty()) {
            ctx += " [trace=";
            ctx += trace;
            ctx += "]";
        }
    }
    throw ShapeMismatch(expected, got, std::move(ctx));
}

void ErrorBuilder::dtype_mismatch(Dtype expected, Dtype got, const std::string& detail) const {
    std::string ctx = resolve_op_name(op_);
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    if (trace_enabled()) {
        const std::string trace = ErrorContext::trace();
        if (!trace.empty()) {
            ctx += " [trace=";
            ctx += trace;
            ctx += "]";
        }
    }
    throw DtypeMismatch(std::string(dtype_name(expected)), std::string(dtype_name(got)),
                        std::move(ctx));
}

void ErrorBuilder::device_mismatch(Device expected, Device got, const std::string& detail) const {
    std::string ctx = resolve_op_name(op_);
    if (!detail.empty()) {
        ctx += ": ";
        ctx += detail;
    }
    if (trace_enabled()) {
        const std::string trace = ErrorContext::trace();
        if (!trace.empty()) {
            ctx += " [trace=";
            ctx += trace;
            ctx += "]";
        }
    }
    throw DeviceMismatch(std::string(device_name(expected)), std::string(device_name(got)),
                         std::move(ctx));
}

void refuse_metal_dtype(Dtype dt) {
    const ErrorBuilder err("to_mlx_dtype");
    if (dt == Dtype::C128)
        err.not_implemented("complex128 is not supported on GPU (MLX has no complex128). "
                            "Cast to complex64 first, or keep the tensor on CPU. "
                            "linalg.eig on an f64 matrix produces one — it is the only "
                            "dtype wide enough to hold its eigenvalues.");
    err.not_implemented("float64 is not supported on GPU (MLX-Metal limitation). "
                        "Cast to float32 first, or keep the tensor on CPU.");
}

}  // namespace lucid
