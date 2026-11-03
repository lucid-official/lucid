// lucid/_C/core/GradMode.cpp
//
// Thread-local gradient mode storage and NoGradGuard implementation.
// g_grad_enabled starts as true in every new thread — the default assumption
// is that gradient computation is desired unless the caller opts out.

#include "GradMode.h"

namespace lucid {

namespace {
// One bool per thread.  No synchronisation needed because reads and writes
// only ever occur on the owning thread.
thread_local bool g_grad_enabled = true;
thread_local bool g_inference_enabled = false;
}  // namespace

bool GradMode::is_enabled() {
    return g_grad_enabled && !g_inference_enabled;
}

void GradMode::set_enabled(bool value) {
    g_grad_enabled = value;
}

bool GradMode::raw_enabled() {
    return g_grad_enabled;
}

bool GradMode::inference_enabled() {
    return g_inference_enabled;
}

void GradMode::set_inference_enabled(bool value) {
    g_inference_enabled = value;
}

NoGradGuard::NoGradGuard() : prev_(g_grad_enabled) {
    g_grad_enabled = false;
}

// Restore the mode that was active before this guard was constructed,
// correctly handling nested guards of the same or different kind.
NoGradGuard::~NoGradGuard() {
    g_grad_enabled = prev_;
}

GradModeGuard::GradModeGuard(bool enabled) : prev_(g_grad_enabled) {
    g_grad_enabled = enabled;
}

GradModeGuard::~GradModeGuard() {
    g_grad_enabled = prev_;
}

}  // namespace lucid
