// lucid/_C/core/Device.h
//
// Engine-side device tag identifying which physical processing unit owns a
// tensor's storage.
//
// Lucid is Apple Silicon-only, so there are exactly two device targets and
// they map one-to-one onto Lucid's two compute streams:
//
//   * ``CPU`` — Apple Accelerate (vDSP / vForce / BLAS / LAPACK) operating
//     on posix_memalign-backed host buffers.
//   * ``GPU`` — MLX lazy graph executing on the Metal GPU.
//
// The Python-side :class:`lucid.device` is a thin wrapper around this enum;
// :attr:`Tensor.device` ultimately reads from :class:`TensorImpl`'s
// :class:`TensorMeta::device` field.
//
// See Also
// --------
// :class:`Dtype`      — element type tag.
// :class:`TensorImpl` — owner of the per-tensor :class:`Device` value.

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string_view>

namespace lucid {

// Memory domain that owns a tensor's underlying storage.
//
// The underlying ``uint8_t`` representation is stable across releases: the
// enumerator values are serialised into checkpoints and crossed across the
// Python/C++ ABI boundary, so reordering or inserting in the middle would
// silently break old archives.  Append new entries at the end only.
//
// Attributes
// ----------
// CPU : Device
//     Host memory, directly addressable from any thread via a raw pointer.
//     All Apple Accelerate / BLAS / vDSP / LAPACK operations dispatch here.
//     The CPU stream also services the linalg carve-out where MLX is run
//     against CPU storage (MLX-on-CPU).
// GPU : Device
//     MLX lazy graph node executing on the Metal GPU.  Buffers are
//     GPU-private — dereferencing the underlying pointer from the CPU
//     before graph evaluation causes a SIGBUS.  See :func:`TensorImpl::eval`
//     for the explicit materialisation entry point.
//
// Notes
// -----
// The carve-outs of the strict CPU=Accelerate / GPU=MLX split are documented
// in Lucid's hard rule H3: linalg dispatches MLX on CPU storage, and
// data-dependent output shapes round-trip through the CPU stream regardless
// of the input device.
enum class Device : std::uint8_t {
    CPU,
    GPU,
};

// Returns the canonical lower-case name for a device tag.
//
// The string form is what the engine's error messages print, so it is the
// public device name: ``metal``, as ``tensor.device`` says, not the enum's.
//
// Parameters
// ----------
// d : Device
//     Device tag to convert.
//
// Returns
// -------
// std::string_view
//     ``"cpu"`` for :attr:`Device::CPU`, ``"metal"`` for :attr:`Device::GPU`.
//     The returned view points into a static string literal — safe to store
//     beyond the call site.
//
// Raises
// ------
// std::logic_error
//     The enum was extended without updating this switch.  Indicates a
//     missing recompile of all translation units.
//
// Examples
// --------
// ::
//
//     const auto name = device_name(Device::GPU);   // "metal"
constexpr std::string_view device_name(Device d) {
    switch (d) {
    case Device::CPU:
        return "cpu";
    case Device::GPU:
        return "metal";
    }
    throw std::logic_error("device_name: unknown Device");
}

}  // namespace lucid
