// lucid/_C/version.h
//
// Compile-time version identifiers and runtime accessors for the Lucid
// engine shared library.
//
// The three integer macros (``LUCID_VERSION_MAJOR/MINOR/PATCH``) and
// the string macro (``LUCID_VERSION_STRING``) describe the
// human-facing SemVer of the engine; the integer ``LUCID_ABI_VERSION``
// is a separate, monotonically increasing counter that the Python
// bindings compare at import time against the value baked into the
// extension module so a mismatched header / library pair can fail
// loudly with a clear error rather than crashing later at an
// undefined-behaviour boundary.
//
// Notes
// -----
// SemVer convention used here:
//
//   * **MAJOR** — bumped on ABI-breaking changes (struct layout,
//     removed exports, altered calling conventions, dependency
//     pinning that excludes prior wheels).
//   * **MINOR** — bumped on backwards-compatible feature additions
//     (new ops, new public classes, new factory functions).
//   * **PATCH** — bumped on bug fixes that preserve both ABI and API.
//
// :macro:`LUCID_ABI_VERSION` is independent of the MAJOR component:
// it bumps **every** time the binary interface changes, even for
// internal layout shifts that keep the public API the same.  This
// catches header / library skew during local rebuilds where the
// SemVer might not have advanced.
//
// See Also
// --------
// :file:`lucid/_C/api.h` — symbol-visibility macros that gate the ABI.

#pragma once

#include "api.h"

// Semantic version components.  Increment MAJOR on ABI-breaking changes,
// MINOR for backwards-compatible feature additions, PATCH for bug fixes.

// Major component of the engine SemVer; bumped on ABI-breaking changes.
//
// Notes
// -----
// Set to ``0`` during the pre-1.0 development phase, signalling that
// the public ABI is still in flux.  Will increment to ``1`` when the
// engine reaches a frozen-interface milestone.
#define LUCID_VERSION_MAJOR 0

// Minor component of the engine SemVer; bumped on backwards-compatible
// feature additions.
//
// Notes
// -----
// Increments when new public symbols are added but existing ones keep
// the same signature and semantics; clients built against an older
// minor revision continue to work without recompilation.
#define LUCID_VERSION_MINOR 16

// Patch component of the engine SemVer; bumped on bug fixes that
// preserve both the public API and the ABI.
//
// Notes
// -----
// Resets to ``0`` whenever MAJOR or MINOR advances.
#define LUCID_VERSION_PATCH 1

// Human-readable version string embedded into the shared library.

// Human-readable SemVer string embedded into the shared library.
//
// Notes
// -----
// Format is ``"<major>.<minor>.<patch>[-<pre-release-tag>]"``.  The
// optional pre-release tag (``-dev``, ``-rc1`` …) marks unstable
// builds; release tarballs strip the tag.  Returned verbatim by
// :func:`version_string`.
#define LUCID_VERSION_STRING "0.16.1"

// Monotonically increasing ABI generation counter.  Bump this whenever the
// C++ binary interface changes in a way that makes old .so / .dylib objects
// incompatible with new headers (e.g., changed struct layout, removed
// symbols, altered calling conventions).

// Monotonically increasing ABI generation counter.
//
// Independent of :macro:`LUCID_VERSION_MAJOR` — bumps every time the
// binary interface shifts, including internal struct layout changes
// that do not warrant a SemVer major bump.  Compared at Python import
// time via :func:`abi_version` against the value compiled into the
// extension module; a mismatch raises a clear "header / library skew"
// error before any tensor is constructed.
//
// Notes
// -----
// Bump rules:
//
//   * Any change to a struct exported across the ABI boundary.
//   * Any addition / removal / signature change of a :macro:`LUCID_API`
//     function.
//   * Any change to virtual-table layout of an exported polymorphic
//     class.
//
// New features that are purely additive (new free functions, new
// classes, new enumerators appended to the end of an enum) still
// require a bump because their absence in an older library breaks
// the new extension module.
#define LUCID_ABI_VERSION 37

namespace lucid {

// Returns the engine's SemVer string at runtime.
//
// Returns
// -------
// const char*
//     Pointer to a static, null-terminated string equal to
//     :macro:`LUCID_VERSION_STRING`.  Lifetime is the program's.
//
// Notes
// -----
// Surfaced to Python as ``lucid.__version__`` and used in error
// messages that need to mention the running engine build.
LUCID_API const char* version_string();

// Returns the engine's SemVer major component at runtime.
//
// Returns
// -------
// int
//     Value of :macro:`LUCID_VERSION_MAJOR` at the time the engine
//     shared library was built.
//
// See Also
// --------
// :func:`version_minor` — minor component.
// :func:`version_patch` — patch component.
LUCID_API int version_major();

// Returns the engine's SemVer minor component at runtime.
//
// Returns
// -------
// int
//     Value of :macro:`LUCID_VERSION_MINOR` at the time the engine
//     shared library was built.
LUCID_API int version_minor();

// Returns the engine's SemVer patch component at runtime.
//
// Returns
// -------
// int
//     Value of :macro:`LUCID_VERSION_PATCH` at the time the engine
//     shared library was built.
LUCID_API int version_patch();

// Returns the engine's ABI generation counter at runtime.
//
// Returns
// -------
// int
//     Value of :macro:`LUCID_ABI_VERSION` at the time the engine
//     shared library was built.
//
// Notes
// -----
// The Python bindings call this on first import and compare it
// against the constant baked into the extension module's translation
// unit.  A mismatch indicates that the loaded ``.dylib`` was built
// from headers different from the ones the bindings saw, and the
// import is aborted with a diagnostic naming both versions.
//
// Examples
// --------
// Pseudo-code for the import-time check::
//
//     if (engine.abi_version() != LUCID_ABI_VERSION)
//         throw ImportError("Lucid ABI mismatch …");
LUCID_API int abi_version();

}  // namespace lucid
