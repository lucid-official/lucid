// lucid/_C/bindings/bind_errors.cpp
//
// Maps every C++ exception type defined in core/Error.h to a corresponding
// Python exception class and installs a pybind11 exception translator so that
// C++ throws propagate across the language boundary as typed Python exceptions.
// All custom exception classes are subclasses of `LucidError` (itself a
// subclass of Python's `RuntimeError`) so that Python catch-all handlers for
// `RuntimeError` still work.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <variant>

#include "../core/Error.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// Creates a Python subclass of LucidError and exports it as m.<name>.
// ``__module__`` is given explicitly: ``type()`` otherwise records the
// module of its caller, which here is ``importlib._bootstrap``, and every
// traceback printed ``importlib._bootstrap.DtypeMismatch``.
//
// A class that shares its name with a builtin also derives from it, so
// ``except NotImplementedError`` and ``except IndexError`` (and
// ``LookupError``) catch the engine's as they would Python's own; before,
// they slipped past both.
py::object make_subclass(py::module_& m,
                         const py::object& builtins,
                         const py::object& lucid_error_cls,
                         const char* name,
                         py::object builtin_base = py::none()) {
    py::tuple bases = builtin_base.is_none()
                          ? py::tuple(py::make_tuple(lucid_error_cls))
                          : py::tuple(py::make_tuple(lucid_error_cls, builtin_base));
    py::dict ns;
    ns["__module__"] = m.attr("__name__");
    py::object cls = builtins.attr("type")(py::str(name), bases, ns);
    m.attr(name) = cls;
    return cls;
}

// LucidError's type object, for the translator below; set by register_errors.
PyObject* g_lucid_error_type = nullptr;

// A safety net, not a contract: an op that reads a storage kind the tensor
// does not have raises a LucidError rather than a bare RuntimeError.  The
// phantom contract is held before this, where an op reads or views a phantom
// (refuse_phantom_outside_mode); what lands here is an op that std::gets a
// saved phantom Storage under the mode without going through the dispatcher.
void translate_storage_mismatch(std::exception_ptr p) {
    try {
        if (p)
            std::rethrow_exception(p);
    } catch (const std::bad_variant_access& e) {
        const std::string msg = std::string("a tensor's storage is not the kind this op reads (a "
                                            "phantom tensor has no values to read): ") +
                                e.what();
        PyErr_SetString(g_lucid_error_type, msg.c_str());
    }
}

}  // namespace

// Registers the LucidError hierarchy as Python exception classes and installs
// the global exception translator.  Must be called before any op bindings so
// that errors thrown during subsequent registration are caught correctly.
void register_errors(py::module_& m) {
    // LucidError is the base; created via py::exception so pybind11 owns
    // the type object.  All sub-classes are created dynamically via the
    // Python type() builtin to avoid needing a separate py::exception<>
    // instantiation for each leaf type.  Static locals ensure the class
    // objects survive the module's lifetime.
    static py::object lucid_error_cls =
        py::exception<LucidError>(m, "LucidError", PyExc_RuntimeError);

    auto builtins = py::module_::import("builtins");

    // A refusal of one kind is catchable as the builtin a caller would reach
    // for, whichever layer raised it: a shape or device mismatch is a
    // ``ValueError`` (the Python layer raised that for the same refusal), a
    // dtype mismatch a ``TypeError``, an allocation that cannot be met a
    // ``MemoryError``.  Each stays a ``LucidError``, and so a
    // ``RuntimeError``, as the reference framework raises them.
    static py::object oom_cls =
        make_subclass(m, builtins, lucid_error_cls, "OutOfMemory", builtins.attr("MemoryError"));
    static py::object shape_mismatch_cls =
        make_subclass(m, builtins, lucid_error_cls, "ShapeMismatch", builtins.attr("ValueError"));
    static py::object dtype_mismatch_cls =
        make_subclass(m, builtins, lucid_error_cls, "DtypeMismatch", builtins.attr("TypeError"));
    static py::object device_mismatch_cls =
        make_subclass(m, builtins, lucid_error_cls, "DeviceMismatch", builtins.attr("ValueError"));
    static py::object version_mismatch_cls =
        make_subclass(m, builtins, lucid_error_cls, "VersionMismatch");
    static py::object gpu_unavailable_cls =
        make_subclass(m, builtins, lucid_error_cls, "GpuNotAvailable");
    static py::object index_error_cls =
        make_subclass(m, builtins, lucid_error_cls, "IndexError", builtins.attr("IndexError"));
    static py::object not_implemented_cls = make_subclass(
        m, builtins, lucid_error_cls, "NotImplementedError", builtins.attr("NotImplementedError"));
    static py::object invalid_argument_cls =
        make_subclass(m, builtins, lucid_error_cls, "InvalidArgument", builtins.attr("ValueError"));
    // A matrix the routine could not factor: the reference's class of the
    // same name is a plain ``RuntimeError``, and so is this one.
    static py::object linalg_error_cls = make_subclass(m, builtins, lucid_error_cls, "LinAlgError");
    linalg_error_cls.attr("__doc__") =
        "A matrix a linear-algebra routine could not factor: singular, not "
        "positive-definite, or an iteration that did not converge (LAPACK's "
        "``info > 0``).  A ``RuntimeError``; argument refusals are not this "
        "class.";

    // The translator is called for every active exception that crosses the
    // C++/Python boundary.  More-derived types are checked first so that they
    // shadow the base LucidError catch at the end.  Pybind11's own exceptions
    // (``py::builtin_exception``) are re-thrown unchanged; ``std::invalid_argument``
    // (raised by MLX when an array primitive refuses its arguments) becomes
    // ``InvalidArgument``, which is both a ``LucidError`` and a ``ValueError``.
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const py::builtin_exception&) {
            throw;
        } catch (const OutOfMemory& e) {
            PyErr_SetString(oom_cls.ptr(), e.what());
        } catch (const ShapeMismatch& e) {
            PyErr_SetString(shape_mismatch_cls.ptr(), e.what());
        } catch (const DtypeMismatch& e) {
            PyErr_SetString(dtype_mismatch_cls.ptr(), e.what());
        } catch (const DeviceMismatch& e) {
            PyErr_SetString(device_mismatch_cls.ptr(), e.what());
        } catch (const VersionMismatch& e) {
            PyErr_SetString(version_mismatch_cls.ptr(), e.what());
        } catch (const GpuNotAvailable& e) {
            PyErr_SetString(gpu_unavailable_cls.ptr(), e.what());
        } catch (const IndexError& e) {
            PyErr_SetString(index_error_cls.ptr(), e.what());
        } catch (const NotImplementedError& e) {
            PyErr_SetString(not_implemented_cls.ptr(), e.what());
        } catch (const InvalidArgument& e) {
            PyErr_SetString(invalid_argument_cls.ptr(), e.what());
        } catch (const LinAlgError& e) {
            PyErr_SetString(linalg_error_cls.ptr(), e.what());
        } catch (const LucidError& e) {
            PyErr_SetString(lucid_error_cls.ptr(), e.what());
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(invalid_argument_cls.ptr(), e.what());
        }
    });

    // Registered last, so pybind11 asks it first; anything else it rethrows
    // on to the translator above.
    g_lucid_error_type = lucid_error_cls.ptr();
    py::register_exception_translator(&translate_storage_mismatch);
}

}  // namespace lucid::bindings
