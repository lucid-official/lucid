// lucid/_C/bindings/bind_random.cpp
//
// Exposes the random number generation subsystem to Python.  Registers:
//   - Generator class (Philox-based PRNG: seed, counter, next_uint32x4)
//   - default_generator() — returns the process-global Generator by reference
//   - set_deterministic() / is_deterministic() — global determinism flag
//   - Tensor-valued random ops: rand, uniform, randn, normal, randint, bernoulli
//
// All random ops accept an optional `generator` keyword argument.  Passing
// None selects default_generator(); passing an explicit Generator instance
// allows reproducible per-call sequences independent of global state.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../core/Determinism.h"
#include "../core/Generator.h"
#include "../core/Shape.h"
#include "../core/TensorImpl.h"
#include "../random/Random.h"

namespace py = pybind11;

namespace lucid::bindings {

namespace {

// Converts a Python list of int64 dimensions into the internal Shape type.
Shape vec_to_shape(const std::vector<std::int64_t>& v) {
    Shape s;
    s.reserve(v.size());
    for (auto d : v)
        s.push_back(d);
    return s;
}

// Converts a pybind11 object that is either None or a Generator* into a raw
// pointer acceptable by the random op functions.  nullptr means "use default".
Generator* resolve_gen(py::object gen_obj) {
    if (gen_obj.is_none())
        return nullptr;
    return gen_obj.cast<Generator*>();
}

}  // namespace

// Registers the Generator class, the global generator accessor, determinism
// controls, and all stochastic tensor-creation ops.
void register_random(py::module_& m) {
    // Generator wraps a Philox counter-based PRNG.  Python holds the object by
    // value (no shared_ptr); the caller is responsible for keeping the instance
    // alive when passing it to random ops.
    py::class_<Generator>(m, "Generator")
        .def(py::init<std::uint64_t>(), py::arg("seed") = 0)
        .def("set_seed", &Generator::set_seed, py::arg("seed"))
        .def_property_readonly("seed", &Generator::seed)
        // counter is read/write to allow checkpointing and restoring RNG state.
        .def_property("counter", &Generator::counter, &Generator::set_counter)
        .def("next_uniform_float", &Generator::next_uniform_float)
        // next_uint32x4 draws four raw uint32 values at once; returned as a
        // Python tuple because pybind11 cannot marshal a C array directly.
        .def("next_uint32x4",
             [](Generator& g) {
                 std::uint32_t out[4];
                 g.next_uint32x4(out);
                 return py::make_tuple(out[0], out[1], out[2], out[3]);
             })
        .def("__repr__", [](const Generator& g) {
            return "Generator(seed=" + std::to_string(g.seed()) +
                   ", counter=" + std::to_string(g.counter()) + ")";
        });

    // default_generator returns a borrowed reference to the static per-process
    // generator; Python must not delete it.
    m.def("default_generator", &default_generator, py::return_value_policy::reference);

    m.def("set_deterministic", &Determinism::set_enabled, py::arg("value"));

    m.def("is_deterministic", &Determinism::is_enabled);

    // Each random op converts the Python int-list shape to the internal Shape
    // type and resolves the optional generator before forwarding to the
    // corresponding C++ random op.
    m.def(
        "rand",
        [](std::vector<std::int64_t> shape, Dtype dt, Device device, py::object gen) {
            return rand_op(vec_to_shape(shape), dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("generator") = py::none(), "Uniform [0, 1) tensor.");

    m.def(
        "uniform",
        [](std::vector<std::int64_t> shape, double low, double high, Dtype dt, Device device,
           py::object gen) {
            return uniform_op(vec_to_shape(shape), low, high, dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("low"), py::arg("high"), py::arg("dtype") = Dtype::F32,
        py::arg("device") = Device::CPU, py::arg("generator") = py::none(),
        "Uniform [low, high) tensor.");

    m.def(
        "randn",
        [](std::vector<std::int64_t> shape, Dtype dt, Device device, py::object gen) {
            return randn_op(vec_to_shape(shape), dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("generator") = py::none(),
        "Standard normal N(0, 1) tensor (Box-Muller via Lucid Philox).");

    m.def(
        "normal",
        [](std::vector<std::int64_t> shape, double mean, double std, Dtype dt, Device device,
           py::object gen) {
            return normal_op(vec_to_shape(shape), mean, std, dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("mean") = 0.0, py::arg("std") = 1.0,
        py::arg("dtype") = Dtype::F32, py::arg("device") = Device::CPU,
        py::arg("generator") = py::none(), "Normal(mean, std) tensor.");

    m.def(
        "randint",
        [](std::vector<std::int64_t> shape, std::int64_t low, std::int64_t high, Dtype dt,
           Device device, py::object gen) {
            return randint_op(vec_to_shape(shape), low, high, dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("low"), py::arg("high"), py::arg("dtype") = Dtype::I64,
        py::arg("device") = Device::CPU, py::arg("generator") = py::none(),
        "Uniform integer in [low, high).");

    m.def(
        "bernoulli",
        [](std::vector<std::int64_t> shape, double p, Dtype dt, Device device, py::object gen) {
            return bernoulli_op(vec_to_shape(shape), p, dt, device, resolve_gen(gen));
        },
        py::arg("shape"), py::arg("p"), py::arg("dtype") = Dtype::F32,
        py::arg("device") = Device::CPU, py::arg("generator") = py::none(),
        "Bernoulli(p) tensor (each cell is 1.0 or 0.0 with probability p).");
}

}  // namespace lucid::bindings
