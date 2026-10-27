"""Models that draw random numbers, and how a package can hold one.

Core ML has no random operation. A draw takes no inputs, so it is a
constant as far as the compiler is concerned, and it folds at build
time — the package returns one sample for the life of the file. That is
refused by default, and the refusal is the most important one this
subsystem makes: nothing about such a package looks wrong. It loads, the
numbers are plausible, and a variational encoder's latent never moves.

It also stopped seven families in the zoo from being exported at all,
which is the largest single blocker measured: variational encoders,
world models with a stochastic latent, score-based samplers.

``Draws.AS_INPUT`` lifts each draw to an input the caller fills. Nothing
is lost by it, because the draw was never part of the network:
``mu + sigma * eps`` is a function of ``eps``, and that the eager model
makes its own on the way past is an artifact of running in Python. The
handle draws for a caller who passes nothing, so the package still
answers differently on each prediction the way the model does; a caller
who passes one gets a deterministic function, which is the other reason
to want this.

Comparing one against its eager model needs care. Running the model
again draws different numbers, so its values cannot be the reference —
the two sides would be computing different things and the difference
would be the draw. ``verify`` feeds the package the sample the trace
drew and compares against what the trace answered, which is the only
pairing where both sides saw the same numbers. A reopened handle does
not have them and says so.
"""

import pytest

import lucid
import lucid.coreml as cml
import lucid.nn as nn
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


class _Reparameterised(nn.Module):
    """A variational encoder's trick, which is the case this is for."""

    def __init__(self) -> None:
        super().__init__()
        self.mu = nn.Linear(8, 4)
        self.logvar = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        mu, logvar = self.mu(x), self.logvar(x)
        return mu + (logvar * 0.5).exp() * lucid.randn(1, 4)


class _TwoDraws(nn.Module):
    """One normal and one uniform, which must not be filled alike."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        return self.fc(x) + lucid.randn(1, 4) * 0.0 + lucid.rand(1, 4)


def _sampler(tmp_path, name="lifted", **kwargs):
    lucid.manual_seed(0)
    model = _Reparameterised().eval()
    x = lucid.randn(1, 8)
    exported = cml.export(
        model, x, f"{tmp_path}/{name}.mlpackage", draws=cml.Draws.AS_INPUT, **kwargs
    )
    return model, x, exported


class TestTheDefaultStillRefuses:
    def test_a_draw_is_refused_unless_asked_for(self, tmp_path: object) -> None:
        """The refusal is the safe direction and stays the default.

        A caller who has not thought about where the sample comes from
        should not get a package that answers with the same one forever.
        """
        lucid.manual_seed(0)
        with pytest.raises(cml.UnsupportedOp, match="fixed sample"):
            cml.export(
                _Reparameterised().eval(),
                lucid.randn(1, 8),
                f"{tmp_path}/refused.mlpackage",
            )


class TestADrawBecomesAnInput:
    def test_it_is_declared(self, tmp_path: object) -> None:
        _model, _x, exported = _sampler(tmp_path)
        try:
            assert exported.noise_inputs == [("noise_0", (1, 4))]
            assert exported.input_names == ["input", "noise_0"]
        finally:
            exported.close()

    def test_the_package_still_samples(self, tmp_path: object) -> None:
        """Two predictions of one input must differ.

        This is the whole difference from the folded package the refusal
        exists to prevent: that one would return the same numbers here.
        """
        _model, x, exported = _sampler(tmp_path, "samples")
        try:
            first, second = exported.predict(x), exported.predict(x)
            assert float((first - second).abs().max().item()) > 0.0
        finally:
            exported.close()

    def test_supplying_the_draw_makes_it_deterministic(self, tmp_path: object) -> None:
        """The other reason to want this: a reproducible sampler."""
        _model, x, exported = _sampler(tmp_path, "fixed")
        try:
            eps = lucid.randn(1, 4)
            first = exported.predict({"input": x, "noise_0": eps})
            second = exported.predict({"input": x, "noise_0": eps})
            assert float((first - second).abs().max().item()) == 0.0
        finally:
            exported.close()

    def test_a_uniform_draw_is_not_filled_with_a_normal_one(self, tmp_path: object) -> None:
        """Which operation drew is recorded, because it has to be.

        Filling ``rand``'s input from ``randn`` is a package that runs
        and is wrong — the samples are in the wrong range and nothing
        reports it. Over enough draws a uniform stays inside [0, 1) and
        a normal does not.
        """
        lucid.manual_seed(0)
        model = _TwoDraws().eval()
        x = lucid.zeros(1, 8)
        exported = cml.export(model, x, f"{tmp_path}/kinds.mlpackage", draws=cml.Draws.AS_INPUT)
        try:
            assert len(exported.noise_inputs) == 2
            bias = float(model.fc.bias.max().item())
            for _ in range(20):
                answered = exported.predict(x)
                # fc(0) + 0 * randn + rand, so what is left is the uniform.
                uniform = float((answered.max() - bias).item())
                assert -1e-4 <= uniform < 1.0 + 1e-4
        finally:
            exported.close()


class TestComparingOneAgainstItsModel:
    def test_it_agrees_with_what_the_trace_answered(self, tmp_path: object) -> None:
        model, x, exported = _sampler(tmp_path, "verified")
        try:
            assert exported.verify(model, x, relative=True) < 1e-5
        finally:
            exported.close()

    def test_a_reopened_handle_refuses_rather_than_guessing(self, tmp_path: object) -> None:
        """The samples are not written into the file.

        A reopened handle could run the eager model and compare — and
        would be measuring two different draws, which would read as a
        broken export at around the magnitude of the noise.
        """
        model, x, exported = _sampler(tmp_path, "reopened")
        path = exported.path
        exported.close()

        reopened = cml.load(path)
        try:
            with pytest.raises(ValueError, match="random draws as inputs"):
                reopened.verify(model, x)
        finally:
            reopened.close()

    def test_a_reopened_handle_still_predicts(self, tmp_path: object) -> None:
        """Only the comparison is lost, not the package."""
        _model, x, exported = _sampler(tmp_path, "reopened_predict")
        path = exported.path
        exported.close()

        reopened = cml.load(path)
        try:
            got = reopened.predict({"input": x, "noise_0": lucid.randn(1, 4)})
            assert tuple(got.shape) == (1, 4)
        finally:
            reopened.close()


class TestAReopenedHandleKnowsWhichInputsAreDraws:
    """Nothing in the file distinguishes them, so the export writes it.

    To Core ML a lifted draw is an ordinary input; there is no field
    saying otherwise. Without recording it the reopened package would ask
    its caller for a sample they did not know it wanted — and in a
    deployment, reopening is the ordinary path, so that is the shape most
    callers would meet. It goes in the creator-defined metadata, which
    Core ML carries untouched.
    """

    def test_it_reads_them_back(self, tmp_path: object) -> None:
        _model, _x, exported = _sampler(tmp_path, "declared")
        path = exported.path
        wanted = exported.noise_inputs
        exported.close()

        reopened = cml.load(path)
        try:
            assert reopened.noise_inputs == wanted
        finally:
            reopened.close()

    def test_it_draws_for_a_caller_who_passes_none(self, tmp_path: object) -> None:
        """The point of recording it: the package still behaves like one."""
        _model, x, exported = _sampler(tmp_path, "self_drawing")
        path = exported.path
        exported.close()

        reopened = cml.load(path)
        try:
            first, second = reopened.predict(x), reopened.predict(x)
            assert tuple(first.shape) == (1, 4)
            assert float((first - second).abs().max().item()) > 0.0
        finally:
            reopened.close()

    def test_the_distribution_survives_too(self, tmp_path: object) -> None:
        """A uniform draw filled from a normal is a package that is wrong.

        Recording the name without the operation that made it would do
        exactly that, and nothing would report it.
        """
        lucid.manual_seed(0)
        model = _TwoDraws().eval()
        x = lucid.zeros(1, 8)
        exported = cml.export(
            model, x, f"{tmp_path}/kinds_reopened.mlpackage", draws=cml.Draws.AS_INPUT
        )
        path = exported.path
        exported.close()

        reopened = cml.load(path)
        try:
            bias = float(model.fc.bias.max().item())
            for _ in range(20):
                uniform = float((reopened.predict(x).max() - bias).item())
                assert -1e-4 <= uniform < 1.0 + 1e-4
        finally:
            reopened.close()


class TestTheFamiliesThisUnblocks:
    """Measured, and the reason the option exists.

    Each of these refused outright before. They are checked against what
    their own trace answered, which is exact for a float32 export.
    """

    @pytest.mark.parametrize(
        ("factory", "make_inputs", "draws"),
        [
            ("vae", lambda: lucid.randn(1, 3, 32, 32), 1),
            ("hvae", lambda: lucid.randn(1, 3, 32, 32), 3),
            ("score_sde_ve", lambda: (lucid.randn(1, 3, 32, 32), lucid.zeros(1)), 1),
        ],
        ids=["vae", "hvae", "score_sde_ve"],
    )
    def test_a_generative_family_exports(self, factory, make_inputs, draws, tmp_path) -> None:
        import lucid.models as M

        lucid.manual_seed(0)
        model = M.create_model(factory).eval()
        inputs = make_inputs()
        exported = cml.export(
            model,
            inputs,
            f"{tmp_path}/{factory}.mlpackage",
            draws=cml.Draws.AS_INPUT,
        )
        try:
            assert len(exported.noise_inputs) == draws
            assert exported.verify(model, inputs, relative=True) < 1e-4
        finally:
            exported.close()
