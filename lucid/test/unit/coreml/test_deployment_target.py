"""Which systems a package will run on, said out loud.

Three features move a program from the ``CoreML7`` opset to ``CoreML8``,
and until this existed they moved it silently: carrying state,
palettizing weights, and writing several entry points into one package.
A caller who needed iOS 17 got a package that loads on iOS 18 and
nowhere earlier, with nothing in the export saying so — the kind of
thing you find out from a device, after shipping.

Two directions, both here. Ask for a floor the package cannot meet and
the export stops, naming the feature that raised it and the argument
that asked for it. Ask for nothing and the package tells you what floor
it ended up with.

There is no target below ``IOS17`` because the emitters do not write
below it: ``gather`` and ``scatter`` carry ``validate_indices``, which
iOS 16 has no field for. That is a change to the emitters, not a missing
enum member, and the test at the end says so rather than leaving the
absence to be read as an oversight.
"""

import pytest

import lucid
import lucid.nn as nn
import lucid.coreml as cml
from lucid._C import engine as _C_engine

pytestmark = pytest.mark.skipif(
    not hasattr(_C_engine, "coreml"),
    reason="the engine was built without the Core ML writer",
)


def _small() -> nn.Module:
    """Small enough that nothing in it is worth compressing."""
    return nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).eval()


def _compressible() -> nn.Module:
    """Weights above the threshold, so a request to compress takes."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, padding=1),
    ).eval()


class _Carries(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("total", lucid.zeros(1, 3))

    def forward(self, x: lucid.Tensor) -> tuple[lucid.Tensor, lucid.Tensor]:
        running = x + self.total
        return running, running


class TestAPackageKnowsItsFloor:
    def test_an_ordinary_export_stays_where_the_writer_writes(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        exported = cml.export(_small(), lucid.randn(1, 3, 32, 32), f"{tmp_path}/plain.mlpackage")
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS17
        finally:
            exported.close()

    def test_palettization_raises_it_and_says_so(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        exported = cml.export(
            _compressible(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/palette.mlpackage",
            weights=cml.Palettize(bits=4),
        )
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS18
        finally:
            exported.close()

    def test_asking_for_palettization_is_not_the_same_as_getting_it(self, tmp_path: object) -> None:
        """The package answers, not the request.

        Weights below the threshold are left alone — the tables cost
        more than they save — so a model of small layers comes out a
        CoreML7 program however it was asked for. Reporting IOS18 there
        would tell somebody they need a newer system than they do.
        """
        lucid.manual_seed(0)
        exported = cml.export(
            _small(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/nothing_to_palettize.mlpackage",
            weights=cml.Palettize(bits=4),
        )
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS17
        finally:
            exported.close()

    def test_reopening_a_package_agrees_with_exporting_it(self, tmp_path: object) -> None:
        """One file, one answer.

        An exported handle knows its floor because the export settled
        it; a reopened one reads it off the program. They are the same
        package and must not disagree about it.
        """
        lucid.manual_seed(0)
        path = f"{tmp_path}/reopened.mlpackage"
        exported = cml.export(
            _compressible(),
            lucid.randn(1, 3, 32, 32),
            path,
            weights=cml.Palettize(bits=4),
        )
        written = exported.deployment_target
        exported.close()

        reopened = cml.load(path)
        try:
            assert reopened.deployment_target is written
        finally:
            reopened.close()

    def test_several_entry_points_raise_it(self, tmp_path: object) -> None:
        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 3)

            def forward(self, x: lucid.Tensor) -> lucid.Tensor:
                return self.fc(x)

        lucid.manual_seed(0)
        model = _Net().eval()
        handles = cml.export_functions(
            {"whole": (model, lucid.randn(1, 4)), "one": (model, lucid.randn(2, 4))},
            f"{tmp_path}/many.mlpackage",
        )
        try:
            assert all(
                handle.deployment_target is cml.DeploymentTarget.IOS18
                for handle in handles.values()
            )
        finally:
            for handle in handles.values():
                handle.close()


class TestAFloorItCannotMeetIsRefused:
    """Named, and while it is still a Python call.

    The message has to carry the feature and the argument: "IOS17 is not
    possible" leaves the caller to work out which of their options did
    it, and they may be using all three.
    """

    def test_palettization_against_ios17(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        with pytest.raises(ValueError) as excinfo:
            cml.export(
                _small(),
                lucid.randn(1, 3, 32, 32),
                f"{tmp_path}/refused.mlpackage",
                weights=cml.Palettize(bits=4),
                minimum_deployment_target=cml.DeploymentTarget.IOS17,
            )
        message = str(excinfo.value)
        assert "palettization" in message
        assert "Palettize" in message
        assert "IOS17" in message

    def test_state_against_ios17(self, tmp_path: object) -> None:
        lucid.manual_seed(0)
        with pytest.raises(ValueError) as excinfo:
            cml.export(
                _Carries().eval(),
                lucid.randn(1, 3),
                f"{tmp_path}/stateful.mlpackage",
                state=[cml.State(input="total", output="1")],
                minimum_deployment_target=cml.DeploymentTarget.IOS17,
            )
        assert "state" in str(excinfo.value)

    def test_asking_for_ios18_is_never_refused(self, tmp_path: object) -> None:
        """A caller who is already on the newer floor pays nothing.

        And may still get an older floor back. The target is a ceiling
        on what the caller can deploy to, not an instruction to use it:
        a package that came out CoreML7 runs on iOS 17 and everything
        after, which is a better answer than the one they asked for.
        """
        lucid.manual_seed(0)
        exported = cml.export(
            _small(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/eighteen.mlpackage",
            weights=cml.Palettize(bits=4),
            minimum_deployment_target=cml.DeploymentTarget.IOS18,
        )
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS17
        finally:
            exported.close()

    def test_asking_for_ios18_and_using_it(self, tmp_path: object) -> None:
        """The other half: a model that really does palettize."""
        lucid.manual_seed(0)
        exported = cml.export(
            _compressible(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/eighteen_used.mlpackage",
            weights=cml.Palettize(bits=4),
            minimum_deployment_target=cml.DeploymentTarget.IOS18,
        )
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS18
        finally:
            exported.close()

    def test_a_plain_export_accepts_the_lower_floor(self, tmp_path: object) -> None:
        """Nothing raised it, so nothing to refuse."""
        lucid.manual_seed(0)
        exported = cml.export(
            _small(),
            lucid.randn(1, 3, 32, 32),
            f"{tmp_path}/seventeen.mlpackage",
            minimum_deployment_target=cml.DeploymentTarget.IOS17,
        )
        try:
            assert exported.deployment_target is cml.DeploymentTarget.IOS17
        finally:
            exported.close()


def test_there_is_deliberately_no_target_below_ios17():
    """The floor of the enum is the floor of the emitters.

    An older target would need ``gather`` and ``scatter`` written
    without ``validate_indices``, which is an iOS 17 field. Adding a
    member here without doing that would promise a package this writer
    cannot produce.
    """
    assert {member.name for member in cml.DeploymentTarget} == {"IOS17", "IOS18"}
