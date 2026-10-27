"""What the machine running the Core ML tests can and cannot do.

A hosted CI runner is a virtual machine, and Core ML inside one is not
Core ML on a device: there is no Neural Engine, so every placement claim
reads 0.0.  Tests ask these questions directly rather than inferring the
machine from the answers they are there to check.
"""

import functools
import glob
import subprocess
from pathlib import Path

import pytest


@functools.cache
def under_hypervisor() -> bool:
    """Whether this process is running inside a virtual machine.

    ``kern.hv_vmm_present`` is 1 under a hypervisor and 0 on hardware.
    """
    try:
        done = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "kern.hv_vmm_present"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return done.stdout.strip() == "1"


@functools.cache
def has_neural_engine() -> bool:
    """Whether Core ML has a Neural Engine to place work on here.

    Asked of Core ML's own device list rather than read off a compute
    plan: a plan is what the placement tests check, and inferring the
    hardware from it would skip exactly the case that matters — an
    engine that is there and unused.
    """
    try:
        from coremltools.models.compute_device import (  # noqa: PLC0415
            MLComputeDevice,
            MLNeuralEngineComputeDevice,
        )
    except ImportError:
        return True  # cannot tell, so assert as if it were there
    return any(
        isinstance(device, MLNeuralEngineComputeDevice)
        for device in MLComputeDevice.get_all_compute_devices()
    )


def require_neural_engine() -> None:
    if not has_neural_engine():
        pytest.skip("no Neural Engine on this machine; a hosted CI runner is a VM")


#: Where Core ML keeps the bundles it specialises a loaded package into,
#: relative to the home it was given: one directory per load, under the
#: cache of whichever app the process is.
_BUNDLES = ("Library", "Caches", "*", "com.apple.e5rt.e5bundlecache", "*", "*")


def bundles_under(home: Path) -> set[str]:
    """Every bundle Core ML has specialised under ``home``."""
    return set(glob.glob(str(home.joinpath(*_BUNDLES))))
