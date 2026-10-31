"""lucid.optim — parameter update rules and learning-rate schedules.

Thirteen optimizers over the shared ``Optimizer`` base — ``SGD``, ``Adam``,
``AdamW``, ``LBFGS``, ``RMSprop``, ``Adagrad``, ``Adadelta``, ``Adamax``,
``RAdam``, ``NAdam``, ``ASGD``, ``Rprop``, ``SparseAdam`` — plus sixteen
schedulers in ``lucid.optim.lr_scheduler``: fourteen schedules and the
``SequentialLR`` / ``ChainedScheduler`` composites that wrap them.

Weight averaging (SWA and EMA) lives in ``lucid.optim.swa_utils``:
``AveragedModel``, ``SWALR``, ``update_bn`` and the ``get_*_avg_fn`` factories.

Updates run under ``lucid.no_grad()`` and write into the parameter in place.
That is load-bearing rather than incidental: assigning the result of a
*differentiable* op to a parameter makes it a non-leaf, after which backward
flows through it and accumulates nothing, and training silently becomes a
no-op that no test or loss curve reports as an error.

Checkpointing: ``Optimizer`` and every scheduler expose ``state_dict`` /
``load_state_dict``, so a resumed run continues the same schedule instead of
restarting it.  Schedulers deliberately do not serialise the callables
``LambdaLR`` and ``MultiplicativeLR`` hold — reconstruct those with the same
function, then load the rest of the state onto them.
"""

from lucid.optim.optimizer import Optimizer
from lucid.optim.sgd import SGD
from lucid.optim.adam import Adam, AdamW
from lucid.optim.lbfgs import LBFGS
from lucid.optim.others import (
    RMSprop,
    Adagrad,
    Adadelta,
    Adamax,
    RAdam,
    NAdam,
    ASGD,
    Rprop,
    SparseAdam,
)
from lucid.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    MultiStepLR,
    CosineAnnealingLR,
    LambdaLR,
    CyclicLR,
    ReduceLROnPlateau,
    NoamScheduler,
    MultiplicativeLR,
    LinearLR,
    ConstantLR,
    PolynomialLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    SequentialLR,
    ChainedScheduler,
)
from lucid.optim import swa_utils

__all__ = [
    "Optimizer",
    "SGD",
    "Adam",
    "AdamW",
    "LBFGS",
    "RMSprop",
    "Adagrad",
    "Adadelta",
    "Adamax",
    "RAdam",
    "NAdam",
    "ASGD",
    "Rprop",
    "SparseAdam",
    "StepLR",
    "ExponentialLR",
    "MultiStepLR",
    "CosineAnnealingLR",
    "LambdaLR",
    "CyclicLR",
    "ReduceLROnPlateau",
    "NoamScheduler",
    "MultiplicativeLR",
    "LinearLR",
    "ConstantLR",
    "PolynomialLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "SequentialLR",
    "ChainedScheduler",
]
