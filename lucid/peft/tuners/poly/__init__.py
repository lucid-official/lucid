r"""``lucid.peft.tuners.poly`` — Poly and Multi-Head Routing: low-rank skills mixed per task by a learned router.

Polytropon (Ponti et al., 2022, arXiv:2202.13914) gives every adapted linear layer ``n_skills``
low-rank adapters and a router with one learned logit vector per task.  A batch row's update is the
mixture of the skills its task's logits weight, :math:`y = Wx + \tfrac{1}{r}\,x\,A_t B_t` with
:math:`A_t = \sum_s m_{t,s}A_s` and likewise :math:`B_t`.  Multi-Head Routing (Caccia et al., 2022,
arXiv:2211.03831) routes each of ``n_splits`` blocks of the features separately.  The router logits
start near zero, so every skill is mixed equally, and :math:`B` starts at zero, so the adapted model
starts as the base.  Pick it for multitask fine-tuning where tasks should share some skills and a
new task can be learned by routing alone; each call names its tasks (``task_ids``).

The update depends on the task of each row, so it does not fold into the weights and nothing merges.
The layer type is ``Linear`` (an ``nn.Linear``); ``PolyConfig`` and ``PolyModel`` are here and
exported from ``lucid.peft``.  Importing the package registers the method under ``PeftType.POLY``.
"""

from lucid.peft.tuners.poly._config import PolyConfig
from lucid.peft.tuners.poly._layer import Linear, PolyLayer
from lucid.peft.tuners.poly._model import PolyModel

__all__ = ["Linear", "PolyConfig", "PolyLayer", "PolyModel"]
