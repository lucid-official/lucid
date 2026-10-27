"""Per-directory pytest config for ``lucid/test/parity/models/vision``.

Skips collection of parity tests that depend on model families which are
declared in the test tree but not yet implemented in ``lucid.models``.
Mirrors the same pattern used by ``lucid/test/unit/models/conftest.py``.
"""

import importlib

_PENDING_MODULES = {"test_inception_resnet_v2.py": ("lucid.models.vision.inception_resnet",)}

collect_ignore: list[str] = []
for _file, _modules in _PENDING_MODULES.items():
    for _mod in _modules:
        try:
            importlib.import_module(_mod)
        except ModuleNotFoundError:
            collect_ignore.append(_file)
            break
