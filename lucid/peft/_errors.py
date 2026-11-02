"""The errors and the warning ``lucid.peft`` raises, rooted so a caller can catch the library's own."""


class PeftError(Exception):
    """The base of the errors ``lucid.peft`` raises on its own account.

    Catch it to handle any of them at once; the library's other errors are the
    builtin ones (``ValueError``, ``TypeError``, ``NotImplementedError``) it raises
    for a bad argument.

    Examples
    --------
    >>> from lucid.peft import NoMatchingPeftModuleError, PeftError
    >>> issubclass(NoMatchingPeftModuleError, PeftError)
    True
    """


class NoMatchingPeftModuleError(PeftError, ValueError):
    """An adapter's targets matched no module, or no parameter, of the model.

    Raised when ``target_modules`` (or ``target_parameters``, or
    ``modules_to_save``) selects nothing, which most often points at a
    misconfiguration.  It subclasses ``ValueError``, which was raised before it
    existed, so ``except ValueError`` keeps catching it; code that adds such an
    adapter on purpose can catch this class (or :class:`PeftError`) and go on.

    Examples
    --------
    >>> import lucid.nn as nn
    >>> from lucid.peft import LoraConfig, NoMatchingPeftModuleError, get_peft_model
    >>> try:
    ...     get_peft_model(nn.Sequential(nn.Linear(2, 2)), LoraConfig(target_modules=["missing"]))
    ... except NoMatchingPeftModuleError as error:
    ...     isinstance(error, ValueError)
    True
    """


class PeftWarning(UserWarning):
    """The base of the warnings ``lucid.peft`` issues on its own account.

    Examples
    --------
    >>> import warnings
    >>> from lucid.peft import PeftWarning
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     warnings.warn("an adapter is left unmerged", PeftWarning)
    >>> caught[0].category.__name__
    'PeftWarning'
    """
