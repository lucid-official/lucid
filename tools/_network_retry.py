"""Telling a dropped network from a wrong answer, and waiting one out.

``tools.check_pretrained_parity`` fetches two checkpoints per comparison, a
hundred and sixty times in one job.  A resolver that is gone for a few seconds
says nothing about any conversion, yet it was reported as one: on 2026-11-03
``vgg_16_cls`` and ``vgg_19_bn_cls`` failed with ``URLError: nodename nor
servname provided`` sixteen seconds before DNS came back, and that alone
turned the weekly run red.

Two things live here and nothing else: :func:`is_network_failure` decides
whether an exception is the network's doing, and :class:`Backoff` spends one
run-wide allowance of waiting on trying again.  Deciding what a persistent
failure *means* — it is not a parity result — stays with the tool.
"""

from collections.abc import Callable, Iterator, Sequence
import socket
import time
import urllib.error

#: Statuses below 500 that still mean "ask again": request timeout, too early
#: and rate limiting.  Every other 4xx is the server's considered answer — a
#: checkpoint that moved or went private — and retrying will not change it.
_RETRY_STATUSES = frozenset({408, 425, 429})


def _chain(exc: BaseException) -> Iterator[BaseException]:
    """``exc`` and what it was raised from, nearest first, as a traceback shows it.

    A loader that cannot reach its hub raises its own error *from* the
    transport's: transformers' ``OSError("We couldn't connect ...")`` sits on
    a ``LocalEntryNotFoundError`` on an ``httpx.ConnectError`` on the
    ``socket.gaierror`` that was the actual fault.
    """
    seen: set[int] = set()
    link: BaseException | None = exc
    while link is not None and id(link) not in seen:
        seen.add(id(link))
        yield link
        nearer = link.__cause__
        if nearer is None and not link.__suppress_context__:
            nearer = link.__context__
        link = nearer


def _status_of(exc: BaseException) -> int | None:
    """The HTTP status ``exc`` carries, whichever client raised it."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _is_transport_failure(exc: BaseException) -> bool:
    status = _status_of(exc)
    if status is not None:
        return status >= 500 or status in _RETRY_STATUSES
    if isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, socket.gaierror)):
        return True
    # A resolver failure that lost its class on the way up.  ``getaddrinfo``'s
    # EAI_NONAME is 8 on macOS; nothing here execs a file, so ENOEXEC (also 8)
    # cannot be what this is.
    return isinstance(exc, OSError) and exc.errno == socket.EAI_NONAME


def is_network_failure(exc: BaseException) -> bool:
    """Whether the network, not the code or the data, is what raised ``exc``.

    ``FileNotFoundError`` and its kin are deliberately not here: macOS numbers
    ``EAI_AGAIN`` 2, the same as ``ENOENT``, so matching on errno alone would
    call a missing ``hubconf.py`` a network outage.
    """
    return any(_is_transport_failure(link) for link in _chain(exc))


class Backoff:
    """Waits between attempts, out of one allowance for the whole run.

    Parameters
    ----------
    delays : sequence of float
        Seconds to wait before the first, second, ... retry; the number of
        retries a job gets is its length.
    allowance : float
        Seconds of waiting the run may spend in total.  An outage that never
        ends then costs this much on top of the run rather than a full set of
        ``delays`` for each of the hundred and sixty checkpoints, which would
        outlast the job's timeout.
    sleep : callable
        Called with each wait; tests pass a recorder so that nothing sleeps.
    """

    def __init__(
        self,
        delays: Sequence[float] = (5.0, 15.0, 45.0),
        allowance: float = 600.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._delays = tuple(delays)
        self._left = allowance
        self._sleep = sleep

    def wait(self, retry: int) -> bool:
        """Sleep before retry number ``retry`` (from 0); False when none is left."""
        if retry >= len(self._delays) or self._delays[retry] > self._left:
            return False
        delay = self._delays[retry]
        self._left -= delay
        self._sleep(delay)
        return True
