#!/usr/bin/env python3
"""tools/check_weight_fit.py — assert every published checkpoint still fits
the factory that offers it.

A factory and the checkpoint it ships drift apart silently.  Nothing about
``num_encoder_layers=6`` looks wrong next to a config whose default is 0,
and the model builds, trains and exports either way — the only thing that
breaks is ``pretrained=True``, which no test exercised.  Three families
were shipping unloadable weights when this was written:

  * ``sk_resnet_18/34_cls`` — a paper floor (L=32) the checkpoints were
    trained without, so the two narrow stages wanted 32-wide attention
    against a 16-wide checkpoint.
  * ``maskformer_resnet50/101`` — a 6-layer transformer encoder the
    checkpoints have no weights for, contradicting the config's own
    comment.
  * ``resnest_200/269_cls`` — a dropout that moves the classifier to
    ``classifier.1``, against a checkpoint converted from a flat head.

**No weight data is downloaded.**  A safetensors file opens with a
length-prefixed JSON header naming every tensor and its shape, so two
HTTP range requests (a few KB) answer the question for a checkpoint of
any size.

**The factory is called, not imitated.**  An earlier version of this
check rebuilt the model itself and reported six false positives, because
factories do things a reimplementation does not: load into a submodule
(``model.dit``, ``model.diamond``), derive config from the entry
(per-game action counts), or switch config on whether overrides were
passed.  Here ``load_weight_entry`` is intercepted instead — the factory
runs its real path and hands over the very module the load targets, then
the download is skipped.

**Nothing is allocated either.**  Only names and shapes are compared, so
the factory runs under :func:`lucid.nn._shadow.shadow_alloc`, where every
tensor is shape metadata.  Building the real weights was nearly all of
this check's time — 67 minutes of a nightly run, BERT-large alone 16 s,
against milliseconds in shadow.  A factory shadow mode cannot follow is
built for real instead, and the run says which.  The headers are then
read over parallel connections.

What a pass does *not* mean
---------------------------
Structure, not identity.  This reads the header and compares names and
shapes; it never hashes the file.  A checkpoint replaced by a different
revision of the same architecture passes here and is caught only by
``_verify_sha256`` at download time, which needs all the bytes.

There is no cheap stand-in for that: the Hub's ``ETag`` is not the
content digest — checked against four entries that download and verify
cleanly, and it differed from the declared SHA-256 for every one.

And a fit says the weights *load*, not that they are any good.  No
accuracy is verified anywhere in this repository; see
``obsidian/api/api-python-models.md`` on registration not being
validation.

Run::

    python -m tools.check_weight_fit                   # every checkpoint
    python -m tools.check_weight_fit --family sknet    # one family
    python -m tools.check_weight_fit --model resnet_18_cls
    python -m tools.check_weight_fit --list            # print, do not check

Exit codes
----------
0 — every checkpoint fits, or none was reachable to test.
1 — at least one checkpoint does not fit its factory.
2 — the network could not be reached for some checkpoint (reported
    separately from a mismatch, because "unknown" is not "broken").
"""

import argparse
import json
import struct
import sys
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from typing import Iterator

warnings.filterwarnings("ignore")

import lucid.models  # noqa: F401  — populates the weights registry
import lucid.weights as weights_mod
from lucid.models import create_model
from lucid.nn import Module
from lucid.weights import WeightEntry, WeightsEnum
from lucid.models._registry import _REGISTRY
from lucid.nn._shadow import shadow_alloc
from lucid.weights._registry import _WEIGHTS_BY_MODEL

_TIMEOUT = 90

#: Connections reading headers at once.  Each read is two small range
#: requests, so the run waits on round trips, not on bandwidth.
_CONNECTIONS = 16

#: Tries per header before a network error counts as unreachable.
_ATTEMPTS = 3


def _normalise(shape: object) -> tuple[int, ...]:
    """Treat a 0-d scalar and a length-1 vector as the same thing.

    ``num_batches_tracked`` is a scalar in the model and is written as
    ``[1]`` by the safetensors converter.  ``load_state_dict`` accepts
    either, so a check that did not would flag every BatchNorm model.
    """
    dims = tuple(shape)  # type: ignore[call-overload]
    return () if dims in ((), (1,)) else dims


def _header(url: str) -> dict[str, object]:
    """Read a safetensors header without fetching the tensors."""
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        size = struct.unpack("<Q", response.read(8))[0]
    request = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + size - 1}"})
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        return json.loads(response.read(size).decode())


def _header_retried(url: str) -> dict[str, object]:
    """:func:`_header`, tried again after a network error.

    Sixteen connections at once meet the odd dropped one: a first run
    reported a V-JEPA 2 checkpoint unreachable on a single connect timeout
    that the next request did not repeat.  An HTTP error is an answer, not
    a dropped connection, and is not retried.
    """
    for attempt in range(1, _ATTEMPTS + 1):
        try:
            return _header(url)
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError):
            if attempt == _ATTEMPTS:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


@contextmanager
def _load_intercepted() -> Iterator[list[tuple[Module, WeightEntry]]]:
    """Run factories with the download-and-load step replaced by a note.

    What the factory passes here is the ground truth this check needs:
    the module that actually receives the checkpoint, after every
    config decision the factory made.
    """
    seen: list[tuple[Module, WeightEntry]] = []
    original = weights_mod.load_weight_entry

    def spy(
        model: Module, weights: WeightsEnum | WeightEntry, *, name: str, strict: bool = True
    ) -> object:
        entry = weights.entry if isinstance(weights, WeightsEnum) else weights
        seen.append((model, entry))
        return None

    weights_mod.load_weight_entry = spy  # type: ignore[assignment]
    try:
        yield seen
    finally:
        weights_mod.load_weight_entry = original  # type: ignore[assignment]


def _tags(enum: type[WeightsEnum]) -> list[str]:
    """Every real member, with ``DEFAULT`` folded into what it aliases."""
    return [tag for tag in enum.__members__ if tag != "DEFAULT"]


def _target(model_name: str, tag: str, *, shadow: bool) -> dict[str, object]:
    """What the factory loads for ``tag``, and the shapes it loads it into."""
    with _load_intercepted() as seen:
        try:
            with shadow_alloc() if shadow else nullcontext():
                create_model(model_name, pretrained=tag)
                if not seen:
                    return {"error": "the factory did not load any weights for this tag"}
                target, entry = seen[-1]
                wanted = {k: _normalise(v.shape) for k, v in target.state_dict().items()}
        except ValueError as exc:
            if not seen:
                # A factory may share an enum with a sibling and refuse the
                # sibling's tags — diamond and diamond_csgo do, because the
                # registry maps a factory to an enum and not to a subset of
                # its tags.  Refusing is the correct answer, so it is not a
                # failure; but it must be recorded, or a checkpoint every
                # factory declines would vanish from the report.
                return {"declined": str(exc)[:160]}
            return {"error": f"ValueError: {exc}"[:200]}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return {"entry": entry, "wanted": wanted}


def _build(model_name: str, tag: str) -> dict[str, object]:
    """:func:`_target` in shadow, or for real where shadow mode cannot follow.

    A refusal is retried for real too: shadow mode raising inside a
    factory would otherwise pass for the factory declining the tag.
    """
    built = _target(model_name, tag, shadow=True)
    if "entry" in built:
        return built
    built = _target(model_name, tag, shadow=False)
    if "entry" in built:
        built["real"] = True
    return built


def _read_headers(urls: list[str]) -> dict[str, dict[str, object] | str]:
    """Every URL's header, or why it could not be read, over parallel connections."""
    found: dict[str, dict[str, object] | str] = {}
    with ThreadPoolExecutor(max_workers=_CONNECTIONS) as pool:
        futures = {pool.submit(_header_retried, url): url for url in urls}
        for done, future in enumerate(as_completed(futures), 1):
            url = futures[future]
            try:
                found[url] = future.result()
            except (urllib.error.URLError, OSError, ValueError) as exc:
                found[url] = f"{type(exc).__name__}: {exc}"[:160]
            if done % 25 == 0 or done == len(urls):
                # A line now and then, so a slow run can be told from a
                # hung one in CI, where nobody can look at the process.
                print(f"  {done}/{len(urls)} headers read", flush=True)
    return found


def _fit(built: dict[str, object], header: dict[str, object]) -> dict[str, object]:
    """Compare a checkpoint's header against the module it loads into."""
    entry = built["entry"]
    wanted = built["wanted"]
    assert isinstance(entry, WeightEntry) and isinstance(wanted, dict)
    stored = {
        entry.key_map.get(key, key): _normalise(value["shape"])  # type: ignore[index]
        for key, value in header.items()
        if key != "__metadata__"
    }
    return {
        "tensors": len(stored),
        "shape_mismatch": sorted(k for k in stored if k in wanted and wanted[k] != stored[k]),
        "checkpoint_only": sorted(set(stored) - set(wanted)),
        "model_only": sorted(set(wanted) - set(stored)),
    }


def _describe(model_name: str, tag: str, report: dict[str, object]) -> list[str]:
    """One line per thing wrong, naming enough to act on."""
    lines: list[str] = []
    where = f"{model_name} [{tag}]"
    for kind, label in (
        ("shape_mismatch", "shape differs"),
        ("checkpoint_only", "in the checkpoint, not the model"),
        ("model_only", "in the model, no weights for it"),
    ):
        keys = report.get(kind) or []
        assert isinstance(keys, list)
        if keys:
            shown = ", ".join(keys[:4])
            more = f" (+{len(keys) - 4} more)" if len(keys) > 4 else ""
            lines.append(f"  {where}: {len(keys)} {label} — {shown}{more}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument(
        "--family", help="only this family, as registered (e.g. sknet, not sk_resnet)"
    )
    parser.add_argument("--model", help="only this factory")
    parser.add_argument(
        "--list", action="store_true", help="print what would be checked, then stop"
    )
    args = parser.parse_args()

    targets = sorted(_WEIGHTS_BY_MODEL.items())
    if args.model:
        targets = [(n, e) for n, e in targets if n == args.model]
        if not targets:
            print(f"no registered weights for {args.model!r}", file=sys.stderr)
            return 1
    elif args.family:
        # A factory's family is registry metadata, not a name prefix:
        # the sknet family ships sk_resnet_18_cls, which shares no
        # substring with it.
        targets = [
            (n, e) for n, e in targets if getattr(_REGISTRY.get(n), "family", None) == args.family
        ]
        if not targets:
            known = sorted(
                {
                    fam
                    for n in _WEIGHTS_BY_MODEL
                    if (fam := getattr(_REGISTRY.get(n), "family", None))
                }
            )
            print(
                f"no family {args.family!r} ships weights. Families that do:\n"
                f"  {', '.join(known)}",
                file=sys.stderr,
            )
            return 1

    checkpoints = sum(len(_tags(enum)) for _, enum in targets)
    if not checkpoints:
        # Reporting "OK" for an empty selection would let a typo'd
        # filter pass a CI job that checked nothing.
        print("no checkpoints selected — nothing was checked", file=sys.stderr)
        return 1
    if args.list:
        for name, enum in targets:
            print(f"{name}: {', '.join(_tags(enum))}")
        print(f"\n{len(targets)} factories, {checkpoints} checkpoints")
        return 0

    print(f"checking {checkpoints} checkpoints across {len(targets)} factories")
    bad: list[str] = []
    unreachable: list[str] = []
    checked = 0
    declined = 0
    # A checkpoint is fine if *some* factory can load it; one that every
    # factory declines is orphaned, and a refusal added by mistake would
    # otherwise hide it.
    fitted: set[str] = set()
    offered: set[str] = set()

    builds: list[tuple[str, str, dict[str, object]]] = []
    built_for_real: list[str] = []
    for index, (name, enum) in enumerate(targets, 1):
        print(f"  [{index}/{len(targets)}] {name}", flush=True)
        for tag in _tags(enum):
            built = _build(name, tag)
            builds.append((name, tag, built))
            if built.get("real"):
                built_for_real.append(f"{name} [{tag}]")
    if built_for_real:
        print(f"built for real, shadow mode could not follow: {', '.join(built_for_real)}")

    urls = sorted({e.url for *_, b in builds if isinstance(e := b.get("entry"), WeightEntry)})
    headers = _read_headers(urls)

    for name, tag, built in builds:
        offered.add(tag)
        if "declined" in built:
            declined += 1
            continue
        if "error" in built:
            bad.append(f"  {name} [{tag}]: {built['error']}")
            continue
        entry = built["entry"]
        assert isinstance(entry, WeightEntry)
        header = headers[entry.url]
        if isinstance(header, str):
            unreachable.append(f"  {name} [{tag}]: {header}")
            fitted.add(tag)  # unknown, not orphaned
            continue
        checked += 1
        problems = _describe(name, tag, _fit(built, header))
        if not problems:
            fitted.add(tag)
        bad.extend(problems)

    for tag in sorted(offered - fitted):
        bad.append(f"  [{tag}]: no factory in this run could load this checkpoint")

    print(f"{checked} checkpoints read", end="")
    print(f", {declined} declined by a sibling factory" if declined else "")
    if unreachable:
        print(f"\n{len(unreachable)} could not be reached:", file=sys.stderr)
        for line in unreachable:
            print(line, file=sys.stderr)
    if bad:
        print(f"\n{len(bad)} problem(s):", file=sys.stderr)
        for line in bad:
            print(line, file=sys.stderr)
        print(
            "\nA checkpoint that does not fit means pretrained=True raises for "
            "that factory. Either the factory's config drifted from the "
            "checkpoint, or the entry needs a key_map.",
            file=sys.stderr,
        )
        return 1

    print("[check_weight_fit] OK — every checkpoint fits its factory.")
    return 2 if unreachable else 0


if __name__ == "__main__":
    sys.exit(main())
