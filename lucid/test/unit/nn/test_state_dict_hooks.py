"""The ``Module`` state-dict hook API against the reference (LCD-792).

``register_state_dict_pre_hook`` / ``register_state_dict_post_hook`` and the
private ``_register_state_dict_hook`` / ``_register_load_state_dict_pre_hook``
did not exist, so nothing could record per-module metadata on save (the way
``spectral_norm`` records its version) or migrate an old checkpoint on load
without overriding ``_load_from_state_dict``.  The call order, the arguments
each hook receives and what a return value means are pinned against the
reference on a nested module.
"""

import copy
import pickle

import pytest

import lucid
import lucid.nn as nn


def _tree(lib):
    return lib.nn.Sequential(lib.nn.Linear(2, 2), lib.nn.Sequential(lib.nn.Linear(2, 2)))


def _record_save(lib, log: list):
    net = _tree(lib)
    for path, sub in net.named_modules():

        def pre(module, prefix, keep_vars, path=path):
            log.append(("pre", path, prefix, keep_vars))

        def post(module, state_dict, prefix, local_metadata, path=path):
            log.append(("post", path, prefix, sorted(state_dict), dict(local_metadata)))

        sub.register_state_dict_pre_hook(pre)
        sub.register_state_dict_post_hook(post)
    return net


@pytest.mark.parity
@pytest.mark.parametrize("keep_vars", [False, True])
def test_save_hooks_fire_in_the_references_order_with_its_arguments(ref, keep_vars) -> None:
    ours_log, theirs_log = [], []
    ours = _record_save(lucid, ours_log)
    theirs = _record_save(ref, theirs_log)
    ours.state_dict(prefix="m.", keep_vars=keep_vars)
    theirs.state_dict(prefix="m.", keep_vars=keep_vars)
    assert ours_log == theirs_log and len(ours_log) == 8


@pytest.mark.parity
def test_a_post_hook_edits_the_state_dict_and_its_metadata_in_place(ref) -> None:
    def post(module, state_dict, prefix, local_metadata):
        state_dict[prefix + "marker"] = 1
        local_metadata["tag"] = prefix

    results = []
    for lib in (lucid, ref):
        net = _tree(lib)
        net[1].register_state_dict_post_hook(post)
        sd = net.state_dict()
        results.append((list(sd), dict(sd._metadata)))
    assert results[0] == results[1]
    assert results[0][1]["1"] == {"version": 1, "tag": "1."}


@pytest.mark.parity
def test_a_public_post_hook_must_return_none(ref) -> None:
    for lib in (lucid, ref):
        net = lib.nn.Linear(2, 2)
        net.register_state_dict_post_hook(lambda m, sd, p, lm: sd)
        with pytest.raises(RuntimeError, match="must return None"):
            net.state_dict()


@pytest.mark.parity
def test_the_private_post_hook_may_replace_the_state_dict_at_the_root(ref) -> None:
    out = []
    for lib in (lucid, ref):
        net = _tree(lib)
        net._register_state_dict_hook(lambda m, sd, prefix, lm: {"only": 1})
        out.append(dict(net.state_dict()))
        child = _tree(lib)
        child[1]._register_state_dict_hook(lambda m, sd, prefix, lm: {"only": 1})
        out.append(sorted(child.state_dict()))
    assert out[0] == out[2] == {"only": 1}
    # A replacement returned by a child is dropped, as in the reference.
    assert out[1] == out[3] and "1.0.weight" in out[1]


@pytest.mark.parity
def test_the_private_form_refuses_a_hook_registered_publicly(ref) -> None:
    def hook(module, state_dict, prefix, local_metadata) -> None:
        return None

    for lib in (lucid, ref):
        net = lib.nn.Linear(2, 2)
        net.register_state_dict_post_hook(hook)
        with pytest.raises(RuntimeError, match="register_state_dict_post_hook"):
            net._register_state_dict_hook(hook)
        hook._from_public_api = False


@pytest.mark.parity
def test_handles_remove_their_hook(ref) -> None:
    for lib in (lucid, ref):
        net = lib.nn.Linear(2, 2)
        seen = []
        handles = [
            net.register_state_dict_pre_hook(lambda m, p, k: seen.append("pre")),
            net.register_state_dict_post_hook(lambda m, sd, p, lm: seen.append("post")),
            net._register_state_dict_hook(lambda m, sd, p, lm: seen.append("old")),
        ]
        net.state_dict()
        assert seen == ["pre", "post", "old"]
        for handle in handles:
            handle.remove()
            handle.remove()
        seen.clear()
        net.state_dict()
        assert seen == []


def _record_load(lib, log: list, private: bool):
    net = _tree(lib)
    sd = _tree(lib).state_dict()
    for path, sub in net.named_modules():
        if private:

            def pre(
                state_dict, prefix, local_metadata, strict, missing, unexpected, errors, path=path
            ):
                log.append(("pre", path, prefix, strict, sorted(local_metadata)))

            sub._register_load_state_dict_pre_hook(pre)
        else:

            def pre(
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing,
                unexpected,
                errors,
                path=path,
            ):
                log.append(("pre", path, prefix, strict, sorted(local_metadata)))

            sub.register_load_state_dict_pre_hook(pre)

        def post(module, incompatible, path=path):
            log.append(("post", path, list(incompatible.missing_keys)))

        sub.register_load_state_dict_post_hook(post)
    return net, sd


def _load_logs(ref, private):
    logs = []
    for lib in (lucid, ref):
        log: list = []
        net, sd = _record_load(lib, log, private)
        net.load_state_dict(sd)
        logs.append(log)
    return logs


@pytest.mark.parity
@pytest.mark.parametrize("private", [False, True])
def test_load_pre_hooks_fire_in_the_references_order(ref, private) -> None:
    ours, theirs = _load_logs(ref, private)
    pre = lambda log: [e for e in log if e[0] == "pre"]  # noqa: E731
    assert len(pre(ours)) == 4 and pre(ours) == pre(theirs)


@pytest.mark.parity
def test_every_load_post_hook_fires_once(ref) -> None:
    ours, theirs = _load_logs(ref, False)
    post = lambda log: sorted(e for e in log if e[0] == "post")  # noqa: E731
    assert len(post(ours)) == 4 and post(ours) == post(theirs)


@pytest.mark.parity
@pytest.mark.xfail(
    strict=True,
    reason="LCD-792 follow-up: load post-hooks run after the whole walk; the reference runs "
    "each right after its own subtree",
)
def test_load_post_hooks_interleave_with_the_walk_as_in_the_reference(ref) -> None:
    ours, theirs = _load_logs(ref, False)
    assert [e[:3] for e in ours] == [e[:3] for e in theirs]


@pytest.mark.parity
def test_a_module_less_pre_hook_is_not_handed_the_module(ref) -> None:
    for lib in (lucid, ref):
        net = lib.nn.Linear(2, 2)
        got = []
        net._register_load_state_dict_pre_hook(lambda *args: got.append(args))
        net.load_state_dict(lib.nn.Linear(2, 2).state_dict())
        assert got and isinstance(got[0][0], dict)
        got.clear()
        net._load_state_dict_pre_hooks.clear()
        net._register_load_state_dict_pre_hook(lambda *args: got.append(args), with_module=True)
        net.load_state_dict(lib.nn.Linear(2, 2).state_dict())
        assert got[0][0] is net


def test_a_pre_hook_can_rename_a_key_the_checkpoint_carries() -> None:
    net = nn.Linear(2, 2)
    old = {"w": net.weight.detach(), "bias": net.bias.detach()}

    def rename(state_dict, prefix, *rest):
        if prefix + "w" in state_dict:
            state_dict[prefix + "weight"] = state_dict.pop(prefix + "w")

    net._register_load_state_dict_pre_hook(rename)
    assert net.load_state_dict(old).missing_keys == []


def test_hooks_follow_a_deep_copy_to_the_copy() -> None:
    got: list = []

    def pre(module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
        got.append(module)

    net = nn.Linear(2, 2)
    net.register_load_state_dict_pre_hook(pre)
    sd = net.state_dict()
    twin = copy.deepcopy(net)
    twin.load_state_dict(sd)
    assert got == [twin]


def test_a_module_with_a_module_bound_hook_pickles_clean() -> None:
    net = nn.Linear(2, 2)
    net._register_load_state_dict_pre_hook(_noop_pre_hook, with_module=True)
    revived = pickle.loads(pickle.dumps(net))
    (wrapped,) = revived._load_state_dict_pre_hooks.values()
    assert wrapped.module() is revived
    revived.load_state_dict(net.state_dict())


def _noop_pre_hook(module, state_dict, prefix, local_metadata, strict, missing, unexpected, errors):
    return None


def test_a_module_pickled_before_the_registries_existed_loads() -> None:
    net = nn.Linear(2, 2)
    state = dict(net.__dict__)
    for name in ("_state_dict_hooks", "_state_dict_pre_hooks"):
        del state[name]
    fresh = nn.Linear.__new__(nn.Linear)
    fresh.__setstate__(state)
    assert fresh._state_dict_hooks == {} and "weight" in fresh.state_dict()
