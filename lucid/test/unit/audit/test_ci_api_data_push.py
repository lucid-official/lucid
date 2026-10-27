"""The nightly api-data refresh survives a main that moves under it (LCD-553).

``docs-api-data.yml`` rebuilds for about two hours and then pushes; anything
landing on main meanwhile made that push fail and threw the rebuild away.
``scripts/ci_api_data_push.sh`` now carries the refresh onto the new tip.
These run that script for real, against a bare repository standing in for
GitHub and a shallow clone standing in for the runner, with the real drift
check (``web/scripts/build-api-data-cached.py``) and a stand-in generator
whose pages record the source they were built from and the commit they were
built at.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[4]
API = "web/public/api-data"

_GENERATOR = '''\
"""Stand-in for build-api-data.py: one page per slug, naming its source.

Like the real one, a page lists a factory, ``<leaf>_small``, carrying its
entry of ``_summaries.json``.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web" / "public" / "api-data"
MODULE_MANIFEST = {
    "lucid.alpha": "lucid.alpha",
    "lucid.beta": "lucid.beta",
    "lucid.models.gamma": "lucid.models.gamma",
}
SYNTH = ("lucid.ops",)


def main() -> None:
    args = sys.argv[1:]
    slugs = args[args.index("--slugs") + 1 :] if "--slugs" in args else [
        *MODULE_MANIFEST, *SYNTH
    ]
    sha = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
    ).strip()
    summaries = json.loads((OUT / "_summaries.json").read_text())
    for slug in slugs:
        parts = slug.split(".")[1:]
        if slug in MODULE_MANIFEST:
            src = ROOT.joinpath("lucid", *parts, "__init__.py").read_text()
        elif slug in SYNTH:
            src = "".join(p.read_text() for p in sorted((ROOT / "lucid").rglob("*.py")))
        else:
            continue
        factory = f"{parts[-1]}_small"
        page = {
            "slug": slug,
            "source": src,
            "sha": sha,
            "functions": [
                {
                    "name": factory,
                    "signature": "()",
                    "model_summary": summaries.get(factory),
                }
            ],
        }
        (OUT / f"{slug}.json").write_text(json.dumps(page, indent=2) + "\\n")
    if os.environ.get("STUB_ON_BUILD"):
        subprocess.run(os.environ["STUB_ON_BUILD"], shell=True, check=True)


if __name__ == "__main__":
    main()
'''

_META = '''\
"""Stand-in post-processor that can be told to fail."""
import os
import sys

sys.exit(3 if os.environ.get("STUB_POST_FAIL") else 0)
'''

_CROSS_LINKS = '''\
"""Stand-in post-processor: an index of the pages, rebuilt every run."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "public" / "api-data"
pages = sorted(p.stem for p in OUT.glob("lucid*.json"))
(OUT / "_cross_links.json").write_text(json.dumps(pages, indent=2) + "\\n")
'''

_POST_PROCESSORS = ("build-usedby", "link-citations", "build-bibtex")
_SUMMARIES = '{"gamma_small": {"params": 1}}\n'


def _env(tmp_path: Path) -> dict[str, str]:
    # Nothing of the calling git may leak in: run under a pre-commit hook,
    # GIT_DIR / GIT_INDEX_FILE would point every command below at the real
    # repository.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    config = tmp_path / "gitconfig"
    # No automatic gc: a push's background repack of the origin raced a
    # local clone copying the same objects ("failed to copy file").
    config.write_text(
        "[user]\n\tname = ci\n\temail = ci@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n"
        "[gc]\n\tauto = 0\n[receive]\n\tautogc = false\n"
    )
    env.update(GIT_CONFIG_GLOBAL=str(config), GIT_CONFIG_NOSYSTEM="1", PYTHON_BIN=sys.executable)
    return env


def _git(cwd: Path, env: dict[str, str], *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _build(root: Path, env: dict[str, str], *args: str) -> None:
    subprocess.run(
        [sys.executable, "scripts/build-api-data.py", *args],
        cwd=root / "web",
        env=env,
        check=True,
        capture_output=True,
    )


class _Fixture:
    """``origin`` (GitHub), ``dev`` (whoever lands on main), ``runner``."""

    def __init__(self, tmp_path: Path) -> None:
        self.env = _env(tmp_path)
        self.origin = tmp_path / "origin.git"
        self.dev = tmp_path / "dev"
        self.runner = tmp_path / "runner"
        subprocess.run(["git", "init", "-q", "--bare", str(self.origin)], env=self.env, check=True)
        d = self.dev
        d.mkdir()
        _git(d, self.env, "init", "-q")
        _write(d, "lucid/alpha/__init__.py", "ALPHA = 1\n")
        _write(d, "lucid/beta/__init__.py", "BETA = 1\n")
        _write(d, "lucid/models/gamma/__init__.py", "GAMMA = 1\n")
        _write(d, "web/scripts/build-api-data.py", _GENERATOR)
        _write(d, "web/scripts/build-cross-links.py", _CROSS_LINKS)
        _write(d, "web/scripts/build-meta.py", _META)
        for name in _POST_PROCESSORS:
            _write(d, f"web/scripts/{name}.py", "")
        shutil.copy(
            ROOT / "web/scripts/build-api-data-cached.py",
            d / "web/scripts/build-api-data-cached.py",
        )
        (d / "scripts").mkdir()
        shutil.copy(ROOT / "scripts/ci_api_data_push.sh", d / "scripts/ci_api_data_push.sh")
        _write(d, f"{API}/_summaries.json", _SUMMARIES)
        _git(d, self.env, "add", "-A")
        _git(d, self.env, "commit", "-qm", "init")
        _build(d, self.env)
        self.check(d, write=True)
        _git(d, self.env, "add", "-A")
        _git(d, self.env, "commit", "-qm", "docs(api-data): first build")
        _git(d, self.env, "remote", "add", "origin", str(self.origin))
        _git(d, self.env, "push", "-q", "origin", "HEAD:refs/heads/main")
        # The drift the nightly is woken for.
        self.land("lucid/alpha/__init__.py", "ALPHA = 2\n")

    def land(self, rel: str, text: str | None) -> str:
        """Commit one change on main, as a land during the rebuild would."""
        _git(self.dev, self.env, "pull", "-q", "--ff-only", "origin", "main")
        if text is None:
            _git(self.dev, self.env, "rm", "-rq", rel)
        else:
            _write(self.dev, rel, text)
            _git(self.dev, self.env, "add", "-A")
        _git(self.dev, self.env, "commit", "-qm", f"change {rel}")
        _git(self.dev, self.env, "push", "-q", "origin", "HEAD:refs/heads/main")
        return _git(self.dev, self.env, "rev-parse", "HEAD")

    def land_with_pages(self, changes: dict[str, str], *slugs: str) -> str:
        """A land that carries api-data: source, its pages and the baseline.

        No ``slugs`` regenerates every page, as a manual full build does.
        """
        _git(self.dev, self.env, "pull", "-q", "--ff-only", "origin", "main")
        for rel, text in changes.items():
            _write(self.dev, rel, text)
        _build(self.dev, self.env, *(("--slugs", *slugs) if slugs else ()))
        self.check(self.dev, write=True)
        _git(self.dev, self.env, "add", "-A")
        _git(self.dev, self.env, "commit", "-qm", "land with api-data")
        _git(self.dev, self.env, "push", "-q", "origin", "HEAD:refs/heads/main")
        return _git(self.dev, self.env, "rev-parse", "HEAD")

    def nightly_build(self) -> None:
        """Check out as actions/checkout does — depth 1 — and rebuild."""
        subprocess.run(
            ["git", "clone", "-q", "--depth", "1", f"file://{self.origin}", str(self.runner)],
            env=self.env,
            check=True,
        )
        _build(self.runner, self.env)

    def push(self, **env: str) -> subprocess.CompletedProcess[str]:
        out = self.runner.parent / "github_output"
        summary = self.runner.parent / "step_summary"
        out.write_text("")
        summary.write_text("")
        github = {"GITHUB_OUTPUT": str(out), "GITHUB_STEP_SUMMARY": str(summary)}
        return subprocess.run(
            ["scripts/ci_api_data_push.sh"],
            cwd=self.runner,
            env={**self.env, **github, **env},
            capture_output=True,
            text=True,
            check=False,
        )

    def pushed(self) -> str:
        return (self.runner.parent / "github_output").read_text()

    def summary(self) -> str:
        return (self.runner.parent / "step_summary").read_text()

    def check(self, root: Path, write: bool = False) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "scripts/build-api-data-cached.py"]
            + (["--write-cache"] if write else ["--check"]),
            cwd=root / "web",
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def main_tree(self, tmp_path: Path) -> Path:
        """A fresh clone of what main now holds."""
        where = tmp_path / "after"
        subprocess.run(
            ["git", "clone", "-q", str(self.origin), str(where)], env=self.env, check=True
        )
        return where

    def page(self, root: Path, slug: str) -> dict[str, Any]:
        return dict(json.loads((root / API / f"{slug}.json").read_text()))


@pytest.fixture
def repo(tmp_path: Path) -> _Fixture:
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    return _Fixture(tmp_path)


def test_a_main_that_did_not_move_takes_the_refresh_as_built(
    repo: _Fixture, tmp_path: Path
) -> None:
    repo.nightly_build()
    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pushed=true" in repo.pushed()
    assert "rebuilding" not in result.stdout
    after = repo.main_tree(tmp_path)
    assert repo.page(after, "lucid.alpha")["source"] == "ALPHA = 2\n"
    assert repo.check(after).returncode == 0


def test_a_main_that_moved_gets_the_refresh_with_its_moved_pages_rebuilt(
    repo: _Fixture, tmp_path: Path
) -> None:
    repo.nightly_build()
    moved = repo.land("lucid/beta/__init__.py", "BETA = 2\n")
    # A summary for a factory no page lists: no page to rebuild for it.
    summaries = '{"gamma_small": {"params": 1}, "zeta_small": {"params": 9}}\n'
    repo.land(f"{API}/_summaries.json", summaries)

    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pushed=true" in repo.pushed()
    # The beta page, whose source moved, and the synthesised ops page, which
    # reads the whole tree; not the alpha page.
    assert "rebuilding 2 page(s) there" in result.stdout, result.stdout

    after = repo.main_tree(tmp_path)
    log = _git(after, repo.env, "log", "--format=%s", "-4").splitlines()
    assert log[0].startswith("docs(api-data): nightly refresh"), log
    assert log[1:3] == [f"change {API}/_summaries.json", "change lucid/beta/__init__.py"]
    beta = repo.page(after, "lucid.beta")
    assert beta["source"] == "BETA = 2\n"
    # Rebuilt at the tip it was carried onto — a commit on main, so its
    # permalinks resolve — and not at the refresh commit itself.
    assert beta["sha"] == _git(after, repo.env, "rev-parse", "--short", "HEAD^")
    _git(after, repo.env, "merge-base", "--is-ancestor", moved, beta["sha"])
    assert repo.page(after, "lucid.alpha")["source"] == "ALPHA = 2\n"
    assert "BETA = 2" in repo.page(after, "lucid.ops")["source"]
    # A file the refresh does not own keeps what main gave it.
    assert (after / API / "_summaries.json").read_text() == summaries
    assert repo.check(after).returncode == 0, repo.check(after).stdout


def test_a_page_both_sides_regenerated_is_rebuilt_at_the_tip(
    repo: _Fixture, tmp_path: Path
) -> None:
    """The 10-23 shape: a land carrying api-data for a page the refresh owns.

    Both commits rewrote the alpha page and the drift baseline, so the pick
    conflicts on both, and the refresh's side wins them.  Its alpha page
    describes ALPHA = 2 while main's source says 3, so the drift check
    against the refresh's own baseline names it, and it is rebuilt.
    """
    repo.nightly_build()
    moved = repo.land_with_pages(
        {"lucid/alpha/__init__.py": "ALPHA = 3\n"}, "lucid.alpha", "lucid.ops"
    )

    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rebuilding 2 page(s) there" in result.stdout, result.stdout
    after = repo.main_tree(tmp_path)
    alpha = repo.page(after, "lucid.alpha")
    assert alpha["source"] == "ALPHA = 3\n"
    assert alpha["sha"] == moved[:7]
    assert "ALPHA = 3" in repo.page(after, "lucid.ops")["source"]
    assert repo.check(after).returncode == 0, repo.check(after).stdout


def test_a_full_regeneration_on_main_does_not_rebuild_anything(
    repo: _Fixture, tmp_path: Path
) -> None:
    """A manual full build landing in the window (ba8610cc5, 10-23).

    Built after an unrelated land, so every page conflicts — the two builds
    name different commits — and no source moved, so the refresh's pages
    stand; each names a commit on main either way.
    """
    repo.nightly_build()
    repo.land("README.md", "unrelated\n")
    repo.land_with_pages({})

    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rebuilding 0 page(s) there" in result.stdout, result.stdout
    after = repo.main_tree(tmp_path)
    for slug in ("lucid.alpha", "lucid.beta", "lucid.models.gamma", "lucid.ops"):
        _git(after, repo.env, "merge-base", "--is-ancestor", repo.page(after, slug)["sha"], "HEAD")
    assert repo.page(after, "lucid.alpha")["source"] == "ALPHA = 2\n"
    assert repo.check(after).returncode == 0, repo.check(after).stdout


def test_a_summary_that_moved_rebuilds_the_page_embedding_it(
    repo: _Fixture, tmp_path: Path
) -> None:
    """No source hash sees _summaries.json, which model pages embed."""
    repo.nightly_build()
    moved = repo.land(f"{API}/_summaries.json", '{"gamma_small": {"params": 2}}\n')

    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rebuilding 1 page(s) there" in result.stdout, result.stdout
    after = repo.main_tree(tmp_path)
    gamma = repo.page(after, "lucid.models.gamma")
    assert gamma["functions"][0]["model_summary"] == {"params": 2}
    assert gamma["sha"] == moved[:7]
    assert repo.check(after).returncode == 0, repo.check(after).stdout


@pytest.mark.parametrize(
    ("env", "command"),
    [
        # In a subshell, inside a function.
        ({"STUB_POST_FAIL": "1"}, '`"$PYTHON_BIN" "scripts/$step.py"` failed with exit 3'),
        # In the main loop.
        ({"LUCID_API_DATA_REMOTE": "nowhere"}, '`git fetch -q "$REMOTE" "$BRANCH"` failed'),
    ],
    ids=["post-processor", "fetch"],
)
def test_a_failing_command_still_reports_as_a_failed_push(
    repo: _Fixture, env: dict[str, str], command: str
) -> None:
    repo.nightly_build()
    result = repo.push(**env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert f"::error::api-data was not pushed: {command}" in result.stderr
    # Reported once, though the failure fails its callers too.
    assert result.stderr.count("::error::") == 1, result.stderr
    assert "pushed=false" in repo.pushed()
    assert "### api-data was not pushed" in repo.summary()


def test_a_module_main_deleted_stays_deleted(repo: _Fixture, tmp_path: Path) -> None:
    repo.nightly_build()
    repo.land("lucid/alpha", None)
    repo.land(f"{API}/lucid.alpha.json", None)

    result = repo.push()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "rebuilding 1 page(s) there" in result.stdout, result.stdout
    after = repo.main_tree(tmp_path)
    assert not (after / API / "lucid.alpha.json").exists()
    assert "ALPHA" not in repo.page(after, "lucid.ops")["source"]
    assert repo.check(after).returncode == 0, repo.check(after).stdout


def test_a_main_that_keeps_moving_fails_loudly_naming_the_stale_slugs(repo: _Fixture) -> None:
    repo.nightly_build()
    repo.land("lucid/beta/__init__.py", "BETA = 2\n")
    # Every partial rebuild sees main move again before its push.
    again = (
        f"cd {repo.dev} && git pull -q --ff-only origin main"
        " && echo '# again' >> lucid/beta/__init__.py"
        " && git commit -qam again && git push -q origin HEAD:refs/heads/main"
    )
    result = repo.push(LUCID_API_DATA_PUSH_ATTEMPTS="2", STUB_ON_BUILD=again)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "::error::api-data was not pushed: the push was refused 2 times" in result.stderr
    assert "  moved on main during the rebuild:\n" in result.stderr
    assert "    - lucid.beta" in result.stderr
    assert "    - lucid.ops" in result.stderr
    assert "    - lucid.alpha" not in result.stderr
    assert "pushed=false" in repo.pushed()
    tip = _git(repo.dev, repo.env, "ls-remote", str(repo.origin), "refs/heads/main")
    assert _git(repo.dev, repo.env, "log", "--format=%s", "-1", tip.split()[0]) == "again"


def test_a_generator_that_moved_fails_loudly(repo: _Fixture) -> None:
    repo.nightly_build()
    repo.land("web/scripts/build-api-data.py", _GENERATOR + "# changed\n")
    result = repo.push()
    assert result.returncode == 1, result.stdout + result.stderr
    assert "build-api-data.py changed on main during the rebuild" in result.stderr
    assert "    - every page (the generator moved)" in result.stderr
    assert "pushed=false" in repo.pushed()


def test_the_workflow_pushes_through_the_script() -> None:
    workflow = (ROOT / ".github/workflows/docs-api-data.yml").read_text()
    assert "./scripts/ci_api_data_push.sh" in workflow
    assert "git push" not in workflow
