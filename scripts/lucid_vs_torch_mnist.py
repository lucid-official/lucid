#!/usr/bin/env python3
"""Lucid (compiled) vs torch — LeNet-5 / MNIST training comparison.

Trains an *identical* LeNet-5 on MNIST in both frameworks from a bit-identical
weight init and an identical batch order, then reports loss, accuracy, per-step
speed and device-memory side by side.

The Lucid side runs through ``lucid.compile.fused_step`` — forward + loss +
autodiff backward + optimizer update lowered into one MPSGraph executable.
The torch side runs an ordinary eager Adam step (``torch.compile`` on the MPS
backend is prototype-only / unavailable on 3.14, so eager is the fair baseline).

Ways to run
-----------
1. **Default — fully isolated.**  Each framework trains to completion in its
   OWN subprocess, one after the other (torch finishes entirely, then lucid),
   so there is zero shared state / GPU contention.  The parent then draws the
   combined figure.  Parity is preserved: a shared init-weight file + the
   seeded batch order make the two runs comparable per-iteration::

       python scripts/lucid_vs_torch_mnist.py

2. One framework only (the building block the default orchestrates)::

       python scripts/lucid_vs_torch_mnist.py --only torch    # → /tmp/lvt_torch.json
       python scripts/lucid_vs_torch_mnist.py --only lucid    # → /tmp/lvt_lucid.json
       python scripts/lucid_vs_torch_mnist.py --plot-only     # redraw, no retrain

3. Both in a single shared process (no subprocess), still sequential::

       python scripts/lucid_vs_torch_mnist.py --inproc

The final figure has three horizontal panels:

    [1] running train/val loss   (per-iter train + per-epoch val, lucid vs torch)
    [2] time-running train loss  (loss vs cumulative wall-clock, lucid vs torch)
    [3] device memory            (avg / peak bars, lucid vs torch)

This script imports torch directly; that is fine here because Hard-Rule H5
binds the ``lucid/`` runtime package only — ``scripts/`` is outside its scope
(the hard-rule audit greps ``lucid/`` exclusively).
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

import torch
import torch.nn as tnn
import torchvision

import lucid
import lucid.nn as lnn
import lucid.nn.functional as F
import lucid.optim as loptim
from lucid.compile import fused_step

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover

    def tqdm(it, **_):
        return it


# ───────────────────────────── data ─────────────────────────────

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def load_mnist(root: str, limit: int | None) -> tuple[np.ndarray, ...]:
    """Return (train_x, train_y, test_x, test_y) as numpy, normalised + padded
    to 1x32x32 (the canonical LeNet-5 input size).  The same arrays feed both
    frameworks, so the data is provably identical."""
    tr = torchvision.datasets.MNIST(root, train=True, download=True)
    te = torchvision.datasets.MNIST(root, train=False, download=True)

    def prep(ds: object) -> tuple[np.ndarray, np.ndarray]:
        x = ds.data.numpy().astype("float32") / 255.0  # (N,28,28)
        x = (x - MNIST_MEAN) / MNIST_STD
        x = np.pad(x, ((0, 0), (2, 2), (2, 2)))  # → (N,32,32)
        x = x[:, None, :, :]  # → (N,1,32,32)
        y = ds.targets.numpy().astype("int64")
        return np.ascontiguousarray(x), np.ascontiguousarray(y)

    tx, ty = prep(tr)
    ex, ey = prep(te)
    if limit is not None:
        tx, ty = tx[:limit], ty[:limit]
        ex, ey = ex[: max(1000, limit // 4)], ey[: max(1000, limit // 4)]
    return tx, ty, ex, ey


# ───────────────────────────── models ─────────────────────────────
# Identical submodule registration order in both frameworks so that
# ``named_parameters()`` aligns 1:1 and init weights load by position.


class TorchLeNet(tnn.Module):
    def __init__(self, act: str, pool: str, num_classes: int = 10) -> None:
        super().__init__()
        A = tnn.ReLU if act == "relu" else tnn.Tanh
        P = (lambda: tnn.MaxPool2d(2, 2)) if pool == "max" else (lambda: tnn.AvgPool2d(2, 2))
        self.c1, self.a1, self.s2 = tnn.Conv2d(1, 6, 5), A(), P()
        self.c3, self.a3, self.s4 = tnn.Conv2d(6, 16, 5), A(), P()
        self.c5, self.a5 = tnn.Conv2d(16, 120, 5), A()
        self.f6, self.a6 = tnn.Linear(120, 84), A()
        self.out = tnn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.s2(self.a1(self.c1(x)))
        x = self.s4(self.a3(self.c3(x)))
        x = self.a5(self.c5(x))
        x = torch.flatten(x, 1)
        x = self.a6(self.f6(x))
        return self.out(x)


class LucidLeNet(lnn.Module):
    def __init__(self, act: str, pool: str, num_classes: int = 10) -> None:
        super().__init__()

        def A() -> lnn.Module:
            return lnn.ReLU() if act == "relu" else lnn.Tanh()

        def P() -> lnn.Module:
            return lnn.MaxPool2d(2, stride=2) if pool == "max" else lnn.AvgPool2d(2, stride=2)

        self.c1, self.a1, self.s2 = lnn.Conv2d(1, 6, 5), A(), P()
        self.c3, self.a3, self.s4 = lnn.Conv2d(6, 16, 5), A(), P()
        self.c5, self.a5 = lnn.Conv2d(16, 120, 5), A()
        self.f6, self.a6 = lnn.Linear(120, 84), A()
        self.out = lnn.Linear(84, num_classes)

    def forward(self, x: lucid.Tensor) -> lucid.Tensor:
        x = self.s2(self.a1(self.c1(x)))
        x = self.s4(self.a3(self.c3(x)))
        x = self.a5(self.c5(x))
        x = x.reshape(x.shape[0], -1)
        x = self.a6(self.f6(x))
        return self.out(x)


# ─────────────────────── shared init weights ───────────────────────


def ensure_init(path: str, act: str, pool: str, seed: int) -> None:
    """Create ``path`` (an .npz of positional init weights) if missing, so that
    every run — torch or lucid, same process or separate — starts from a
    bit-identical initialisation.  Deterministic in ``seed``."""
    if os.path.exists(path):
        return
    torch.manual_seed(seed)
    m = TorchLeNet(act, pool)
    arrays = {
        f"p{i}": p.detach().cpu().numpy().astype("float32")
        for i, (_, p) in enumerate(m.named_parameters())
    }
    np.savez(path, **arrays)


def load_init_torch(model: tnn.Module, path: str, dev: str) -> None:
    z = np.load(path)
    with torch.no_grad():
        for i, (_, p) in enumerate(model.named_parameters()):
            p.copy_(torch.from_numpy(z[f"p{i}"]).to(dev))


def load_init_lucid(model: lnn.Module, path: str, dev: str) -> None:
    z = np.load(path)
    for i, (_, p) in enumerate(model.named_parameters()):
        p.copy_(lucid.tensor(z[f"p{i}"], dtype=lucid.float32, device=dev))


# ───────────────────────────── eval ─────────────────────────────


def eval_torch(
    model: tnn.Module, x: np.ndarray, y: np.ndarray, dev: str, ce_sum: tnn.Module, bs: int = 256
) -> tuple[float, float]:
    model.eval()
    correct, loss_sum = 0, 0.0
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb = torch.from_numpy(x[i : i + bs]).to(dev)
            yb = torch.from_numpy(y[i : i + bs]).to(dev)
            out = model(xb)
            loss_sum += float(ce_sum(out, yb))
            correct += int((out.argmax(1).cpu().numpy() == y[i : i + bs]).sum())
    model.train()
    return correct / len(x), loss_sum / len(x)


def eval_lucid(
    model: lnn.Module, x: np.ndarray, y: np.ndarray, dev: str, bs: int = 256
) -> tuple[float, float]:
    model.eval()
    correct, loss_sum = 0, 0.0
    with lucid.no_grad():
        for i in range(0, len(x), bs):
            xb = lucid.tensor(x[i : i + bs], dtype=lucid.float32, device=dev)
            yb = lucid.tensor(y[i : i + bs], dtype=lucid.int64, device=dev)
            out = model(xb)
            loss_sum += float(F.cross_entropy(out, yb, reduction="sum").item())
            pred = out.argmax(dim=1).numpy().astype(y.dtype)
            correct += int((pred == y[i : i + bs]).sum())
    model.train()
    return correct / len(x), loss_sum / len(x)


# ─────────────────── single-framework training run ───────────────────


def run_one(fw: str, args: argparse.Namespace, init_path: str) -> dict:
    """Train ONE framework end-to-end (isolated) and return a result dict.

    ``fw`` is ``"torch"`` or ``"lucid"``.  The seeded batch order is identical
    across frameworks, so per-iter losses from two separate runs align by index.
    """
    is_lucid = fw == "lucid"
    if is_lucid:
        dev = "metal" if lucid.metal.is_available() else "cpu"
        use_compile = (dev == "metal") and not args.no_compile
    else:
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        use_compile = False

    # Seed governs the per-epoch permutations — same seed ⇒ same batch order in
    # both the torch run and the lucid run, even in separate processes.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    lucid.manual_seed(args.seed)

    tx, ty, ex, ey = load_mnist(args.data_root, args.limit)
    tr_probe_x, tr_probe_y = tx[:10000], ty[:10000]

    label = "compiled" if (is_lucid and use_compile) else "eager"
    print(f"\n── running {fw} ({label})  device={dev!r} ──")
    print(
        f"   data: train={len(tx)} test={len(ex)} | epochs={args.epochs} "
        f"batch={args.batch_size} lr={args.lr}"
    )

    if is_lucid:
        model = LucidLeNet(args.activation, args.pool).to(dev)
        load_init_lucid(model, init_path, dev)
        opt = loptim.Adam(list(model.parameters()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        if use_compile:
            step = fused_step(model, F.cross_entropy, opt)
        else:

            def step(x: lucid.Tensor, y: lucid.Tensor) -> lucid.Tensor:
                opt.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                opt.step()
                return loss

        sync = (lambda: lucid.metal.synchronize()) if dev == "metal" else (lambda: None)
        cur_mem = (
            (lambda: float(lucid.metal.memory_allocated())) if dev == "metal" else (lambda: 0.0)
        )
        if dev == "metal":
            lucid.metal.reset_peak_memory_stats()
    else:
        model = TorchLeNet(args.activation, args.pool).to(dev)
        load_init_torch(model, init_path, dev)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)
        ce = tnn.CrossEntropyLoss()

        def step(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
            return loss

        sync = (lambda: torch.mps.synchronize()) if dev == "mps" else (lambda: None)
        cur_mem = (
            (lambda: float(torch.mps.current_allocated_memory())) if dev == "mps" else (lambda: 0.0)
        )

    ce_sum = tnn.CrossEntropyLoss(reduction="sum")  # for torch eval loss

    def to_dev(xb: np.ndarray, yb: np.ndarray):
        if is_lucid:
            return (
                lucid.tensor(xb, dtype=lucid.float32, device=dev),
                lucid.tensor(yb, dtype=lucid.int64, device=dev),
            )
        return torch.from_numpy(xb).to(dev), torch.from_numpy(yb).to(dev)

    bs, n = args.batch_size, len(tx)
    n_batches = n // bs

    ep_list, tl_loss, vl_loss, tr_acc, va_acc = [], [], [], [], []
    it_loss, it_cumt, step_ms = [], [], []
    mem_sum = mem_peak = 0.0
    mem_cnt = 0
    cumt = 0.0
    gidx = 0

    for ep in range(args.epochs):
        perm = np.random.permutation(n)[: n_batches * bs].reshape(n_batches, bs)
        model.train()
        ep_loss = 0.0
        bar = tqdm(range(n_batches), desc=f"{fw} epoch {ep + 1}/{args.epochs}", ncols=104)
        for bi in bar:
            idx = perm[bi]
            xb, yb = to_dev(tx[idx], ty[idx])
            t0 = time.perf_counter()
            loss = step(xb, yb)
            sync()
            dt = time.perf_counter() - t0

            lv = float(loss.item())
            ep_loss += lv
            cumt += dt
            it_loss.append(lv)
            it_cumt.append(cumt)
            if gidx > 0:  # skip global first iter (compile warmup)
                step_ms.append(dt * 1e3)
            m = cur_mem()
            mem_sum += m
            mem_peak = max(mem_peak, m)
            mem_cnt += 1
            gidx += 1
            bar.set_postfix(loss=f"{lv:.3f}", ms=f"{dt * 1e3:.1f}")

        if is_lucid:
            ta, _ = eval_lucid(model, tr_probe_x, tr_probe_y, dev)
            va, vl = eval_lucid(model, ex, ey, dev)
        else:
            ta, _ = eval_torch(model, tr_probe_x, tr_probe_y, dev, ce_sum)
            va, vl = eval_torch(model, ex, ey, dev, ce_sum)
        ep_list.append(ep + 1)
        tl_loss.append(ep_loss / n_batches)
        vl_loss.append(vl)
        tr_acc.append(ta)
        va_acc.append(va)
        print(
            f"   ▸ epoch {ep + 1}: train-loss={ep_loss / n_batches:.4f} "
            f"val-loss={vl:.4f} test-acc={va:.4f}"
        )

    peak_alloc = (
        float(lucid.metal.max_memory_allocated()) if is_lucid and dev == "metal" else mem_peak
    )
    driver = float(torch.mps.driver_allocated_memory()) if (not is_lucid and dev == "mps") else 0.0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    cfg = {
        "epochs": args.epochs,
        "batch": args.batch_size,
        "lr": args.lr,
        "act": args.activation,
        "pool": args.pool,
        "seed": args.seed,
        "limit": args.limit,
    }
    return {
        "fw": fw,
        "label": label,
        "device": dev,
        "compiled": use_compile,
        "cfg": cfg,
        "ep_list": ep_list,
        "tl_loss": tl_loss,
        "vl_loss": vl_loss,
        "tr_acc": tr_acc,
        "va_acc": va_acc,
        "it_loss": it_loss,
        "it_cumt": it_cumt,
        "step_ms_mean": float(np.mean(step_ms)) if step_ms else float("nan"),
        "cumt_total": cumt,
        "n_batches": n_batches,
        "avg_mem": mem_sum / max(mem_cnt, 1),
        "peak_mem": peak_alloc,
        "driver": driver,
        "rss": float(rss),
    }


# ───────────────────────────── reporting ─────────────────────────────


def _mb(v: float) -> float:
    return v / 1024 / 1024


def combined_report(rt: dict, rl: dict, plot_path: str, draw: bool) -> None:
    """Print the side-by-side table and (optionally) draw the 3-panel figure
    from two result dicts (which may come from separate processes)."""
    ep = rt["ep_list"]
    if rt.get("cfg") != rl.get("cfg"):
        print("\n  ⚠ config mismatch between the two runs — results may not be comparable:")
        print(f"      torch={rt.get('cfg')}")
        print(f"      lucid={rl.get('cfg')}")
        print("    Re-run both with the same flags (e.g. delete /tmp/lvt_*.json first).")
    print("\n" + "=" * 74 + "\n  COMBINED SUMMARY  (torch vs lucid)\n" + "=" * 74)
    print("  ep | train-loss        | val-loss          | test-acc")
    print("     |  torch   lucid     |  torch   lucid     |  torch   lucid")
    for i in range(min(len(ep), len(rl["ep_list"]))):
        print(
            f"   {ep[i]} | {rt['tl_loss'][i]:.4f}  {rl['tl_loss'][i]:.4f}   | "
            f"{rt['vl_loss'][i]:.4f}  {rl['vl_loss'][i]:.4f}   | "
            f"{rt['va_acc'][i]:.4f}  {rl['va_acc'][i]:.4f}"
        )

    m = min(len(rt["it_loss"]), len(rl["it_loss"]))
    dl = [abs(rt["it_loss"][i] - rl["it_loss"][i]) for i in range(m)]
    print(
        f"\n  수치 정확도 (per-iter |Δloss|, aligned by index):  "
        f"max={max(dl):.2e}  mean={float(np.mean(dl)):.2e}  final={dl[-1]:.2e}"
    )
    mt, ml = rt["step_ms_mean"], rl["step_ms_mean"]
    print("\n  속도 (per-iter step, warmup excluded):")
    print(f"     torch = {mt:7.2f} ms/iter   (wall-clock {rt['cumt_total']:.1f}s)")
    print(
        f"     lucid = {ml:7.2f} ms/iter   (wall-clock {rl['cumt_total']:.1f}s)  "
        f"→ {mt / ml:.2f}× (>1 = lucid faster)"
    )
    print("\n  메모리 사용량 (isolated process, device allocator):")
    print(
        f"     torch  avg={_mb(rt['avg_mem']):7.1f}  peak={_mb(rt['peak_mem']):7.1f} MB "
        f"(driver {_mb(rt['driver']):.1f}) | RSS {_mb(rt['rss']):.0f} MB"
    )
    print(
        f"     lucid  avg={_mb(rl['avg_mem']):7.1f}  peak={_mb(rl['peak_mem']):7.1f} MB"
        f"                  | RSS {_mb(rl['rss']):.0f} MB"
    )
    if rl["compiled"]:
        print("       ⚠ lucid device-alloc UNDER-reports under compile (MPSGraph buffers")
        print("         invisible to the MLX allocator); RSS is the more honest lucid number.")
    print("=" * 74)

    if draw:
        save_plots(plot_path, rt, rl)


def save_plots(path: str, rt: dict, rl: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def smooth(a: list[float], w: int = 25) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        return a if len(a) < w else np.convolve(a, np.ones(w) / w, mode="valid")

    C = {"torch": "#ff7f0e", "lucid": "#1f77b4"}
    res = {"torch": rt, "lucid": rl}
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    # [1] running train / val loss — per-iter train (raw + smoothed), val markers
    #     at each epoch boundary (val is only evaluated once per epoch).
    for k in ("torch", "lucid"):
        it = res[k]["it_loss"]
        xs = np.arange(1, len(it) + 1)
        ax[0].plot(xs, it, color=C[k], alpha=0.20, lw=0.7)  # raw per-iter
        ys = smooth(it)
        ax[0].plot(xs[len(xs) - len(ys) :], ys, color=C[k], lw=1.6, label=f"{k} train")
        nb = res[k]["n_batches"]
        vx = [e * nb for e in res[k]["ep_list"]]
        ax[0].plot(vx, res[k]["vl_loss"], "--s", color=C[k], alpha=0.85, label=f"{k} val")
    ax[0].set_yscale("log")
    ax[0].set(
        title="Running train / val loss (per-iter)",
        xlabel="iteration",
        ylabel="cross-entropy loss (log)",
    )
    ax[0].legend()
    ax[0].grid(alpha=0.3, which="both")

    # [2] time-running training loss (loss vs cumulative wall-clock)
    for k in ("torch", "lucid"):
        ys = smooth(res[k]["it_loss"])
        xs = np.asarray(res[k]["it_cumt"])[len(res[k]["it_cumt"]) - len(ys) :]
        lbl = f"{k} ({res[k]['label']})"
        ax[1].plot(xs, ys, color=C[k], label=lbl)
    ax[1].set(
        title="Time-running training loss",
        xlabel="cumulative train wall-clock (s)",
        ylabel="train loss (smoothed)",
    )
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    # [3] device memory (avg / peak bars)
    x = np.arange(2)
    w = 0.36
    tv = [_mb(rt["avg_mem"]), _mb(rt["peak_mem"])]
    lv = [_mb(rl["avg_mem"]), _mb(rl["peak_mem"])]
    b1 = ax[2].bar(x - w / 2, tv, w, color=C["torch"], label="torch")
    b2 = ax[2].bar(x + w / 2, lv, w, color=C["lucid"], label="lucid")
    ax[2].bar_label(b1, fmt="%.0f", padding=2, fontsize=8)
    ax[2].bar_label(b2, fmt="%.0f", padding=2, fontsize=8)
    ax[2].set_xticks(x, ["avg", "peak"])
    ax[2].set(title="Device memory (allocator, MB)", ylabel="MB")
    ax[2].legend()
    ax[2].grid(alpha=0.3, axis="y")
    if rl["compiled"]:
        ax[2].text(
            0.5,
            -0.16,
            "⚠ lucid under-reports: MPSGraph buffers invisible to MLX allocator",
            transform=ax[2].transAxes,
            ha="center",
            fontsize=8,
            color="gray",
        )

    fig.suptitle("LeNet-5 / MNIST — lucid (compiled) vs torch", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    fig.savefig(path, dpi=120)
    print(f"\n  plot saved → {path}")


# ───────────────────────────── main ─────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description="LeNet-5 / MNIST: lucid (compiled) vs torch")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    ap.add_argument("--pool", choices=["max", "avg"], default="max")
    ap.add_argument("--limit", type=int, default=None, help="subset train size for a quick smoke")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="/tmp/mnist-data")
    ap.add_argument(
        "--only",
        choices=["torch", "lucid"],
        default=None,
        help="train ONLY this framework in this process (isolated); saves a JSON",
    )
    ap.add_argument("--no-compile", action="store_true", help="run Lucid eager instead of compiled")
    ap.add_argument("--no-plot", action="store_true", help="skip the final figure")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="don't train — just redraw the figure from existing JSON results",
    )
    ap.add_argument(
        "--inproc",
        action="store_true",
        help="train both frameworks sequentially in THIS process "
        "(default spawns an isolated subprocess per framework instead)",
    )
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--result-dir", default="/tmp")
    ap.add_argument(
        "--init-path",
        default=None,
        help="shared init weights (.npz); default derives from arch+seed",
    )
    ap.add_argument("--plot-path", default="/tmp/lucid_vs_torch_mnist.png")
    args = ap.parse_args()

    # Derive the init path from arch+seed so changing them never silently reuses
    # a mismatched-shape init file.
    init_path = args.init_path or os.path.join(
        args.result_dir, f"lvt_init_{args.activation}_{args.pool}_s{args.seed}.npz"
    )
    jpath = {fw: os.path.join(args.result_dir, f"lvt_{fw}.json") for fw in ("torch", "lucid")}

    def load(fw: str) -> dict | None:
        if os.path.exists(jpath[fw]):
            with open(jpath[fw]) as f:
                return json.load(f)
        return None

    # ── plot-only: redraw from saved results, no training ──
    if args.plot_only:
        rt, rl = load("torch"), load("lucid")
        if rt is None or rl is None:
            missing = [fw for fw in ("torch", "lucid") if load(fw) is None]
            print(f"  missing results for: {missing}. Run with --only <fw> first.")
            return
        combined_report(rt, rl, args.plot_path, draw=not args.no_plot)
        return

    # ── default: orchestrate — train each framework fully in its OWN process,
    #    one after the other (complete isolation), then draw the combined figure.
    if args.only is None and not args.inproc:
        passthrough = [
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--activation",
            args.activation,
            "--pool",
            args.pool,
            "--seed",
            str(args.seed),
            "--data-root",
            args.data_root,
            "--result-dir",
            args.result_dir,
            "--init-path",
            init_path,
            "--plot-path",
            args.plot_path,
        ]
        if args.limit is not None:
            passthrough += ["--limit", str(args.limit)]
        if args.no_compile:
            passthrough += ["--no-compile"]
        for fw in ("torch", "lucid"):
            print(f"\n{'#' * 74}\n#  isolated subprocess: training {fw} (full run)\n{'#' * 74}")
            subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--only", fw, "--child", *passthrough],
                check=True,
            )
        rt, rl = load("torch"), load("lucid")
        if rt is not None and rl is not None:
            combined_report(rt, rl, args.plot_path, draw=not args.no_plot)
        return

    ensure_init(init_path, args.activation, args.pool, args.seed)

    targets = [args.only] if args.only else ["torch", "lucid"]
    results: dict[str, dict] = {}
    for fw in targets:
        results[fw] = run_one(fw, args, init_path)
        with open(jpath[fw], "w") as f:
            json.dump(results[fw], f)
        print(f"   saved → {jpath[fw]}")

    # A child subprocess just trains + saves; the parent draws the combined figure.
    if args.child:
        return

    # ── draw combined figure if both results are available ──
    rt = results.get("torch") or load("torch")
    rl = results.get("lucid") or load("lucid")
    if rt is not None and rl is not None:
        combined_report(rt, rl, args.plot_path, draw=not args.no_plot)
    else:
        have, need = (args.only, "lucid" if args.only == "torch" else "torch")
        print(f"\n  {have} done. Now run:  python scripts/lucid_vs_torch_mnist.py --only {need}")
        print("  (it will auto-draw the combined figure once both JSONs exist.)")


if __name__ == "__main__":
    main()
