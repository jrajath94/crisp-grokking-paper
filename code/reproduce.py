"""Reproduction script for CRISP: grokking phase-transition experiment.

Reproduces the 120-run sweep on modular arithmetic (a+b) mod 47 reported in
the paper: 5 widths x 8 training fractions x 3 seeds.

Usage:
    python reproduce.py --output-dir results/

Output:
    results/sweep_results.json  -- per-run training histories
    results/summary.json        -- per-condition aggregates

Hardware: single GPU (any CUDA device) or CPU (slower). Total runtime ~30 min
on an A100, ~2 hr on CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


P = 47                              # prime modulus
TOTAL_PAIRS = P * P                 # 2209
WIDTHS = [32, 48, 64, 96, 128]
FRACTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = [42, 137, 256]
STEPS = 10_000
EVAL_EVERY = 250
LR = 0.03
WD = 0.3
GROK_THRESHOLD = 0.90
MEMORIZE_THRESHOLD = 0.99


@dataclass
class RunResult:
    width: int
    train_frac: float
    seed: int
    n_train: int
    step: list
    train_acc: list
    test_acc: list
    train_loss: list
    test_loss: list
    eff_rank: list
    grokked: bool
    grok_step: int | None
    mem_step: int | None
    final_test_acc: float
    final_test_loss: float


class ModAddMLP(nn.Module):
    """Two-layer MLP with one-hot input, trained on (a+b) mod P."""

    def __init__(self, width: int, p: int = P) -> None:
        super().__init__()
        self.p = p
        self.fc1 = nn.Linear(2 * p, width, bias=True)
        self.fc2 = nn.Linear(width, p, bias=True)

    def forward(self, x_ab: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(x_ab[:, 0], self.p).float()
        b = F.one_hot(x_ab[:, 1], self.p).float()
        x = torch.cat([a, b], dim=-1)
        h = F.relu(self.fc1(x))
        return self.fc2(h)

    @torch.no_grad()
    def effective_rank(self, x_ab: torch.Tensor) -> float:
        a = F.one_hot(x_ab[:, 0], self.p).float()
        b = F.one_hot(x_ab[:, 1], self.p).float()
        x = torch.cat([a, b], dim=-1)
        h = F.relu(self.fc1(x))
        _, s, _ = torch.linalg.svd(h - h.mean(0, keepdim=True), full_matrices=False)
        s = s / (s.sum() + 1e-12)
        entropy = -(s * (s + 1e-12).log()).sum().item()
        return float(math.exp(entropy))


def build_dataset(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs = torch.tensor([(a, b) for a in range(P) for b in range(P)], dtype=torch.long)
    labels = (pairs[:, 0] + pairs[:, 1]) % P
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(TOTAL_PAIRS, generator=g)
    return pairs[perm], labels[perm]


def split_train_test(pairs: torch.Tensor, labels: torch.Tensor, frac: float) -> tuple:
    n_train = int(round(frac * TOTAL_PAIRS))
    return pairs[:n_train], labels[:n_train], pairs[n_train:], labels[n_train:]


def compute_metrics(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits, y).item()
        acc = (logits.argmax(-1) == y).float().mean().item()
    return acc, loss


def run_single(width: int, frac: float, seed: int, device: torch.device) -> RunResult:
    torch.manual_seed(seed)
    np.random.seed(seed)

    pairs, labels = build_dataset(seed)
    x_tr, y_tr, x_te, y_te = split_train_test(pairs, labels, frac)
    x_tr, y_tr = x_tr.to(device), y_tr.to(device)
    x_te, y_te = x_te.to(device), y_te.to(device)

    model = ModAddMLP(width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    hist = {"step": [], "train_acc": [], "test_acc": [],
            "train_loss": [], "test_loss": [], "eff_rank": []}
    grok_step: int | None = None
    mem_step: int | None = None

    for step in range(1, STEPS + 1):
        model.train()
        logits = model(x_tr)
        loss = F.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()

        if step % EVAL_EVERY == 0 or step == STEPS:
            tr_acc, tr_loss = compute_metrics(model, x_tr, y_tr)
            te_acc, te_loss = compute_metrics(model, x_te, y_te)
            er = model.effective_rank(x_tr[: min(512, len(x_tr))])

            hist["step"].append(step)
            hist["train_acc"].append(tr_acc)
            hist["test_acc"].append(te_acc)
            hist["train_loss"].append(tr_loss)
            hist["test_loss"].append(te_loss)
            hist["eff_rank"].append(er)

            if mem_step is None and tr_acc >= MEMORIZE_THRESHOLD:
                mem_step = step
            if grok_step is None and mem_step is not None and te_acc >= GROK_THRESHOLD:
                grok_step = step

    return RunResult(
        width=width, train_frac=frac, seed=seed, n_train=len(x_tr),
        step=hist["step"], train_acc=hist["train_acc"], test_acc=hist["test_acc"],
        train_loss=hist["train_loss"], test_loss=hist["test_loss"],
        eff_rank=hist["eff_rank"],
        grokked=grok_step is not None, grok_step=grok_step, mem_step=mem_step,
        final_test_acc=hist["test_acc"][-1] if hist["test_acc"] else 0.0,
        final_test_loss=hist["test_loss"][-1] if hist["test_loss"] else float("inf"),
    )


def aggregate(results: dict) -> dict:
    per_cond: dict = {}
    for r in results.values():
        key = f"w{r['width']}_f{r['train_frac']}"
        per_cond.setdefault(key, []).append(r)

    summary_per_cond = {}
    grokked_total = 0
    for key, runs in per_cond.items():
        grok_rate = sum(r["grokked"] for r in runs) / len(runs)
        mean_acc = float(np.mean([r["final_test_acc"] for r in runs]))
        delays = [r["grok_step"] - (r["mem_step"] or 0)
                  for r in runs if r["grok_step"] is not None]
        mean_delay = float(np.mean(delays)) if delays else None
        summary_per_cond[key] = {
            "grok_rate": grok_rate,
            "mean_test_acc": mean_acc,
            "mean_delay": mean_delay,
        }
        grokked_total += sum(r["grokked"] for r in runs)

    return {
        "total_runs": len(results),
        "grokked_runs": grokked_total,
        "grokking_rate": grokked_total / max(len(results), 1),
        "per_condition": summary_per_cond,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results")
    ap.add_argument("--widths", nargs="+", type=int, default=WIDTHS)
    ap.add_argument("--fractions", nargs="+", type=float, default=FRACTIONS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results: dict = {}
    total = len(args.widths) * len(args.fractions) * len(args.seeds)
    i = 0
    for w in args.widths:
        for f in args.fractions:
            for s in args.seeds:
                i += 1
                print(f"[{i}/{total}] width={w} frac={f} seed={s}")
                r = run_single(w, f, s, device)
                all_results[f"w{w}_f{f}_s{s}"] = r.__dict__

    (out / "sweep_results.json").write_text(json.dumps(all_results, indent=2))
    (out / "summary.json").write_text(json.dumps(aggregate(all_results), indent=2))
    print(f"Wrote {out / 'sweep_results.json'} and {out / 'summary.json'}")


if __name__ == "__main__":
    main()
