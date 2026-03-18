#!/usr/bin/env python3
"""Parse experiment log and generate figures + summary.

Reads the training log output (from the crashed run) and reconstructs
all results, then generates 6 publication-quality figures.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy import stats as sp_stats

P = 47
LOG_FILE = sys.argv[1] if len(sys.argv) > 1 else "/private/tmp/claude-501/-Users-rj-research-claw/b759b9b5-5db8-4e4b-b654-d38a17bcc34e/tasks/brboi280n.output"
FIG_DIR = Path(__file__).resolve().parent / "figures"
RES_DIR = Path(__file__).resolve().parent / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# Parse log
print("Parsing log...")
results = {}
current_key = None
current_hist = None

with open(LOG_FILE) as f:
    for line in f:
        # Match run header: [N/120] width=W frac=F seed=S
        m = re.match(r'\[(\d+)/\d+\] width=(\d+) frac=([\d.]+) seed=(\d+)', line)
        if m:
            if current_key and current_hist:
                results[current_key] = current_hist
            w, frac, seed = int(m.group(2)), float(m.group(3)), int(m.group(4))
            current_key = f"w{w}_f{frac}_s{seed}"
            current_hist = {
                "width": w, "train_frac": frac, "seed": seed,
                "n_train": int(P*P*frac),
                "step": [], "train_acc": [], "test_acc": [],
                "train_loss": [], "test_loss": [], "eff_rank": [],
            }
            continue

        # Match step data: [w=W frac=F s=S] step N tr_acc=X te_acc=X ...
        m = re.match(r'\s+\[w=\d+ frac=[\d.]+ s=\d+\] step\s+(\d+)\s+tr_acc=([\d.]+)\s+te_acc=([\d.]+)\s+tr_loss=([\d.]+)\s+te_loss=([\d.]+)\s+eff_rank=([\d.]+)', line)
        if m and current_hist is not None:
            current_hist["step"].append(int(m.group(1)))
            current_hist["train_acc"].append(float(m.group(2)))
            current_hist["test_acc"].append(float(m.group(3)))
            current_hist["train_loss"].append(float(m.group(4)))
            current_hist["test_loss"].append(float(m.group(5)))
            current_hist["eff_rank"].append(float(m.group(6)))

# Save last run
if current_key and current_hist:
    results[current_key] = current_hist

# Detect grokking
for key, hist in results.items():
    memorized = False
    grok_step = None
    for ta, tea, s in zip(hist["train_acc"], hist["test_acc"], hist["step"]):
        if ta > 0.99:
            memorized = True
        if memorized and tea > 0.80:
            grok_step = s
            break
    hist["grokked"] = grok_step is not None
    hist["grok_step"] = grok_step
    hist["final_test_acc"] = hist["test_acc"][-1] if hist["test_acc"] else 0
    hist["final_test_loss"] = hist["test_loss"][-1] if hist["test_loss"] else 99

    # Find memorization step
    mem_step = None
    for ta, s in zip(hist["train_acc"], hist["step"]):
        if ta > 0.99:
            mem_step = s
            break
    hist["mem_step"] = mem_step
    if grok_step and mem_step:
        hist["delay"] = grok_step - mem_step
    else:
        hist["delay"] = None

print(f"Parsed {len(results)} runs")

# Summary
WIDTHS = sorted(set(r["width"] for r in results.values()))
FRACS = sorted(set(r["train_frac"] for r in results.values()))
SEEDS = sorted(set(r["seed"] for r in results.values()))

grokked = sum(1 for r in results.values() if r["grokked"])
print(f"Grokked: {grokked}/{len(results)} ({grokked/len(results)*100:.1f}%)")

# Per-condition summary
summary = {"total_runs": len(results), "grokked_runs": grokked,
           "grokking_rate": grokked/len(results), "per_condition": {}}
for w in WIDTHS:
    for f in FRACS:
        conds = [r for r in results.values() if r["width"]==w and r["train_frac"]==f]
        grok_rate = sum(1 for c in conds if c["grokked"]) / len(conds) if conds else 0
        mean_acc = np.mean([c["final_test_acc"] for c in conds]) if conds else 0
        delays = [c["delay"] for c in conds if c["delay"] is not None]
        mean_delay = np.mean(delays) if delays else None
        summary["per_condition"][f"w{w}_f{f}"] = {
            "grok_rate": grok_rate, "mean_test_acc": float(mean_acc),
            "mean_delay": float(mean_delay) if mean_delay else None
        }
        status = f"GROK {grok_rate*100:.0f}%" if grok_rate > 0 else "no"
        delay_str = f"delay={mean_delay:.0f}" if mean_delay else ""
        print(f"  w={w:3d} f={f:.1f} -> {status:10s} acc={mean_acc:.3f} {delay_str}")

# Save
with open(RES_DIR / "summary.json", "w") as fp:
    json.dump(summary, fp, indent=2, default=str)
with open(RES_DIR / "sweep_results.json", "w") as fp:
    json.dump(results, fp, indent=2, default=lambda x: str(x) if isinstance(x, (float,)) and (np.isinf(x) or np.isnan(x)) else x)
print(f"Saved to {RES_DIR}/")

# ── FIGURES ──────────────────────────────────────────────────────

plt.rcParams.update({
    "font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 300,
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})
COLORS = plt.cm.tab10.colors

def agg(width, frac, key):
    vals = []
    for s in SEEDS:
        k = f"w{width}_f{frac}_s{s}"
        if k in results and results[k][key]:
            vals.append(np.array(results[k][key]))
    if not vals:
        return None, None, None
    ml = min(len(v) for v in vals)
    vals = [v[:ml] for v in vals]
    arr = np.stack(vals)
    steps = np.array(results[f"w{width}_f{frac}_s{SEEDS[0]}"]["step"][:ml])
    return arr.mean(0), arr.std(0), steps

# Fig 1: Grokking curves
print("Fig 1: grokking curves")
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
show_widths = [w for w in WIDTHS if w >= 48][:3]
for ax, w in zip(axes, show_widths):
    frac = 0.7
    tr_m, tr_s, steps = agg(w, frac, "train_acc")
    te_m, te_s, _ = agg(w, frac, "test_acc")
    if tr_m is not None:
        ax.plot(steps, tr_m, color=COLORS[0], label="Train", lw=2)
        ax.fill_between(steps, np.clip(tr_m-tr_s,0,1), np.clip(tr_m+tr_s,0,1), alpha=0.15, color=COLORS[0])
        ax.plot(steps, te_m, color=COLORS[3], label="Test", lw=2)
        ax.fill_between(steps, np.clip(te_m-te_s,0,1), np.clip(te_m+te_s,0,1), alpha=0.15, color=COLORS[3])
    ax.set_title(f"Width $d_h = {w}$", fontweight="bold")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.99, ls="--", color="gray", alpha=0.3, lw=0.8)
    ax.axhline(0.80, ls=":", color="gray", alpha=0.3, lw=0.8)
    ax.set_xlabel("Training step")
    if ax == axes[0]:
        ax.set_ylabel("Accuracy")
        ax.legend(loc="center right", framealpha=0.9)
fig.suptitle(f"Grokking Dynamics: $(a+b) \\,\\mathrm{{mod}}\\, {P}$, train fraction = 0.7", fontsize=15, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig1_grokking_curves.pdf")
fig.savefig(FIG_DIR / "fig1_grokking_curves.png")
plt.close()

# Fig 2: Phase diagram
print("Fig 2: phase diagram")
fig, ax = plt.subplots(figsize=(9, 5))
grid = np.zeros((len(WIDTHS), len(FRACS)))
for i, w in enumerate(WIDTHS):
    for j, f in enumerate(FRACS):
        accs = [results[f"w{w}_f{f}_s{s}"]["final_test_acc"]
                for s in SEEDS if f"w{w}_f{f}_s{s}" in results]
        grid[i, j] = np.mean(accs) if accs else 0

im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn",
               vmin=0, vmax=1, interpolation="bilinear",
               extent=[FRACS[0]-0.05, FRACS[-1]+0.05, -0.5, len(WIDTHS)-0.5])
ax.set_yticks(range(len(WIDTHS)))
ax.set_yticklabels(WIDTHS)
ax.set_xlabel("Train fraction $\\alpha = n / n_{\\mathrm{total}}$", fontsize=13)
ax.set_ylabel("Hidden width $d_h$", fontsize=13)
ax.set_title(f"Phase Diagram: $(a+b) \\,\\mathrm{{mod}}\\, {P}$", fontweight="bold")
plt.colorbar(im, ax=ax, label="Final test accuracy")

# n_c boundary
nc_obs = {}
for w in WIDTHS:
    for f in FRACS:
        key = f"w{w}_f{f}"
        rate = summary["per_condition"].get(key, {}).get("grok_rate", 0)
        if rate >= 0.5 and w not in nc_obs:
            nc_obs[w] = f
if len(nc_obs) >= 2:
    ws_idx = [WIDTHS.index(w) for w in sorted(nc_obs.keys())]
    fs = [nc_obs[w] for w in sorted(nc_obs.keys())]
    ax.plot(fs, ws_idx, "k--", lw=2.5, label="Observed $n_c$ boundary")
    ax.legend(loc="upper left")

plt.tight_layout()
fig.savefig(FIG_DIR / "fig2_phase_diagram.pdf")
fig.savefig(FIG_DIR / "fig2_phase_diagram.png")
plt.close()

# Fig 3: n_c vs width
print("Fig 3: critical n_c")
fig, ax = plt.subplots(figsize=(7, 5))
nc_data = {}
for w in WIDTHS:
    for f in FRACS:
        rate = summary["per_condition"].get(f"w{w}_f{f}", {}).get("grok_rate", 0)
        if rate >= 0.5 and w not in nc_data:
            nc_data[w] = f * P * P

if nc_data:
    ws = np.array(sorted(nc_data.keys()), dtype=float)
    ncs = np.array([nc_data[w] for w in sorted(nc_data.keys())], dtype=float)
    alpha = np.median(ncs / (ws * np.log(P)))

    ax.scatter(ws, ncs, s=100, color=COLORS[0], zorder=5, edgecolors="black", lw=0.8, label="Observed $n_c$")
    wl = np.linspace(min(ws)*0.7, max(ws)*1.3, 100)
    ax.plot(wl, alpha*wl*np.log(P), "--", color=COLORS[3], lw=2.5,
            label=f"CRISP: $n_c = {alpha:.2f} \\cdot w \\cdot \\ln({P})$")

    pred = alpha * ws * np.log(P)
    ss_res = ((ncs - pred)**2).sum()
    ss_tot = ((ncs - ncs.mean())**2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10) if ss_tot > 0 else float('nan')
    ax.text(0.05, 0.92, f"$R^2 = {r2:.3f}$\n$\\hat{{\\alpha}} = {alpha:.3f}$",
            transform=ax.transAxes, fontsize=13, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.7))

ax.set_xlabel("Hidden width $d_h$", fontsize=13)
ax.set_ylabel("Critical dataset size $n_c$", fontsize=13)
ax.set_title("Critical Dataset Size Scales with Width", fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig3_critical_nc.pdf")
fig.savefig(FIG_DIR / "fig3_critical_nc.png")
plt.close()

# Fig 4: Superposition dynamics
print("Fig 4: superposition dynamics")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, w in zip(axes, [64, 128]):
    for fi, f in enumerate([0.3, 0.5, 0.7, 0.9]):
        er_m, er_s, steps = agg(w, f, "eff_rank")
        if er_m is not None:
            ax.plot(steps, er_m, color=COLORS[fi], lw=1.8, label=f"$\\alpha={f}$")
            ax.fill_between(steps, er_m-er_s, er_m+er_s, alpha=0.1, color=COLORS[fi])
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective rank")
    ax.set_title(f"Width $d_h = {w}$", fontweight="bold")
    ax.legend(fontsize=10)
fig.suptitle("Representation Dynamics: Effective Rank During Training", fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig4_superposition_dynamics.pdf")
fig.savefig(FIG_DIR / "fig4_superposition_dynamics.png")
plt.close()

# Fig 5: Scaling law
print("Fig 5: scaling law")
fig, ax = plt.subplots(figsize=(8, 5.5))
for wi, w in enumerate(WIDTHS):
    ns, losses, loss_errs = [], [], []
    for f in FRACS:
        lvs = [results[f"w{w}_f{f}_s{s}"]["final_test_loss"]
               for s in SEEDS if f"w{w}_f{f}_s{s}" in results]
        if lvs:
            ns.append(f * P * P)
            losses.append(np.mean(lvs))
            loss_errs.append(np.std(lvs))
    if ns:
        ax.errorbar(ns, losses, yerr=loss_errs, fmt="o-", color=COLORS[wi],
                     lw=1.8, ms=6, capsize=3, label=f"$d_h={w}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Dataset size $n$", fontsize=13)
ax.set_ylabel("Final test loss", fontsize=13)
ax.set_title(f"Neural Scaling Curves: $(a+b) \\,\\mathrm{{mod}}\\, {P}$", fontweight="bold")
ax.legend(fontsize=11)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig5_scaling_law.pdf")
fig.savefig(FIG_DIR / "fig5_scaling_law.png")
plt.close()

# Fig 6: Grokking delay
print("Fig 6: grokking delay")
fig, ax = plt.subplots(figsize=(7, 5))
delays_all, excess_all, widths_all = [], [], []
for w in WIDTHS:
    if w not in nc_data:
        continue
    nc = nc_data[w]
    for f in FRACS:
        n = f * P * P
        if n <= nc:
            continue
        for s in SEEDS:
            r = results.get(f"w{w}_f{f}_s{s}")
            if r and r["delay"] and r["delay"] > 0:
                delays_all.append(r["delay"])
                excess_all.append((n - nc) / nc)
                widths_all.append(w)

if delays_all:
    for wi, w in enumerate(sorted(set(widths_all))):
        mask = [i for i, ww in enumerate(widths_all) if ww == w]
        dx = [excess_all[i] for i in mask]
        dy = [delays_all[i] for i in mask]
        ax.scatter(dx, dy, s=60, alpha=0.7, color=COLORS[wi], edgecolors="black",
                   lw=0.3, label=f"$d_h={w}$", zorder=5)

    # Fit power law
    lx = np.log(excess_all)
    ly = np.log(delays_all)
    slope, intercept, r, p, se = sp_stats.linregress(lx, ly)
    xf = np.linspace(min(excess_all)*0.8, max(excess_all)*1.2, 100)
    ax.plot(xf, np.exp(intercept)*xf**slope, "k--", lw=2,
            label=f"$\\tau \\propto \\Delta^{{{slope:.2f}}}$ ($R^2={r**2:.2f}$)")
    ax.set_xscale("log")
    ax.set_yscale("log")

ax.set_xlabel("$(n - n_c) / n_c$", fontsize=13)
ax.set_ylabel("Grokking delay (steps)", fontsize=13)
ax.set_title("Grokking Delay vs Distance from Critical Point", fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(FIG_DIR / "fig6_grokking_delay.pdf")
fig.savefig(FIG_DIR / "fig6_grokking_delay.png")
plt.close()

print(f"\nAll 6 figures saved to {FIG_DIR}/")
print("Done.")
