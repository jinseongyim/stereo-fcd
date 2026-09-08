"""
Figure 1 for the note.

Three panels, each from a different experiment, labelled with its pool so the
reader is never invited to compare across them numerically:

  (a) ChEBI-20 / MolT5-large. FCD before and after inverting every double bond,
      drawn against its own +/-1 sigma resampling band. The bars are the same
      height; that is the point.
  (b) The same molecules, scored against per-molecule ground truth with RDKit.
      Per-bond E/Z correctness collapses 98.97% -> 1.03% (actual inverted molecules; step 35).
  (c) ChEMBL controlled pool. What FCD *does* respond to, in sigma units.
      flip100 and strip_ez carry bootstrap 95% CIs over 15 independent draws;
      the two substitution conditions were run once, in a draw whose sigma
      agrees with the replicated estimate to within 2%.

Sources: c2_full_results.json, cip_recount.json,
         flip_vs_strip_onepool.json, flip_vs_strip_repeated.json
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).parent

c2 = json.loads((HERE / "c2_full_results.json").read_text())
cc = json.loads((HERE / "c2_full_correct.json").read_text())  # correct-inversion headline
sd = json.loads((HERE / "cip_recount.json").read_text())
rep = json.loads((HERE / "flip_vs_strip_repeated.json").read_text())

REAL = "#4C72B0"
INV = "#C44E52"
GREY = "#8C8C8C"
BAND = "#B8CBE3"

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
})

fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(9.0, 3.0))

# ---------------------------------------------------------------- panel (a)
f_real = cc["real"]
f_inv = cc["inverted"]
f_sd = cc["noise_sd"]
f_dsd = cc["delta_in_sd"]

# markers, not bars: the y-axis is zoomed to a few sigma and truncated bars
# would exaggerate the (absent) difference
axA.axhspan(f_real - f_sd, f_real + f_sd, color=BAND, alpha=0.6, lw=0,
            label=r"$\pm 1\,\sigma_{\rm noise}$ about 'as generated'")
axA.axhline(f_real, color=GREY, lw=0.8, ls="-", zorder=2)
axA.plot([0], [f_real], "o", ms=11, color=REAL, zorder=4)
axA.plot([1], [f_inv], "o", ms=11, color=INV, zorder=4)
axA.set_xticks([0, 1])
axA.set_xticklabels(["as generated", "every C=C\ninverted"])
axA.set_xlim(-0.55, 1.55)
axA.set_ylabel("FCD  (lower = better)")

lo = f_real - 3.0 * f_sd
hi = f_real + 3.6 * f_sd
axA.set_ylim(lo, hi)
axA.set_title("(a)  what the metric sees", loc="left", fontweight="bold")
axA.annotate(rf"$\Delta = {f_dsd:.3f}\,\sigma$", xy=(0.5, f_real + 2.4 * f_sd),
             ha="center", fontsize=9.5)
axA.legend(frameon=False, loc="lower center", fontsize=7)
axA.set_xlabel(f"ChEBI-20 test · MolT5-large · n = {c2['n_gen']}",
               fontsize=7, color=GREY, labelpad=7)

# ---------------------------------------------------------------- panel (b)
ok_asis = 100 * sd["correct"] / sd["assigned"]
ok_inv = 100 * sd["correct_inverted"] / sd["assigned"]

axB.bar([0, 1], [ok_asis, ok_inv], width=0.55, color=[REAL, INV], zorder=3)
axB.set_xticks([0, 1])
axB.set_xticklabels(["as generated", "every C=C\ninverted"])
axB.set_ylabel("per-bond E/Z correct")
axB.yaxis.set_major_formatter(PercentFormatter())
axB.set_ylim(0, 108)
axB.set_title("(b)  what actually changed", loc="left", fontweight="bold")
for x, v in [(0, ok_asis), (1, ok_inv)]:
    axB.text(x, v + 3.5, f"{v:.2f}%", ha="center", fontsize=9.5)
axB.set_xlabel(f"skeleton-matched predictions · n = {sd['assigned']} stereo bonds",
               fontsize=7, color=GREY, labelpad=7)

# ---------------------------------------------------------------- panel (c)
labels = [
    "every E/Z toggled",
    "E/Z deleted,\nchirality kept",
    "2% molecules\nreplaced",
    "5% molecules\nreplaced",
]
# all four conditions come from the same 15-draw replication, so they share one
# noise estimate (sigma from baseline spread across draws) and each carries a
# 95% CI over draws -- no protocol mixing between panels
vals = [rep["flip100"]["mean_delta_in_sd"],
        rep["strip_ez"]["mean_delta_in_sd"],
        abs(rep["swap2"]["mean_delta_in_sd"]),
        abs(rep["swap5"]["mean_delta_in_sd"])]
cis = [rep["flip100"]["ci95_delta_in_sd"],
       rep["strip_ez"]["ci95_delta_in_sd"],
       [abs(x) for x in rep["swap2"]["ci95_delta_in_sd"]],
       [abs(x) for x in rep["swap5"]["ci95_delta_in_sd"]]]
cols = [INV, GREY, GREY, GREY]

# dots, not bars: on a log axis a bar's length depends on where xlim is set,
# so only the marker position is meaningful
for y, v, c, ci in zip([3, 2, 1, 0], vals, cols, cis):
    if ci is not None:
        axC.plot(ci, [y, y], "-", color=c, lw=2.2, solid_capstyle="butt", zorder=3)
    axC.plot([v], [y], "o", ms=10, color=c, zorder=4)
axC.grid(axis="x", which="major", color="0.9", lw=0.7, zorder=0)
axC.set_yticks([3, 2, 1, 0])
axC.set_yticklabels(labels, fontsize=8)
axC.set_xscale("log")
axC.set_xlim(3e-3, 3e3)
axC.set_ylim(-0.55, 3.55)
axC.set_xlabel("FCD response  ($\\sigma_{\\rm noise}$, log scale)\n"
               f"ChEMBL, n = {rep['n']} per set; {rep['n_draws']} draws, "
               f"bars = 95% CI over draws",
               fontsize=7.5, color="black", labelpad=6)
axC.axvline(1.0, color="k", ls=":", lw=0.9, zorder=2)
axC.text(1.25, -0.42, r"$1\,\sigma$", fontsize=7.5, ha="left")
axC.set_title("(c)  what it responds to instead", loc="left", fontweight="bold")
for y, v in zip([3, 2, 1, 0], vals):
    axC.text(v * 1.6, y, f"{v:.3f}" if v < 0.1 else f"{v:.1f}",
             va="center", fontsize=8.5)

fig.tight_layout(pad=1.1)
for ext in ("png", "pdf"):
    out = HERE / f"figure2.{ext}"
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")

print(f"\n(a) FCD {f_real:.6f} -> {f_inv:.6f}   = {f_dsd:.4f} sigma")
print(f"(b) per-bond E/Z {ok_asis:.2f}% -> {ok_inv:.2f}%  (n={sd['assigned']})")
print(f"(c) flip100 {vals[0]:.4g} | strip_ez {vals[1]:.2f} | "
      f"swap2 {vals[2]:.2f} | swap5 {vals[3]:.2f}   "
      f"-> strip/flip = {vals[1]/vals[0]:.0f}x, swap2/flip = {vals[2]/vals[0]:.0f}x")
