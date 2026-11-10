#!/usr/bin/env python3
"""The result that a table cannot show as well as a picture.

The figure exists because its point is a *shape*, not a value: what the learned
state adds behind a fixed selection is the distance between two curves, and
that distance widens and then narrows with scale. A reader can extract it from
the table, but only by doing the subtraction and the plotting mentally.

Conventions chosen for a paper rather than a screen: vector PDF, no colour
carrying information that greyscale would lose (markers and line styles differ
too), font sizes matched to 11pt body text, and no gridlines or frames beyond
what is needed to read a value.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# ---------------------------------------------------------------- figure 1
# What the state adds behind a fixed selection, across scale. Lower block of
# the scale table in the paper (v2, re-evaluated after the scoring fix).
sizes = [0.6, 1.7, 4, 8, 14]
h2o_slots = [96.3, 103.0, 104.1, 98.0, 94.0]     # H2O per head 512 + 64 slots
h2o_alone = [77.8, 80.2, 80.7, 80.6, 81.5]       # H2O per head, all 576
ours = [81.5, 89.9, 91.6, 88.5, 82.6]            # speculative 512 + 64 slots
ours_alone = [46.3, 51.1, 55.9, 60.0, 62.8]      # same anchors, no state

gain_spec = [a - b for a, b in zip(ours, ours_alone)]
gain_snap = [29.5, 33.6, 32.1, 24.4, 19.6]       # SnapKV 512 + slots minus 576
gain_h2o = [a - b for a, b in zip(h2o_slots, h2o_alone)]

# Two panels. A single panel shading the band between each pair was tried and
# the two bands overlapped into what looked like a third; the gain is the
# point, so it gets its own axis instead of being left as a band to read.
fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.4, 2.6),
                             gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.75})
for top, bot, col, mk in ((h2o_slots, h2o_alone, "#1a1a1a", "o"),
                          (ours, ours_alone, "#777777", "s")):
    ax.plot(sizes, top, mk + "-", color=col, lw=1.6, ms=4.5, zorder=4)
    ax.plot(sizes, bot, mk + "--", color=col, lw=1.1, ms=3.5, zorder=3,
            markerfacecolor="white")
# Direct labels; H2O alone and our method end within a point of each other at
# 14B, so those two are nudged apart rather than printed on top of each other.
for y, txt, col in ((h2o_slots[-1], "H2O 512 + 64 slots", "#1a1a1a"),
                    (ours[-1] + 3.5, "ours: 512 + 64 slots", "#777777"),
                    (h2o_alone[-1] - 3.5, "H2O 576 alone", "#1a1a1a"),
                    (ours_alone[-1], "our anchors alone", "#777777")):
    ax.text(17.5, y, txt, fontsize=7, color=col, va="center", ha="left")
ax.set_ylabel("recovery (%)")
ax.set_ylim(40, 110)
ax.set_title("(a) with and without the slots", loc="left")

for g, col, mk, ls, lab in ((gain_spec, "#777777", "s", "-", "behind our anchors"),
                            (gain_snap, "#444444", "^", "-.", "behind SnapKV"),
                            (gain_h2o, "#1a1a1a", "o", "-", "behind H2O")):
    bx.plot(sizes, g, mk + ls, color=col, lw=1.4, ms=4, label=lab)
bx.set_ylabel("what the slots add (points)")
bx.set_ylim(0, 45)
bx.legend(loc="lower left", frameon=False, handlelength=2.2)
bx.set_title("(b) the difference", loc="left")

for a in (ax, bx):
    a.set_xscale("log")
    a.set_xticks(sizes)
    a.set_xticklabels(["0.6B", "1.7B", "4B", "8B", "14B"])
    a.minorticks_off()
    a.set_xlabel("backbone size")
ax.set_xlim(0.5, 14 * 2.6)
bx.set_xlim(0.5, 18)
fig.savefig(HERE / "fig_scale.pdf")

# v1 also drew fig_inversion.pdf (rank of four policies on text vs needle).
# Withdrawn in v2: its needle ranking came from the broken selector scoring.
plt.close(fig)

for f in ("fig_scale.pdf",):
    print(f"{f}: {(HERE / f).stat().st_size} bytes")
