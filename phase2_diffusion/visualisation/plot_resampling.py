"""
Visualisation: Variable-length Q-path → Fixed H=3 waypoints via arc-length resampling.
For inclusion in the paper.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Reproducible example ────────────────────────────────────────────────────
GRID_SIZE = 12

raw_path_grid = np.array([
    [0,0],[0,1],[0,2],[0,3],[0,4],[0,5],
    [1,5],[2,5],[3,5],[4,5],[5,5]
], dtype=float)   # shape (11, 2)

H = 3

# ── Helpers (mirror project code) ───────────────────────────────────────────

def normalize(coords, gs=GRID_SIZE):
    return 2.0 * coords / (gs - 1) - 1.0

def denormalize(coords, gs=GRID_SIZE):
    return (coords + 1.0) / 2.0 * (gs - 1)

def resample_path(path, H):
    diffs  = np.diff(path, axis=0)
    dists  = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(dists)])
    total  = cumlen[-1]
    target = np.linspace(0.0, total, H)
    out = np.zeros((H, 2))
    for i, s in enumerate(target):
        idx  = int(np.searchsorted(cumlen, s, side="right")) - 1
        idx  = np.clip(idx, 0, len(path) - 2)
        frac = (s - cumlen[idx]) / max(float(dists[idx]), 1e-8)
        out[i] = path[idx] + frac * (path[idx+1] - path[idx])
    return out, cumlen, target

# ── Compute ──────────────────────────────────────────────────────────────────
path_norm             = normalize(raw_path_grid)
wp_norm, cumlen, tgts = resample_path(path_norm, H)
wp_grid               = denormalize(wp_norm)

# ── Colour palette ───────────────────────────────────────────────────────────
C_PATH   = "#4C72B0"
C_WP     = "#DD4444"
C_RAW    = "#888888"
C_FILL   = "#EEF2FF"
C_GRID   = "#CCCCCC"
C_ARC    = "#2E8B57"
C_TARGET = "#DD4444"
C_BAND   = "#FFF3CD"

# ── Figure layout ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 5.2))
fig.patch.set_facecolor("white")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

# ═══════════════════════════════════════════════════════════════════════════
# Panel A — raw Q-path on grid
# ═══════════════════════════════════════════════════════════════════════════
ax1 = fig.add_subplot(gs[0])

# grid background
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        rect = mpatches.FancyBboxPatch(
            (c, GRID_SIZE - 1 - r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=C_FILL, edgecolor=C_GRID, linewidth=0.5
        )
        ax1.add_patch(rect)

# raw path
cols = raw_path_grid[:, 1] + 0.5
rows = (GRID_SIZE - 1 - raw_path_grid[:, 0]) + 0.5

ax1.plot(cols, rows, "-", color=C_PATH, linewidth=2.2, zorder=3, label="Q-agent path")
ax1.scatter(cols[1:-1], rows[1:-1], s=35, color=C_PATH, zorder=4)
ax1.scatter(cols[0],  rows[0],  s=120, color="#2ECC71", zorder=5,
            marker="^", label="Start (0,0)")
ax1.scatter(cols[-1], rows[-1], s=120, color="#E74C3C", zorder=5,
            marker="*", label="End (5,5)")

# step labels
for i, (r, c) in enumerate(raw_path_grid):
    ax1.text(c + 0.5, GRID_SIZE - 1 - r + 0.5 + 0.28,
             str(i), fontsize=5.5, ha="center", va="bottom",
             color="#333333", zorder=6)

ax1.set_xlim(0, GRID_SIZE)
ax1.set_ylim(0, GRID_SIZE)
ax1.set_aspect("equal")
ax1.set_xticks(range(GRID_SIZE + 1))
ax1.set_yticks(range(GRID_SIZE + 1))
ax1.tick_params(labelsize=6)
ax1.set_xticklabels(range(GRID_SIZE + 1), fontsize=6)
ylab = list(range(GRID_SIZE, -1, -1))
ax1.set_yticklabels(ylab, fontsize=6)
ax1.set_title("(A)  Raw Q-path  (10 steps, 11 positions)", fontsize=10, fontweight="bold", pad=8)
ax1.set_xlabel("Column", fontsize=8)
ax1.set_ylabel("Row", fontsize=8)
ax1.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

# annotate step count
ax1.text(0.5, 0.04, "Shape:  (11, 2)  — variable length",
         transform=ax1.transAxes, fontsize=7.5,
         ha="center", color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#CCBB00", alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════
# Panel B — arc-length curve + target sampling
# ═══════════════════════════════════════════════════════════════════════════
ax2 = fig.add_subplot(gs[1])
ax2.set_facecolor("#FAFAFA")

n_steps = len(cumlen)
x_idx   = np.arange(n_steps)

# arc-length curve
ax2.plot(x_idx, cumlen, "o-", color=C_ARC, linewidth=2.2,
         markersize=7, markerfacecolor="white", markeredgewidth=1.8, zorder=4,
         label="Cumulative arc-length")

# label each point with its cumlen value
for i, cl in enumerate(cumlen):
    ax2.text(i, cl + 0.04, f"{cl:.3f}", fontsize=5.8,
             ha="center", va="bottom", color="#333333")

# H=3 target lines  (horizontal dashed → vertical drop)
colours_t = ["#2ECC71", "#E67E22", "#E74C3C"]
wp_labels = ["w₀ = 0.000", "w₁ = 0.909", "w₂ = 1.818"]
wp_grid_labels = ["(0,0)", "(0,5)", "(5,5)"]
hit_indices = [0, 5, 10]   # which path indices the targets hit

for ti, (s, col, wpl, gpl, hi) in enumerate(
        zip(tgts, colours_t, wp_labels, wp_grid_labels, hit_indices)):
    # horizontal dashed line from y-axis to curve
    ax2.axhline(s, xmin=0, xmax=hi / (n_steps - 1),
                color=col, linestyle="--", linewidth=1.4, alpha=0.8)
    # vertical drop from curve to x-axis
    ax2.axvline(hi, ymin=0, ymax=s / cumlen[-1],
                color=col, linestyle=":", linewidth=1.4, alpha=0.7)
    # highlight the hit point
    ax2.scatter(hi, s, s=110, color=col, zorder=6, edgecolors="white", linewidth=1.2)
    # label
    ax2.text(-0.18, s, wpl, fontsize=7, color=col, va="center", fontweight="bold")
    ax2.text(hi, -0.12, f"step {hi}\n{gpl}",
             fontsize=6.2, ha="center", va="top", color=col)

ax2.set_xlim(-0.5, n_steps - 0.5)
ax2.set_ylim(-0.25, cumlen[-1] + 0.22)
ax2.set_xticks(x_idx)
ax2.set_xticklabels([str(i) for i in x_idx], fontsize=7)
ax2.set_xlabel("Step index along Q-path", fontsize=8)
ax2.set_ylabel("Cumulative arc-length (normalised)", fontsize=8)
ax2.set_title("(B)  Arc-length curve  →  H=3 equally-spaced targets",
              fontsize=10, fontweight="bold", pad=8)
ax2.legend(loc="upper left", fontsize=7, framealpha=0.9)
ax2.grid(axis="y", linestyle=":", alpha=0.4)

ax2.text(0.5, 0.04,
         "linspace(0, 1.818, 3) = [0.000,  0.909,  1.818]",
         transform=ax2.transAxes, fontsize=7.5,
         ha="center", color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4", edgecolor="#CCBB00", alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════
# Panel C — resampled H=3 waypoints on grid
# ═══════════════════════════════════════════════════════════════════════════
ax3 = fig.add_subplot(gs[2])

# grid background
for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        rect = mpatches.FancyBboxPatch(
            (c, GRID_SIZE - 1 - r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=C_FILL, edgecolor=C_GRID, linewidth=0.5
        )
        ax3.add_patch(rect)

# faded original path for reference
ax3.plot(cols, rows, "-", color=C_PATH, linewidth=1.0,
         alpha=0.25, zorder=2, label="Original path (reference)")
ax3.scatter(cols, rows, s=18, color=C_PATH, alpha=0.25, zorder=3)

# resampled waypoints
wp_c = wp_grid[:, 1] + 0.5
wp_r = (GRID_SIZE - 1 - wp_grid[:, 0]) + 0.5

wp_colors = ["#2ECC71", "#E67E22", "#E74C3C"]
wp_names  = ["w₀", "w₁", "w₂"]
grid_names = ["(0, 0)", "(0, 5)", "(5, 5)"]
norm_names  = ["(−1.00, −1.00)", "(−1.00, −0.09)", "(−0.09, −0.09)"]

# absolute label positions in the empty right half of the grid
label_xy = [(7.2, 10.8), (7.2, 8.5), (7.2, 6.2)]

for i in range(H):
    ax3.scatter(wp_c[i], wp_r[i], s=220, color=wp_colors[i],
                zorder=6, edgecolors="white", linewidth=1.5,
                label=f"{wp_names[i]}  grid {grid_names[i]}")
    lx, ly = label_xy[i]
    ax3.annotate(
        f"{wp_names[i]}  {grid_names[i]}\n{norm_names[i]}",
        xy=(wp_c[i], wp_r[i]),
        xytext=(lx, ly),
        fontsize=6.5, color=wp_colors[i], fontweight="bold", zorder=7,
        arrowprops=dict(arrowstyle="-", color=wp_colors[i], lw=1.0, alpha=0.7),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor=wp_colors[i], linewidth=1.1, alpha=0.95)
    )

# connect waypoints with dashed line
ax3.plot(wp_c, wp_r, "--", color="#888888", linewidth=1.6,
         alpha=0.7, zorder=4)

ax3.set_xlim(0, GRID_SIZE)
ax3.set_ylim(0, GRID_SIZE)
ax3.set_aspect("equal")
ax3.set_xticks(range(GRID_SIZE + 1))
ax3.set_yticks(range(GRID_SIZE + 1))
ax3.tick_params(labelsize=6)
ax3.set_xticklabels(range(GRID_SIZE + 1), fontsize=6)
ax3.set_yticklabels(ylab, fontsize=6)
ax3.set_title("(C)  Resampled H=3 waypoints  (diffusion input)", fontsize=10, fontweight="bold", pad=8)
ax3.set_xlabel("Column", fontsize=8)
ax3.set_ylabel("Row", fontsize=8)
ax3.legend(loc="upper right", fontsize=6.5, framealpha=0.9)

ax3.text(0.5, 0.04, "Shape:  (3, 2)  — fixed  →  diffusion input",
         transform=ax3.transAxes, fontsize=7.5,
         ha="center", color="#555555",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#D5F5E3", edgecolor="#27AE60", alpha=0.9))


# ═══════════════════════════════════════════════════════════════════════════
# Global title + save
# ═══════════════════════════════════════════════════════════════════════════
fig.suptitle(
    "Variable-length Q-path  →  Fixed H=3 Waypoints via Arc-length Resampling",
    fontsize=12, fontweight="bold", y=1.02
)

out = os.path.join(_ROOT, "results", "resampling_pipeline.png")
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved → {out}")
