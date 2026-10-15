"""Shared figure helpers; importing applies the stylesheet.

Linestyles: solid = analytical, dashed = fit to analytical data, dotted = fit to RWZ data.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullLocator

from core import peak_frame_times
from plot_config import PlotConfig

config = PlotConfig()
config.apply_style()

LS_FIT_ANALYTICAL = "--"
LS_FIT_PERTURBATIVE = ":"


def fit_linestyle(use_rwz):
    return LS_FIT_PERTURBATIVE if use_rwz else LS_FIT_ANALYTICAL


def overtone_groups(n_max):
    """Split overtones 0..n_max into a low and a high half."""
    mid = (n_max + 1) // 2
    return [list(range(mid)), list(range(mid, n_max + 1))]


def mark_times(ax, meta, **kw):
    """Vertical dotted lines at the light-ring and bounce times."""
    times = peak_frame_times(meta)
    ax.axvline(times["t_lr"], color=config.color_lr, ls=":", **kw)
    ax.axvline(times["t_bounce"], color=config.color_bounce, ls=":", **kw)


def lambda_label(lam):
    """LaTeX label for ridge strength ``lam`` (lambda^2 in the paper)."""
    if lam == 0:
        return r"$\lambda^2=0$"
    exponent = np.log10(lam)
    if abs(exponent - round(exponent)) < 1e-9:
        return rf"$\lambda^2=10^{{{int(round(exponent))}}}$"
    return rf"$\lambda^2=10^{{{exponent:.1f}}}$"


def amplitude_grid(n_max, row_labels=("Co-rotating", "Counter-rotating"),
                   ylabel="Amplitude", height=1.7):
    """2x2 grid: rows co-/counter-rotating, columns low/high overtones. Returns (fig, axes, n_groups)."""
    n_groups = overtone_groups(n_max)
    fig, axes = plt.subplots(
        2, 2,
        figsize=(config.fig_width_2, config.fig_height * height),
        sharex="col",
        sharey="row",
        gridspec_kw={"hspace": 0.1, "wspace": 0.04},
    )
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.93, bottom=0.10)

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(f"{label}\n{ylabel}")
        axes[row, 1].tick_params(labelleft=False)
    for col, group in enumerate(n_groups):
        axes[0, col].tick_params(labelbottom=False)
        axes[0, col].set_title(rf"$n = {group[0]}$--${group[-1]}$")
        axes[1, col].set_xlabel(r"$t_0$ [$M$]")
    return fig, axes, n_groups


def overtone_stack(n_max, ncols=2, width_ratios=None, **adjust):
    """One row per overtone, low/high-n blocks split by a hidden spacer row.

    Returns (fig, axes, grid_row, mid): ``grid_row(n)`` is overtone n's row; n < mid is the top block.
    """
    mid = (n_max + 1) // 2
    height_ratios = [1] * mid + [0.25] + [1] * (n_max + 1 - mid)
    gridspec_kw = {"hspace": 0.1, "wspace": 0.04, "height_ratios": height_ratios}
    if width_ratios is not None:
        gridspec_kw["width_ratios"] = width_ratios
    fig, axes = plt.subplots(n_max + 2, ncols,
                             figsize=(config.fig_width_2, config.fig_height * 3),
                             gridspec_kw=gridspec_kw)
    fig.set_layout_engine("none")
    fig.subplots_adjust(**{"left": 0.16, "right": 0.97, "top": 0.95, "bottom": 0.11,
                           **adjust})
    for ax in axes[mid]:
        ax.set_visible(False)
    return fig, axes, (lambda n: n if n < mid else n + 1), mid


def log_ylim(arrays, pad_frac=0.0):
    """(ymin, ymax) spanning ``arrays``, padded by ``pad_frac`` of the log range (min one decade)."""
    lo = np.log10(min(np.min(a) for a in arrays))
    hi = np.log10(max(np.max(a) for a in arrays))
    pad = pad_frac * max(hi - lo, 1.0)
    return 10 ** (lo - pad), 10 ** (hi + pad)


def plot_sweep_stack(t0_array, C_norm_list, labels, t_td, A_td, meta, modes, n_max,
                     x_lim, x_lim_high, y_log_pad, legend_ncol):
    """Overtone-per-row stack: analytical |A| (black) against one dashed fit per ``C_norm_list`` entry."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    colors = config.colors2(len(C_norm_list))
    fig, axes, grid_row, mid = overtone_stack(n_max)

    for n in range(n_max + 1):
        xlim = x_lim if n < mid else x_lim_high
        mask_td = (t_td >= xlim[0]) & (t_td <= xlim[1])
        mask_t0 = (t0_array >= xlim[0]) & (t0_array <= xlim[1])

        row_vals = [C_norm[mask_t0, mode_col[(sign, n)]]
                    for sign in (1, -1) if (sign, n) in mode_col
                    for C_norm in C_norm_list]
        row_vals += [np.abs(A_td[p, n])[mask_td] for p in (0, 1)]
        row_ylim = log_ylim(row_vals, y_log_pad if n in (0, 1) else 0.0)

        for col, sign in enumerate((1, -1)):
            ax = axes[grid_row(n), col]
            ax.semilogy(t_td[mask_td], np.abs(A_td[col, n])[mask_td], color="k", lw=0.5)
            col_idx = mode_col.get((sign, n))
            if col_idx is not None:
                for color, C_norm in zip(colors, C_norm_list):
                    ax.semilogy(t0_array, C_norm[:, col_idx], color=color, ls="--", lw=0.6)
            mark_times(ax, meta, lw=0.5)
            ax.set_xlim(*xlim)
            ax.set_ylim(*row_ylim)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.text(0.06, 0.94, rf"$n={n}$", transform=ax.transAxes, va="top", ha="left")
            if col == 0:
                ax.set_ylabel(r"$|\hat{C}|$")
            else:
                ax.tick_params(labelleft=False)
            if n not in (mid - 1, n_max):
                ax.tick_params(labelbottom=False)

    axes[0, 0].set_title("Co-rotating")
    axes[0, 1].set_title("Counter-rotating")
    for col in range(2):
        axes[grid_row(n_max), col].set_xlabel(r"$t_0$ [$M$]")

    handles = [Line2D([0], [0], color=c, ls="--", lw=1.2, label=label)
               for c, label in zip(colors, labels)]
    handles.append(Line2D([0], [0], color="k", lw=1.0, label=r"$|\hat{A}(t)|$"))
    fig.legend(handles=handles, loc="lower center", ncol=legend_ncol, fontsize=6,
               frameon=False, bbox_to_anchor=(0.56, 0.0))
    return fig


def save(fig, name, **kw):
    """Save to figs/<name>.pdf, with no minor ticks on log axes."""
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if axis.get_scale() == "log":
                axis.set_minor_locator(NullLocator())
    os.makedirs("figs", exist_ok=True)
    path = f"figs/{name}.pdf"
    fig.savefig(path, dpi=300, **kw)
    print(f"Saved {path}")
    return path
