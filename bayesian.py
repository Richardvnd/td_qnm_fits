"""BGP mode selection and Bayesian amplitude fits shared by the 3_ scripts."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import ScaledTranslation
import bgp_qnm_fits as bgp

from core import load_package, get_analytical_h22np
from fitting import regularized_bgp_samples
from plotting import config, amplitude_grid, mark_times


def resolution_noise_var(package_dir, t0, n_max, data_type="strain"):
    """Noise level |h_40k - h_20k| at ``t0`` from two integration resolutions, after alignment.

    Returns (noise, alignment shift in samples).
    """
    h = {}
    for res in (20, 40):
        meta, data = load_package(package_dir, data_type=data_type, resolution=res)
        t, h_modes = get_analytical_h22np(data, meta, n_max=n_max)
        h[res] = h_modes.sum(axis=(0, 1))

    corr  = np.fft.ifft(np.fft.fft(h[40]) * np.conj(np.fft.fft(h[20])))
    shift = int(np.argmax(np.abs(corr)))
    if shift > len(corr) // 2:
        shift -= len(corr)

    residual = h[40] - np.roll(h[20], shift)
    return np.abs(residual[np.argmin(np.abs(t - t0))]), shift


def _tuned_params(noise_var, data_type):
    """White-noise kernel parameters with the noise level set to ``noise_var``."""
    tuned = bgp.get_tuned_param_dict("WN", data_type=data_type)['0001'].copy()
    tuned[(2, 2)]['sigma_max'] = noise_var
    return tuned


def run_bgp_select(t, h22, noise_var, t0_grid, n_max, T, data_type, Mf=1.0, chif=0.0):
    """Run BGP_select at every t0 in ``t0_grid``; return the selected modes per t0."""
    tuned = _tuned_params(noise_var, data_type)
    candidate_modes = [(2, 2, n, s) for n in range(n_max + 1) for s in (1, -1)]

    modes_list = []
    for t0 in t0_grid:
        print(f"  BGP_select t0={t0:.1f}", flush=True)
        sel = bgp.BGP_select(
            t, {(2, 2): h22}, [], Mf, chif, tuned, bgp.kernel_WN,
            t0=t0, candidate_modes=candidate_modes, log_threshold=np.log(0.9999),
            candidate_type="sequential", num_draws=1000, T=T,
            spherical_modes=[(2, 2)], include_chif=False, include_Mf=False,
            data_type=data_type,
        )
        modes_list.append([tuple(m) for m in sel.full_modes])
    return modes_list


def bgp_sweep(t, h22, get_modes, mode_keys, noise_var, t0_array, T, data_type,
              lambda_reg=0.0, num_samples=1000, Mf=1.0, chif=0.0):
    """BGP_fit at each t0 with modes ``get_modes(t0)``.

    Returns (median, low, high) dicts keyed by (n, sign) of decay-corrected |C| over t0 (NaN if unfitted).
    """
    tuned = _tuned_params(noise_var, data_type)
    median, low, high = ({key: np.full(len(t0_array), np.nan) for key in mode_keys}
                         for _ in range(3))

    for i, t0 in enumerate(t0_array):
        modes_t0 = get_modes(t0)
        if not modes_t0:
            continue
        print(f"  BGP fit t0={t0:.1f} [{i+1}/{len(t0_array)}]", flush=True)
        fit = bgp.BGP_fit(
            t, {(2, 2): h22}, modes_t0, Mf, chif, tuned, bgp.kernel_WN,
            t0=t0, num_samples=num_samples, t0_method="closest", T=T,
            spherical_modes=[(2, 2)], include_chif=False, include_Mf=False,
            data_type=data_type, decay_corrected=False,
        ).fit
        med, lo, hi = regularized_bgp_samples(
            fit["fisher_matrix"], fit["b_vector"], fit["ref_params"], modes_t0, t0,
            chif=chif, Mf=Mf, lambda_reg=lambda_reg, num_samples=num_samples,
        )
        for j, (_, _, n, sign) in enumerate(modes_t0):
            median[(n, sign)][i] = med[j]
            low[(n, sign)][i] = lo[j]
            high[(n, sign)][i] = hi[j]
    return median, low, high


def plot_mode_content(modes_list, t0_grid, n_max, meta, label):
    """Bars marking where BGP_select keeps each overtone (top co-, bottom counter-rotating)."""
    n_ot  = n_max + 1
    dt    = np.median(np.diff(t0_grid)) if len(t0_grid) > 1 else 1.0
    bar_height = 0.3

    fig, axes = plt.subplots(2, 1, figsize=(config.fig_width_2, config.fig_height * 1.5),
                             sharex=True, gridspec_kw={"hspace": 0.25})
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.12)

    for ax, sign, title in zip(axes, (1, -1), ("Co-rotating ($+$)", "Counter-rotating ($-$)")):
        for i, (t0, modes_at_t0) in enumerate(zip(t0_grid, modes_list)):
            width = (t0_grid[i + 1] - t0) if i < len(t0_grid) - 1 else dt
            for l, m, n, s in set(modes_at_t0):
                if (l, m) != (2, 2) or n > n_max or s != sign:
                    continue
                ax.broken_barh([(t0 - width / 2, width)],
                               (n_max - n - bar_height / 2, bar_height),
                               facecolors=[config.colors[n]],
                               alpha=max(0.15, 1.0 - 0.10 * n), edgecolor="none")

        ax.set_yticks(list(range(n_ot)))
        ax.set_yticklabels([rf"$n={n_max - i}$" for i in range(n_ot)])
        ax.set_ylabel(r"Overtone $n$")
        ax.set_xlim(t0_grid[0], t0_grid[-1])
        ax.set_ylim(-0.5, n_ot - 0.5)
        ax.set_title(f"{title} — {label}")
        mark_times(ax, meta)
        ax.tick_params(axis="y", direction="out", which="both", right=False)
        ax.xaxis.set_minor_locator(FixedLocator(np.arange(t0_grid[0], t0_grid[-1] + dt, 5)))

    axes[-1].set_xlabel(r"Start time $t_0 \, [M]$")
    return fig


CONST_COLOR, CONST_ALPHA, CONST_LW = "r", 0.5, 1.2   # late-time constant amplitudes, as fig. 1


def _bgp_legend(ax, fig, ref_label, fit_ls, show_const=False):
    """Linestyle legend, nudged 6 pt up to clear the curves below it."""
    nudge_up = ScaledTranslation(0, 6 / 72, fig.dpi_scale_trans)
    handles = [Line2D([0], [0], color="k", label=ref_label),
               Line2D([0], [0], color="k", ls=fit_ls, label=r"$|\hat{C}(t_0)|$")]
    if show_const:
        handles.append(Line2D([0], [0], color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW,
                              label=r"$|\hat{A}_{\rm const}|$"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              bbox_transform=ax.transAxes + nudge_up)


def _draw_const(ax, A_const_row, n_group, x_lim):
    """Late-time constant amplitudes as horizontal lines behind the data."""
    ax.hlines(np.abs(A_const_row[list(n_group)]), *x_lim, colors=CONST_COLOR, alpha=CONST_ALPHA,
              lw=CONST_LW, zorder=0)


def _draw_bgp(ax, n, sign, t0_array, bands, t_ref, A_ref_row, fit_ls):
    """Reference amplitude (faded) plus posterior median and 25-75% band."""
    color = config.colors[n]
    ax.semilogy(t_ref, np.abs(A_ref_row), color=color, alpha=0.5)
    median, low, high = bands
    if (n, sign) in median:
        ax.fill_between(t0_array, low[(n, sign)], high[(n, sign)], color=color, alpha=0.25)
        ax.semilogy(t0_array, median[(n, sign)], color=color, ls=fit_ls)


def _overtone_legend(ax, n_group, ncol):
    ax.add_artist(ax.legend(
        handles=[Line2D([0], [0], color=config.colors[n], label=rf"$n={n}$") for n in n_group],
        fontsize=7, ncol=ncol, loc="lower right", frameon=False))


def plot_bgp_amplitudes(t0_array, bands, t_ref, A_ref, meta, n_max, x_lims, y_lim,
                        ref_label=r"$|\hat{A}(t)|$", fit_ls="--", A_const=None):
    """BGP medians and 25-75% bands against ``A_ref`` on the 2x2 grid; ``x_lims`` per column.

    ``A_const`` (2, n_max + 1), if given, is drawn as horizontal lines.
    """
    fig, axes, n_groups = amplitude_grid(n_max)
    for row, sign in enumerate((1, -1)):
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            for n in n_group:
                _draw_bgp(ax, n, sign, t0_array, bands, t_ref, A_ref[row, n], fit_ls)
            if A_const is not None:
                _draw_const(ax, A_const[row], n_group, x_lims[col])
            mark_times(ax, meta)
            ax.set_xlim(*x_lims[col])
            ax.set_ylim(*y_lim)
            if row == 0:
                _overtone_legend(ax, n_group, ncol=2 if col == 0 else 1)
    _bgp_legend(axes[1, 1], fig, ref_label, fit_ls, show_const=A_const is not None)
    return fig


def plot_bgp_amplitudes_low(t0_array, bands, t_ref, A_ref, meta, n_max, x_lim, y_lim,
                            ref_label=r"$|\hat{A}(t)|$", fit_ls="--", A_const=None):
    """Half-width :func:`plot_bgp_amplitudes` for the low overtones only."""
    n_group = list(range((n_max + 1) // 2))
    fig, axes = plt.subplots(2, 1, figsize=(config.fig_width_2 / 2, config.fig_height * 1.7),
                             sharex=True, gridspec_kw={"hspace": 0.1})
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.22, right=0.95, top=0.93, bottom=0.10)

    for row, (sign, row_label) in enumerate([(1, "Co-rotating"), (-1, "Counter-rotating")]):
        ax = axes[row]
        for n in n_group:
            _draw_bgp(ax, n, sign, t0_array, bands, t_ref, A_ref[row, n], fit_ls)
        if A_const is not None:
            _draw_const(ax, A_const[row], n_group, x_lim)
        mark_times(ax, meta)
        ax.set_xlim(*x_lim)
        ax.set_ylim(*y_lim)
        ax.set_ylabel(f"{row_label}\nAmplitude")

    axes[0].tick_params(labelbottom=False)
    axes[0].set_title(rf"$n = {n_group[0]}$--${n_group[-1]}$")
    _overtone_legend(axes[0], n_group, ncol=2)
    axes[-1].set_xlabel(r"$t_0$ [$M$]")
    _bgp_legend(axes[-1], fig, ref_label, fit_ls, show_const=A_const is not None)
    return fig
