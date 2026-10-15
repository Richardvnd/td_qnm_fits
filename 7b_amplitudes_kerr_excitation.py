"""Fig. 7 plus the Della Rocca et al. (arXiv:2512.07959) excitation coefficients as constant lines (red).

C_22n (n = 0, 1, 2, prograde only) are critical-plunge-from-ISCO values, so they apply to the Teukode
(test-particle) curves. Each line is |C_n/C_0| exp[(Im w_n - Im w_0) U0], where U0 is the paper's
retarded time u at Teukode t = 0, so it shares the (2,2,0,+) = 1 normalisation.

Inputs, all digitised from the paper's vector figures:
- C_22n(a = 0.6921): their Fig. 3 (|C| and arg vs a/M); this matches their Table I to 4 s.f. at a = 0.4, 0.9.
- U_LR = -5.18 M: light-ring retarded time in their u. Their Fig. 2 (a = 0.68) mode sums equal
  k sum_n C_22n exp(-i w_n u) with one real k = 0.332 for n = 0-2 (0.2% in |.|, 0.07 rad in phase).
- DELTA = -7.58 M: Teukode t = (u - U_LR) - DELTA, from matching the Teukode h22 to their full
  waveform (Fig. 2, black), 2.8% rms residual. Varying the match window moves it by +/-0.3 M.
"""

import numpy as np
from matplotlib.lines import Line2D

from core import load_teukode_data, load_sxs_data, mode_set, qnm_frequencies
from fitting import fit_amplitudes
from plotting import config, amplitude_grid, save

# ── Configuration ──────────────────────────────────────────────────────────────
SXS_ID        = "SXS:BBH:0305"
TEUK_DIR      = "data_all"   # private Teukode waveform
TEUK_CHIF     = 0.6921
TEUK_MF       = 1.0
TEUK_T_SHIFT  = 6.0        # [M] shift that aligns the Teukode fits with NR
LAMBDA_REG    = 1e-4
N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.1)
T_FIT         = 100.0
INCLUDE_RETRO = True

NORM_T0 = 10.0   # [M] normalise (2,2,0,+) amplitude to 1 at this t0

# Della Rocca et al. excitation coefficients and time alignment (see docstring)
C_EXC = np.array([-0.0601 - 0.6883j, -0.2336 + 0.3538j, 0.1624 - 0.2003j])   # C_22n / m_p, n = 0, 1, 2
U_LR  = -5.18    # [M]
DELTA = -7.58    # [M]
DELTA_ERR = 0.3  # [M], drawn as a band
CONST_COLOR, CONST_ALPHA, CONST_LW = "r", 0.5, 1.2

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_LIM = (1e-4, 1e11)

MODES = mode_set(N_MAX, INCLUDE_RETRO)


def excitation_lines(delta):
    """|C_n/C_0| exp[(Im w_n - Im w_0) U0] for n = 0, 1, 2 (prograde), with U0 = U_LR + delta."""
    omega = qnm_frequencies([(2, 2, n, 1) for n in range(len(C_EXC))], TEUK_CHIF, TEUK_MF)
    u0 = U_LR + delta
    return np.abs(C_EXC / C_EXC[0]) * np.exp((omega.imag - omega[0].imag) * u0)


def plot_amplitudes(t0_array, C_nr, C_teuk, A_exc, A_exc_band, modes):
    """NR (dash-dot) vs Teukode (dotted) amplitudes; excitation-coefficient constants in red."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    fig, axes, n_groups = amplitude_grid(N_MAX, row_labels=("Prograde", "Retrograde"),
                                         ylabel=r"$|\hat{C}|$", height=1.5)

    for row, sign in enumerate((1, -1)):
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            xlim = X_LIM if col == 0 else X_LIM_HIGH
            for n in n_group:
                idx = mode_col[(sign, n)]
                color = config.colors[n]
                ax.semilogy(t0_array, C_nr[:, idx], color=color, ls="-.", label=rf"$n={n}$")
                ax.semilogy(t0_array, C_teuk[:, idx], color=color, ls=":")
                if sign == 1 and n < len(A_exc):
                    ax.axhspan(*A_exc_band[:, n], color=CONST_COLOR, alpha=0.15, lw=0, zorder=0)
                    ax.hlines(A_exc[n], *xlim, colors=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, zorder=0)
            ax.set_xlim(*xlim)
            ax.set_ylim(*Y_LIM)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="upper left", frameon=False))

    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", ls="-.", label="NR"),
        Line2D([0], [0], color="k", ls=":", label="Perturbative"),
        Line2D([0], [0], color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, label=r"$|C_{22n}|$ (critical plunge)"),
    ], loc="lower right")
    return fig


def main():
    print(f"Loading {SXS_ID}...", flush=True)
    t_nr, h22_nr, Mf, chif = load_sxs_data(SXS_ID)

    print(f"Loading Teukode data (a={TEUK_CHIF})...", flush=True)
    t_teuk, h22_teuk = load_teukode_data(TEUK_DIR, t_shift=TEUK_T_SHIFT)

    print("Fitting NR and Teukode data...", flush=True)
    C_nr = fit_amplitudes(t_nr, h22_nr, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG,
                          Mf=Mf, chif=chif, spherical_modes=[(2, 2)])
    C_teuk = fit_amplitudes(t_teuk, h22_teuk, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG,
                            Mf=TEUK_MF, chif=TEUK_CHIF, spherical_modes=[(2, 2)])

    ref_col = MODES.index((2, 2, 0, 1))
    ref_idx = np.argmin(np.abs(T0_ARRAY - NORM_T0))
    C_nr   = C_nr / C_nr[ref_idx, ref_col]
    C_teuk = C_teuk / C_teuk[ref_idx, ref_col]

    A_exc = excitation_lines(DELTA)
    A_exc_band = np.sort([excitation_lines(DELTA - DELTA_ERR), excitation_lines(DELTA + DELTA_ERR)], axis=0)
    for n, a in enumerate(A_exc):
        fit_vals = C_teuk[:, MODES.index((2, 2, n, 1))]
        print(f"  n={n}: excitation line = {a:.3g} [{A_exc_band[0, n]:.3g}, {A_exc_band[1, n]:.3g}]   "
              f"Teukode fit at t0 = 0/5/10/15/20: "
              + " ".join(f"{fit_vals[np.argmin(np.abs(T0_ARRAY - t))]:.3g}" for t in (0, 5, 10, 15, 20)))

    save(plot_amplitudes(T0_ARRAY, C_nr, C_teuk, A_exc, A_exc_band, MODES), "7b_amplitudes_kerr_excitation")


if __name__ == "__main__":
    main()
