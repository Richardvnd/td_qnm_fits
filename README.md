# Dynamical Black Hole Spectroscopy: Time-dependent Quasinormal Mode Fits

Code accompanying the paper:

> **Dynamical Black Hole Spectroscopy: Time-dependent Quasinormal Mode Fits**
> Richard Dyer, Gregorio Carullo and Marina De Amicis
> _In preparation._

```bibtex
@article{Dyer:TimeDependentQNM,
    author        = {Dyer, Richard and Carullo, Gregorio and De Amicis, Marina},
    title         = {Dynamical Black Hole Spectroscopy: Time-dependent Quasinormal Mode Fits},
    year          = {2026},
    note          = {In preparation},
}
```

## Overview

This repository contains the code for comparing time-dependent quasinormal
mode (QNM) amplitudes from an analytical ringdown framework against amplitudes
recovered by conventional (static) ringdown fits. We sweep the fit start time
`t0` and track how each overtone amplitude evolves, contrasting least-squares
and Bayesian (BGP) fits against the analytical predictions for Regge-Wheeler-
Zerilli (RWZ), numerical relativity and Teukode waveforms.

## Dependencies

- [`bgp_qnm_fits`](https://github.com/BGP-QNM-FITS/bgp_qnm_fits) — Bayesian
  ringdown fitting and mode selection (`BGP_fit`, `BGP_select`), used by the
  `3_`/`3a_` scripts.
- [`qnm`](https://pypi.org/project/qnm/) — Kerr QNM frequencies and spherical-
  spheroidal mixing coefficients, used directly by the least-squares fits.
- [`sxs`](https://pypi.org/project/sxs/) — SXS catalogue waveforms (scripts
  `7_` and `8_`).
- `numpy`, `scipy`, `matplotlib` (with a LaTeX installation for `text.usetex`).

## Data

The analytical QNM amplitudes are included in this repository. The raw numerical
RWZ and Teukode waveforms are not included, and are available upon request.

- **`data_public/`** (in this repository) — everything needed for the analytical results:
  - **Analytical QNM amplitude packages** —
    `data_public/QNM_amplitude_packages/qnm_l2_m2_{DATASET}_N7/` with `DATASET` in
    `{ID00, ID10}` (quasi-circular inspiral and radial infall). Each contains
    `metadata.json` and `amplitudes.npz`, whose `A_22npm` array holds the complex
    time-dependent amplitude of every overtone, shape (branch +/-, n = 0-7, time).
  - **Higher-resolution package** —
    `data_public/QNM_amplitude_packages_v2/qnm_l2_m2_ID00_N7_Nint40000/`
    (N_int = 40000, used for the noise estimate in `3_`/`3a_`).
  - **Constant-amplitude table** — `data_public/table_l2m2_late_times_r2.5569_ID00.dat`
    (late-time constant QNM amplitudes and redshift terms for ID00).
- **Perturbative waveforms** — the raw perturbative waveforms, produced with the private
  RWZHyp and Teukode codes, are currently private. Scripts that fit the RWZ or Teukode data
  (e.g. `2_`, `4_`, `6_`, `7_`, `8_`) need these files.
- **NR** — `SXS:BBH:0305` and `SXS:BBH:3991`, loaded through `sxs.load(...)`
  and cached by `sxs`.

## Scripts

Each numbered script writes `figs/<script name>.pdf`:

| Script | Figure |
| --- | --- |
| `1_amplitudes.py` | Least-squares amplitudes vs analytical TD amplitudes (quasi-circular). |
| `2_mode_stability.py` | Stability of the n=0, 1 amplitudes as overtones are added to the fit. |
| `3_bayesian_content.py` | BGP mode selection and Bayesian amplitudes (results cached in `cache/`). |
| `3a_bayesian_content.py` | The same on the constant-amplitude control waveform. |
| `4_mismatch_pert.py` | Mismatch vs t0, fits to the perturbative (RWZ) data. |
| `5_mismatch_analytical.py` | Mismatch vs t0, fits to the analytical data. |
| `6_redshift.py` | Fits to analytical and RWZ data without decay correction, with the horizon-mode decay. |
| `7_amplitudes_kerr_NR.py` | NR (`SXS:BBH:0305`) vs Teukode (Kerr, a=0.6921) amplitudes. |
| `8_amplitudes_radial_NR.py` | Radial infall: analytical vs RWZ and NR (`SXS:BBH:3991`) fits. |
| `9_amplitudes_T.py` | Least-squares amplitudes for a range of fitting windows T. |
| `10_amplitudes_reg.py` | Least-squares amplitudes for a range of ridge strengths λ². |
| `11_picard.py` | Discrete Picard plot of the fit at t0 = 20M. |
| `12_condition_number.py` | Regularised condition number vs number of modes. |
| `13_truncation_vs_regularisation.py` | Truncated (n ≤ 3, λ²=0) vs regularised (n ≤ 7) fits. |
| `14_amplitudes_comp.py` | Fits to analytical and perturbative data, quasi-circular and radial. |
| `15_const_amplitudes.py` | TD amplitudes and fits with the constant (late-time) amplitudes. |
| `16_single_amplitudes.py` | Single-mode isolation test for each overtone. |

Shared code lives in `core.py` (data loading and the analytical model),
`fitting.py` (least-squares fits), `bayesian.py` (BGP fits for `3_`/`3a_`), and
`plotting.py`, `plot_config.py` and `stylesheet.mplstyle` (figure layout and
styling).

Run a script from the repository root, e.g.:

```bash
python 1_amplitudes.py
```

Configuration (dataset, highest overtone, `t0` grid, regularisation, etc.) is set via
the constants near the top of each script.

## License

This project is licensed under the terms of the MIT License. See [LICENSE](LICENSE) for details.
