"""Least-squares ringdown fitting functions for td_qnm_fits."""

import numpy as np
import qnm

from core import mismatch, qnm_frequencies


def _mu(ell, m, ellp, mp, nprime, sign, chif, s=-2):
    """Mixing of spheroidal QNM (ellp, mp, nprime, sign) into spherical (ell, m); valid for n' <= 7."""
    if mp != m:
        return 0.0 + 0.0j
    ks = qnm.modes_cache(s=s, l=ellp, m=mp * sign, n=nprime)
    C = ks(a=chif)[2]
    mu = C[ell - max(abs(m), abs(s))]
    if sign == -1:
        mu = (-1) ** (ell + ellp) * np.conj(mu)
    return mu


def _mu_list(indices, chif, s=-2):
    """Mixing coefficients for 6-tuple indices; a 4-tuple is a Kronecker delta on (l, m)."""
    mus = []
    for k in indices:
        if len(k) == 4:
            ell, m, ellp, mp = k
            mus.append(1.0 + 0.0j if (ell, m) == (ellp, mp) else 0.0 + 0.0j)
        elif len(k) == 6:
            mus.append(_mu(*k, chif, s))
        else:
            raise ValueError(f"mixing index must have length 4 or 6, got {len(k)}")
    return mus


def _mixing_coefficients(spherical_modes, modes, chif):
    """mu_lists[i][j]: mixing of spheroidal mode j into spherical channel i."""
    return [_mu_list([lm + mode for mode in modes], chif) for lm in spherical_modes]


def _window(times, t0, T, t0_method):
    """Fit window [t0, t0 + T): a mask ('geq') or a slice between nearest samples ('closest')."""
    if t0_method == 'geq':
        return (times >= t0) & (times < t0 + T)
    i0 = np.argmin((times - t0) ** 2)
    i1 = np.argmin((times - t0 - T) ** 2)
    return slice(i0, i1)


def _ridge_solve(a, data, n_modes, t0, lambda_reg=0.0, verbose=False):
    """Solve a @ C = data: lstsq, or (A^H A + λI) C = A^H d if lambda_reg > 0.

    Returns (C, condition number of a, smallest singular value of a).
    """
    s = np.linalg.svd(a, compute_uv=False)
    kappa = s[0] / s[-1]

    if verbose:
        msg = f"t0={t0:.2f}  κ(A)={kappa:.2e}  σ_min={s[-1]:.2e}"
        if lambda_reg > 0:
            sqrt_lam = np.sqrt(lambda_reg)
            n_reg = int(np.sum(s < sqrt_lam))
            msg += (f"  √λ={sqrt_lam:.2e}"
                    f"  f(σ_min)={s[-1]**2/(s[-1]**2+lambda_reg):.2e}"
                    f"  n_reg={n_reg}/{n_modes}")
        print(msg)

    if lambda_reg > 0:
        AhA = a.conj().T @ a
        C = np.linalg.solve(AhA + lambda_reg * np.eye(n_modes), a.conj().T @ data)
    else:
        C, *_ = np.linalg.lstsq(a, data, rcond=None)

    return C, kappa, float(s[-1])


def ringdown_fit(times, data, modes, Mf, chif, t0, t0_method='geq', T=100,
                 lambda_reg=0.0, verbose=False, frequencies=None):
    """Single-channel least-squares ringdown fit; ``frequencies`` may be precomputed."""
    window = _window(times, t0, T, t0_method)
    times, data = times[window], data[window]

    if frequencies is None:
        frequencies = qnm_frequencies(modes, chif, Mf)
    frequencies = np.array(frequencies)
    a = np.exp(-1j * np.outer(times - t0, frequencies))

    C, kappa, s_min = _ridge_solve(a, data, len(modes), t0,
                                   lambda_reg=lambda_reg, verbose=verbose)

    model = a @ C
    return {
        'C': C,
        'mismatch': mismatch(times, model, data),
        'model': model,
        'model_times': times,
        'data': data,
        't0': t0,
        'modes': modes,
        'mode_labels': [str(m) for m in modes],
        'frequencies': frequencies,
        'condition_number': kappa,
        's_min': s_min,
    }


def ls_mismatch_sweep(times, data, t0_array, modes, T, Mf=1.0, chif=0.0,
                      lambda_reg=0.0):
    """Least-squares fit mismatch over [t0, t0 + T] for each t0."""
    frequencies = qnm_frequencies(modes, chif, Mf)
    mm = np.empty(len(t0_array))
    for i, t0 in enumerate(t0_array):
        fit = ringdown_fit(times, data, modes, Mf, chif, t0, T=T,
                           lambda_reg=lambda_reg, frequencies=frequencies)
        mm[i] = fit["mismatch"]
    return mm


def multimode_ringdown_fit(times, data_dict, modes, Mf, chif, t0,
                           t0_method='geq', T=100, lambda_reg=0.0, verbose=False,
                           spherical_modes=None, frequencies=None, mu_lists=None):
    """Least-squares fit of spheroidal QNMs to spherical channels {(l, m): h}, with mode mixing.

    ``frequencies`` and ``mu_lists`` may be precomputed for sweeps.
    """
    if spherical_modes is None:
        spherical_modes = list(data_dict.keys())

    window = _window(times, t0, T, t0_method)
    times_w = times[window]
    data_w = {lm: data_dict[lm][window] for lm in spherical_modes}

    data = np.concatenate([data_w[lm] for lm in spherical_modes])

    if frequencies is None:
        frequencies = qnm_frequencies(modes, chif, Mf)
    frequencies = np.array(frequencies)

    if mu_lists is None:
        mu_lists = _mixing_coefficients(spherical_modes, modes, chif)

    phases = np.exp(-1j * np.outer(times_w - t0, frequencies))
    a = np.concatenate([phases * np.asarray(mu) for mu in mu_lists])

    C, kappa, s_min = _ridge_solve(a, data, len(modes), t0,
                                   lambda_reg=lambda_reg, verbose=verbose)

    model_full = a @ C
    n_t = len(times_w)
    model_dict = {lm: model_full[i * n_t:(i + 1) * n_t]
                  for i, lm in enumerate(spherical_modes)}

    mm = mismatch(times_w,
                 [model_dict[lm] for lm in spherical_modes],
                 [data_w[lm] for lm in spherical_modes])

    return {
        'C': C,
        'mismatch': mm,
        'model': model_dict,
        'model_times': times_w,
        't0': t0,
        'modes': modes,
        'frequencies': frequencies,
        'condition_number': float(kappa),
        's_min': s_min,
    }


def run_amplitude_sweep(t, h, modes, t0_array, T, lambda_reg=0.0,
                        Mf=1.0, chif=0.0, spherical_modes=None, verbose=False):
    """Fit at each t0; mixing-aware if ``h`` is a dict or ``spherical_modes`` is given.

    Returns (C_array, model_list, condition numbers, smallest singular values).
    """
    if isinstance(h, dict):
        use_mixing = True
        data_dict = h
        if spherical_modes is None:
            spherical_modes = list(h.keys())
    elif spherical_modes is not None:
        use_mixing = True
        if len(spherical_modes) != 1:
            raise ValueError(
                "Pass h as a data_dict {(l,m): array} to fit more than one "
                "spherical channel."
            )
        data_dict = {spherical_modes[0]: np.asarray(h)}
    else:
        if len({(mode[0], mode[1]) for mode in modes}) > 1:
            raise ValueError(
                "Modes span multiple (l,m) families — pass spherical_modes=[(l,m)] "
                "(with h as that channel's array) or h as a data_dict "
                "{(l,m): array} for a mixing-aware fit."
            )
        use_mixing = False

    frequencies = qnm_frequencies(modes, chif, Mf)
    if use_mixing:
        mu_lists = _mixing_coefficients(spherical_modes, modes, chif)

    C_list, model_list, kappa_list, s_min_list = [], [], [], []
    for t0 in t0_array:
        if use_mixing:
            fit = multimode_ringdown_fit(t, data_dict, modes=modes, Mf=Mf, chif=chif,
                                         t0=t0, T=T, lambda_reg=lambda_reg,
                                         spherical_modes=spherical_modes, verbose=verbose,
                                         frequencies=frequencies, mu_lists=mu_lists)
            model_list.append(fit["model"])
        else:
            fit = ringdown_fit(t, h, modes=modes, Mf=Mf, chif=chif, t0=t0, T=T,
                               lambda_reg=lambda_reg, verbose=verbose,
                               frequencies=frequencies)
            model_list.append(np.asarray(fit["model"]))
        C_list.append(fit["C"])
        kappa_list.append(fit["condition_number"])
        s_min_list.append(fit["s_min"])
    return np.array(C_list), model_list, np.array(kappa_list), np.array(s_min_list)


def normalize_amplitudes(C_array, modes, t0_array, *, chif, Mf):
    """Decay-corrected |C(t0) exp(+i omega t0)|, comparable to get_analytical_amplitude.

    ``chif``/``Mf`` are keyword-only and required so Kerr fits are never corrected with Schwarzschild frequencies.
    """
    frequencies = qnm_frequencies(modes, chif, Mf)

    return np.abs(C_array * np.exp(1j * np.outer(t0_array, frequencies)))


def fit_amplitudes(t, h, modes, t0_array, T, lambda_reg=0.0, Mf=1.0, chif=0.0,
                   spherical_modes=None):
    """Decay-corrected fitted amplitudes, shape (len(t0_array), len(modes))."""
    C_array, *_ = run_amplitude_sweep(t, h, modes, t0_array, T, lambda_reg=lambda_reg,
                                      Mf=Mf, chif=chif, spherical_modes=spherical_modes)
    return normalize_amplitudes(C_array, modes, t0_array, chif=chif, Mf=Mf)


def regularized_bgp_samples(fisher, b_vector, ref_params, modes, t0, *,
                             chif, Mf, lambda_reg=0.0, num_samples=1000):
    """50th/25th/75th percentiles of decay-corrected |C| from a ridge-regularised BGP posterior.

    Assumes a fixed remnant (include_Mf = include_chif = False).
    """
    F   = np.array(fisher, dtype=float)
    b   = np.array(b_vector, dtype=float)
    ref = np.array(ref_params, dtype=float)
    n2  = F.shape[0]

    # Prior N(0, I/lambda); BGP's b = A^H K^-1 (d - A ref), so A^H K^-1 d = b + F ref.
    F_reg    = F + lambda_reg * np.eye(n2)
    mean_reg = np.linalg.solve(F_reg, b + F @ ref)
    covariance = np.linalg.inv(F_reg)
    covariance = (covariance + covariance.T) / 2

    samples = np.random.multivariate_normal(mean_reg, covariance, size=num_samples)
    # Columns alternate real, imag per mode.

    n_modes  = len(modes)
    A_low    = np.zeros(n_modes)
    A_median = np.zeros(n_modes)
    A_high   = np.zeros(n_modes)

    frequencies = qnm_frequencies(modes, chif, Mf)

    for j, omega in enumerate(frequencies):
        C_tilde   = np.sqrt(samples[:, 2 * j] ** 2 + samples[:, 2 * j + 1] ** 2) \
                    * np.abs(np.exp(1j * omega * t0))
        A_low  [j] = np.percentile(C_tilde, 25)
        A_median[j] = np.percentile(C_tilde, 50)
        A_high [j] = np.percentile(C_tilde, 75)

    return A_median, A_low, A_high

