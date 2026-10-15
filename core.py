"""Data loading and analytical model functions for td_qnm_fits."""

import json
import os
import re
import warnings

import numpy as np
import qnm
from scipy.signal import savgol_filter

# Late-time cutoff [M, peak frame], below every package's frozen tail (ID00 58.7M, ID10 59.5M).
T_END_SAFE = 57.0

# np.trapz became np.trapezoid in numpy 2.0; the sxs/bgp env is on numpy 1.26.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz

# Packages at integration resolutions N_int = 10k/20k/40k; all share one observer grid.
_RESOLUTION_ROOT = "QNM_amplitude_packages_v2"
_VALID_RESOLUTIONS = (10, 20, 40)


def _resolve_package_dir(package_dir, resolution=20):
    """Package path at resolution N_int (thousands); 20 is the default package."""
    if resolution not in _VALID_RESOLUTIONS:
        raise ValueError(f"resolution must be one of {_VALID_RESOLUTIONS} "
                         f"(N_int in thousands), got {resolution!r}")
    if resolution == 20:
        return package_dir

    package_dir = str(package_dir).rstrip("/")
    parent, _, base = package_dir.rpartition("/")
    base = re.sub(r"_Nint\d+$", "", base)
    data_root = parent.rpartition("/")[0]
    prefix = f"{data_root}/" if data_root else ""
    return f"{prefix}{_RESOLUTION_ROOT}/{base}_Nint{resolution * 1000}"


# Public inputs (in the repository) and private raw waveforms (RWZ, Teukode; gitignored, on request).
PUBLIC_DATA  = "data_public"
PRIVATE_DATA = "data_all"
PACKAGE_ROOT = f"{PUBLIC_DATA}/QNM_amplitude_packages"


def _private_path(path):
    """Mirror of a ``PUBLIC_DATA`` path under ``PRIVATE_DATA``, where the raw waveforms live."""
    return os.path.join(PRIVATE_DATA, os.path.relpath(path, PUBLIC_DATA))


def package_path(dataset):
    """Path of the amplitude package for ``dataset`` ('ID00'/'ID10')."""
    return f"{PACKAGE_ROOT}/qnm_l2_m2_{dataset}_N7"


def mode_set(n_max, include_retro=True, ell=2, m=2):
    """[(l, m, n, sign)]: co-rotating n = 0..n_max, then optionally counter-rotating."""
    signs = (1, -1) if include_retro else (1,)
    return [(ell, m, n, sign) for sign in signs for n in range(n_max + 1)]


def _branch_frequencies(meta, n_max, sign):
    """QNM frequencies of overtones 0..n_max on one branch."""
    return qnm_frequencies([(meta["ell"], meta["m"], n, sign) for n in range(n_max + 1)])


# Load data

def load_package(package_dir, data_type='strain', resolution=20):
    """Load (meta, amplitude arrays) for a package."""
    package_dir = _resolve_package_dir(package_dir, resolution)
    with open(f"{package_dir}/metadata.json") as f:
        meta = json.load(f)

    data = dict(np.load(f"{package_dir}/amplitudes.npz"))

    if data_type == 'psi4':
        data["A_22npm"] = convert_analytical_to_psi4(data["A_22npm"],
                                                      data['tau_minus_rho_plus'],
                                                      l=meta["ell"], m=meta["m"])

    return meta, data


def peak_frame_times(meta):
    """Key times in the strain-peak frame: t_start, t_end, t_lr, t_bounce (+ t_peak_abs)."""
    t_peak_abs = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    std = meta["stored_time_domain"]
    return {
        "t_peak_abs": t_peak_abs,
        "t_start": std["tau_stored_start"] - t_peak_abs,
        "t_end":   std["tau_stored_end"]   - t_peak_abs,
        "t_lr":    std["tau_light_ring"]   - t_peak_abs,
        "t_bounce": std["tau_bounce"]      - t_peak_abs,
    }


# Table labels: "220" = (2, 2, 0), "2-20" = (2, -2, 0); single-digit l, |m|.
_MODE_LABEL_RE = re.compile(r"^(\d)(-?\d)(\d+)$")

CONSTANT_TABLE = f"{PUBLIC_DATA}/table_l2m2_late_times_r2.5569_ID00.dat"
_CONSTANT_COLUMNS = {'constant': 'Aqnm', 'redshift': 'Red0'}


def constant_amplitude_scale(kind, meta, n_max):
    """Factor putting the table on the peak-frame axes, shape (2, n_max + 1).

    A_hat(t) = sqrt(24) [Aqnm e^{-i om t_peak} + Red0 e^{-t_peak/4} e^{t(i om - 1/4)}].
    """
    t_peak = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    sqrt24 = meta["strain_conversion_factor"]

    if kind == 'redshift':
        return np.full((2, n_max + 1), sqrt24 * np.exp(-t_peak / 4), dtype=complex)

    factor = np.zeros((2, n_max + 1), dtype=complex)
    for p, sign in enumerate((1, -1)):
        factor[p] = sqrt24 * np.exp(-1j * _branch_frequencies(meta, n_max, sign) * t_peak)
    return factor


def load_constant_amplitudes(kind='constant', n_max=None, path=CONSTANT_TABLE,
                              data=None, meta=None, scale=True):
    """Constant ('Aqnm') or redshift ('Red0') table amplitudes, shape (2, n_max + 1).

    With ``data`` and ``meta``, returns (t, A(t)) on the package grid, NaN before the bounce.
    """
    if kind not in _CONSTANT_COLUMNS:
        raise ValueError(f"kind must be one of {sorted(_CONSTANT_COLUMNS)}, got {kind!r}")
    column = _CONSTANT_COLUMNS[kind]

    # Explicit dtype: auto-detection reads '220' as an int, then fails on '2-20'.
    table = np.genfromtxt(path, names=True, encoding=None,
                          dtype=["U8"] + ["f8"] * 4)
    table = np.atleast_1d(table)

    if f"Real_{column}" not in table.dtype.names:
        raise ValueError(f"{path} has no '{column}' column "
                         f"(available: {table.dtype.names})")

    amplitudes = {}
    for row in table:
        label = str(row["Mode"])
        match = _MODE_LABEL_RE.match(label)
        if match is None:
            raise ValueError(f"Cannot parse mode label {label!r} in {path}")
        _, m, n = (int(g) for g in match.groups())
        p = 0 if m > 0 else 1
        amplitudes[(p, n)] = complex(row[f"Real_{column}"], row[f"Imaginary_{column}"])

    if n_max is None:
        n_max = max(n for _, n in amplitudes)

    A_const = np.full((2, n_max + 1), np.nan, dtype=complex)
    for (p, n), value in amplitudes.items():
        if n <= n_max:
            A_const[p, n] = value

    if scale:
        if meta is None:
            raise ValueError(
                "load_constant_amplitudes(scale=True) needs `meta` from "
                "load_package to build the conversion factor; pass meta=..., "
                "or scale=False to get the raw table values.")
        if path == CONSTANT_TABLE and meta.get("id") != "00":
            warnings.warn(
                f"{CONSTANT_TABLE} was computed for ID00 but meta is for "
                f"ID{meta.get('id')}; the scaling uses this package's t_peak.",
                RuntimeWarning, stacklevel=2)
        A_const = A_const * constant_amplitude_scale(kind, meta, n_max)

    if data is None or meta is None:
        return A_const

    t_peak_abs = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    t = data["tau"] - t_peak_abs
    t = t[t <= T_END_SAFE]
    t_bounce = peak_frame_times(meta)["t_bounce"]
    window = np.where(t >= t_bounce, 1.0, np.nan)
    A_t = A_const[:, :, None] * window
    if kind == 'redshift':
        for p, sign in enumerate((1, -1)):
            omegas = _branch_frequencies(meta, n_max, sign)
            A_t[p] *= np.exp(t[None, :] * (1j * omegas[:, None] - 0.25))
    return t, A_t


def _find_peak_after_transient(tau, h, tau_skip_until):
    """Index of max |h| after ``tau_skip_until``."""
    i_start = np.argmin(np.abs(tau - tau_skip_until))
    return np.argmax(np.abs(h[i_start:])) + i_start


def load_rwz(package_dir, data_type='strain', meta=None, clip_to_window=True,
             resolution=20):
    """Numerical RWZ waveform (t, h22); peak frame and clipped to the package window if ``meta`` given.

    Read from the ``PRIVATE_DATA`` mirror of ``package_dir``.
    """
    package_dir = _resolve_package_dir(package_dir, resolution)
    path = f"{_private_path(package_dir)}/rwz_reference.dat"
    raw = np.loadtxt(path, skiprows=1)
    h22 = raw[:, 4] + 1j * raw[:, 5]
    if data_type == 'psi4':
        h22 = _second_time_derivative(h22, raw[:, 0])
    tau = raw[:, 0]
    if meta is not None:
        tau = tau - meta["coefficient_provenance"]["rwz_h22_peak_tau"]
        if clip_to_window:
            times = peak_frame_times(meta)
            mask = (tau >= times["t_start"]) & (tau <= T_END_SAFE)
            tau, h22 = tau[mask], h22[mask]
    return tau, h22


def load_teukode_data(package_dir=PRIVATE_DATA, t_shift=0.0):
    """Teukode (Kerr, a=0.6921) waveform (t, h22), t = 0 at the strain peak minus ``t_shift``."""
    path = f"{package_dir}/h_teukHH10_a0.6921_r04.200_nu1.000e-03_l2_m2_hres.dat"
    raw = np.loadtxt(path, delimiter=',')
    h22 = raw[:, 1] + 1j * raw[:, 2]
    tau = raw[:, 0]
    i_peak = _find_peak_after_transient(tau, h22, tau[0] + 50.0)
    tau -= tau[i_peak] + t_shift
    return tau, h22


def load_sxs_data(sxs_id, data_type='strain', l=2, m=2, download=False):
    """SXS (l, m) mode, t = 0 at the strain peak. Returns (t, h, Mf, chif).

    psi4 is d^2h/dt^2 by finite differences, matching the RWZ side.
    """
    import sxs

    sim   = sxs.load(sxs_id, download=download)
    h_obj = sim.strain
    h22   = h_obj.data[:, h_obj.index(l, m)]
    t     = h_obj.t.copy()

    meta = sim.metadata
    Mf   = float(meta["remnant_mass"])
    chif = float(np.linalg.norm(meta["remnant_dimensionless_spin"]))

    t = t - t[np.argmax(np.abs(h22))]

    if data_type == 'psi4':
        h22 = _second_time_derivative(h22, t)

    return t, h22, Mf, chif


def load_analytical_model(package_dir, data_type='strain', meta=None, data=None,
                          amplitudes='time_dependent', resolution=20):
    """Analytical QNM waveform (t, h22) in the peak frame.

    ``amplitudes``: 'time_dependent' (package), or 'constant'/'redshift' (table, from the bounce).
    """
    if meta is None or data is None:
        meta, data = load_package(package_dir, data_type=data_type,
                                  resolution=resolution)

    if data_type == 'psi4':
        warnings.warn(
            "psi4 analytical waveforms are estimated with a Savitzky-Golay "
            "derivative before summing the QNM modes.",
            RuntimeWarning,
            stacklevel=2,
        )

    if amplitudes == 'time_dependent':
        return get_analytical_model(data, meta, n_max=meta["n_max"])

    if data_type == 'psi4':
        raise ValueError("Constant-amplitude tables are strain amplitudes; "
                         "data_type='psi4' is only supported with "
                         "amplitudes='time_dependent'.")

    n_max = meta["n_max"]
    t, A_const = load_constant_amplitudes(kind=amplitudes, n_max=n_max,
                                          data=data, meta=meta)
    # Peak-frame amplitudes, so each mode is A(t) exp(-i omega t).
    h22 = np.zeros(len(t), dtype=complex)
    for p, sign in enumerate((1, -1)):
        omegas = _branch_frequencies(meta, n_max, sign)
        h22 = h22 + np.sum(A_const[p] * np.exp(-1j * omegas[:, None] * t), axis=0)
    return t, h22


def load_waveform(package_dir, use_rwz=False, data_type='strain', meta=None, data=None):
    """Data to fit: RWZ (``use_rwz``) or the analytical reconstruction, as (t, h22)."""
    if use_rwz:
        return load_rwz(package_dir, data_type=data_type, meta=meta)
    return load_analytical_model(package_dir, data_type=data_type, meta=meta, data=data)


def _savgol_derivative(y, x, deriv=1, window_length=501, polyorder=3):
    """Savitzky-Golay derivative of complex sampled data."""
    y = np.asarray(y)
    x = np.asarray(x)

    window_length = min(window_length, len(x))
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(window_length, polyorder + 1)
    if window_length > len(x):
        window_length = len(x) if len(x) % 2 == 1 else len(x) - 1
    if window_length <= polyorder:
        raise ValueError("window_length must be larger than polyorder")

    delta = float(np.median(np.diff(x)))
    real = savgol_filter(np.real(y), window_length=window_length, polyorder=polyorder,
                         deriv=deriv, delta=delta, mode="interp")
    imag = savgol_filter(np.imag(y), window_length=window_length, polyorder=polyorder,
                         deriv=deriv, delta=delta, mode="interp")
    return real + 1j * imag


def convert_analytical_to_psi4(A, tau, l=2, m=2):
    """Strain amplitudes to psi4: A'' - 2i om A' - om^2 A, (2, n+1, N) layout."""
    amplitude = A.copy()
    n_max = amplitude.shape[1] - 1
    for p, sign in enumerate((1, -1)):
        omegas = qnm_frequencies([(l, m, n, sign) for n in range(n_max + 1)])
        for n in range(n_max + 1):
            series = amplitude[p, n, :]
            A_dot = _savgol_derivative(series, tau, deriv=1)
            A_doubledot = _savgol_derivative(series, tau, deriv=2)
            amplitude[p, n, :] = (A_doubledot
                                 - 2j * omegas[n] * A_dot
                                 - omegas[n]**2 * series)
    return amplitude


def _second_time_derivative(h, t):
    """d^2h/dt^2 by repeated np.gradient (strain to psi4)."""
    dh_dt = np.gradient(h, t, edge_order=2)
    return np.gradient(dh_dt, t, edge_order=2)


# General functions

def mismatch(times, wf1, wf2):
    """Mismatch (qnmfits definition); lists of waveforms are summed over channels."""
    if isinstance(wf1, (list, tuple)):
        num   = np.real(sum(_trapezoid(w1 * np.conj(w2), x=times)
                            for w1, w2 in zip(wf1, wf2)))
        denom = np.sqrt(sum(_trapezoid(np.real(w1 * np.conj(w1)), x=times) for w1 in wf1)
                        * sum(_trapezoid(np.real(w2 * np.conj(w2)), x=times) for w2 in wf2))
    else:
        num   = np.real(_trapezoid(wf1 * np.conj(wf2), x=times))
        denom = np.sqrt(_trapezoid(np.real(wf1 * np.conj(wf1)), x=times)
                        * _trapezoid(np.real(wf2 * np.conj(wf2)), x=times))
    return 1.0 - num / denom


# Get analytical predictions

def qnm_frequencies(modes, chif=0.0, Mf=1.0, s=-2):
    """Frequencies of (l, m, n, sign) modes; sign -1 gives the mirror mode -conj(omega)."""
    frequencies = []
    for l, m, n, sign in modes:
        ks = qnm.modes_cache(s=s, l=l, m=m * sign, n=n)
        omega = ks(a=chif)[0] / Mf
        if sign == -1:
            omega = -np.conj(omega)
        frequencies.append(omega)
    return np.array(frequencies)


# Arrays of shape (2, n+1, N): axis 0 index 0 = co-rotating, 1 = counter-rotating.


def get_analytical_amplitude(data, meta, n_max=0, signs=(1, -1)):
    """Analytical amplitudes (t, A) in the peak frame, phased to match normalize_amplitudes."""
    t_peak_abs  = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    phase_shift = meta["phase_time_shift"]
    t = data["tau"] - t_peak_abs
    A_array = data["A_22npm"].copy()
    for sign in signs:
        p = 0 if sign == 1 else 1
        omegas = _branch_frequencies(meta, n_max, sign)
        A_array[p, :n_max + 1] *= np.exp(-1j * omegas * (t_peak_abs - phase_shift))[:, None]
    mask = t <= T_END_SAFE
    return t[mask], A_array[:, :, mask]


def get_analytical_h22np(data, meta, n_max=0, signs=(1, -1)):
    """Per-mode analytical waveforms (t, h_modes) in the peak frame."""
    if n_max is None:
        n_max = meta["n_max"]
    t_peak_abs  = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    tau_mrp = data["tau_minus_rho_plus"]
    t = data["tau"] - t_peak_abs
    h_modes = np.zeros((2, n_max + 1, len(t)), dtype=complex)
    for sign in signs:
        p = 0 if sign == 1 else 1
        omegas = _branch_frequencies(meta, n_max, sign)
        h_modes[p] = data["A_22npm"][p, :n_max + 1] * np.exp(-1j * omegas[:, None] * tau_mrp)
    mask = t <= T_END_SAFE
    return t[mask], h_modes[:, :, mask]


def get_analytical_model(data, meta, n_max=None, signs=(1, -1)):
    """h22 = sum over modes of A(tau) exp(-i omega tau_-); returns (t, h22)."""
    t, h_modes = get_analytical_h22np(data, meta, n_max=n_max, signs=signs)
    return t, h_modes.sum(axis=(0, 1))


def analytical_constant_amplitude_mismatch_sweep(t, data, t0_array, amp_data, meta,
                                                 modes, T):
    """Mismatch vs t0 of the analytical model with amplitudes frozen at A(t0)."""
    t_peak_abs  = meta["coefficient_provenance"]["rwz_h22_peak_tau"]
    phase_shift = meta["phase_time_shift"]
    frequencies = qnm_frequencies(modes)

    t_ana   = amp_data["tau"] - t_peak_abs
    A       = amp_data["A_22npm"]
    tau_minus = t + t_peak_abs - phase_shift

    phases, A_at_t0 = {}, {}
    for (_, _, n, sign), omega in zip(modes, frequencies):
        p = 0 if sign == 1 else 1
        phases[(p, n)] = np.exp(-1j * omega * tau_minus)
        A_at_t0[(p, n)] = (np.interp(t0_array, t_ana, A[p, n, :].real)
                           + 1j * np.interp(t0_array, t_ana, A[p, n, :].imag))

    mm = np.empty(len(t0_array))
    for i, t0 in enumerate(t0_array):
        window = (t >= t0) & (t < T+t0)
        if not np.any(window):
            raise ValueError(
                f"analytical_constant_amplitude_mismatch_sweep: no data samples in "
                f"the window [{t0}, {t0+T}) at t0={t0} (data spans "
                f"[{t[0]}, {t[-1]}]) — t0_array/T extend past the available data."
            )
        model = np.zeros(int(np.count_nonzero(window)), dtype=complex)
        for key in phases:
            model += A_at_t0[key][i] * phases[key][window]
        mm[i] = mismatch(t[window], model, data[window])
    return mm
