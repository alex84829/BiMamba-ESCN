# -*- coding: utf-8 -*-
"""Feature extraction helpers for BiMamba-ESCN.

The training code expects each subject/recording to be saved as an .npz file with keys:
  psd:      [n_epochs, C, F]
  de_bands: [n_epochs, C, F]
  plv:      [n_epochs, C*C, F]
  wpli:     [n_epochs, C*C, F]

This file provides lightweight NumPy/SciPy implementations for already-epoched EEG arrays.
For full clinical preprocessing, apply your MNE pipeline first, then call these functions.
"""

from typing import Dict, Tuple
import numpy as np
from scipy.signal import welch, hilbert, butter, sosfiltfilt, iirnotch, filtfilt


FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpass_filter(data: np.ndarray, sfreq: float = 256.0, l_freq: float = 1.0, h_freq: float = 45.0, order: int = 4):
    sos = butter(order, [l_freq, h_freq], btype="bandpass", fs=sfreq, output="sos")
    return sosfiltfilt(sos, data, axis=-1)


def notch_filter(data: np.ndarray, sfreq: float = 256.0, notch_freq: float = 50.0, q: float = 30.0):
    b, a = iirnotch(w0=notch_freq, Q=q, fs=sfreq)
    return filtfilt(b, a, data, axis=-1)


def compute_psd_bands(epochs: np.ndarray, sfreq: float = 256.0, bands: Dict[str, Tuple[float, float]] = FREQ_BANDS) -> np.ndarray:
    """Compute band PSD using Welch. Input: [E, C, samples]. Output: [E, C, F]."""
    freqs, pxx = welch(epochs, fs=sfreq, nperseg=min(int(sfreq * 2), epochs.shape[-1]), axis=-1)
    out = []
    for low, high in bands.values():
        idx = (freqs >= low) & (freqs < high)
        out.append(pxx[..., idx].mean(axis=-1))
    return np.stack(out, axis=-1).astype(np.float32)


def compute_de_bands(epochs: np.ndarray, sfreq: float = 256.0, bands: Dict[str, Tuple[float, float]] = FREQ_BANDS) -> np.ndarray:
    """Compute differential entropy per band. Input: [E, C, samples]. Output: [E, C, F]."""
    outs = []
    for low, high in bands.values():
        x = bandpass_filter(epochs, sfreq=sfreq, l_freq=low, h_freq=high)
        var = np.var(x, axis=-1) + 1e-8
        de = 0.5 * np.log(2 * np.pi * np.e * var)
        outs.append(de)
    return np.stack(outs, axis=-1).astype(np.float32)


def _band_phase(epochs: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    x = bandpass_filter(epochs, sfreq=sfreq, l_freq=low, h_freq=high)
    analytic = hilbert(x, axis=-1)
    return np.angle(analytic)


def compute_plv_wpli(epochs: np.ndarray, sfreq: float = 256.0, bands: Dict[str, Tuple[float, float]] = FREQ_BANDS):
    """
    Compute pair-wise PLV and wPLI.

    Input:
      epochs: [E, C, samples]
    Output:
      plv, wpli: [E, C*C, F]
    """
    e, c, _ = epochs.shape
    plv_all, wpli_all = [], []
    for low, high in bands.values():
        phase = _band_phase(epochs, sfreq, low, high)  # [E,C,S]
        plv_band = np.zeros((e, c, c), dtype=np.float32)
        wpli_band = np.zeros((e, c, c), dtype=np.float32)
        for i in range(c):
            for j in range(c):
                phase_diff = phase[:, i, :] - phase[:, j, :]
                complex_phase = np.exp(1j * phase_diff)
                plv_band[:, i, j] = np.abs(complex_phase.mean(axis=-1))
                im = np.imag(complex_phase)
                wpli_band[:, i, j] = np.abs(im.mean(axis=-1)) / (np.mean(np.abs(im), axis=-1) + 1e-8)
        plv_all.append(plv_band.reshape(e, c * c))
        wpli_all.append(wpli_band.reshape(e, c * c))
    return np.stack(plv_all, axis=-1).astype(np.float32), np.stack(wpli_all, axis=-1).astype(np.float32)


def extract_all_features(epochs: np.ndarray, sfreq: float = 256.0):
    """Return dict compatible with np.savez for BiMamba-ESCN training."""
    epochs = np.asarray(epochs, dtype=np.float32)
    psd = compute_psd_bands(epochs, sfreq=sfreq)
    de_bands = compute_de_bands(epochs, sfreq=sfreq)
    plv, wpli = compute_plv_wpli(epochs, sfreq=sfreq)
    return {"psd": psd, "de_bands": de_bands, "plv": plv, "wpli": wpli}
