"""FSK demodulation helpers: channel masking and peak interpolation."""

import numpy as np

from . import constants


def build_chan_mask(F0: float, synth_res_val: float) -> np.ndarray:
    """Boolean mask for 64-FSK bandwidth around *F0*."""
    a = F0 - 1 * synth_res_val
    b = F0 + (constants.NUM_FSK_BINS + 1) * synth_res_val
    return (constants.fft_freqs >= min(a, b)) & (constants.fft_freqs <= max(a, b))


def interp_peak(psd: np.ndarray, bin_idx: int, freqs: np.ndarray) -> float:
    """Parabolic interpolation around an FFT peak for sub-bin accuracy."""
    n = len(psd)
    b = bin_idx
    if b <= 0 or b >= n - 1:
        return float(freqs[b])
    alpha = psd[b - 1]
    beta = psd[b]
    gamma = psd[b + 1]
    denom = alpha - 2 * beta + gamma
    if abs(denom) < 1e-30:
        return float(freqs[b])
    delta = 0.5 * (alpha - gamma) / denom
    bin_spacing = freqs[1] - freqs[0] if n > 1 else 1.0
    return float(freqs[b] + delta * bin_spacing)


def demod_one_symbol(
    sig_segment: np.ndarray,
    F0: float,
    synth_res_val: float,
    chan_mask: np.ndarray,
) -> tuple[int, float, float]:
    """Return ``(fsk_bin, peak_freq, peak_power)`` for one symbol slot."""
    spectrum = np.fft.fft(sig_segment)
    psd = np.abs(spectrum) ** 2
    psd_masked = psd.copy()
    psd_masked[~chan_mask] = 0.0
    peak_bin = int(np.argmax(psd_masked))
    peak_freq = interp_peak(psd_masked, peak_bin, constants.fft_freqs)
    fsk_bin = int(round((peak_freq - F0) / synth_res_val))
    fsk_bin = max(0, min(constants.NUM_FSK_BINS - 1, fsk_bin))
    return fsk_bin, peak_freq, float(psd_masked[peak_bin])


def demod_one_symbol_with_collision(
    sig_segment: np.ndarray,
    F0: float,
    synth_res_val: float,
    chan_mask: np.ndarray,
    collision_threshold: float = 0.5,
    null_bins: int = 2,
) -> tuple[int, float, float, bool]:
    """Like :func:`demod_one_symbol` but also detects potential collisions.

    Returns ``(fsk_bin, peak_freq, peak_power, collision_detected)``.

    A collision is flagged when a secondary peak (outside ±null_bins of the
    primary) exceeds ``collision_threshold`` times the primary peak power.
    """
    spectrum = np.fft.fft(sig_segment)
    psd = np.abs(spectrum) ** 2
    psd_masked = psd.copy()
    psd_masked[~chan_mask] = 0.0

    peak_bin = int(np.argmax(psd_masked))
    peak_freq = interp_peak(psd_masked, peak_bin, constants.fft_freqs)
    peak_power = float(psd_masked[peak_bin])

    psd_second = psd_masked.copy()
    null_start = max(0, peak_bin - null_bins)
    null_end = min(len(psd_second), peak_bin + null_bins + 1)
    psd_second[null_start:null_end] = 0.0
    second_peak_power = float(np.max(psd_second))
    collision = second_peak_power > (collision_threshold * peak_power)

    fsk_bin = int(round((peak_freq - F0) / synth_res_val))
    fsk_bin = max(0, min(constants.NUM_FSK_BINS - 1, fsk_bin))
    return fsk_bin, peak_freq, peak_power, collision
