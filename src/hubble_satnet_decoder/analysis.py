"""Per-packet signal analysis for the Hubble satellite PHY.

Given a decoded packet (or a bare preamble detection) plus its corrected symbol edges, return
raw per-symbol timing, amplitude, and frequency/channel measurements as plain data -- no
formatting. The caller owns capture, preamble detection, and symbol-edge
finding.

Per-symbol tone frequency reuses the decoder's ``interp_peak`` (sub-bin accurate, consistent
with demod), and the channel grid is anchored on the decoder's own ``F0_hz`` /
``measured_synth_res`` rather than re-derived from the preamble.

Callers must have run ``configure()`` first (this module reads ``constants.fft_freqs`` /
``samples_per_symbol`` / ``slot_samples``, which ``configure()`` populates).
"""
import numpy as np

from . import constants
from .demod import interp_peak

# Packet geometry / hop schedule

def packet_symbol_grid(spec: dict) -> tuple[int, int]:
    """Return ``(n_sym, slot)``: total symbols (preamble + header + PDU) and samples per
    symbol slot, for the packet's PHY version."""
    ver = spec.get("phy_ver", 1)
    slot = constants.slot_samples.get(ver, constants.slot_samples[1])["slot"]
    n_sym = (constants.PREAMBLE_LEN + constants.NUM_HEADER_SYMS
             + (spec.get("num_pdu_symbols") or 0))
    return n_sym, slot


def rotated_hop_sequence(channel_num, hop_seq_idx):
    """Hop sequence rotated to start at ``channel_num`` (index h = channel for hop h), or
    None if the hop parameters are missing / inconsistent."""
    if channel_num is None or hop_seq_idx is None or hop_seq_idx >= len(constants.HOPPING_SEQS):
        return None
    seq = constants.HOPPING_SEQS[hop_seq_idx]
    if channel_num not in seq:
        return None
    i = seq.index(channel_num)
    return seq[i:] + seq[:i]


# DSP primitives

def symbol_amplitudes_dbfs(iq, edges, full_scale: float = constants.ADC_FULL_SCALE):
    """Per-symbol RMS amplitude and per-gap noise-floor RMS, both in dBFS."""
    sym_dbfs, gap_dbfs = [], []
    for i, (s, e) in enumerate(edges):
        body = iq[s:e]
        if len(body):
            sym_dbfs.append(20.0 * np.log10(
                np.sqrt(np.mean(np.abs(body) ** 2)) / full_scale + 1e-30))
        if i < len(edges) - 1:
            gap = iq[e:edges[i + 1][0]]
            if len(gap):
                gap_dbfs.append(20.0 * np.log10(
                    np.sqrt(np.mean(np.abs(gap) ** 2)) / full_scale + 1e-30))
    return sym_dbfs, gap_dbfs


def _dominant_freq(sym_iq, band=None) -> float:
    """Dominant baseband frequency (Hz) of a symbol slice: FFT -> argmax (optionally restricted
    to ``band=(lo, hi)``) -> parabolic ``interp_peak`` for sub-bin accuracy. Uses the decoder's
    ``fft_freqs`` (unshifted: DC at index 0), so the measurement matches demod."""
    n = constants.samples_per_symbol
    x = np.zeros(n, dtype=np.complex128)
    m = min(len(sym_iq), n)
    x[:m] = sym_iq[:m]
    psd = np.abs(np.fft.fft(x)) ** 2
    freqs = constants.fft_freqs
    psd[0] = 0.0                        # blank only the DC bin (LO leakage), not a wide band
    if band is not None:
        psd = np.where((freqs >= band[0]) & (freqs <= band[1]), psd, 0.0)
    return interp_peak(psd, int(np.argmax(psd)), freqs)

# Per-symbol analyses (each answers one question about one packet)

def analyze_timing(edges, slot) -> dict:
    """Per-symbol duration, inter-symbol gap, and clock drift.

    Drift is measured at each symbol's slot midpoint (the mean of this symbol's and the next
    symbol's start deviation from an ideal grid anchored on the first start), so a single
    mis-placed start edge is halved rather than shown in full. The last symbol has no
    following start, so no slot midpoint -> no drift.
    """
    sr = constants.SAMPLE_RATE
    if len(edges) < 2:
        return {"per_symbol": [], "sym_mean_ms": None, "gap_mean_ms": None,
                "total_drift_us": None, "drift_us_per_sym": None}
    n = len(edges)
    start_drift = [(s - (edges[0][0] + i * slot)) / sr * 1e6 for i, (s, _e) in enumerate(edges)]
    center = [(start_drift[i] + start_drift[i + 1]) / 2 if i < n - 1 else None for i in range(n)]

    rows, durs, gaps = [], [], []
    for i, (s, e) in enumerate(edges):
        dur = (e - s) / sr * 1e6
        gap = (edges[i + 1][0] - e) / sr * 1e6 if i < n - 1 else None
        drift = center[i]
        if i == 0:
            rate = 0.0
        elif drift is None:
            rate = None
        else:
            rate = round(drift - center[i - 1], 2)
        durs.append(dur)
        if gap is not None:
            gaps.append(gap)
        rows.append({"idx": i, "duration_us": round(dur, 2),
                     "gap_us": round(gap, 2) if gap is not None else None,
                     "drift_us": round(drift, 2) if drift is not None else None,
                     "rate_us_per_sym": rate})
    return {"per_symbol": rows,
            "sym_mean_ms": round(float(np.mean(durs)) / 1e3, 4),
            "gap_mean_ms": round(float(np.mean(gaps)) / 1e3, 4) if gaps else None,
            "total_drift_us": round(start_drift[-1], 2),
            "drift_us_per_sym": round(start_drift[-1] / (n - 1), 2)}


def analyze_amplitude(iq, edges) -> dict:
    """Per-symbol RMS amplitude (dBFS) and SNR above the inter-symbol noise floor."""
    amps, gaps = symbol_amplitudes_dbfs(iq, edges)
    if not amps:
        return {"per_symbol": [], "mean_dbfs": None, "dropoff_db": None,
                "noise_floor_dbfs": None, "snr_db": None}
    floor = float(np.median(gaps)) if gaps else None
    rows = [{"idx": i, "amp_dbfs": round(float(a), 2),
             "snr_db": round(float(a) - floor, 2) if floor is not None else None}
            for i, a in enumerate(amps)]
    return {"per_symbol": rows,
            "mean_dbfs": round(float(np.mean(amps)), 2),
            "dropoff_db": round(float(max(amps) - min(amps)), 2),
            "noise_floor_dbfs": round(floor, 2) if floor is not None else None,
            "snr_db": round(float(np.mean(amps)) - floor, 2) if floor is not None else None}


def _channel_spacing(chipset, step):
    """Device channel spacing (Hz): the decoder's per-chipset table value, else derived by
    quantising CHANNEL_SPACING to the synth-res grid."""
    dcs = constants.DEVICE_CHANNEL_SPACING.get(chipset)
    if dcs:
        return float(dcs), "device-table"
    return float(round(constants.CHANNEL_SPACING / step) * step), "derived"


def analyze_channels(iq, spec, edges):
    """Per-symbol dominant tone vs the expected channel window.

    The channel grid is anchored on the decoder's ``F0_hz`` (value-0 tone of the start channel)
    and ``measured_synth_res`` (intra-channel FSK step) -- not re-derived from the preamble.
    Each symbol's tone is measured band-limited to the packet's channels (rejecting out-of-band
    spurs) and checked against the window of the channel its hop position predicts. Returns None
    when the decode fields needed to build the grid are absent (e.g. a non-decoding detection).
    """
    rotated = rotated_hop_sequence(spec.get("channel_num"), spec.get("hop_seq_idx"))
    F0, step = spec.get("F0_hz"), spec.get("measured_synth_res")
    if rotated is None or F0 is None or step is None or len(edges) < 2:
        return None

    ch = rotated[0]
    sps, hop = constants.samples_per_symbol, constants.NUM_SYM_PER_HOP
    width = constants.NUM_FSK_BINS * step
    spacing, spacing_src = _channel_spacing(spec.get("chipset"), step)

    # Only the channels this packet actually hops through (one per NUM_SYM_PER_HOP block), so
    # the spur-reject band spans just those channels rather than the whole 19-channel range.
    n_hops = -(-len(edges) // hop)                          # ceil(n_sym / hop)
    used = [rotated[h % len(rotated)] for h in range(n_hops)]
    win_lo = {c: F0 + (c - ch) * spacing for c in used}     # each used channel's window lower edge
    band = (min(win_lo.values()) - width, max(win_lo.values()) + 2 * width)

    # FSK tones sit at win_lo + V*step for V in 0..63, so give the extreme tones (value 0 at
    # win_lo, value 63 at the top) a half-bin guard instead of landing exactly on the edge.
    # One half bin is enough to avoid a tone being mis-classified as out-of-window
    half = step / 2
    rows = []
    for i, (s, _e) in enumerate(edges):
        c = used[i // hop]
        fr = _dominant_freq(iq[s:s + sps], band)
        lo, hi = win_lo[c] - half, win_lo[c] - half + width
        ok = lo <= fr < hi
        rows.append({"idx": i, "channel": int(c), "freq_hz": round(fr, 1),
                     "in_window": bool(ok),
                     "off_by_hz": 0.0 if ok else round(min(abs(fr - lo), abs(fr - hi)), 1)})
    return {"calibration": {"f0_hz": round(float(F0), 1), "step_hz": round(float(step), 2),
                            "channel_width_hz": round(width, 1), "spacing_hz": round(spacing, 1),
                            "spacing_source": spacing_src,
                            "channels": [int(c) for c in used]},
            "per_symbol": rows,
            "n_in_window": sum(r["in_window"] for r in rows),
            "all_valid": all(r["in_window"] for r in rows)}


def analyze_chipset(spec) -> dict:
    """Chipset + synth-res as reported by the decoder, with the chipset's nominal table value."""
    name = spec.get("chipset")
    meas = spec.get("measured_synth_res")
    return {"chipset": name,
            "measured_synth_res": round(float(meas), 2) if meas is not None else None,
            "nominal_synth_res": constants.SYNTH_RES.get(name)}


def expected_hops(spec):
    """Expected channel-hop schedule (start channel + rotated sequence). None if unavailable."""
    rotated = rotated_hop_sequence(spec.get("channel_num"), spec.get("hop_seq_idx"))
    if rotated is None:
        return None
    n_sym, _slot = packet_symbol_grid(spec)
    n_hops = -(-n_sym // constants.NUM_SYM_PER_HOP)                 # ceil division
    return {"hop_seq_idx": int(spec["hop_seq_idx"]), "start_channel": int(rotated[0]),
            "expected_channels": [int(rotated[h % len(rotated)]) for h in range(n_hops)]}


# Entry point

def analyze_packet(iq, spec, edges) -> dict:
    """Raw per-symbol signal analysis for one packet.

    iq    : IQ array the edges index into (absolute sample indices).
    spec  : a decoded packet dict, or a minimal detection ``{start_sample, phy_ver,
            num_pdu_symbols}``. Channel/chipset/hop outputs degrade to None when the
            corresponding decode fields are absent.
    edges : corrected ``(start, end)`` sample pairs for the packet's symbols.

    Returns ``{timing, amplitude, channels|None, chipset, hops|None}`` -- plain data, no
    formatting.
    """
    _n, slot = packet_symbol_grid(spec)
    return {
        "timing": analyze_timing(edges, slot),
        "amplitude": analyze_amplitude(iq, edges),
        "channels": analyze_channels(iq, spec, edges),
        "chipset": analyze_chipset(spec),
        "hops": expected_hops(spec),
    }
