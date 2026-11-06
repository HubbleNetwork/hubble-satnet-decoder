"""Build synthetic PHY v1 packets as IQ, mirroring what _decode_v1 expects.

Test helper only: an ideal 64-FSK transmitter (no phase continuity, no
filtering) that can plant wrong symbols to exercise Reed-Solomon correction.
"""

import numpy as np
import reedsolo as rs

from hubble_satnet_decoder import constants
from hubble_satnet_decoder.whitening import lfsr7_symbols


def _rs_codeword(data_syms: list[int], n: int) -> list[int]:
    return list(rs.rs_encode_msg(data_syms, n - len(data_syms), fcr=1))


def _bits_to_syms(bits: str, nsym: int) -> list[int]:
    bits = bits.ljust(nsym * 6, "0")
    return [int(bits[i * 6:(i + 1) * 6], 2) for i in range(nsym)]


def v1_codewords(*, channel=4, hop_seq_idx=1, pkt_len_idx=1, seq=321,
                 ntw_id=0x1234ABCD, auth_tag=0xCAFEF00D, payload=0xDEADBEEF):
    """Return ``(header_codeword, pdu_codeword)``: RS-encoded, before whitening."""
    # The header has 4 bits of channel: 16-18 alias to 0-2 (the decoder
    # resolves them by frequency)
    hdr_bits = f"{1:04b}{pkt_len_idx:02b}{hop_seq_idx:02b}{channel & 0xF:04b}"
    header = _rs_codeword(_bits_to_syms(hdr_bits, 2), constants.RS_N_V1[0])

    k = constants.RS_K_V1[pkt_len_idx + 1]
    n = constants.RS_N_V1[pkt_len_idx + 1]
    payload_bits = constants.PAYLOAD_LEN_BYTES_V1[pkt_len_idx] * 8
    mac_bits = f"{0:02b}{seq:010b}{ntw_id:032b}{auth_tag:032b}"
    if payload_bits:
        mac_bits += f"{payload:0{payload_bits}b}"
    pdu = _rs_codeword(_bits_to_syms(mac_bits, k), n)
    return header, pdu


def v1_signal(*, channel=4, hop_seq_idx=1, pkt_len_idx=1, chipset="nordic",
              start_s=0.2, total_s=1.5, amplitude=0.05, noise=1e-4,
              header_errors=None, pdu_errors=None, seed=7, **mac):
    """IQ for one v1 packet.

    ``header_errors`` / ``pdu_errors`` map a codeword position to the symbol
    actually sent there (PDU positions are before whitening, like the
    decoder's ``pdu_syms_rx``).
    """
    header, pdu = v1_codewords(channel=channel, hop_seq_idx=hop_seq_idx,
                               pkt_len_idx=pkt_len_idx, **mac)
    header = list(header)
    pdu = list(pdu)
    for pos, sym in (header_errors or {}).items():
        header[pos] = sym
    for pos, sym in (pdu_errors or {}).items():
        pdu[pos] = sym
    whitened = [s ^ w for s, w in zip(pdu, lfsr7_symbols(channel, len(pdu)))]
    symbols = list(constants.PREAMBLE_CODE_V1) + header + whitened

    # Each FSK tone is one synth-res step above the last. Symbol 0 is the
    # lowest tone, and the 64 tones are centered on the channel frequency.
    synth_res = constants.SYNTH_RES[chipset]
    hop_spacing = constants.DEVICE_CHANNEL_SPACING[chipset]
    channel_center = (channel - constants.LO_CHANNEL) * constants.CHANNEL_SPACING
    half_tone_span = (constants.NUM_FSK_BINS - 1) / 2 * synth_res
    f0 = channel_center - half_tone_span
    hopping = constants.HOPPING_SEQS[hop_seq_idx]
    hop_index = hopping.index(channel)

    fs = constants.SAMPLE_RATE
    nsym = constants.samples_per_symbol
    slot = constants.slot_samples[1]["slot"]
    rng = np.random.default_rng(seed)
    n_total = int(total_s * fs)
    sig = (rng.standard_normal(n_total) + 1j * rng.standard_normal(n_total)) * noise
    t = np.arange(nsym) / fs
    start = int(start_s * fs)
    for i, sym in enumerate(symbols):
        ch = hopping[(hop_index + i // constants.NUM_SYM_PER_HOP) % constants.NUM_CHANNELS]
        freq = f0 + (ch - channel) * hop_spacing + sym * synth_res
        s0 = start + i * slot
        sig[s0:s0 + nsym] += amplitude * np.exp(2j * np.pi * freq * t)
    return sig.astype(np.complex64)
