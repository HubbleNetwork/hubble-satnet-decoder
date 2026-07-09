"""hubble-satnet-decoder — Hubble PHY preamble detector, FSK decoder, spectrogram."""

from .analysis import analyze_packet, packet_symbol_grid
from .chipset import get_chipset_stats, reset_chipset_stats
from .constants import (
    CHANNEL_SPACING,
    DEVICE_CHANNEL_SPACING,
    HOPPING_SEQS,
    PREAMBLE_CODE_V1,
    RS_K_V1,
    RS_N_V1,
    SYNTH_RES,
    configure,
)
from .decoder import decode_signal
from .detector import detect_preambles
from .spectrogram import compute_spec_chunk

__all__ = [
    "CHANNEL_SPACING",
    "DEVICE_CHANNEL_SPACING",
    "HOPPING_SEQS",
    "PREAMBLE_CODE_V1",
    "RS_K_V1",
    "RS_N_V1",
    "SYNTH_RES",
    "analyze_packet",
    "compute_spec_chunk",
    "configure",
    "decode_signal",
    "detect_preambles",
    "packet_symbol_grid",
    "get_chipset_stats",
    "reset_chipset_stats",
]
