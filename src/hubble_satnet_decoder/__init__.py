"""hubble-satnet-decoder — Hubble PHY preamble detector, FSK decoder, spectrogram."""

from .chipset import get_chipset_stats, reset_chipset_stats
from . import constants
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


def enable_collision_logging(enabled: bool = True) -> None:
    """Enable or disable collision detection logging.

    When enabled, prints [COLLISION] messages when secondary peaks
    are detected during symbol demodulation (indicates potential
    packet collisions or interference).
    """
    constants.LOG_COLLISIONS = enabled
    
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
    "compute_spec_chunk",
    "configure",
    "decode_signal",
    "detect_preambles",
    "enable_collision_logging",
    "get_chipset_stats",
    "reset_chipset_stats",
]
