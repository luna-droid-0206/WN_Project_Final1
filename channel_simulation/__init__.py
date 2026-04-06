"""Channel simulation package."""

from .rayleigh_fading import (
    generate_rayleigh_clarke,
    generate_rician_fading,
    generate_spatially_correlated_fading
)
from .ofdm_system import OFDMConfig
from .channel_models import ChannelSimulator, compute_mcs_ber_lut, label_mcs_from_snr

__all__ = [
    'generate_rayleigh_clarke',
    'generate_rician_fading',
    'generate_spatially_correlated_fading',
    'OFDMConfig',
    'ChannelSimulator',
    'compute_mcs_ber_lut',
    'label_mcs_from_snr'
]
