"""
Channel model definitions: Rayleigh, TDL, and CDL models.
This module provides a unified interface for generating wireless channel conditions.
"""

from typing import Optional, Tuple, List, Dict
from pathlib import Path

import numpy as np

from .rayleigh_fading import (
    generate_rayleigh_clarke,
    generate_rician_fading,
    generate_spatially_correlated_fading
)
from .ofdm_system import OFDMConfig, compute_channel_frequency_response


class ChannelSimulator:
    """
    Main channel simulator class.
    Generates time-frequency channel coefficients for MCS dataset.
    """

    def __init__(
        self,
        ofdm_config: Optional[OFDMConfig] = None,
        channel_type: str = "rayleigh",
        channel_params: Optional[Dict] = None,
        **kwargs
    ):
        """
        Args:
            ofdm_config: OFDM system configuration
            channel_type: "rayleigh", "tdl", "cdl", or "awgn" (flat)
            channel_params: Model-specific parameters:
                For "rayleigh": {"doppler_hz": 100.0, "delay_spread_s": 0.001}
                For "tdl": {"profile": "TDL-C", "doppler_hz": 100.0}
                For "cdl": {"profile": "CDL-D", "doppler_hz": 100.0} (stub - use MATLAB if available)
        """
        self.ofdm_config = ofdm_config or OFDMConfig()
        self.channel_type = channel_type
        self.channel_params = channel_params or {}

        # Pre-computed channel cache (for fast generation if using MATLAB pre-generated)
        self._cached_h_freq = None
        self._cache_idx = 0

    def load_precomputed_channels(self, filepath: str):
        """
        Load pre-generated channel coefficients from .npy file.
        File should contain array of shape [n_samples, n_subcarriers] complex.
        Useful if using MATLAB for TDL/CDL generation.

        Args:
            filepath: Path to .npy file
        """
        self._cached_h_freq = np.load(filepath)
        self._cache_idx = 0
        print(f"[ChannelSimulator] Loaded {len(self._cached_h_freq)} precomputed channels from {filepath}")

    def generate_single_channel(
        self,
        snr_db: float,
        doppler_hz: Optional[float] = None,
        delay_spread_s: Optional[float] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate channel frequency response for one OFDM symbol.

        Args:
            snr_db: SNR in dB (used to scale channel power, optionally add noise)
            doppler_hz: Override Doppler frequency
            delay_spread_s: Override delay spread (for frequency selectivity)

        Returns:
            h_freq: Complex channel per subcarrier [n_subcarriers]
            h_time: (if applicable) impulse response [n_taps]
        """
        doppler = doppler_hz or self.channel_params.get("doppler_hz", 100.0)
        delay_spread = delay_spread_s or self.channel_params.get("delay_spread_s", 0.0)

        if self.channel_type == "awgn":
            # Flat fading (AWGN channel, no fading - just noise)
            h_freq = np.ones(self.ofdm_config.n_subcarriers, dtype=complex)
            h_time = None

        elif self.channel_type == "rayleigh":
            # Rayleigh fading with optional delay spread
            if delay_spread > 0:
                # Frequency-selective: generate taps and FFT
                n_taps = min(10, int(5 * delay_spread * self.ofdm_config.sampling_rate_hz) + 1)
                h_taps = np.zeros(n_taps, dtype=complex)
                for i in range(n_taps):
                    h_taps[i] = generate_rayleigh_clarke(1, doppler, self.ofdm_config.sampling_rate_hz)[0]
                # Apply PDP
                tau = np.arange(n_taps) / self.ofdm_config.sampling_rate_hz
                pdp = np.exp(-tau / (delay_spread**2))
                pdp /= pdp.sum()
                h_taps *= np.sqrt(pdp)
                h_freq = compute_channel_frequency_response(h_taps, self.ofdm_config.n_subcarriers, self.ofdm_config.subcarrier_spacing_hz)
                h_time = h_taps
            else:
                # Flat fading: same gain across all subcarriers
                h = generate_rayleigh_clarke(1, doppler, self.ofdm_config.sampling_rate_hz)[0]
                h_freq = np.full(self.ofdm_config.n_subcarriers, h, dtype=complex)
                h_time = np.array([h])

        elif self.channel_type == "tdl":
            # Use simplified TDL implementation or cached MATLAB data
            h_taps = self._generate_tdl_taps(doppler, **kwargs)
            h_freq = compute_channel_frequency_response(h_taps, self.ofdm_config.n_subcarriers, self.ofdm_config.subcarrier_spacing_hz)
            h_time = h_taps

        elif self.channel_type == "rician":
            K = self.channel_params.get("K_factor", 5.0)
            h = generate_rician_fading(1, doppler, self.ofdm_config.sampling_rate_hz, K)[0]
            h_freq = np.full(self.ofdm_config.n_subcarriers, h, dtype=complex)
            h_time = np.array([h])

        else:
            raise ValueError(f"Unknown channel type: {self.channel_type}")

        return h_freq, h_time

    def _generate_tdl_taps(self, doppler_hz: float, profile: str = "TDL-C") -> np.ndarray:
        """Generate TDL taps (simplified)."""
        if self._cached_h_freq is not None:
            # Use pre-generated data: draw random channel and convert to taps?
            # For now, fallback to Rayleigh with delay spread
            return generate_rayleigh_clarke(5, doppler_hz, self.ofdm_config.sampling_rate_hz) * np.array([1.0, 0.9, 0.7])
        else:
            # Simplified TDL-C (3 taps)
            taps = np.zeros(3, dtype=complex)
            powers = [1.0, 0.9, 0.5]
            for i in range(3):
                taps[i] = generate_rayleigh_clarke(1, doppler_hz, self.ofdm_config.sampling_rate_hz)[0] * np.sqrt(powers[i])
            return taps

    def generate_channel_sequence(
        self,
        seq_len: int,
        snr_db_range: Tuple[float, float] = (0.0, 30.0),
        doppler_hz: Optional[float] = None,
        normalize_power: bool = True,
        **kwargs
    ) -> np.ndarray:
        """
        Generate a time sequence of channel frequency responses.

        Args:
            seq_len: Number of OFDM symbols in sequence
            snr_db_range: Range of SNR values to sample (min, max)
            doppler_hz: Doppler frequency for temporal correlation
            normalize_power: If True, normalize each channel to unit power

        Returns:
            h_seq: [seq_len, n_subcarriers, 2] (real/imag channels)
        """
        doppler = doppler_hz or self.channel_params.get("doppler_hz", 100.0)

        # Sample random SNRs from range (for scaling)
        snr_values = np.random.uniform(snr_db_range[0], snr_db_range[1], size=seq_len)
        # Linear SNR values (snr = |h|^2 / N0) - we'll use to scale |h|
        # Not adding noise yet; just varying channel gain

        h_seq_real = np.zeros((seq_len, self.ofdm_config.n_subcarriers))
        h_seq_imag = np.zeros((seq_len, self.ofdm_config.n_subcarriers))

        for t in range(seq_len):
            h_freq, _ = self.generate_single_channel(
                snr_db=snr_values[t],
                doppler_hz=doppler,
                **kwargs
            )

            if normalize_power:
                # Scale to have average power 1 across subcarriers
                power = np.mean(np.abs(h_freq)**2)
                if power > 0:
                    h_freq = h_freq / np.sqrt(power)

            # Scale by SNR to get desired average power
            scale = 10**(snr_values[t] / 20)  # Linear SNR scale to magnitude
            h_freq = h_freq * scale

            h_seq_real[t] = h_freq.real
            h_seq_imag[t] = h_freq.imag

        # Stack into [seq_len, n_subcarriers, 2]
        h_seq = np.stack([h_seq_real, h_seq_imag], axis=-1)

        return h_seq


# Utility: Generate MCS lookup table (LTE CQI -> required SNR for BER=10^-3)
def compute_mcs_ber_lut(
    snr_range_db: Tuple[float, float],
    mcs_table: Dict[int, Tuple[str, float]],
    snr_step: float = 0.5
) -> np.ndarray:
    """
    Pre-compute a lookup table of BER vs SNR for each MCS.
    For quick oracle labeling without full link simulation.

    WARNING: This uses a simplified analytical model or pre-tabulated values.
    For more accuracy, use a pre-computed table from standards or simulations.

    Args:
        snr_range_db: (min_snr, max_snr) in dB
        mcs_table: Dictionary {mcs_idx: (modulation, code_rate)}
        snr_step: Step size for SNR grid in dB

    Returns:
        lut: 2D array [snr_bins, n_mcs] with BER values
    """
    snr_bins = np.arange(snr_range_db[0], snr_range_db[1] + snr_step, snr_step)
    n_snr = len(snr_bins)
    n_mcs = len(mcs_table)

    # For quick implementation, use approximate analytical BER formulas
    # In production, replace with pre-tabulated values from link simulations
    lut = np.zeros((n_snr, n_mcs))

    for i, snr_lin in enumerate(10**(snr_bins / 10)):
        for mcs_idx, (mod, rate) in mcs_table.items():
            # Simplified: BER ≈ c * Q(sqrt(Es/N0 * d_min^2))
            # Different modulations have different constants
            if mod == "QPSK":
                ber = 0.5 * np.exp(-snr_lin * rate / 2)  # Approximate
            elif mod == "16QAM":
                ber = 0.375 * np.exp(-snr_lin * rate / 5)  # Very rough
            elif mod == "64QAM":
                ber = 0.25 * np.exp(-snr_lin * rate / 14)  # Even rougher
            else:
                ber = 1e-3  # Default

            # Clamp to reasonable range
            lut[i, mcs_idx - 1] = np.clip(ber, 1e-12, 1.0)

    return lut, snr_bins


def label_mcs_from_snr(
    snr_db: np.ndarray,
    lut: np.ndarray,
    snr_bins: np.ndarray,
    target_ber: float = 1e-3
) -> np.ndarray:
    """
    Assign MCS labels given SNR using pre-computed BER lookup table.

    Args:
        snr_db: Array of SNR values (dB) per sample
        lut: BER lookup table [n_snr_bins, n_mcs]
        snr_bins: SNR bin centers
        target_ber: Maximum acceptable BER threshold

    Returns:
        mcs_labels: Integer MCS indices (1-15) or 0 if no MCS satisfies
    """
    # Find SNR bin indices
    bin_indices = np.digitize(snr_db, snr_bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(snr_bins)-1)

    mcs_labels = np.zeros(len(snr_db), dtype=int)

    for i in range(len(snr_db)):
        ber_profile = lut[bin_indices[i], :]
        # Find highest MCS that satisfies BER <= target_ber
        satisfied = np.where(ber_profile <= target_ber)[0]
        if len(satisfied) > 0:
            mcs_labels[i] = satisfied[-1] + 1  # MCS indices from 1-15
        else:
            mcs_labels[i] = 0  # No transmission (or lowest MCS)

    return mcs_labels


if __name__ == "__main__":
    # Test channel simulator
    print("Testing ChannelSimulator...")
    config = OFDMConfig(n_subcarriers=72)
    sim = ChannelSimulator(config, channel_type="rayleigh", channel_params={"doppler_hz": 100.0, "delay_spread_s": 0.0})

    # Generate a sequence
    seq = sim.generate_channel_sequence(seq_len=10, snr_db_range=(5, 15))
    print(f"Sequence shape: {seq.shape}")  # [10, 72, 2]
    print(f"Mean power: {np.mean(np.sum(seq**2, axis=-1)):.4f}")

    # Test LUT generation
    from training.config import OracleConfig
    lut, bins = compute_mcs_ber_lut((0, 30), OracleConfig.MCS_TABLE)
    print(f"\nBER LUT shape: {lut.shape}")
    print(f"SNR bins: {bins[0]} to {bins[-1]} ({len(bins)} bins)")

    # Test MCS labeling
    test_snrs = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    labels = label_mcs_from_snr(test_snrs, lut, bins)
    print(f"\nSNR -> MCS mapping:")
    for snr, label in zip(test_snrs, labels):
        print(f"  SNR={snr:.1f} dB -> MCS={label}")