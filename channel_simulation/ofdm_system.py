"""
OFDM system configuration and utility functions.
Defines standard OFDM numerology for LTE/5G NR.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class OFDMConfig:
    """Configuration for an OFDM system (LTE/5G NR compatible)."""
    # Subcarrier parameters
    n_subcarriers: int = 72  # Total subcarriers (e.g., 6 resource blocks * 12 subcarriers = 72)
    subcarrier_spacing_khz: float = 15.0  # 15 kHz standard for LTE/5G

    # Time domain
    ofdm_symbol_duration_ms: float = 0.071  # 1/14 ms for CP length ~16.67 μs
    # Or calculate: Ts = 1/Δf where Δf = subcarrier_spacing (Hz)

    # Cyclic prefix
    cp_length_samples: int = 512  # For 2048 FFT size, CP ~1/14
    useful_symbol_length_samples: int = 2048  # FFT size
    total_symbol_length_samples: int = 2048 + 512  # 2560

    # Bandwidth
    bandwidth_khz: float = 0.0  # Computed from n_subcarriers and spacing

    def __post_init__(self):
        if self.bandwidth_khz == 0.0:
            self.bandwidth_khz = self.n_subcarriers * self.subcarrier_spacing_khz

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.subcarrier_spacing_khz * 1000.0

    @property
    def sampling_rate_hz(self) -> float:
        """The sampling rate must be at least equal to the FFT size / useful symbol duration."""
        # Typically: N_fft * subcarrier_spacing (with some guard bands)
        # For LTE: 2048-point FFT with 15 kHz subcarrier spacing -> 30.72 MHz
        n_fft = self.useful_symbol_length_samples
        return n_fft * self.subcarrier_spacing_hz * (1 + self.cp_length_samples / self.useful_symbol_length_samples)

    @property
    def symbol_duration_s(self) -> float:
        """Total OFDM symbol duration including CP."""
        return self.total_symbol_length_samples / self.sampling_rate_hz

    @property
    def useful_symbol_duration_s(self) -> float:
        """Useful OFDM symbol duration (without CP)."""
        return self.useful_symbol_length_samples / self.sampling_rate_hz


def compute_snr_to_ebno(snr_db: float, code_rate: float) -> float:
    """
    Convert SNR_dB to Eb/No_dB given the code rate.
    Eb/No = SNR / code_rate (in linear scale), or +10log10(1/code_rate) in dB.

    Args:
        snr_db: Signal-to-Noise Ratio per RE (resource element) in dB
        code_rate: Number of information bits per coded bit (0 < code_rate ≤ 1)

    Returns:
        Eb/No in dB
    """
    return snr_db - 10 * np.log10(code_rate)


def generate_freq_selective_channel(
    n_subcarriers: int,
    delay_profile: str,
    rms_delay_spread: float,
    sampling_rate_hz: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a frequency-selective (multipath) channel impulse response.

    Args:
        n_subcarriers: Number of subcarriers (frequency resolution)
        delay_profile: "exponential", "uniform", or "tapped"
        rms_delay_spread: RMS delay spread in seconds
        sampling_rate_hz: Sampling rate (determines time resolution)

    Returns:
        h_time: Channel impulse response (time domain). Shape (n_taps,)
        delays: Delay of each tap in seconds
    """
    # Number of taps needed: t_max ≈ 5 * RMS_delay_spread / Ts
    t_resolution = 1.0 / sampling_rate_hz
    max_delay = 5.0 * rms_delay_spread
    n_taps = int(np.ceil(max_delay / t_resolution)) + 1

    if delay_profile == "exponential":
        # Exponential power delay profile: PDP[i] ~ exp(-τᵢ/τ₀) where τ₀ = RMS²
        tau_0 = rms_delay_spread**2
        delays = np.arange(n_taps) * t_resolution
        pdp = np.exp(-delays / tau_0)
        pdp /= pdp.sum()
    elif delay_profile == "uniform":
        # Uniform PDP over max delay
        delays = np.arange(n_taps) * t_resolution
        pdp = np.ones(n_taps) / n_taps
    elif delay_profile == "tapped":
        # 3-tap model (like EPA, EVA, EPY)
        # For simplicity, use discrete taps
        n_taps = 3
        delays = np.array([0, rms_delay_spread/2, rms_delay_spread])
        pdp = np.array([1/3, 1/3, 1/3])
    else:
        raise ValueError(f"Unknown delay profile: {delay_profile}")

    # Complex channel taps (Rayleigh for now)
    taps = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) * np.sqrt(pdp)

    return taps, delays


def compute_channel_frequency_response(
    h_time: np.ndarray,
    n_subcarriers: int,
    subcarrier_spacing_hz: float
) -> np.ndarray:
    """
    Compute frequency response by FFT of time-domain impulse response.

    Args:
        h_time: Channel impulse response (complex taps) [n_taps]
        n_subcarriers: Number of subcarriers to compute
        subcarrier_spacing_hz: Subcarrier spacing

    Returns:
        h_freq: Frequency response at each subcarrier [n_subcarriers]
    """
    # Zero-pad or truncate to n_subcarriers
    H = np.fft.fft(h_time, n=n_subcarriers)
    # Normalize to unit average power (can scale later for SNR)
    H = H / np.sqrt(np.mean(np.abs(H)**2))
    return H


# Standard LTE/5G channel models (3GPP) - simplified Python versions
# For authentic TDL/CDL, use MATLAB's comm toolbox or the DeepMIMO dataset

class ChannelModel:
    """Base class for wireless channel models."""

    def __init__(
        self,
        ofdm_config: Optional[OFDMConfig] = None,
        fs_hz: Optional[float] = None,
        **kwargs
    ):
        self.ofdm_config = ofdm_config or OFDMConfig()
        self.fs_hz = fs_hz or self.ofdm_config.sampling_rate_hz

    def generate_impulse_response(self, n_samples: int, f_d_hz: float) -> np.ndarray:
        """
        Generate time-varying channel impulse response.

        Args:
            n_samples: Number of OFDM symbols
            f_d_hz: Doppler frequency (determines mobility)

        Returns:
            h_time: [n_samples, n_taps] complex taps per OFDM symbol
        """
        raise NotImplementedError

    def get_frequency_response(self, h_time: np.ndarray) -> np.ndarray:
        """Convert impulse response to frequency domain for given OFDM config."""
        n_sub = self.ofdm_config.n_subcarriers
        return np.array([
            compute_channel_frequency_response(h_time[i, :], n_sub, self.ofdm_config.subcarrier_spacing_hz)
            for i in range(h_time.shape[0])
        ])


class RayleighChannelModel(ChannelModel):
    """Simple Rayleigh fading with optional delay spread."""

    def __init__(
        self,
        delay_spread_s: float = 0.0,
        max_doppler_hz: float = 100.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.delay_spread_s = delay_spread_s
        self.max_doppler_hz = max_doppler_hz

        # For delay spread: create taps
        if delay_spread_s > 0:
            self.n_taps = min(10, int(5 * delay_spread_s * self.fs_hz) + 1)
            self.delays = np.arange(self.n_taps) / self.fs_hz
            # Exponential PDP
            tau_0 = delay_spread_s**2
            self.pdp = np.exp(-self.delays / tau_0)
            self.pdp /= self.pdp.sum()
        else:
            self.n_taps = 1
            self.pdp = np.array([1.0])

    def generate_impulse_response(self, n_samples: int, f_d_hz: float) -> np.ndarray:
        # Generate Rayleigh fading for each tap with appropriate time correlation
        h_taps = np.zeros((n_samples, self.n_taps), dtype=complex)
        for tap in range(self.n_taps):
            # Different taps can have independent fading (frequency selectivity)
            h_taps[:, tap] = generate_rayleigh_clarke(
                n_samples,
                f_d_hz,
                self.fs_hz,
                n_paths=200
            ) * np.sqrt(self.pdp[tap])

        return h_taps


class TDLChannelModel(ChannelModel):
    """
    Simplified Tapped Delay Line (TDL) model.
    Reference: 3GPP TR 38.901, Table 7.7.3-1 (TDL-A, TDL-B, TDL-C, etc.)
    This is a weaker simulation; for accurate results, use MATLAB/DeepMIMO.
    """
    # Simplified TDL tap delays and powers (normalized)
    # Format: (delay in samples, power in linear)
    TDL_PROFILES = {
        "TDL-A": [(0, 1.0), (2, 0.9), (4, 0.5), (7, 0.3), (11, 0.2)],
        "TDL-B": [(0, 1.0), (1, 0.9), (2, 0.6), (3, 0.3)],
        "TDL-C": [(0, 1.0), (2, 1.0), (4, 0.9), (7, 0.7), (11, 0.5)],  # High delay spread
    }

    def __init__(self, profile: str = "TDL-C", **kwargs):
        super().__init__(**kwargs)
        self.profile = profile
        if profile not in self.TDL_PROFILES:
            raise ValueError(f"Unknown TDL profile: {profile}. Available: {list(self.TDL_PROFILES.keys())}")

        self.taps_info = self.TDL_PROFILES[profile]
        self.n_taps = len(self.taps_info)
        self.delays_samples = np.array([d for d, p in self.taps_info])
        self.powers = np.array([p for d, p in self.taps_info])
        self.powers /= self.powers.sum()  # Normalize

    def generate_impulse_response(self, n_samples: int, f_d_hz: float) -> np.ndarray:
        h_taps = np.zeros((n_samples, self.n_taps), dtype=complex)
        for tap in range(self.n_taps):
            h_taps[:, tap] = generate_rayleigh_clarke(n_samples, f_d_hz, self.fs_hz, n_paths=200) * np.sqrt(self.powers[tap])
        return h_taps


if __name__ == "__main__":
    # Test channel models
    config = OFDMConfig(n_subcarriers=72)
    ch_model = TDLChannelModel(profile="TDL-C", ofdm_config=config, fs_hz=15000)

    n_symbols = 20
    f_d = 100.0

    print(f"Generating {n_symbols} OFDM symbols...")
    h_time = ch_model.generate_impulse_response(n_symbols, f_d)
    print(f"Impulse response shape: {h_time.shape}")

    # Convert to frequency domain
    h_freq = ch_model.get_frequency_response(h_time)
    print(f"Frequency response shape: {h_freq.shape}")

    # Check average power per subcarrier
    avg_power = np.mean(np.abs(h_freq)**2, axis=0)
    print(f"Mean power across frequency: {np.mean(avg_power):.4f}, std: {np.std(avg_power):.4f}")