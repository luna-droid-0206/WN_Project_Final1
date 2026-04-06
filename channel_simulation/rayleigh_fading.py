"""
Rayleigh/Rician fading channel simulation with Doppler effects.
Implements Clarke's/Jake's model for generating time-correlated fading coefficients.
"""

import numpy as np
from typing import Tuple, Optional


def generate_rayleigh_clarke(
    n_samples: int,
    f_d: float,
    fs: float,
    n_paths: int = 100
) -> np.ndarray:
    """
    Generate Rayleigh fading coefficients using Clarke's method (sum of sinusoids).
    References: Jakes' model, Rayleigh fading with specified Doppler spread.

    Args:
        n_samples: Number of time samples to generate
        f_d: Doppler frequency (Hz). For UE speed v: f_d = v * fc / c
             e.g., v=120 km/h, fc=2.1 GHz -> f_d ≈ 233 Hz
        fs: Sampling frequency (Hz). For OFDM: 1/T_symbol, e.g., 15 kHz subcarrier spacing
        n_paths: Number of paths to sum (higher = more accurate, slower)

    Returns:
        Complex fading coefficients h[n] with Rayleigh distribution.
        Shape: (n_samples,)
    """
    # Clarke's model: h(t) = sum_{n=1}^{N} (1/sqrt(N)) * exp(j*2π*f_n*t + j*θ_n)
    # where f_n = f_d * cos(θ_n) and θ_n uniform [0, 2π]

    # Generate evenly spaced angles for better properties
    theta = np.linspace(0, 2*np.pi, n_paths, endpoint=False)
    phases = np.random.uniform(0, 2*np.pi, n_paths)

    # Doppler frequencies for each path
    f_n = f_d * np.cos(theta)

    # Time vector
    t = np.arange(n_samples) / fs

    # Sum sinusoids
    h = np.zeros(n_samples, dtype=complex)
    for i in range(n_paths):
        h += (1.0 / np.sqrt(n_paths)) * np.exp(1j * (2 * np.pi * f_n[i] * t + phases[i]))

    # Normalize to unit average power (E[|h|^2] = 1)
    power = np.mean(np.abs(h)**2)
    if power > 0:
        h = h / np.sqrt(power)

    return h


def generate_rician_fading(
    n_samples: int,
    f_d: float,
    fs: float,
    K: float = 5.0,
    n_paths: int = 100
) -> np.ndarray:
    """
    Generate Rician fading coefficients (LOS + scattered components).

    Args:
        n_samples: Number of time samples
        f_d: Doppler frequency (Hz)
        fs: Sampling frequency (Hz)
        K: Rician K-factor (ratio of LOS to scattered power)
           K=0 -> Rayleigh; K=∞ -> pure sinusoid
        n_paths: Number of scattered paths

    Returns:
        Complex fading coefficients h[n]
    """
    # LOS component
    theta_los = np.random.uniform(0, 2*np.pi)
    t = np.arange(n_samples) / fs
    f_los = f_d * np.cos(theta_los)
    h_los = np.sqrt(K / (K + 1)) * np.exp(1j * (2 * np.pi * f_los * t + theta_los))

    # Scattered component (Rayleigh)
    h_scat = generate_rayleigh_clarke(n_samples, f_d, fs, n_paths) * np.sqrt(1 / (K + 1))

    return h_los + h_scat


def generate_spatially_correlated_fading(
    n_antennas: int,
    n_samples: int,
    f_d: float,
    fs: float,
    correlation: float = 0.5,
    model: str = "rayleigh"
) -> np.ndarray:
    """
    Generate spatially correlated fading across multiple antennas.

    Args:
        n_antennas: Number of transmit/receive antennas
        n_samples: Number of time samples
        f_d: Doppler frequency
        fs: Sampling frequency
        correlation: Antenna correlation coefficient (0=independent, 1=identical)
        model: "rayleigh" or "rician"

    Returns:
        Complex fading matrix [n_samples, n_antennas]
    """
    if model == "rayleigh":
        gen_func = generate_rayleigh_clarke
    elif model == "rician":
        gen_func = lambda n_samples, f_d, fs: generate_rician_fading(n_samples, f_d, fs, K=5.0)
    else:
        raise ValueError(f"Unknown model: {model}")

    # Generate independent fading for each antenna
    h_indep = np.zeros((n_samples, n_antennas), dtype=complex)
    for ant in range(n_antennas):
        h_indep[:, ant] = gen_func(n_samples, f_d, fs)

    if correlation == 0:
        return h_indep

    # Apply spatial correlation using Cholesky decomposition
    # Correlation matrix: R[i,j] = correlation^(|i-j|)
    R = np.zeros((n_antennas, n_antennas), dtype=complex)
    for i in range(n_antennas):
        for j in range(n_antennas):
            R[i, j] = correlation ** abs(i - j)

    # Cholesky: R = L * L^H
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        # If R is not positive definite, use identity
        L = np.eye(n_antennas)

    # Correlate: h_corr = h_indep @ L^H
    h_corr = h_indep @ L.conj().T

    # Normalize each antenna to unit power
    for ant in range(n_antennas):
        power = np.mean(np.abs(h_corr[:, ant])**2)
        if power > 0:
            h_corr[:, ant] /= np.sqrt(power)

    return h_corr


def compute_autocorrelation(h: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """
    Compute autocorrelation of fading envelope (|h|).
    Useful for validating generated channel.

    Args:
        h: Complex fading coefficients [n_samples]
        max_lag: Maximum lag to compute

    Returns:
        Autocorrelation array of length max_lag
    """
    envelope = np.abs(h)
    n = len(envelope)
    acorr = np.correlate(envelope - np.mean(envelope), envelope - np.mean(envelope), mode='full')
    acorr = acorr[n-1 : n-1+max_lag]
    acorr /= acorr[0]  # Normalize
    return acorr


def compute_doppler_spectrum(h: np.ndarray, fs: float, n_fft: int = 2048) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Doppler spectrum of fading envelope.

    Args:
        h: Complex fading coefficients [n_samples]
        fs: Sampling frequency (Hz)
        n_fft: FFT size

    Returns:
        freqs: Frequency bins (Hz), spectrum: Power spectral density
    """
    envelope = np.abs(h)
    # Apply window to reduce spectral leakage
    window = np.hanning(len(envelope))
    spectrum = np.abs(np.fft.fft(envelope * window, n=n_fft))**2

    freqs = np.fft.fftfreq(n_fft, d=1/fs)
    # Keep only positive frequencies
    positive_mask = freqs >= 0
    return freqs[positive_mask], spectrum[positive_mask]


def add_awgn(h: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Add Additive White Gaussian Noise to complex channel.
    Useful for testing: some systems observe channel+noise.

    Args:
        h: Noiseless complex channel coefficients
        snr_db: Signal-to-Noise Ratio in dB (linear E[|h|^2] / N0)

    Returns:
        y = h + n where n ~ CN(0, N0)
    """
    # Signal power (average)
    signal_power = np.mean(np.abs(h)**2)

    # Noise power
    noise_power = signal_power / (10**(snr_db / 10))

    # Complex Gaussian noise
    noise = np.random.normal(0, np.sqrt(noise_power/2), size=h.shape) + \
            1j * np.random.normal(0, np.sqrt(noise_power/2), size=h.shape)

    return h + noise


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Test the fading generator
    print("Testing Rayleigh fading generation...")

    f_d = 100.0  # 100 Hz Doppler (≈ 50 km/h at 2 GHz)
    fs = 15000.0  # 15 kHz sampling (LTE subcarrier spacing)
    n_samples = 10000

    h = generate_rayleigh_clarke(n_samples, f_d, fs, n_paths=200)

    print(f"Generated {len(h)} samples")
    print(f"Mean power: {np.mean(np.abs(h)**2):.4f} (should be 1.0)")
    print(f"Std power: {np.std(np.abs(h)**2):.4f}")
    print(f"Rayleigh distribution test (K-factor ≈ 0): {np.mean(np.abs(h)**4) / (np.mean(np.abs(h)**2)**2 - 1e-6):.2f}")

    # Autocorrelation
    acorr = compute_autocorrelation(h, max_lag=20)
    print(f"\nAutocorrelation at lag 1: {acorr[1]:.4f} (higher Doppler -> faster decay)")
    print(f"Autocorrelation at lag 5: {acorr[4]:.4f}")

    # Doppler spectrum
    freqs, spectrum = compute_doppler_spectrum(h, fs)
    print(f"\nDoppler spectrum peak frequency: {freqs[np.argmax(spectrum)]:.1f} Hz (expected ≈ {f_d} Hz)")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].plot(np.arange(1000), np.abs(h[:1000]))
    axes[0, 0].set_title("Fading envelope (first 1000 samples)")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("|h(t)|")

    axes[0, 1].hist(np.abs(h), bins=50, density=True, alpha=0.7)
    axes[0, 1].set_title("Amplitude distribution (Rayleigh)")
    axes[0, 1].set_xlabel("|h|")
    axes[0, 1].set_ylabel("PDF")

    axes[1, 0].plot(acorr)
    axes[1, 0].set_title("Autocorrelation of envelope")
    axes[1, 0].set_xlabel("Lag")
    axes[1, 0].set_ylabel("Correlation")

    axes[1, 1].plot(freqs, spectrum)
    axes[1, 1].set_title("Doppler spectrum")
    axes[1, 1].set_xlabel("Frequency (Hz)")
    axes[1, 1].set_ylabel("PSD")
    axes[1, 1].set_xlim(0, f_d * 1.5)

    plt.tight_layout()
    plt.show()