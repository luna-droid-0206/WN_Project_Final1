"""
Data generation pipeline for MCS selection.
Generates channel sequences and labels using oracle policy with BER lookup table.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add project root to Python path for absolute imports when running as script
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from training.config import (
    set_global_seed,
    GLOBAL_SEED,
    DataConfig,
    OracleConfig
)
from channel_simulation import (
    ChannelSimulator,
    OFDMConfig,
    compute_mcs_ber_lut,
    label_mcs_from_snr
)


def compute_snr_from_channel(h_freq: np.ndarray) -> float:
    """
    Compute wideband SNR from frequency-domain channel.

    Args:
        h_freq: Complex channel per subcarrier [n_subcarriers]

    Returns:
        SNR in dB (assuming noise power N0=1 for scaling)
    """
    # Signal power: average |H(f)|^2 across subcarriers
    signal_power = np.mean(np.abs(h_freq)**2)
    # Assuming noise variance = 1 (we'll scale when adding noise later)
    snr_linear = signal_power / 1.0
    return 10 * np.log10(snr_linear + 1e-12)


def generate_dataset(
    n_samples: int,
    channel_type: str = "rayleigh",
    channel_params: Optional[dict] = None,
    snr_range_db: Tuple[float, float] = (0.0, 30.0),
    seq_len: int = 10,
    n_subcarriers: int = 72,
    seed: int = GLOBAL_SEED,
    save_dir: str = "data/",
    split_ratio: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    **kwargs
) -> dict:
    """
    Main dataset generation function.

    Args:
        n_samples: Total number of sequences to generate
        channel_type: "rayleigh", "tdl", "rician", etc.
        channel_params: Model-specific parameters
        snr_range_db: SNR range in dB
        seq_len: Number of OFDM symbols per sequence
        n_subcarriers: Number of subcarriers per OFDM symbol
        seed: Random seed
        save_dir: Directory to save generated data
        split_ratio: Train/val/test split ratios (must sum to 1)

    Returns:
        Dictionary with dataset info and file paths
    """
    set_global_seed(seed)

    print(f"\n{'='*60}")
    print(f"Dataset Generation")
    print(f"{'='*60}")
    print(f"Channel type: {channel_type}")
    print(f"Samples: {n_samples:,}")
    print(f"Sequence length: {seq_len} OFDM symbols")
    print(f"Subcarriers: {n_subcarriers}")
    print(f"SNR range: {snr_range_db[0]} - {snr_range_db[1]} dB")
    print(f"Seed: {seed}")
    print(f"{'='*60}\n")

    # Initialize OFDM and channel simulator
    ofdm_config = OFDMConfig(n_subcarriers=n_subcarriers)
    channel_sim = ChannelSimulator(
        ofdm_config=ofdm_config,
        channel_type=channel_type,
        channel_params=channel_params or {}
    )

    # Pre-compute BER lookup table for oracle labeling
    print("Computing BER lookup table for oracle labeling...")
    lut, snr_bins = compute_mcs_ber_lut(
        snr_range_db,
        OracleConfig.MCS_TABLE,
        snr_step=0.5
    )

    # Generate data
    all_channels = []
    all_snr_estimates = []

    print(f"Generating {n_samples} channel sequences...")
    for i in tqdm(range(n_samples), desc="Generating"):
        # Generate one sequence
        h_seq = channel_sim.generate_channel_sequence(
            seq_len=seq_len,
            snr_db_range=snr_range_db,
            **channel_params
        )
        all_channels.append(h_seq)

        # Estimate wideband SNR from channel power (for labeling)
        # Average power across time and subcarriers
        channel_power = np.mean(np.sum(h_seq**2, axis=-1))  # E[|h|^2]
        snr_db = 10 * np.log10(channel_power / 1.0)  # Assume N0=1
        all_snr_estimates.append(snr_db)

    print("Labeling with oracle MCS policy (SNR+LUT)...")
    snr_array = np.array(all_snr_estimates)
    mcs_labels = label_mcs_from_snr(snr_array, lut, snr_bins, target_ber=OracleConfig.TARGET_BER)

    # Stack channel sequences
    channels_array = np.stack(all_channels, axis=0)  # [n_samples, seq_len, n_subcarriers, 2]
    print(f"Dataset shape: {channels_array.shape}")
    print(f"MCS distribution: {np.bincount(mcs_labels, minlength=DataConfig.N_MCS_CLASSES)}")

    # Split into train/val/test
    n_train = int(n_samples * split_ratio[0])
    n_val = int(n_samples * split_ratio[1])
    n_test = n_samples - n_train - n_val

    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]

    # Save to disk
    os.makedirs(save_dir, exist_ok=True)

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx
    }

    for split_name, split_indices in splits.items():
        split_channels = channels_array[split_indices]
        split_labels = mcs_labels[split_indices]

        split_dir = Path(save_dir) / split_name
        split_dir.mkdir(exist_ok=True)

        # Save as numpy arrays (for fast loading)
        np.save(split_dir / "channels.npy", split_channels)
        np.save(split_dir / "labels.npy", split_labels)

        print(f"[{split_name.upper()}] Saved {len(split_indices)} samples")

    # Save metadata
    metadata = {
        "n_samples": n_samples,
        "seq_len": seq_len,
        "n_subcarriers": n_subcarriers,
        "channel_type": channel_type,
        "channel_params": channel_params,
        "snr_range_db": snr_range_db,
        "seed": seed,
        "n_mcs_classes": DataConfig.N_MCS_CLASSES,
        "split_ratio": split_ratio,
        "splits": {k: len(v) for k, v in splits.items()},
        "mcs_distribution": np.bincount(mcs_labels).tolist(),
        "oracle_target_ber": OracleConfig.TARGET_BER
    }

    metadata_path = Path(save_dir) / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nDataset saved to: {save_dir}")
    print(f"Metadata: {metadata_path}")

    return {
        "train_dir": str(Path(save_dir) / "train"),
        "val_dir": str(Path(save_dir) / "val"),
        "test_dir": str(Path(save_dir) / "test"),
        "metadata_path": str(metadata_path),
        "metadata": metadata
    }


def load_split(split_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a single split (train/val/test).

    Args:
        split_dir: Directory containing channels.npy and labels.npy

    Returns:
        channels: [n_samples, seq_len, n_subcarriers, 2]
        labels: [n_samples] with MCS indices (0-15)
    """
    split_path = Path(split_dir)
    channels = np.load(split_path / "channels.npy")
    labels = np.load(split_path / "labels.npy")
    return channels, labels


def _worker_init_fn(worker_id: int, seed: int = GLOBAL_SEED):
    """Initialize worker with a unique but reproducible seed."""
    worker_seed = (seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def create_dataloader(
    split_dir: str,
    batch_size: int = 128,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = GLOBAL_SEED
) -> torch.utils.data.DataLoader:
    """
    Create PyTorch DataLoader from saved dataset.

    Args:
        split_dir: Directory with .npy files
        batch_size: Batch size
        shuffle: Whether to shuffle (use False for val/test)
        num_workers: Number of DataLoader workers
        seed: Random seed for reproducible shuffling

    Returns:
        DataLoader
    """
    from torch.utils.data import TensorDataset, DataLoader

    channels, labels = load_split(split_dir)

    # Convert to tensors
    channels_tensor = torch.from_numpy(channels).float()
    labels_tensor = torch.from_numpy(labels).long()

    dataset = TensorDataset(channels_tensor, labels_tensor)

    def worker_init_fn(worker_id):
        """Ensure random seeds are different across workers but reproducible."""
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    # Use worker_init_fn only if shuffle needed (train) and num_workers > 0
    from functools import partial
    w_init = partial(_worker_init_fn, seed=seed) if (shuffle and num_workers > 0) else None

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=w_init,
        persistent_workers=True if num_workers > 0 else False
    )

    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MCS selection dataset")
    parser.add_argument("--n_samples", type=int, default=100000, help="Total samples to generate")
    parser.add_argument("--channel_type", type=str, default="rayleigh", choices=["rayleigh", "tdl", "rician", "awgn"])
    parser.add_argument("--doppler_hz", type=float, default=100.0, help="Doppler frequency in Hz")
    parser.add_argument("--delay_spread_s", type=float, default=0.001, help="RMS delay spread (for frequency-selective)")
    parser.add_argument("--output", type=str, default="data/", help="Output directory")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length (OFDM symbols)")
    parser.add_argument("--seed", type=int, default=GLOBAL_SEED, help="Random seed")
    parser.add_argument("--force", action="store_true", help="Overwrite existing data")

    args = parser.parse_args()

    # Check if output already exists
    output_path = Path(args.output)
    if output_path.exists() and not args.force:
        response = input(f"Output directory {args.output} exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborting.")
            exit(0)

    # Generate dataset
    channel_params = {
        "doppler_hz": args.doppler_hz,
        "delay_spread_s": args.delay_spread_s
    }

    info = generate_dataset(
        n_samples=args.n_samples,
        channel_type=args.channel_type,
        channel_params=channel_params,
        snr_range_db=DataConfig.SNR_RANGE_DB,
        seq_len=args.seq_len,
        n_subcarriers=DataConfig.N_SUBCARRIERS,
        seed=args.seed,
        save_dir=args.output
    )

    print("\nDataset generation complete!")
    print(f"Files saved in: {args.output}")

    # Quick test of DataLoader
    print("\nTesting DataLoader...")
    train_loader = create_dataloader(info["train_dir"], batch_size=32, shuffle=True, num_workers=2)
    batch_channels, batch_labels = next(iter(train_loader))
    print(f"Batch channels shape: {batch_channels.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Sample MCS labels: {batch_labels[:10]}")