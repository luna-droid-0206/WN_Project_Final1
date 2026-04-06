"""
Configuration module for the MCS selection project.
Defines global seed, hyperparameters, and utility functions.
"""

import random
import numpy as np
import torch

# ============ GLOBAL SEED FOR REPRODUCIBILITY ============
GLOBAL_SEED = 42

def set_global_seed(seed: int = GLOBAL_SEED):
    """
    Set random seed for all random number generators.
    Call this at the very beginning of every script.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for reproducibility

# ============ DEFAULT HYPERPARAMETERS ============

# Data generation
class DataConfig:
    N_SUBCARRIERS = 72  # LTE/5G typical (12 subcarriers per RB, 6 RBs)
    SEQ_LEN = 10  # Number of past OFDM symbols to use as context
    N_MCS_CLASSES = 16  # MCS indices 0-15: 0=no transmission, 1-15=LTE CQI levels
    SNR_RANGE_DB = (0.0, 30.0)  # Min and max SNR in dB
    SNR_STEP = 0.5  # Step size for SNR sampling
    N_SAMPLES_TRAIN = 80000
    N_SAMPLES_VAL = 10000
    N_SAMPLES_TEST = 10000

# Model architecture (Attention)
class AttentionModelConfig:
    D_MODEL = 128  # Hidden dimension
    N_HEADS = 4  # Number of attention heads
    N_LAYERS = 2  # Number of transformer encoder layers
    DROPOUT = 0.1
    FF_DIM = 256  # Feed-forward dimension (typically 2-4x d_model)
    USE_2D_ATTENTION = False  # If True: attention over time+frequency; if False: temporal only

# Baseline models
class DNNConfig:
    HIDDEN_DIMS = [512, 256, 128, 64]
    DROPOUT = 0.2

class CNNConfig:
    CHANNELS = [32, 64, 128]
    KERNEL_SIZE = 3
    DROPOUT = 0.2

class LSTMConfig:
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2  # Between layers (if NUM_LAYERS > 1)
    BIDIRECTIONAL = False

# Training
class TrainingConfig:
    BATCH_SIZE = 128
    NUM_EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 10  # Early stopping patience (epochs)
    LR_PATIENCE = 5  # ReduceLROnPlateau patience
    LR_FACTOR = 0.5
    GRADIENT_CLIP = 1.0
    USE_AMP = True  # Automatic Mixed Precision (if GPU supports)
    SAVE_BEST_ONLY = True
    CHECKPOINT_DIR = "results/checkpoints/"

# Oracle labeling
class OracleConfig:
    TARGET_BER = 1e-3  # Standard BER threshold for MCS selection
    MCS_TABLE = {
        # LTE CQI table (simplified): MCS index -> (modulation, code_rate)
        1: ("QPSK", 0.076),
        2: ("QPSK", 0.12),
        3: ("QPSK", 0.19),
        4: ("QPSK", 0.3),
        5: ("QPSK", 0.44),
        6: ("QPSK", 0.59),
        7: ("16QAM", 0.37),
        8: ("16QAM", 0.48),
        9: ("16QAM", 0.6),
        10: ("64QAM", 0.45),
        11: ("64QAM", 0.55),
        12: ("64QAM", 0.65),
        13: ("64QAM", 0.75),
        14: ("64QAM", 0.85),
        15: ("64QAM", 0.93),
    }