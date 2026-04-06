# Attention-Based MCS Selection for Wireless Networks

**Wireless Networks Course Project** - Team of 4  
**First application of Transformer attention to MCS prediction**  

## Project Overview

This project implements an attention-based deep learning framework for adaptive Modulation and Coding Scheme (MCS) selection in dynamic wireless environments. The literature survey identified a key research gap: while attention mechanisms (Transformers) have been successfully applied to channel estimation, they have NOT been applied to MCS selection.

**Target Performance**: ≥90% MCS prediction accuracy, matching/exceeding state-of-the-art (Paper 3: 94.3%)

## Tech Stack
- **Primary**: PyTorch 2.0+
- **Optional**: MATLAB for 3GPP TDL/CDL channel generation
- **Environment**: Python 3.9+, CUDA (optional but recommended)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate dataset (Phase 1)
python training/data_generator.py --n_samples 100000 --output data/

# 3. Train attention model (default)
python training/train.py --model attention --config configs/attention.json

# 4. Evaluate all models
python evaluation/compare_baselines.py

# 5. Visualize results
jupyter notebook notebooks/analysis.ipynb
```

## Project Structure

```
wn_project/
├── channel_simulation/     # Rayleigh/Rician fading, OFDM models
│   ├── rayleigh_fading.py
│   ├── channel_models.py   # TDL/CDL (MATLAB or Python)
│   ├── ofdm_system.py
│   └── data_generator.py   # Dataset generation with oracle labeling
├── models/
│   ├── attention_mcs.py    # Primary: Transformer-based model
│   ├── baselines.py        # DNN, CNN, LSTM, CNN+LSTM
│   └── layers.py           # Multi-head attention, positional encoding
├── training/
│   ├── train.py            # Main training script
│   ├── evaluate.py
│   ├── config.py           # Global seed, hyperparameters
│   └── utils.py
├── evaluation/
│   ├── compare_baselines.py
│   ├── metrics.py
│   └── visualization.py    # Attention heatmaps, plots
├── notebooks/
│   └── analysis.ipynb      # Final analysis notebook
├── data/                   # Generated datasets
│   ├── train/
│   ├── val/
│   └── test/
├── results/
│   ├── checkpoints/        # Model weights (.pt files)
│   ├── plots/              # Generated figures
│   └── metrics/            # metrics.json files
└── matlab/                 # Optional MATLAB scripts for TDL/CDL
```

## Key Design Decisions

### 1. Why Attention (Transformer) over LSTM?
- Attention can directly compare any two time steps (LSTM sequential bottleneck)
- Better for outdated CSI scenarios (Paper 9): model learns to focus on informative historical observations
- Fully parallel → faster training/inference

### 2. 15 MCS Classes (4G LTE) vs 29 (5G NR)
- Paper 3 benchmark: 94.3% accuracy on 15-class problem
- Direct comparability to literature
- Smaller classification head → faster inference

### 3. Oracle Labeling via SNR+LUT
- Pre-compute BER lookup table: [SNR × MCS] → BER
- Assign highest MCS satisfying BER ≤ 10⁻³
- ~1000× faster than full link simulation

### 4. Model Size Target: < 2M parameters
- Inference latency target: < 1ms (5G NR deadline)
- Compact architecture: d_model=128-256, n_heads=4-8, n_layers=2-4

## Reproducibility

All experiments use a global random seed for reproducibility:
```python
from training.config import set_global_seed
set_global_seed(42)  # Call this at the start of every script
```

Training results are saved as `results/metrics/{model}_{timestamp}.json` with full hyperparameters and metrics.

## Baseline Models

1. **DNN**: 4-layer fully connected with residuals (512-256-128-64)
2. **CNN**: 1D convolutions over subcarriers + global pooling
3. **LSTM**: 2-layer LSTM (hidden=128)
4. **CNN+LSTM**: 1D conv → LSTM → FC (Paper 2 replication)
5. **Attention** (primary): Multi-head self-attention over temporal window

## Evaluation Metrics

- Primary: MCS prediction accuracy (top-1, top-3)
- Secondary: Throughput ratio (normalized to oracle), inference latency, model size
- Scenarios: AWGN, Rayleigh/Rician, TDL-C, CDL-D, UE speeds 3-120 km/h

## Team Roles (Suggested)

- **Member 1**: Channel simulation & dataset generation
- **Member 2**: Attention model + training infrastructure
- **Member 3**: Baseline models (DNN, CNN, LSTM, CNN+LSTM)
- **Member 4**: Visualization, analysis, report, documentation

## Literature Survey

The complete literature survey (10 papers, 2019-2025) is available in:
- `Literature_Survey.docx` (original)
- `Literature_Survey.md` (converted)

**Key Insight**: Both comprehensive surveys (Papers 7 & 10) explicitly identify **attention mechanisms** as an unexplored but promising direction for MCS selection. This project addresses that gap.

## License

Educational project for Wireless Networks course. Not for commercial use.

---

**Questions?** Refer to the detailed implementation plan at:
`.claude/plans/greedy-finding-tiger.md`