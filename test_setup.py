#!/usr/bin/env python3
"""
Test script to verify all modules import correctly.
Run this after setting up the project to ensure no import errors.
"""

import sys
import traceback

print("="*60)
print("Testing Project Module Imports")
print("="*60)

tests = [
    ("training.config", "from training.config import set_global_seed, GLOBAL_SEED, DataConfig"),
    ("training.utils", "from training.utils import count_parameters, save_metrics"),
    ("models.layers", "from models.layers import TemporalAttentionTransformer, LearnablePositionalEncoding"),
    ("models.attention_mcs", "from models.attention_mcs import AttentionMCSModel"),
    ("models.baselines", "from models.baselines import DNNMCSModel, CNNMCSModel, LSTMMCSModel, CNNLSTMMCSModel, get_model"),
    ("channel_simulation.rayleigh_fading", "from channel_simulation.rayleigh_fading import generate_rayleigh_clarke"),
    ("channel_simulation.ofdm_system", "from channel_simulation.ofdm_system import OFDMConfig"),
    ("channel_simulation.channel_models", "from channel_simulation.channel_models import ChannelSimulator, compute_mcs_ber_lut"),
    ("training.data_generator", "from training.data_generator import generate_dataset, load_split"),
]

failed = []
passed = []

for module, import_stmt in tests:
    try:
        exec(import_stmt, globals())
        print(f"[PASS] {module}")
        passed.append(module)
    except Exception as e:
        print(f"[FAIL] {module}: {e}")
        failed.append((module, str(e)))

print("\n" + "="*60)
print("Import Test Complete")
print("="*60)
print(f"Passed: {len(passed)}/{len(tests)}")
print(f"Failed: {len(failed)}/{len(tests)}")

if failed:
    print("\nFailed imports:")
    for module, error in failed:
        print(f"  - {module}: {error}")
    sys.exit(1)
else:
    print("\n✓ All modules imported successfully!")
    sys.exit(0)