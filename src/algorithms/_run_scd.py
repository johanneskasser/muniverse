# Modified from https://github.com/AgneGris/swarm-contrastive-decomposition.git

import sys
import numpy as np
import torch
from pathlib import Path
import json
import argparse
from typing import Dict, Tuple, Any

from config.structures import set_random_seed, Config
from models.scd import SwarmContrastiveDecomposition
from processing.postprocess import save_results

def train(data_path: str, config_path: str, output_dir: str):
    """Run SCD decomposition on EMG data."""
    # Load and set config
    with open(config_path, 'r') as f:
        alg_config = json.load(f)
    
    # Unpack config into Config dataclass
    config = Config(**alg_config)
    
    # Set random seed
    set_random_seed(seed=42)

    # Load data
    npy_data = np.load(data_path).T
    neural_data = torch.from_numpy(npy_data).to(device=config.device, dtype=torch.float32)
    
    # Apply time window if specified
    if config.end_time == -1:
        neural_data = neural_data[config.start_time * config.sampling_frequency:, :]
    else:
        neural_data = neural_data[
            config.start_time * config.sampling_frequency : 
            config.end_time * config.sampling_frequency, 
            :
        ]

    # Initiate the model and run
    model = SwarmContrastiveDecomposition()
    predicted_timestamps, dictionary = model.run(neural_data, config)

    # Save results: TO-DO convert spikes and predicted sources to suitable format
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decomposition_results.pkl"
    
    save_results(output_path, dictionary)
    print(f"Saved results to {output_path}")
    
    return dictionary, predicted_timestamps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run SCD decomposition on EMG data')
    parser.add_argument('data_path', type=str, help='Path to .npy file containing EMG data')
    parser.add_argument('config_path', type=str, help='Path to algorithm configuration JSON file')
    parser.add_argument('output_dir', type=str, help='Directory to save results')
    
    args = parser.parse_args()
    
    train(args.data_path, args.config_path, args.output_dir)