import json
import os
import subprocess
from pathlib import Path
import tempfile
import shutil
from edfio import Edf, read_edf
import numpy as np
from typing import Dict, Tuple, Any

from .decomposition_methods import upper_bound, basic_cBSS
from ..utils.bidsify_data import load_bids_data

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def prepare_emg_data(edf_path: Path) -> np.ndarray:
    """Load and prepare EMG data from EDF file."""
    # Load EDF file
    raw = read_edf(edf_path)
    n_channels = raw.num_signals
    signals = np.stack(raw.signals[i] for i in range(n_channels))
    return signals

def decompose_scd(
    input_data: Dict,
    input_config: str,
    algorithm_config: str,
    output_dir: Path,
    engine: str = "singularity",
    container: str = None
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.
    
    Args:
        input_data: Dictionary containing BIDS data paths and metadata
        input_config: Path to input configuration JSON file
        algorithm_config: Path to algorithm configuration JSON file
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load configurations
    input_cfg = load_config(input_config)
    algo_cfg = load_config(algorithm_config)
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Prepare EMG data
        emg_data = prepare_emg_data(input_data['emg_path'])

        # TODO: handle setting start and end time, sampling frequency, and cov_var
        
        # Save EMG data as temporary .npy file
        temp_emg_path = temp_dir / "temp_emg.npy"
        np.save(temp_emg_path, emg_data)
        
        # Get the absolute path to the script
        current_dir = Path(__file__).parent
        run_script = current_dir / "_run_scd.sh"
        
        if not run_script.exists():
            raise FileNotFoundError(f"Script not found at {run_script}")
        
        # Build container command
        cmd = [
            str(run_script),
            str(temp_emg_path),
            str(algorithm_config),
            str(output_dir),
            engine,
            str(container)
        ]
        
        # Run container
        try:
            subprocess.run(cmd, check=True, cwd=current_dir)
            print(f"[INFO] Decomposition completed successfully at {output_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Decomposition failed: {e}")
            print(f"[ERROR] Command output: {e.output if hasattr(e, 'output') else 'No output'}")
            print(f"[ERROR] Command stderr: {e.stderr if hasattr(e, 'stderr') else 'No stderr'}")
            raise
        
        # Load results
        results_path = output_dir / "decomposition_results.pkl"
        metadata_path = output_dir / "processing_metadata.json"
        
        print(f"[INFO] Results saved to {results_path}")
        
        return results_path, metadata_path

def decompose_upperbound(
    input_data: Dict,
    input_config: str,
    algorithm_config: str,
    output_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.
    
    Args:
        input_data: Dictionary containing BIDS data paths and metadata
        input_config: Path to input configuration JSON file
        algorithm_config: Path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load configurations
    input_cfg = load_config(input_config)
    algo_cfg = load_config(algorithm_config)
    
    # Prepare EMG data
    emg_data = prepare_emg_data(input_data['emg_path'], input_data['metadata'])
    
    # Initialize and run upperbound
    ub = upper_bound(config=algo_cfg)
    sources, spikes, sil = ub.decompose(emg_data, input_cfg['sampling_frequency'])
    
    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'silhouette': sil
    }
    
    # Prepare metadata
    metadata = {
        'InputConfig': input_cfg,
        'AlgorithmConfig': algo_cfg,
        'InputDataInfo': {
            'EMGPath': str(input_data['emg_path']),
            'Metadata': input_data['metadata']
        },
        'ProcessingInfo': {
            'Method': 'upperbound',
            'NumComponents': len(spikes)
        }
    }
    
    # Save results
    save_decomposition_results(output_dir, results, metadata)
    
    return results, metadata

def decompose_cbss(
    input_data: Dict,
    input_config: str,
    algorithm_config: str,
    output_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.
    
    Args:
        input_data: Dictionary containing BIDS data paths and metadata
        input_config: Path to input configuration JSON file
        algorithm_config: Path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load configurations
    input_cfg = load_config(input_config)
    algo_cfg = load_config(algorithm_config)
    
    # Prepare EMG data
    emg_data = prepare_emg_data(input_data['emg_path'], input_data['metadata'])
    
    # Initialize and run CBSS
    cbss = basic_cBSS(config=algo_cfg)
    sources, spikes, mu_filters = cbss.decompose(emg_data, input_cfg['sampling_frequency'])
    
    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'mu_filters': mu_filters
    }
    
    # Prepare metadata
    metadata = {
        'InputConfig': input_cfg,
        'AlgorithmConfig': algo_cfg,
        'InputDataInfo': {
            'EMGPath': str(input_data['emg_path']),
            'Metadata': input_data['metadata']
        },
        'ProcessingInfo': {
            'Method': 'cbss',
            'NumComponents': len(spikes)
        }
    }
    
    # Save results
    save_decomposition_results(output_dir, results, metadata)
    
    return results, metadata

def save_decomposition_results(output_dir: Path, results: Dict, metadata: Dict):
    """Save decomposition results and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    results_path = output_dir / "decomposition_results.pkl"
    metadata_path = output_dir / "processing_metadata.json"
    
    # Save results
    import pickle
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    # Save metadata
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return results_path, metadata_path 