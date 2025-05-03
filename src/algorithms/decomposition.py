import json
import subprocess
from pathlib import Path
import tempfile
from edfio import read_edf
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional, Union

from .decomposition_methods import upper_bound, basic_cBSS

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def decompose_scd(
    data: Union[str, np.ndarray],
    output_dir: Path,
    algorithm_config: Optional[str] = None,
    engine: str = "singularity",
    container: str = "environment/muniverse_scd.sif",
) -> Tuple[Dict, Dict]:
    """
    Run SCD decomposition using container.
    
    Args:
        data: Either a path to input data file (.npy or .edf) or numpy array of EMG data
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image
        cache_dir: Optional directory for caching
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load algorithm config if provided, otherwise use defaults
    if not algorithm_config:
        config_dir = Path(__file__).parent.parent.parent / "configs"
        algorithm_config = config_dir / "scd.json"
        if not algorithm_config.exists():
            raise FileNotFoundError(f"Default SCD config not found at {algorithm_config}")
    
    # Create temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Convert input to .npy file if needed
        if isinstance(data, np.ndarray):
            temp_emg_path = temp_dir / "temp_emg.npy"
            np.save(temp_emg_path, data)
        elif isinstance(data, str):
            data_path = Path(data)
            if data_path.suffix == '.edf':
                # Load EDF and save as .npy
                raw = read_edf(data_path)
                n_channels = raw.num_signals
                emg_data = np.stack([raw.signals[i].data for i in range(n_channels)])
                temp_emg_path = temp_dir / "temp_emg.npy"
                np.save(temp_emg_path, emg_data)
            else:  # .npy file
                temp_emg_path = data_path
        else:
            raise TypeError("data must be either a file path (str) or numpy array")
        
        # Get the absolute path to the script
        current_dir = Path(__file__).parent
        run_script_path = current_dir / "_run_scd.sh"
        script_path = current_dir / "_run_scd.py"
        
        if not run_script_path.exists():
            raise FileNotFoundError(f"Script not found at {run_script_path}")

        # Build container command
        cmd = [
            str(run_script_path),
            engine,
            container,
            str(script_path),
            str(temp_emg_path),
            str(algorithm_config),
            str(output_dir)
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
    data: np.ndarray,
    algorithm_config: Optional[str],
    output_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Run upperbound decomposition.
    
    Args:
        data: EMG data array (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load algorithm config if provided, otherwise use defaults
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)
    else:
        algo_cfg = {}  # Will use defaults
    
    # Initialize and run upperbound
    ub = upper_bound(config=algo_cfg)
    sources, spikes, sil = ub.decompose(data, fsamp=2048)  # TODO: Make sampling frequency configurable
    
    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'silhouette': sil
    }
    
    # Prepare metadata
    metadata = {
        'AlgorithmConfig': algo_cfg,
        'ProcessingInfo': {
            'Method': 'upperbound',
            'NumComponents': len(spikes)
        }
    }
    
    # Save results
    save_decomposition_results(output_dir, results, metadata)
    
    return results, metadata

def decompose_cbss(
    data: np.ndarray,
    algorithm_config: Optional[str],
    output_dir: Path
) -> Tuple[Dict, Dict]:
    """
    Run CBSS decomposition.
    
    Args:
        data: EMG data array (channels x samples)
        algorithm_config: Optional path to algorithm configuration JSON file
        output_dir: Directory to save results
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Load algorithm config if provided, otherwise use defaults
    if algorithm_config:
        algo_cfg = load_config(algorithm_config)
    else:
        algo_cfg = {}  # Will use defaults
    
    # Initialize and run CBSS
    cbss = basic_cBSS(config=algo_cfg)
    sources, spikes, mu_filters = cbss.decompose(data, fsamp=2048)  # TODO: Make sampling frequency configurable
    
    # Prepare results
    results = {
        'sources': sources,
        'spikes': spikes,
        'mu_filters': mu_filters
    }
    
    # Prepare metadata
    metadata = {
        'AlgorithmConfig': algo_cfg,
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