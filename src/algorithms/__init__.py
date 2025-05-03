"""
Benchmark algorithms for decomposition.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
from ..utils.containers import pull_container, verify_container_engine
from .decomposition import decompose_scd, decompose_upperbound, decompose_cbss

def init():
    """
    Initialize the algorithms module.
    This includes verifying container engines and pulling container images if needed.
    If both Docker and Singularity are available, Singularity will be used by default.

    Returns:
        str: The selected container engine ("docker" or "singularity")
    """
    # Check availability of both engines
    docker_available = verify_container_engine("docker")
    singularity_available = verify_container_engine("singularity")
    
    # Select engine based on availability
    if singularity_available:
        engine = "singularity"
    elif docker_available:
        engine = "docker"
    else:
        raise RuntimeError("No container engine (Docker or Singularity) is available. Please install one first.")
    
    # Get container name (using default)
    container_name = "pranavm19/muniverse:scd"
    
    # Pull container if needed
    pull_container(container_name, engine)
    print(f"[INFO] Algorithms module initialized using {engine}.")
    
    return engine

def decompose_recording(
    input_data: Dict,
    input_config: str,
    algorithm_config: str,
    method: str = "scd",
    output_dir: str = "outputs",
    engine: str = "singularity",
    container: str = None,
    cache_dir: str = None
) -> Tuple[Dict, Dict]:
    """
    Decompose EMG recordings using specified method.
    
    Args:
        input_data: Dictionary containing BIDS data paths and metadata
        input_config: Path to input configuration JSON file
        algorithm_config: Path to algorithm configuration JSON file
        method: Decomposition method to use ("scd", "upperbound", or "cbss")
        output_dir: Directory to save results
        engine: Container engine to use ("docker" or "singularity")
        container: Path to container image (required for SCD method)
        cache_dir: Directory for caching (optional)
    
    Returns:
        Tuple containing:
        - Dictionary with decomposition results
        - Dictionary with processing metadata
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Route to appropriate method
    if method == "scd":
        if container is None:
            raise ValueError("Container path must be provided for SCD method")
        return decompose_scd(
            input_data=input_data,
            input_config=input_config,
            algorithm_config=algorithm_config,
            output_dir=output_dir,
            engine=engine,
            container=container,
            cache_dir=cache_dir
        )
    elif method == "upperbound":
        return decompose_upperbound(
            input_data=input_data,
            input_config=input_config,
            algorithm_config=algorithm_config,
            output_dir=output_dir
        )
    elif method == "cbss":
        return decompose_cbss(
            input_data=input_data,
            input_config=input_config,
            algorithm_config=algorithm_config,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unknown method: {method}. Must be one of: scd, upperbound, cbss") 