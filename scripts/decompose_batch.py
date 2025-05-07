#!/usr/bin/env python3

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the decomposition function
from src.algorithms.decomposition import decompose_upperbound

def find_muap_cache_file(cache_dir, subject_id, muscle, metadata_file=None):
    """Find the appropriate MUAP cache file for the given subject and muscle."""
    cache_dir = Path(cache_dir)
    movement_type = None
    
    # Try to extract movement type from metadata if available
    if metadata_file and metadata_file.exists():
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                if 'movement_type' in metadata:
                    movement_type = metadata['movement_type']
        except (json.JSONDecodeError, OSError):
            pass
    
    # If we have a movement type, look for an exact match
    if movement_type:
        cache_file = cache_dir / f"subject_{subject_id}_{muscle}_{movement_type}_muaps.npy"
        if cache_file.exists():
            return cache_file
    
    # If no exact match or movement type not found, try common movement types
    common_types = ["Flexion-Extension", "Radial-Ulnar-deviation"]
    for movement in common_types:
        cache_file = cache_dir / f"subject_{subject_id}_{muscle}_{movement}_muaps.npy"
        if cache_file.exists():
            return cache_file
            
    # Last resort: look for any matching cache file
    pattern = f"subject_{subject_id}_{muscle}_*_muaps.npy"
    matches = list(cache_dir.glob(pattern))
    if matches:
        return matches[0]  # Return the first match
            
    return None

def process_recordings(input_dir, output_dir, cache_dir=None):
    """Process all recordings in the input directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir) if cache_dir else input_dir / 'cache'
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Find all run directories
    run_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
    
    if not run_dirs:
        print(f"No run directories found in {input_dir}")
        return
        
    print(f"Found {len(run_dirs)} run directories")
    
    # Process each run directory
    for run_dir in run_dirs:
        print(f"\nProcessing directory: {run_dir}")
        
        # Find all EMG files in the directory
        emg_files = list(run_dir.glob("*_emg.npz"))
        
        for emg_file in emg_files:
            # Extract subject ID and muscle from filename
            filename_parts = emg_file.stem.split('_')
            if len(filename_parts) < 3 or filename_parts[-1] != "emg":
                print(f"Warning: Unexpected filename format: {emg_file.name}. Skipping.")
                continue
                
            subject_id = filename_parts[1]
            muscle = filename_parts[2]
            
            # Find corresponding config file
            config_file = run_dir / f"subject_{subject_id}_{muscle}_config_used.json"
            if not config_file.exists():
                print(f"Warning: Config file not found for {emg_file.name}. Skipping.")
                continue
                
            # Find corresponding metadata file
            metadata_file = run_dir / f"subject_{subject_id}_{muscle}_metadata.json"
            
            # Find corresponding MUAP cache file
            muap_cache_file = find_muap_cache_file(cache_dir, subject_id, muscle, metadata_file)
            if not muap_cache_file:
                print(f"Warning: No MUAP cache file found for subject_{subject_id}_{muscle}. Skipping.")
                continue
                
            # Create output directory for this recording
            recording_output_dir = output_dir / f"decomposed_{run_dir.name}_{subject_id}_{muscle}"
            recording_output_dir.mkdir(exist_ok=True)
            
            print(f"Processing: {emg_file.name}")
            print(f"  - EMG file: {emg_file}")
            print(f"  - Config file: {config_file}")
            print(f"  - MUAP cache file: {muap_cache_file}")
            print(f"  - Output directory: {recording_output_dir}")
            
            try:
                # Load EMG data
                emg_data = np.load(emg_file)['emg']
                
                # Run decomposition
                decompose_upperbound(
                    data=emg_data,
                    data_generation_config=config_file,
                    muap_cache_file=muap_cache_file,
                    output_dir=recording_output_dir,
                    algorithm_config={}
                )
                
                print(f"Successfully processed {emg_file.name}")
            except Exception as e:
                print(f"Error processing {emg_file.name}: {str(e)}")
    
    print("All processing complete!")

def main():
    parser = argparse.ArgumentParser(description='EMG signal decomposition processor')
    parser.add_argument('input_dir', help='Directory containing recording runs')
    parser.add_argument('output_dir', help='Directory to store decomposition results')
    parser.add_argument('--cache-dir', help='Directory containing MUAP cache files (defaults to input_dir/cache)')
    
    args = parser.parse_args()
    process_recordings(args.input_dir, args.output_dir, args.cache_dir)

if __name__ == "__main__":
    main()