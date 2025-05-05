import os
from muniverse.utils import bidsify_data
from pathlib import Path

# Get the absolute path to the outputs directory
outputs_path = os.path.abspath("./outputs")
sidecar_path = os.path.abspath("./bids_metadata/neuromotion.json")

# Get all directories that start with 'run_'
run_dirs = [d for d in os.listdir(outputs_path) if d.startswith('run_') and os.path.isdir(os.path.join(outputs_path, d))]

# Sort the directories to process them in order
run_dirs.sort()
root = '/rds/general/user/pm1222/home/muniverse-demo/'
datasetname = 'neuromotion-dev01'

# Process each run directory
for run_dir in run_dirs[:10]:
    print(f"\nProcessing {run_dir}...")
    data_path = os.path.join(outputs_path, run_dir)
    
    try:
        bidsdata = bidsify_data.neuromotion_to_bids(data_path, sidecar_path, root, datasetname)
        print(f"Successfully converted {run_dir} to BIDS format")
    except Exception as e:
        print(f"Error processing {run_dir}: {str(e)}")

print("\nProcessing complete!") 