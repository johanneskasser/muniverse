#!/usr/bin/env python3

import sys
from pathlib import Path
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from src.algorithms.decomposition import decompose_upperbound

# TODO: Extend script to handle multiple recordingsw
# TODO: Handle start and end times; sampling frequency; etc. -- config is modified here and then passed to the function
# TODO: Config is passed as a dict, and then saved as a json file in the temp directory along with the npy data

# Input/output paths
data_path = project_root / 'src/data_generation/res/subject_1_EDI_emg.npz'
output_dir = project_root / 'outputs'
data_generation_config = project_root / 'src/data_generation/res/config_used.json'
muap_cache_file = project_root / 'src/data_generation/res/muap_cache/subject_1_EDI_Flexion-Extension_muaps.npy'

# Run decomposition
decompose_upperbound(
    data=np.load(data_path)['emg'],
    data_generation_config=data_generation_config,
    muap_cache_file=muap_cache_file,
    output_dir=Path(output_dir),
    algorithm_config={}
)