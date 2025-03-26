#!/bin/bash
set -e  # fail on first error

# Run get_data and generate EMG
python scripts/mov2emg.py
