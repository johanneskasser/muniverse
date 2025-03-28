#!/bin/bash
set -e

# Usage:
# ./run_scd.sh docker|singularity path/to/run_scd.py path/to/input.npy path/to/output_dir

ENGINE=$1
SCRIPT_PATH=$2
DATA_PATH=$3
OUTPUT_DIR=$4

if [ "$ENGINE" == "docker" ]; then
  echo "[INFO] Running with Docker"
  docker run --rm -it \
    --gpus all \
    -v $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -v $(realpath $DATA_PATH):/data/input.npy \
    -v $(realpath $OUTPUT_DIR):/output/ \
    pranavm19/muniverse-test:scd \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py /data/input.npy /output/"
elif [ "$ENGINE" == "singularity" ]; then
  echo "[INFO] Running with Singularity"
  singularity run --nv --cleanenv \
    -B $(realpath $SCRIPT_PATH):/opt/scd/run_scd.py \
    -B $(realpath $DATA_PATH):/data/input.npy \
    -B $(realpath $OUTPUT_DIR):/output/ \
    muniverse-test_scd.sif \
    bash -c "source /opt/mambaforge/etc/profile.d/conda.sh && \
             conda activate decomposition && \
             cd /opt/scd/ && \
             python run_scd.py /data/input.npy /output/"
else
  echo "ERROR: Unknown engine '$ENGINE'. Use 'docker' or 'singularity'."
  exit 1
fi
