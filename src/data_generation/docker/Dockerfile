FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Mambaforge
WORKDIR /opt
RUN wget https://github.com/conda-forge/miniforge/releases/download/23.1.0-2/Mambaforge-Linux-x86_64.sh && \
    bash Mambaforge-Linux-x86_64.sh -b -p /opt/mambaforge && \
    rm Mambaforge-Linux-x86_64.sh
ENV PATH=/opt/mambaforge/bin:$PATH

# Clone the NeuroMotion repo
RUN git clone https://github.com/pranavm19/NeuroMotion

# Switch to repo directory
WORKDIR /opt/NeuroMotion

# Create the conda environment
RUN mamba env create -f env.yml && \
    mamba clean --all -f -y && \
    rm -rf ~/.cache ~/.conda

# Install extra dependencies into the conda env
RUN mamba run -n NeuroMotion pip install \
    git+https://github.com/shihan-ma/BioMime.git \
    gdown

RUN mamba run -n NeuroMotion bash get_data.sh

# Activate env by default
# ENV CONDA_DEFAULT_ENV=NeuroMotion
# ENV PATH=/opt/mambaforge/envs/NeuroMotion/bin:$PATH

CMD ["/bin/bash"]
