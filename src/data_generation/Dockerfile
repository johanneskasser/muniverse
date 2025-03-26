FROM condaforge/mambaforge:latest

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt

# Clone the NeuroMotion repo
RUN git clone https://github.com/pranavm19/NeuroMotion

# Switch to repo directory
WORKDIR /opt/NeuroMotion

# Create the conda environment
RUN mamba env create -f env.yml

# Install extra dependencies into the conda env
RUN mamba run -n NeuroMotion pip install \
    git+https://github.com/shihan-ma/BioMime.git \
    gdown

RUN mamba run -n NeuroMotion bash get_data.sh

# Run the shell script and Python conversion
COPY run_neuromotion_test.sh /opt/NeuroMotion/run_neuromotion_test.sh
RUN chmod +x run_neuromotion_test.sh

ENTRYPOINT ["mamba", "run", "--no-capture-output", "-n", "NeuroMotion", "./run_neuromotion_test.sh"]
