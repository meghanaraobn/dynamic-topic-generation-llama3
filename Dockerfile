FROM nvidia/cuda:12.1.0-devel-ubuntu20.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    bzip2 \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# RUN apt install curl -y
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    && bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-py39_4.12.0-Linux-x86_64.sh

# Create the working directory
WORKDIR /code

# Set the environment variable
ENV PATH=~/miniconda3/bin:${PATH}

# Copy the current project folder
COPY environment.yml /code/

# Create the conda environment
RUN /opt/conda/bin/conda env create -f environment.yml

# Add conda activate to .bashrc
RUN echo "source /opt/conda/bin/activate topic_generation_env" >> ~/.bashrc