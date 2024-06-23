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

ENV PATH=~/miniconda3/bin:${PATH}

WORKDIR /code

COPY environment.yml /code/

RUN /opt/conda/bin/conda env create -f environment.yml

CMD nvidia-smi